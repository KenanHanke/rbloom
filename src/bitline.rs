//! This is a primitive BitVec-like structure that acts as a container around
//! all the bit manipulation. Its bytes either live in an owned `Box<[u8]>` or
//! in the memory of a held Python buffer, after the `HEADER_SIZE` byte header.
//! Indexing is done using u64 to avoid address space issues on 32-bit
//! systems, which would otherwise limit the size to 2^32 bits (512MB).
//! Using bytes for the backing store simplifies file I/O as well as file
//! portability across systems, and the performance is equivalent to
//! using usize, even though the latter is arguably more elegant.
//!
//! All methods take `&self`, so it can live in a frozen pyclass shared
//! between threads. The storage sits behind an `RwLock`: reads take it
//! shared, and writes and releasing the buffer take it exclusively.
//! Owned bytes are accessed as plain `[u8]`. Buffer memory can be written
//! through other views at any time (another filter backed by the same
//! object, Python code, other threads), so it is only ever accessed
//! atomically.
//!
//! To rule out deadlocks, no lock is held while calling into Python, and
//! operations on two BitLines take their locks in address order.

use crate::buffer::{HeldBuffer, ReadByte};
use crate::HEADER_SIZE;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::sync::RwLockExt;
use pyo3::{PyTraverseError, PyVisit};
use std::fs::File;
use std::io::{Read, Write};
use std::cmp::Ordering;
use std::mem;
use std::ptr;
use std::sync::atomic::{AtomicU8, Ordering::Relaxed};
use std::sync::{PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard, TryLockError};

#[inline(always)]
fn bit_idx(idx: u64) -> Option<(usize, u32)> {
    let (q, r) = (idx / 8, idx % 8);
    Some((q.try_into().ok()?, r.try_into().ok()?))
}

/// Returns the number of bytes needed to hold `size_in_bits` bits.
pub fn byte_len(size_in_bits: u64) -> PyResult<usize> {
    let Some((q, r)) = bit_idx(size_in_bits) else {
        return Err(PyValueError::new_err("too many bits"));
    };
    Ok(if r == 0 { q } else { q + 1 })
}

fn released_err() -> PyErr {
    PyValueError::new_err("operation forbidden on released Bloom object")
}

enum Storage {
    Owned(Box<[u8]>),
    Buffer(HeldBuffer),
    Released,
}

// A panic while a lock is held can't leave the bits in an invalid state, so poisoning is
// ignored.
fn lock_for_reading<'a>(
    py: Python<'_>,
    lock: &'a RwLock<Storage>,
) -> RwLockReadGuard<'a, Storage> {
    lock.read_py_attached(py)
        .unwrap_or_else(PoisonError::into_inner)
}

fn lock_for_writing<'a>(
    py: Python<'_>,
    lock: &'a RwLock<Storage>,
) -> RwLockWriteGuard<'a, Storage> {
    lock.write_py_attached(py)
        .unwrap_or_else(PoisonError::into_inner)
}

fn lock_order(lhs: &RwLock<Storage>, rhs: &RwLock<Storage>) -> Ordering {
    ptr::from_ref(lhs).cmp(&ptr::from_ref(rhs))
}

#[derive(Clone, Copy)]
enum Bytes<'a> {
    Owned(&'a [u8]),
    Shared(&'a [ReadByte]),
}

enum BytesMut<'a> {
    Owned(&'a mut [u8]),
    Shared(&'a [AtomicU8]),
}

impl HeldBuffer {
    fn filter_bytes(&self) -> &[ReadByte] {
        &self.readable_bytes()[HEADER_SIZE..]
    }

    fn filter_bytes_mut(&self) -> PyResult<&[AtomicU8]> {
        Ok(&self.writable_bytes()?[HEADER_SIZE..])
    }
}

impl Storage {
    fn bytes(&self) -> PyResult<Bytes<'_>> {
        match self {
            Storage::Owned(bytes) => Ok(Bytes::Owned(bytes)),
            Storage::Buffer(buffer) => Ok(Bytes::Shared(buffer.filter_bytes())),
            Storage::Released => Err(released_err()),
        }
    }

    fn bytes_mut(&mut self) -> PyResult<BytesMut<'_>> {
        match self {
            Storage::Owned(bytes) => Ok(BytesMut::Owned(bytes)),
            Storage::Buffer(buffer) => Ok(BytesMut::Shared(buffer.filter_bytes_mut()?)),
            Storage::Released => Err(released_err()),
        }
    }
}

pub trait Load {
    fn get(&self) -> u8;
}

impl Load for u8 {
    #[inline(always)]
    fn get(&self) -> u8 {
        *self
    }
}

impl Load for ReadByte {
    #[inline(always)]
    fn get(&self) -> u8 {
        ReadByte::get(self)
    }
}

trait Modify {
    fn or(self, mask: u8);
    fn and(self, mask: u8);
}

impl Modify for &mut u8 {
    #[inline(always)]
    fn or(self, mask: u8) {
        *self |= mask;
    }

    #[inline(always)]
    fn and(self, mask: u8) {
        *self &= mask;
    }
}

// Checking the current value first skips the atomic read-modify-write when
// nothing would change, which is the common case for a filter.
impl Modify for &AtomicU8 {
    #[inline(always)]
    fn or(self, mask: u8) {
        let current = self.load(Relaxed);
        if current | mask != current {
            self.fetch_or(mask, Relaxed);
        }
    }

    #[inline(always)]
    fn and(self, mask: u8) {
        let current = self.load(Relaxed);
        if current & mask != current {
            self.fetch_and(mask, Relaxed);
        }
    }
}

enum BitOp {
    Or,
    And,
}

pub struct BitLine {
    storage: RwLock<Storage>,
    len: usize,
}

impl BitLine {
    pub fn new(size_in_bits: u64) -> PyResult<Self> {
        Ok(Self::owned(vec![0; byte_len(size_in_bits)?]))
    }

    fn owned(bytes: Vec<u8>) -> Self {
        Self {
            len: bytes.len(),
            storage: RwLock::new(Storage::Owned(bytes.into_boxed_slice())),
        }
    }

    /// Uses the bytes of `buffer` after the header as the bits, without clearing them.
    ///
    /// # Panics
    ///
    /// Panics if `buffer` is shorter than the header.
    pub fn from_buffer(buffer: HeldBuffer) -> Self {
        let len = buffer.filter_bytes().len();
        Self {
            storage: RwLock::new(Storage::Buffer(buffer)),
            len,
        }
    }

    /// Returns an owned BitLine of the same size with no bits set.
    pub fn zeroed_like(&self) -> Self {
        Self::owned(vec![0; self.len])
    }

    /// Returns an owned copy of the current bits.
    pub fn to_owned(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self::owned(self.to_vec(py)?))
    }

    /// Releases the buffer the bits are stored in, after which every access fails. Does
    /// nothing for owned bits.
    pub fn release(&self, py: Python<'_>) {
        let storage = {
            let mut storage = lock_for_writing(py, &self.storage);
            match &*storage {
                Storage::Buffer(_) => mem::replace(&mut *storage, Storage::Released),
                Storage::Owned(_) | Storage::Released => return,
            }
        };
        // Releasing the buffer can run Python code, so it happens after unlocking.
        drop(storage);
    }

    pub fn is_released(&self, py: Python<'_>) -> bool {
        match &*lock_for_reading(py, &self.storage) {
            Storage::Released => true,
            Storage::Owned(_) | Storage::Buffer(_) => false,
        }
    }

    pub fn readonly(&self, py: Python<'_>) -> PyResult<bool> {
        match &*lock_for_reading(py, &self.storage) {
            Storage::Owned(_) => Ok(false),
            Storage::Buffer(buffer) => Ok(buffer.readonly()),
            Storage::Released => Err(released_err()),
        }
    }

    /// Visits the object exporting the buffer, if any, for `__traverse__`.
    pub fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        // The GC can run while another thread holds the lock, possibly blocked, so it must
        // not wait for it. Not visiting the exporter only makes the GC treat it as reachable.
        let storage = match self.storage.try_read() {
            Ok(storage) => storage,
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(TryLockError::WouldBlock) => return Ok(()),
        };
        match &*storage {
            Storage::Buffer(buffer) => buffer.traverse(visit),
            Storage::Owned(_) | Storage::Released => Ok(()),
        }
    }

    pub fn len_bytes(&self) -> usize {
        self.len
    }

    /// Returns the number of bits in the BitLine
    pub fn len(&self) -> u64 {
        self.len as u64 * 8
    }

    fn with_bytes<R>(&self, py: Python<'_>, f: impl FnOnce(Bytes<'_>) -> R) -> PyResult<R> {
        Ok(f(lock_for_reading(py, &self.storage).bytes()?))
    }

    fn with_bytes_mut<R>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(BytesMut<'_>) -> R,
    ) -> PyResult<R> {
        Ok(f(lock_for_writing(py, &self.storage).bytes_mut()?))
    }

    fn with_pair<R>(
        &self,
        py: Python<'_>,
        other: &BitLine,
        f: impl FnOnce(Bytes<'_>, Bytes<'_>) -> R,
    ) -> PyResult<R> {
        let (lhs, rhs) = match lock_order(&self.storage, &other.storage) {
            Ordering::Less => {
                let lhs = lock_for_reading(py, &self.storage);
                (lhs, lock_for_reading(py, &other.storage))
            }
            Ordering::Greater => {
                let rhs = lock_for_reading(py, &other.storage);
                (lock_for_reading(py, &self.storage), rhs)
            }
            Ordering::Equal => {
                let storage = lock_for_reading(py, &self.storage);
                let bytes = storage.bytes()?;
                return Ok(f(bytes, bytes));
            }
        };
        Ok(f(lhs.bytes()?, rhs.bytes()?))
    }

    /// Make sure `other` is not `self` when calling this!
    fn with_pair_mut<R>(
        &self,
        py: Python<'_>,
        other: &BitLine,
        f: impl FnOnce(BytesMut<'_>, Bytes<'_>) -> R,
    ) -> PyResult<R> {
        let (mut lhs, rhs) = match lock_order(&self.storage, &other.storage) {
            Ordering::Less => {
                let lhs = lock_for_writing(py, &self.storage);
                (lhs, lock_for_reading(py, &other.storage))
            }
            Ordering::Greater => {
                let rhs = lock_for_reading(py, &other.storage);
                (lock_for_writing(py, &self.storage), rhs)
            }
            Ordering::Equal => panic!("with_pair_mut called with self as other"),
        };
        Ok(f(lhs.bytes_mut()?, rhs.bytes()?))
    }

    /// Make sure that every index is less than len when calling this!
    #[inline]
    pub fn set_all(
        &self,
        py: Python<'_>,
        indexes: impl IntoIterator<Item = u64>,
    ) -> PyResult<()> {
        // This and `all_set` are on the per-item path, so they match on the storage
        // directly instead of going through `with_bytes_mut`, and are `#[inline]`: without
        // both, owned filters' `add` and `__contains__` get about 10% slower.
        match &mut *lock_for_writing(py, &self.storage) {
            Storage::Owned(bytes) => {
                for index in indexes {
                    let (idx, offset) = bit_idx(index).unwrap();
                    bytes[idx].or(1 << offset);
                }
                Ok(())
            }
            Storage::Buffer(buffer) => {
                let bytes = buffer.filter_bytes_mut()?;
                for index in indexes {
                    let (idx, offset) = bit_idx(index).unwrap();
                    bytes[idx].or(1 << offset);
                }
                Ok(())
            }
            Storage::Released => Err(released_err()),
        }
    }

    /// Make sure that every index is less than len when calling this!
    #[inline]
    pub fn all_set(
        &self,
        py: Python<'_>,
        indexes: impl IntoIterator<Item = u64>,
    ) -> PyResult<bool> {
        match &*lock_for_reading(py, &self.storage) {
            Storage::Owned(bytes) => Ok(all_set(bytes, indexes)),
            Storage::Buffer(buffer) => Ok(all_set(buffer.filter_bytes(), indexes)),
            Storage::Released => Err(released_err()),
        }
    }

    pub fn clear(&self, py: Python<'_>) -> PyResult<()> {
        self.with_bytes_mut(py, |bytes| match bytes {
            BytesMut::Owned(bytes) => bytes.fill(0),
            BytesMut::Shared(bytes) => {
                for byte in bytes {
                    byte.store(0, Relaxed);
                }
            }
        })
    }

    pub fn count_ones_sum(&self, py: Python<'_>) -> PyResult<u64> {
        self.with_bytes(py, |bytes| match bytes {
            Bytes::Owned(bytes) => count_ones_sum(bytes),
            Bytes::Shared(bytes) => count_ones_sum(bytes),
        })
    }

    pub fn is_empty(&self, py: Python<'_>) -> PyResult<bool> {
        self.with_bytes(py, |bytes| match bytes {
            Bytes::Owned(bytes) => is_empty(bytes),
            Bytes::Shared(bytes) => is_empty(bytes),
        })
    }

    pub fn equals(&self, py: Python<'_>, other: &BitLine) -> PyResult<bool> {
        self.with_pair(py, other, |lhs, rhs| match (lhs, rhs) {
            (Bytes::Owned(lhs), Bytes::Owned(rhs)) => lhs == rhs,
            (Bytes::Owned(lhs), Bytes::Shared(rhs)) => all_pairs(lhs, rhs, |l, r| l == r),
            (Bytes::Shared(lhs), Bytes::Owned(rhs)) => all_pairs(lhs, rhs, |l, r| l == r),
            (Bytes::Shared(lhs), Bytes::Shared(rhs)) => all_pairs(lhs, rhs, |l, r| l == r),
        })
    }

    pub fn is_subset(&self, py: Python<'_>, other: &BitLine) -> PyResult<bool> {
        self.all_pairs(py, other, |lhs, rhs| (lhs | rhs) == rhs)
    }

    pub fn is_strict_subset(&self, py: Python<'_>, other: &BitLine) -> PyResult<bool> {
        let mut is_equal = true;
        let is_subset = self.all_pairs(py, other, |lhs, rhs| {
            is_equal &= lhs == rhs;
            (lhs | rhs) == rhs
        })?;
        Ok(is_subset && !is_equal)
    }

    pub fn or_assign(&self, py: Python<'_>, other: &BitLine) -> PyResult<()> {
        self.assign_pairs(py, other, BitOp::Or)
    }

    pub fn and_assign(&self, py: Python<'_>, other: &BitLine) -> PyResult<()> {
        self.assign_pairs(py, other, BitOp::And)
    }

    fn all_pairs(
        &self,
        py: Python<'_>,
        other: &BitLine,
        f: impl FnMut(u8, u8) -> bool,
    ) -> PyResult<bool> {
        self.with_pair(py, other, |lhs, rhs| match (lhs, rhs) {
            (Bytes::Owned(lhs), Bytes::Owned(rhs)) => all_pairs(lhs, rhs, f),
            (Bytes::Owned(lhs), Bytes::Shared(rhs)) => all_pairs(lhs, rhs, f),
            (Bytes::Shared(lhs), Bytes::Owned(rhs)) => all_pairs(lhs, rhs, f),
            (Bytes::Shared(lhs), Bytes::Shared(rhs)) => all_pairs(lhs, rhs, f),
        })
    }

    fn assign_pairs(&self, py: Python<'_>, other: &BitLine, op: BitOp) -> PyResult<()> {
        if ptr::eq(self, other) {
            // x | x and x & x are both x, but a read-only or released filter still has to
            // fail like it would for any other operand.
            return self.with_bytes_mut(py, |_| ());
        }
        self.with_pair_mut(py, other, |lhs, rhs| match (lhs, rhs) {
            (BytesMut::Owned(lhs), Bytes::Owned(rhs)) => assign_pairs(lhs, rhs, op),
            (BytesMut::Owned(lhs), Bytes::Shared(rhs)) => assign_pairs(lhs, rhs, op),
            (BytesMut::Shared(lhs), Bytes::Owned(rhs)) => assign_pairs(lhs, rhs, op),
            (BytesMut::Shared(lhs), Bytes::Shared(rhs)) => assign_pairs(lhs, rhs, op),
        })
    }

    /// Reads the given file from the current position to the end and
    /// returns a BitLine containing the data.
    pub fn load(file: &mut File) -> PyResult<Self> {
        let mut bits = Vec::new();
        file.read_to_end(&mut bits)?;
        Ok(Self::owned(bits))
    }

    /// Given the provided [u8]
    /// returns a BitLine containing the data.
    pub fn load_bytes(bytes: &[u8]) -> PyResult<Self> {
        Ok(Self::owned(bytes.to_vec()))
    }

    /// Writes the BitLine to the given file from the current position.
    pub fn save(&self, py: Python<'_>, file: &mut File) -> PyResult<()> {
        self.with_bytes(py, |bytes| match bytes {
            Bytes::Owned(bytes) => file.write_all(bytes),
            Bytes::Shared(bytes) => {
                let mut copy = vec![0; bytes.len()];
                copy_bytes(bytes, &mut copy);
                file.write_all(&copy)
            }
        })??;
        Ok(())
    }

    /// Copies the bits into `dest`, which must be `len_bytes()` long.
    pub fn copy_into(&self, py: Python<'_>, dest: &mut [u8]) -> PyResult<()> {
        self.with_bytes(py, |bytes| match bytes {
            Bytes::Owned(bytes) => dest.copy_from_slice(bytes),
            Bytes::Shared(bytes) => copy_bytes(bytes, dest),
        })
    }

    fn to_vec(&self, py: Python<'_>) -> PyResult<Vec<u8>> {
        let mut result = vec![0; self.len];
        self.copy_into(py, &mut result)?;
        Ok(result)
    }
}

pub fn copy_bytes<L: Load>(src: &[L], dest: &mut [u8]) {
    for (dest, byte) in dest.iter_mut().zip(src) {
        *dest = byte.get();
    }
}

fn all_set<L: Load>(bytes: &[L], indexes: impl IntoIterator<Item = u64>) -> bool {
    for index in indexes {
        let (idx, offset) = bit_idx(index).unwrap();
        if bytes[idx].get() & (1 << offset) == 0 {
            return false;
        }
    }
    true
}

fn count_ones_sum<L: Load>(bytes: &[L]) -> u64 {
    let mut sum = 0;
    for byte in bytes {
        sum += byte.get().count_ones() as u64;
    }
    sum
}

fn is_empty<L: Load>(bytes: &[L]) -> bool {
    for byte in bytes {
        if byte.get() != 0 {
            return false;
        }
    }
    true
}

fn all_pairs<L: Load, R: Load>(
    lhs: &[L],
    rhs: &[R],
    mut f: impl FnMut(u8, u8) -> bool,
) -> bool {
    for (lhs, rhs) in lhs.iter().zip(rhs) {
        if !f(lhs.get(), rhs.get()) {
            return false;
        }
    }
    true
}

fn assign_pairs<L, R: Load>(lhs: L, rhs: &[R], op: BitOp)
where
    L: IntoIterator<Item: Modify>,
{
    match op {
        BitOp::Or => {
            for (lhs, rhs) in lhs.into_iter().zip(rhs) {
                lhs.or(rhs.get());
            }
        }
        BitOp::And => {
            for (lhs, rhs) in lhs.into_iter().zip(rhs) {
                lhs.and(rhs.get());
            }
        }
    }
}
