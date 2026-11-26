//! An owned buffer export, held until it's dropped, like a `memoryview`.
//!
//! This is similar to pyo3's `PyUntypedBuffer`, with two differences rbloom needs:
//!
//! - The request flags are chosen by the caller, so a buffer can be requested writable only when
//!   it will be written.
//! - It can be visited by `__traverse__`, like `memoryview` does. Otherwise the reference to the
//!   exporting object that every `Py_buffer` holds is hidden from the garbage collector, which
//!   then can't collect reference cycles that go through the exporter.

use pyo3::exceptions::PyBufferError;
use pyo3::ffi::Py_buffer;
use pyo3::prelude::*;
use pyo3::{ffi, PyErr, PyTraverseError, PyVisit};
use std::ffi::c_int;
use std::pin::Pin;
use std::ptr;
use std::slice;
use std::sync::atomic::{AtomicU8, Ordering::Relaxed};

/// Flags sent to the exporter when requesting a buffer.
pub struct PyBufferRequest(c_int);

impl PyBufferRequest {
    /// Creates a writable C-contiguous request (`PyBUF_CONTIG`).
    pub const fn contig() -> Self {
        Self(ffi::PyBUF_CONTIG)
    }

    /// Creates a C-contiguous request that also accepts read-only buffers (`PyBUF_CONTIG_RO`).
    pub const fn contig_ro() -> Self {
        Self(ffi::PyBUF_CONTIG_RO)
    }
}

/// A byte of buffer memory that is only ever read, with relaxed atomic loads.
///
/// Buffers may be read-only memory (for example a read-only `mmap`). Any atomic write to it, or a
/// non-relaxed load, is undefined behavior; relaxed `AtomicU8` loads are not, on every
/// architecture rbloom is built for. Exposing only relaxed loads makes it impossible to write
/// through this type.
#[repr(transparent)]
pub struct ReadByte(AtomicU8);

impl ReadByte {
    #[inline(always)]
    pub fn get(&self) -> u8 {
        self.0.load(Relaxed)
    }
}

/// A buffer exported by a Python object, released when dropped.
///
/// While it is held, the exporter keeps the memory at the same address and size: a `bytearray`
/// can't be resized and an `mmap` can't be closed.
pub struct HeldBuffer {
    // Exporters may make the Py_buffer self-referential, so it must not move.
    raw: Pin<Box<Py_buffer>>,
}

// SAFETY: the buffer memory can only be accessed through atomics (`readable_bytes` and
// `writable_bytes`), and the buffer is only released while attached to the interpreter.
unsafe impl Send for HeldBuffer {}
unsafe impl Sync for HeldBuffer {}

impl HeldBuffer {
    /// Acquires a buffer from `obj` with the given flags.
    ///
    /// # Errors
    ///
    /// Returns the exporter's error if it cannot satisfy `flags`, for example a `BufferError`
    /// when a writable buffer is requested from `bytes`, and a `BufferError` if the exporter
    /// returns a zero-dimensional buffer.
    pub fn get(obj: &Bound<'_, PyAny>, flags: PyBufferRequest) -> PyResult<Self> {
        let mut raw = Box::new(Py_buffer::new());
        let status =
            unsafe { ffi::PyObject_GetBuffer(obj.as_ptr(), Box::as_mut_ptr(&mut raw), flags.0) };
        if status < 0 {
            return Err(PyErr::fetch(obj.py()));
        }
        let buffer = Self {
            raw: Box::into_pin(raw),
        };
        if buffer.raw.ndim == 0 {
            return Err(PyBufferError::new_err("scalar buffers are not supported"));
        }
        Ok(buffer)
    }

    /// Returns the length of the buffer memory in bytes.
    pub fn len_bytes(&self) -> usize {
        self.raw.len as usize
    }

    /// Returns the buffer memory for reading.
    pub fn readable_bytes(&self) -> &[ReadByte] {
        // SAFETY: ReadByte only performs relaxed atomic loads, which are sound even if the memory
        // is read-only or written through other views.
        unsafe { self.bytes() }
    }

    /// Returns the buffer memory for writing.
    ///
    /// # Errors
    ///
    /// Returns a `BufferError` if the buffer is read-only.
    pub fn writable_bytes(&self) -> PyResult<&[AtomicU8]> {
        if self.readonly() {
            return Err(PyBufferError::new_err("buffer is read-only"));
        }
        // SAFETY: the buffer is writable, and atomic accesses keep concurrent writes through
        // other views free of Rust-level data races.
        Ok(unsafe { self.bytes() })
    }

    /// Returns the buffer memory as a slice of `T`.
    ///
    /// Other views of the same memory (from Python code, other threads, or other extensions) may
    /// read and write it at any time, so it must never be accessed as `[u8]`.
    ///
    /// # Safety
    ///
    /// `T` must have the layout of `u8`, and every access through the returned slice must be
    /// sound for memory that other views may write at any time, and that may be read-only.
    unsafe fn bytes<T>(&self) -> &[T] {
        let len = self.len_bytes();
        if len == 0 {
            return &[];
        }
        // SAFETY: a held buffer's memory stays valid, at the same address and size, until the
        // buffer is dropped, which the borrow of `self` prevents. The caller guarantees the
        // accesses are sound.
        unsafe { slice::from_raw_parts(self.raw.buf.cast::<T>(), len) }
    }

    /// Returns whether the memory must not be written.
    pub fn readonly(&self) -> bool {
        self.raw.readonly != 0
    }

    /// Visits the object exporting the buffer, for `__traverse__`.
    pub fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        // Fails the build if a pyo3 update changes `Py`'s size or alignment, which `traverse` relies on.
        const _: () = {
            assert!(size_of::<Option<Py<PyAny>>>() == size_of::<*mut ffi::PyObject>());
            assert!(align_of::<Option<Py<PyAny>>>() == align_of::<*mut ffi::PyObject>());
        };

        // pyo3 offers no way to visit a raw pointer without a `Python` token, which
        // `__traverse__` doesn't get. Instead, this relies on `Py<T>` being a
        // `#[repr(transparent)]` `NonNull<ffi::PyObject>`, which pyo3 doesn't document as stable
        // (it relies on it internally, in `PyBorrowedUnbound`). `Option<Py<PyAny>>` then has the
        // layout of a nullable `*mut ffi::PyObject`.
        // SAFETY: `raw.obj` is null or a valid object, owned by the Py_buffer for as long as
        // `self` is borrowed. Visiting it doesn't take or drop a reference.
        let exporter = unsafe { &*ptr::from_ref(&self.raw.obj).cast::<Option<Py<PyAny>>>() };
        visit.call(exporter)
    }
}

impl Drop for HeldBuffer {
    fn drop(&mut self) {
        let _ = Python::try_attach(|_| {
            // SAFETY: the buffer was acquired by PyObject_GetBuffer and is released exactly once,
            // here.
            unsafe { ffi::PyBuffer_Release(&mut *self.raw) };
        });
        // If attaching failed, the interpreter has been finalized and the memory is gone with it.
    }
}
