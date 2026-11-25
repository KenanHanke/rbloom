use bitline::BitLine;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyType;
use pyo3::{basic::CompareOp, types::PyBytes, types::PyTuple, PyTraverseError, PyVisit};
use std::fs::File;
use std::io::{Read, Write};
use std::mem;
use std::path::PathBuf;
use std::sync::atomic::Ordering::Relaxed;

mod bitline;
mod buffer;

use buffer::{HeldBuffer, PyBufferRequest};

/// Size of the header that precedes the bits in saved filters and in buffers: `k` as a
/// little-endian u64.
const HEADER_SIZE: usize = mem::size_of::<u64>();

#[pyclass(module = "rbloom", frozen)]
struct Bloom {
    filter: BitLine,
    k: u64, // Number of hash functions (implemented via a LCG that uses
    // the original hash as a seed)
    hash_func: Option<Py<PyAny>>,
}

struct FilterParams {
    size_in_bits: u64,
    k: u64,
}

impl FilterParams {
    fn new(expected_items: u64, false_positive_rate: f64) -> PyResult<Self> {
        if false_positive_rate <= 0.0 || false_positive_rate >= 1.0 {
            return Err(PyValueError::new_err(
                "false_positive_rate must be between 0 and 1",
            ));
        }
        if expected_items == 0 {
            return Err(PyValueError::new_err(
                "expected_items must be greater than 0",
            ));
        }
        let size_in_bits =
            -(expected_items as f64) * false_positive_rate.ln() / 2.0f64.ln().powi(2);
        let k = (size_in_bits / expected_items as f64) * 2.0f64.ln();
        Ok(Self {
            size_in_bits: size_in_bits as u64,
            k: k as u64,
        })
    }
}

#[pymethods]
impl Bloom {
    #[new]
    #[pyo3(signature = (expected_items, false_positive_rate, hash_func = None))]
    fn new(
        expected_items: u64,
        false_positive_rate: f64,
        hash_func: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let FilterParams { size_in_bits, k } =
            FilterParams::new(expected_items, false_positive_rate)?;
        Ok(Bloom {
            filter: BitLine::new(size_in_bits)?,
            k,
            hash_func: custom_hash_func(hash_func)?,
        })
    }

    /// Create an empty filter stored in buffer, laid out exactly like a file written by save,
    /// holding the buffer until released. See "Buffer-backed filters" section in the README
    #[classmethod]
    #[pyo3(signature = (buffer, expected_items, false_positive_rate, hash_func = None))]
    fn init_buffer(
        cls: &Bound<'_, PyType>,
        buffer: &Bound<'_, PyAny>,
        expected_items: u64,
        false_positive_rate: f64,
        hash_func: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Bloom> {
        let FilterParams { size_in_bits, k } =
            FilterParams::new(expected_items, false_positive_rate)?;
        let hash_func = custom_hash_func(hash_func)?;
        let len = buffer_len(size_in_bits)?;
        let buffer = HeldBuffer::get(buffer, PyBufferRequest::contig())?;
        write_buffer_header(&buffer, k, len)?;
        let filter = BitLine::from_buffer(buffer);
        filter.clear(cls.py())?;
        Ok(Bloom {
            filter,
            k,
            hash_func,
        })
    }

    /// Number of bytes a buffer must have to back a filter with these parameters
    #[staticmethod]
    fn buffer_size(expected_items: u64, false_positive_rate: f64) -> PyResult<usize> {
        let size_in_bits = FilterParams::new(expected_items, false_positive_rate)?.size_in_bits;
        buffer_len(size_in_bits)
    }

    /// Number of buckets in the filter
    #[getter]
    fn size_in_bits(&self) -> u64 {
        self.filter.len()
    }

    /// Retrieve the hash_func given to __init__
    #[getter]
    fn hash_func<'py>(&self, py: Python<'py>) -> PyResult<&Bound<'py, PyAny>> {
        match self.hash_func.as_ref() {
            Some(hash_func) => Ok(hash_func.bind(py)),
            None => builtin_hash_func(py),
        }
    }

    /// Estimated number of items in the filter
    #[getter]
    fn approx_items(&self, py: Python<'_>) -> PyResult<f64> {
        let len = self.filter.len() as f64;
        let bits_set = self.filter.count_ones_sum(py)? as f64;
        Ok((len / (self.k as f64) * (1.0 - (bits_set) / len).ln()).abs())
    }

    #[pyo3(signature = (o, /))]
    fn add(&self, o: &Bound<'_, PyAny>) -> PyResult<()> {
        let hash = hash(o, &self.hash_func)?;
        let indexes = lcg::generate_indexes(hash, self.k, self.filter.len());
        self.filter.set_all(o.py(), indexes)
    }

    /// Test whether every element in the bloom may be in other
    ///
    /// This can have false positives (return true for a bloom which does not
    /// contain all items in this set), but it will not return a false negative:
    /// If this returns false, this set contains an element which is not in other
    #[pyo3(signature = (other, /))]
    fn issubset(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.with_other_as_bloom(other, |other_bloom| {
            self.filter.is_subset(other.py(), &other_bloom.filter)
        })
    }

    /// Test whether every element in other may be in self
    ///
    /// This can have false positives (return true for a bloom which does not
    /// contain all items in other), but it will not return a false negative:
    /// If this returns false, other contains an element which is not in self
    #[pyo3(signature = (other, /))]
    fn issuperset(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.with_other_as_bloom(other, |other_bloom| {
            other_bloom.filter.is_subset(other.py(), &self.filter)
        })
    }

    fn __contains__(&self, o: &Bound<'_, PyAny>) -> PyResult<bool> {
        let hash = hash(o, &self.hash_func)?;
        let indexes = lcg::generate_indexes(hash, self.k, self.filter.len());
        self.filter.all_set(o.py(), indexes)
    }

    /// Return a new set with elements from the set and all others.
    #[pyo3(signature = (*others))]
    fn union<'py>(&self, others: &Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        let result = Bound::new(others.py(), self.clone_ref(others.py())?)?;
        result.get().update(others)?;
        Ok(result)
    }

    /// Return a new set with elements common to the set and all others.
    #[pyo3(signature = (*others))]
    fn intersection<'py>(&self, others: &Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        let result = Bound::new(others.py(), self.clone_ref(others.py())?)?;
        result.get().intersection_update(others)?;
        Ok(result)
    }

    fn __or__(&self, py: Python<'_>, other: &Bloom) -> PyResult<Bloom> {
        check_compatible(self, other)?;
        let result = self.clone_ref(py)?;
        result.filter.or_assign(py, &other.filter)?;
        Ok(result)
    }

    fn __ior__(&self, py: Python<'_>, other: &Bloom) -> PyResult<()> {
        check_compatible(self, other)?;
        self.filter.or_assign(py, &other.filter)
    }

    fn __and__(&self, py: Python<'_>, other: &Bloom) -> PyResult<Bloom> {
        check_compatible(self, other)?;
        let result = self.clone_ref(py)?;
        result.filter.and_assign(py, &other.filter)?;
        Ok(result)
    }

    fn __iand__(&self, py: Python<'_>, other: &Bloom) -> PyResult<()> {
        check_compatible(self, other)?;
        self.filter.and_assign(py, &other.filter)
    }

    #[pyo3(signature = (*others))]
    fn update(&self, others: &Bound<'_, PyTuple>) -> PyResult<()> {
        for other in others.iter_borrowed() {
            // If the other object is a Bloom, use the bitwise union
            if let Ok(other) = other.cast::<Bloom>() {
                self.__ior__(others.py(), other.get())?;
            }
            // Otherwise, iterate over the other object and add each item
            else {
                for obj in other.try_iter()? {
                    self.add(&obj?)?;
                }
            }
        }
        Ok(())
    }

    #[pyo3(signature = (*others))]
    fn intersection_update(&self, others: &Bound<'_, PyTuple>) -> PyResult<()> {
        // Lazily allocated temp bitset
        let mut temp: Option<Self> = None;
        for other in others.iter_borrowed() {
            // If the other object is a Bloom, use the bitwise intersection
            if let Ok(other) = other.cast::<Bloom>() {
                self.__iand__(others.py(), other.get())?;
            }
            // Otherwise, iterate over the other object and add each item
            else {
                let temp = match &mut temp {
                    Some(temp) => {
                        temp.clear(others.py())?;
                        temp
                    }
                    None => temp.insert(self.zeroed_clone(others.py())),
                };
                for obj in other.try_iter()? {
                    temp.add(&obj?)?;
                }
                self.__iand__(others.py(), temp)?;
            }
        }
        Ok(())
    }

    fn clear(&self, py: Python<'_>) -> PyResult<()> {
        self.filter.clear(py)
    }

    fn copy(&self, py: Python<'_>) -> PyResult<Bloom> {
        self.clone_ref(py)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        // Use a format that makes it clear that the object
        // cannot be reconstructed from the repr
        if self.filter.is_released(py) {
            return Ok(format!(
                "<Bloom size_in_bits={} released>",
                self.size_in_bits()
            ));
        }
        Ok(format!(
            "<Bloom size_in_bits={} approx_items={:.1}>",
            self.size_in_bits(),
            self.approx_items(py)?
        ))
    }

    fn __bool__(&self, py: Python<'_>) -> PyResult<bool> {
        Ok(!self.filter.is_empty(py)?)
    }

    fn __richcmp__(&self, py: Python<'_>, other: &Self, op: CompareOp) -> PyResult<bool> {
        check_compatible(self, other)?;
        match op {
            CompareOp::Eq => self.filter.equals(py, &other.filter),
            CompareOp::Ne => Ok(!self.filter.equals(py, &other.filter)?),
            CompareOp::Le => self.filter.is_subset(py, &other.filter),
            CompareOp::Lt => self.filter.is_strict_subset(py, &other.filter),
            CompareOp::Ge => other.filter.is_subset(py, &self.filter),
            CompareOp::Gt => other.filter.is_strict_subset(py, &self.filter),
        }
    }

    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    /// Load from a file, see "Persistence" section in the README
    #[classmethod]
    fn load(
        _cls: &Bound<'_, PyType>,
        filepath: PathBuf,
        hash_func: &Bound<'_, PyAny>,
    ) -> PyResult<Bloom> {
        let hash_func = Some(loadable_hash_func(hash_func)?);

        let mut file = File::open(filepath)?;

        let mut k_bytes = [0; HEADER_SIZE];
        file.read_exact(&mut k_bytes)?;
        let k = u64::from_le_bytes(k_bytes);

        let filter = BitLine::load(&mut file)?;

        Ok(Bloom {
            filter,
            k,
            hash_func,
        })
    }

    /// Load from a bytes(), see "Persistence" section in the README
    #[classmethod]
    fn load_bytes(
        _cls: &Bound<'_, PyType>,
        bytes: &[u8],
        hash_func: &Bound<'_, PyAny>,
    ) -> PyResult<Bloom> {
        let hash_func = Some(loadable_hash_func(hash_func)?);

        let (k_bytes, rest) = bytes.split_first_chunk::<HEADER_SIZE>().ok_or_else(|| {
            PyValueError::new_err(format!(
                "buffer of {} bytes is too small to hold a saved filter",
                bytes.len()
            ))
        })?;
        let k = u64::from_le_bytes(*k_bytes);

        let filter = BitLine::load_bytes(rest)?;

        Ok(Bloom {
            filter,
            k,
            hash_func,
        })
    }

    /// Use a buffer laid out like a file written by save (e.g. an mmap of one) as the filter's
    /// storage, holding it until released. A filter in a read-only buffer can only be read. See
    /// "Buffer-backed filters" section in the README
    #[classmethod]
    fn load_buffer(
        _cls: &Bound<'_, PyType>,
        buffer: Bound<'_, PyAny>,
        hash_func: &Bound<'_, PyAny>,
    ) -> PyResult<Bloom> {
        let hash_func = Some(loadable_hash_func(hash_func)?);
        let buffer = HeldBuffer::get(&buffer, PyBufferRequest::contig_ro())?;
        let k = read_buffer_header(&buffer)?;
        Ok(Bloom {
            filter: BitLine::from_buffer(buffer),
            k,
            hash_func,
        })
    }

    /// Release the buffer a buffer-backed filter is stored in, like memoryview.release(); any
    /// later use of the filter raises ValueError. Does nothing for other filters.
    fn release(&self, py: Python<'_>) {
        self.filter.release(py);
    }

    fn __enter__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, py: Python<'_>, _args: &Bound<'_, PyTuple>) {
        self.release(py);
    }

    /// Whether the filter is stored in a read-only buffer
    #[getter]
    fn readonly(&self, py: Python<'_>) -> PyResult<bool> {
        self.filter.readonly(py)
    }

    /// Save to a file, see "Persistence" section in the README
    fn save(&self, py: Python<'_>, filepath: PathBuf) -> PyResult<()> {
        if self.hash_func.is_none() {
            return Err(PyValueError::new_err(
                "Cannot save a bloom filter that uses the built-in hash function",
            ));
        }
        let mut file = File::create(filepath)?;
        file.write_all(&self.k.to_le_bytes())?;
        self.filter.save(py, &mut file)?;
        Ok(())
    }

    /// Save to a byte(), see "Persistence" section in the README
    fn save_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        if self.hash_func.is_none() {
            return Err(PyValueError::new_err(
                "Cannot save a bloom filter that uses the built-in hash function",
            ));
        }

        let len = HEADER_SIZE + self.filter.len_bytes();
        PyBytes::new_with(py, len, |data| {
            data[..HEADER_SIZE].copy_from_slice(&self.k.to_le_bytes());
            self.filter.copy_into(py, &mut data[HEADER_SIZE..])
        })
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.hash_func)?;
        self.filter.traverse(&visit)?;
        Ok(())
    }
}

// Non-python methods
impl Bloom {
    fn hash_fn_clone(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.hash_func.as_ref().map(|f| f.clone_ref(py))
    }

    /// Returns a copy that owns its bits, even if self is backed by a buffer.
    fn clone_ref(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Bloom {
            filter: self.filter.to_owned(py)?,
            k: self.k,
            hash_func: self.hash_fn_clone(py),
        })
    }

    fn zeroed_clone(&self, py: Python<'_>) -> Bloom {
        Bloom {
            filter: self.filter.zeroed_like(),
            k: self.k,
            hash_func: self.hash_fn_clone(py),
        }
    }

    /// Extract other as a bloom, or iterate other, and add all items to a temporary bloom
    fn with_other_as_bloom<O>(
        &self,
        other: &Bound<'_, PyAny>,
        f: impl FnOnce(&Bloom) -> PyResult<O>,
    ) -> PyResult<O> {
        match other.cast::<Bloom>() {
            Ok(o) => {
                let o = o.get();
                check_compatible(self, o)?;
                f(o)
            }
            Err(_) => {
                let other_bloom = self.zeroed_clone(other.py());
                for obj in other.try_iter()? {
                    other_bloom.add(&obj?)?;
                }
                f(&other_bloom)
            }
        }
    }
}

/// This implements a linear congruential generator that is
/// used to distribute entropy from the hash over multiple ints.
mod lcg {
    pub struct Random {
        state: u128,
    }

    impl Iterator for Random {
        type Item = u64;

        fn next(&mut self) -> Option<Self::Item> {
            self.state = self
                .state
                .wrapping_mul(47026247687942121848144207491837418733)
                .wrapping_add(1);
            Some((self.state >> 32) as Self::Item)
        }
    }

    pub fn distribute_entropy(hash: i128) -> Random {
        Random {
            state: hash as u128,
        }
    }

    pub fn generate_indexes(hash: i128, k: u64, len: u64) -> impl Iterator<Item = u64> {
        distribute_entropy(hash)
            .take(k as usize)
            .map(move |x: u64| x % len)
    }
}

fn hash(o: &Bound<'_, PyAny>, hash_func: &Option<Py<PyAny>>) -> PyResult<i128> {
    match hash_func {
        Some(hash_func) => {
            let hash_func = hash_func.bind(o.py());
            let hash = hash_func.call1((o,))?;
            Ok(hash.extract()?)
        }
        None => Ok(o.hash()? as i128),
    }
}

/// Checks that a hash function can be used to load a saved filter.
fn loadable_hash_func(hash_func: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if !hash_func.is_callable() {
        return Err(PyTypeError::new_err("hash_func must be callable"));
    }
    if hash_func.is(builtin_hash_func(hash_func.py())?) {
        return Err(PyValueError::new_err(
            "Cannot load a bloom filter that uses the built-in hash function",
        ));
    }
    Ok(hash_func.clone().unbind())
}

/// Returns the hash function to store for a new filter: `None` for the built-in hash.
fn custom_hash_func(hash_func: Option<Bound<'_, PyAny>>) -> PyResult<Option<Py<PyAny>>> {
    match hash_func {
        Some(hash_func) if !hash_func.is(builtin_hash_func(hash_func.py())?) => {
            if !hash_func.is_callable() {
                return Err(PyTypeError::new_err("hash_func must be callable"));
            }
            Ok(Some(hash_func.unbind()))
        }
        _ => Ok(None),
    }
}

/// Returns the number of bytes a buffer needs for the header and `size_in_bits` bits.
fn buffer_len(size_in_bits: u64) -> PyResult<usize> {
    Ok(HEADER_SIZE + bitline::byte_len(size_in_bits)?)
}

/// Checks that `buffer` is exactly `len` bytes, then writes the header for `k`.
fn write_buffer_header(buffer: &HeldBuffer, k: u64, len: usize) -> PyResult<()> {
    let bytes = buffer.writable_bytes()?;
    if bytes.len() != len {
        return Err(PyValueError::new_err(format!(
            "buffer must be exactly {len} bytes for this filter, got {}",
            bytes.len()
        )));
    }
    for (byte, value) in bytes.iter().zip(k.to_le_bytes()) {
        byte.store(value, Relaxed);
    }
    Ok(())
}

/// Checks that `buffer` can hold a saved filter, and returns the `k` in its header.
fn read_buffer_header(buffer: &HeldBuffer) -> PyResult<u64> {
    let bytes = buffer.readable_bytes();
    if bytes.len() <= HEADER_SIZE {
        return Err(PyValueError::new_err(format!(
            "buffer of {} bytes is too small to hold a saved filter",
            bytes.len()
        )));
    }
    let mut k = [0; HEADER_SIZE];
    bitline::copy_bytes(&bytes[..HEADER_SIZE], &mut k);
    Ok(u64::from_le_bytes(k))
}

fn check_compatible(a: &Bloom, b: &Bloom) -> PyResult<()> {
    if a.k != b.k || a.filter.len() != b.filter.len() {
        return Err(PyValueError::new_err(
            "size and max false positive rate must be the same for both filters",
        ));
    }

    // now only the hash function can be different
    match (&a.hash_func, &b.hash_func) {
        (Some(lhs), Some(rhs)) if lhs.is(rhs) => {}
        (&None, &None) => {}
        _ => {
            return Err(PyValueError::new_err(
                "Bloom filters must have the same hash function",
            ))
        }
    }

    Ok(())
}

fn builtin_hash_func(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    static HASH_FUNC: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

    let res = HASH_FUNC.get_or_try_init(py, || -> PyResult<_> {
        let builtins = PyModule::import(py, "builtins")?;
        Ok(builtins.getattr("hash")?.unbind())
    })?;

    Ok(res.bind(py))
}

#[pymodule]
mod rbloom {
    #[pymodule_export]
    use super::Bloom;
}
