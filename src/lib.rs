use bitline::BitLine;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::types::PyType;
use pyo3::{basic::CompareOp, prelude::*, types::PyBytes, types::PyString, types::PyTuple};
use pyo3_file::PyFileLikeObject;
use std::fs::File;
use std::io::{Read, Write};
use std::mem;

#[pyclass(module = "rbloom")]
#[derive(Clone)]
struct Bloom {
    filter: BitLine,
    k: u64, // Number of hash functions (implemented via a LCG that uses
    // the original hash as a seed)
    hash_func: Option<PyObject>,
}

#[derive(Debug)]
enum SaveLoadType {
    Filepath(String),
    FileLike(PyFileLikeObject),
    Bytes(Vec<u8>),
    None,
}

impl SaveLoadType {
    pub fn from_pyobject(saveloadtype: Option<PyObject>) -> PyResult<SaveLoadType> {
        Python::with_gil(|py| match saveloadtype {
            None => Ok(SaveLoadType::None),
            Some(obj) => {
                if let Ok(str) = obj.downcast::<PyString>(py) {
                    Ok(SaveLoadType::Filepath(str.to_string()))
                } else if let Ok(bytes) = obj.downcast::<PyBytes>(py) {
                    Ok(SaveLoadType::Bytes(bytes.as_bytes().to_vec()))
                } else {
                    match PyFileLikeObject::with_requirements(obj, true, false, true) {
                        Ok(f) => Ok(SaveLoadType::FileLike(f)),
                        Err(e) => Err(e),
                    }
                }
            }
        })
    }
}

#[pymethods]
impl Bloom {
    #[new]
    fn new(
        expected_items: u64,
        false_positive_rate: f64,
        hash_func: Option<&PyAny>,
    ) -> PyResult<Self> {
        // Check the inputs
        if let Some(hash_func) = hash_func {
            if !hash_func.is_callable() {
                return Err(PyTypeError::new_err("hash_func must be callable"));
            }
        }
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

        // Calculate the parameters for the filter
        let size_in_bits =
            -1.0 * (expected_items as f64) * false_positive_rate.ln() / 2.0f64.ln().powi(2);
        let k = (size_in_bits / expected_items as f64) * 2.0f64.ln();

        let hash_func = match hash_func {
            // if __builtins__.hash was passed, use None instead
            Some(hash_func) if !hash_func.is(get_builtin_hash_func(hash_func.py())?) => {
                Some(hash_func.to_object(hash_func.py()))
            }
            _ => None,
        };
        // Create the filter
        Ok(Bloom {
            filter: BitLine::new(size_in_bits as u64)?,
            k: k as u64,
            hash_func,
        })
    }

    /// Number of buckets in the filter
    #[getter]
    fn size_in_bits(&self) -> u64 {
        self.filter.len()
    }

    /// Retrieve the hash_func given to __init__
    #[getter]
    fn hash_func<'a>(&'a self, py: Python<'a>) -> PyResult<&'a PyAny> {
        match self.hash_func.as_ref() {
            Some(hash_func) => Ok(hash_func.as_ref(py)),
            None => get_builtin_hash_func(py),
        }
    }

    /// Estimated number of items in the filter
    #[getter]
    fn approx_items(&self) -> f64 {
        let len = self.filter.len() as f64;
        let bits_set = self.filter.sum() as f64;
        (len / (self.k as f64) * (1.0 - (bits_set) / len).ln()).abs()
    }

    #[pyo3(signature = (o, /))]
    fn add(&mut self, o: &PyAny) -> PyResult<()> {
        let hash = hash(o, &self.hash_func)?;
        for index in lcg::generate_indexes(hash, self.k, self.filter.len()) {
            self.filter.set(index);
        }
        Ok(())
    }

    /// Test whether every element in the bloom may be in other
    ///
    /// This can have false positives (return true for a bloom which does not
    /// contain all items in this set), but it will not return a false negative:
    /// If this returns false, this set contains an element which is not in other
    #[pyo3(signature = (other, /))]
    fn issubset(&self, other: &PyAny) -> PyResult<bool> {
        self.with_other_as_bloom(other, |other_bloom| {
            Ok(self.filter.is_subset(&other_bloom.filter))
        })
    }

    /// Test whether every element in other may be in self
    ///
    /// This can have false positives (return true for a bloom which does not
    /// contain all items in other), but it will not return a false negative:
    /// If this returns false, other contains an element which is not in self
    #[pyo3(signature = (other, /))]
    fn issuperset(&self, other: &PyAny) -> PyResult<bool> {
        self.with_other_as_bloom(other, |other_bloom| {
            Ok(other_bloom.filter.is_subset(&self.filter))
        })
    }

    fn __contains__(&self, o: &PyAny) -> PyResult<bool> {
        let hash = hash(o, &self.hash_func)?;
        for index in lcg::generate_indexes(hash, self.k, self.filter.len()) {
            if !self.filter.get(index) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Return a new set with elements from the set and all others.
    #[pyo3(signature = (*others))]
    fn union(&self, others: &PyTuple) -> PyResult<Self> {
        let mut result = self.clone();
        result.update(others)?;
        Ok(result)
    }

    /// Return a new set with elements common to the set and all others.
    #[pyo3(signature = (*others))]
    fn intersection(&self, others: &PyTuple) -> PyResult<Self> {
        let mut result = self.clone();
        result.intersection_update(others)?;
        Ok(result)
    }

    fn __or__(&self, py: Python<'_>, other: &Bloom) -> PyResult<Bloom> {
        check_compatible(self, other)?;
        Ok(Bloom {
            filter: &self.filter | &other.filter,
            k: self.k,
            hash_func: self.hash_fn_clone(py),
        })
    }

    fn __ior__(&mut self, other: &Bloom) -> PyResult<()> {
        check_compatible(self, other)?;
        self.filter |= &other.filter;
        Ok(())
    }

    fn __and__(&self, py: Python<'_>, other: &Bloom) -> PyResult<Bloom> {
        check_compatible(self, other)?;
        Ok(Bloom {
            filter: &self.filter & &other.filter,
            k: self.k,
            hash_func: self.hash_fn_clone(py),
        })
    }

    fn __iand__(&mut self, other: &Bloom) -> PyResult<()> {
        check_compatible(self, other)?;
        self.filter &= &other.filter;
        Ok(())
    }

    #[pyo3(signature = (*others))]
    fn update(&mut self, others: &PyTuple) -> PyResult<()> {
        for other in others.iter() {
            // If the other object is a Bloom, use the bitwise union
            if let Ok(other) = other.extract::<PyRef<Bloom>>() {
                self.__ior__(&other)?;
            }
            // Otherwise, iterate over the other object and add each item
            else {
                for obj in other.iter()? {
                    self.add(obj?)?;
                }
            }
        }
        Ok(())
    }

    #[pyo3(signature = (*others))]
    fn intersection_update(&mut self, others: &PyTuple) -> PyResult<()> {
        // Lazily allocated temp bitset
        let mut temp: Option<Self> = None;
        for other in others.iter() {
            // If the other object is a Bloom, use the bitwise intersection
            if let Ok(other) = other.extract::<PyRef<Bloom>>() {
                self.__iand__(&other)?;
            }
            // Otherwise, iterate over the other object and add each item
            else {
                let temp = temp.get_or_insert_with(|| self.clone());
                temp.clear();
                for obj in other.iter()? {
                    temp.add(obj?)?;
                }
                self.__iand__(temp)?;
            }
        }
        Ok(())
    }

    fn clear(&mut self) {
        self.filter.clear();
    }

    fn copy(&self) -> Bloom {
        self.clone()
    }

    fn __repr__(&self) -> String {
        // Use a format that makes it clear that the object
        // cannot be reconstructed from the repr
        format!(
            "<Bloom size_in_bits={} approx_items={:.1}>",
            self.size_in_bits(),
            self.approx_items()
        )
    }

    fn __bool__(&self) -> bool {
        !self.filter.is_empty()
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        check_compatible(self, other)?;
        Ok(match op {
            CompareOp::Eq => self.filter == other.filter,
            CompareOp::Ne => self.filter != other.filter,
            CompareOp::Le => self.filter.is_subset(&other.filter),
            CompareOp::Lt => self.filter.is_strict_subset(&other.filter),
            CompareOp::Ge => other.filter.is_subset(&self.filter),
            CompareOp::Gt => other.filter.is_strict_subset(&self.filter),
        })
    }

    #[classattr]
    const __hash__: Option<PyObject> = None;

    /// Load from a file, see "Persistence" section in the README
    #[classmethod]
    fn load(_cls: &PyType, src: PyObject, hash_func: &PyAny) -> PyResult<Bloom> {
        // Self::load_from_read(hash_func, || File::open(filepath))
        match SaveLoadType::from_pyobject(Some(src)) {
            Ok(f) => match f {
                SaveLoadType::Filepath(filepath) => {
                    let file = File::open(filepath)?;
                    Self::load_from_read(hash_func, || Ok::<_, PyErr>(file))
                }
                SaveLoadType::FileLike(f) => Self::load_from_read(hash_func, || Ok::<_, PyErr>(f)),
                SaveLoadType::Bytes(b) => Self::load_from_read(hash_func, || {
                    Ok::<&[u8], std::convert::Infallible>(b.as_ref())
                }),
                _ => Err(PyErr::new::<PyTypeError, _>("Incorrect argument type")),
            },
            Err(e) => Err(e),
        }
    }

    /// Save to a file, see "Persistence" section in the README
    fn save(&self, dest: Option<PyObject>) -> PyResult<PyObject> {
        match SaveLoadType::from_pyobject(dest) {
            Ok(f) => match f {
                SaveLoadType::Filepath(filepath) => {
                    let file = File::create(filepath)?;
                    let _ = self.save_to_write(|| Ok::<_, PyErr>(file));
                    Python::with_gil(|py| Ok(().to_object(py)))
                }
                SaveLoadType::FileLike(f) => {
                    let _ = self.save_to_write(|| Ok::<_, PyErr>(f));
                    Python::with_gil(|py| Ok(().to_object(py)))
                }
                SaveLoadType::Bytes(_) | SaveLoadType::None => {
                    let mut bytes = Vec::with_capacity(
                        mem::size_of_val(&self.k) + (self.filter.len() / 8) as usize,
                    );
                    self.save_to_write(|| Ok::<_, std::convert::Infallible>(&mut bytes))?;
                    Python::with_gil(|py| Ok(PyBytes::new(py, &bytes).into()))
                }
            },
            Err(e) => Err(e),
        }
    }
}

// Non-python methods
impl Bloom {
    fn hash_fn_clone(&self, py: Python<'_>) -> Option<PyObject> {
        self.hash_func.as_ref().map(|f| f.clone_ref(py))
    }

    fn load_from_read<F, R, E>(hash_func: &PyAny, get_data: F) -> PyResult<Self>
    where
        F: FnOnce() -> Result<R, E>,
        R: Read,
        PyErr: From<E>,
    {
        // check that the hash_func is callable
        if !hash_func.is_callable() {
            return Err(PyTypeError::new_err("hash_func must be callable"));
        }
        // check that the hash_func isn't the built-in hash function
        if hash_func.is(get_builtin_hash_func(hash_func.py())?) {
            return Err(PyValueError::new_err(
                "Cannot load a bloom filter that uses the built-in hash function",
            ));
        }
        let hash_func = Some(hash_func.to_object(hash_func.py()));

        let mut data = get_data()?;

        let mut k_bytes = [0; mem::size_of::<u64>()];
        data.read_exact(&mut k_bytes)?;
        let k = u64::from_le_bytes(k_bytes);

        let filter = BitLine::load(data)?;

        Ok(Bloom {
            filter,
            k,
            hash_func,
        })
    }

    fn save_to_write<F, W, E>(&self, get_writer: F) -> PyResult<()>
    where
        F: FnOnce() -> Result<W, E>,
        W: Write,
        PyErr: From<E>,
    {
        if self.hash_func.is_none() {
            return Err(PyValueError::new_err(
                "Cannot save a bloom filter that uses the built-in hash function",
            ));
        }
        let mut writer = get_writer()?;
        writer.write_all(&self.k.to_le_bytes())?;
        self.filter.save(writer)?;
        Ok(())
    }

    fn zeroed_clone(&self, py: Python<'_>) -> Bloom {
        Bloom {
            filter: BitLine::new(self.filter.len()).unwrap(),
            k: self.k,
            hash_func: self.hash_fn_clone(py),
        }
    }

    /// Extract other as a bloom, or iterate other, and add all items to a temporary bloom
    fn with_other_as_bloom<O>(
        &self,
        other: &PyAny,
        f: impl FnOnce(&Bloom) -> PyResult<O>,
    ) -> PyResult<O> {
        match other.extract::<PyRef<Bloom>>() {
            Ok(o) => {
                check_compatible(self, &o)?;
                f(&o)
            }
            Err(_) => {
                let mut other_bloom = self.zeroed_clone(other.py());
                for obj in other.iter()? {
                    other_bloom.add(obj?)?;
                }
                f(&other_bloom)
            }
        }
    }
}

/// This is a primitive BitVec-like structure that uses a Box<[u8]> as
/// the backing store; it exists here to avoid the need for a dependency
/// on bitvec and to act as a container around all the bit manipulation.
/// Indexing is done using u64 to avoid address space issues on 32-bit
/// systems, which would otherwise limit the size to 2^32 bits (512MB).
/// Using u8 for the backing store simplifies file I/O as well as file
/// portability across systems, and the performance is equivalent to
/// using usize, even though the latter is arguably more elegant.
mod bitline {
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use std::io::{Read, Write};

    #[inline(always)]
    fn bit_idx(idx: u64) -> Option<(usize, u32)> {
        let (q, r) = (idx / 8, idx % 8);
        Some((q.try_into().ok()?, r.try_into().ok()?))
    }

    #[derive(Clone, PartialEq, Eq)]
    pub struct BitLine {
        bits: Box<[u8]>,
    }

    impl BitLine {
        pub fn new(size_in_bits: u64) -> PyResult<Self> {
            match bit_idx(size_in_bits) {
                Some((q, r)) => {
                    let size = if r == 0 { q } else { q + 1 };
                    Ok(Self {
                        bits: vec![0; size].into_boxed_slice(),
                    })
                }
                None => Err(PyValueError::new_err("too many bits")),
            }
        }

        /// Make sure that index is less than len when calling this!
        pub fn set(&mut self, index: u64) {
            let (idx, offset) = bit_idx(index).unwrap();
            self.bits[idx] |= 1 << offset;
        }

        /// Make sure that index is less than len when calling this!
        pub fn get(&self, index: u64) -> bool {
            let (idx, offset) = bit_idx(index).unwrap();
            self.bits[idx] & (1 << offset) != 0
        }

        /// Returns the number of bits in the BitLine
        pub fn len(&self) -> u64 {
            self.bits.len() as u64 * 8
        }

        pub fn clear(&mut self) {
            self.bits.fill(0);
        }

        pub fn sum(&self) -> u64 {
            self.bits.iter().map(|x| x.count_ones() as u64).sum()
        }

        pub fn is_empty(&self) -> bool {
            self.bits.iter().all(|&word| word == 0)
        }

        pub fn is_subset(&self, other: &BitLine) -> bool {
            all_pairs(self, other, |lhs, rhs| (lhs | rhs) == rhs)
        }

        pub fn is_strict_subset(&self, other: &BitLine) -> bool {
            let mut is_equal = true;
            let is_subset = all_pairs(self, other, |lhs, rhs| {
                is_equal &= lhs == rhs;
                (lhs | rhs) == rhs
            });
            is_subset && !is_equal
        }

        /// Reads the given file from the current position to the end and
        /// returns a BitLine containing the data.
        pub fn load<R: Read>(mut file: R) -> PyResult<Self> {
            let mut bits = Vec::new();
            file.read_to_end(&mut bits)?;
            Ok(Self {
                bits: bits.into_boxed_slice(),
            })
        }

        /// Writes the BitLine to the given file from the current position.
        pub fn save<W: Write>(&self, mut file: W) -> PyResult<()> {
            file.write_all(&self.bits)?;
            Ok(())
        }
    }

    fn all_pairs(lhs: &BitLine, rhs: &BitLine, mut f: impl FnMut(u8, u8) -> bool) -> bool {
        lhs.bits
            .iter()
            .zip(rhs.bits.iter())
            .all(move |(&lhs, &rhs)| f(lhs, rhs))
    }

    impl std::ops::BitAnd for BitLine {
        type Output = Self;

        fn bitand(mut self, rhs: Self) -> Self::Output {
            self &= rhs;
            self
        }
    }

    impl std::ops::BitAnd for &BitLine {
        type Output = BitLine;

        fn bitand(self, rhs: Self) -> Self::Output {
            let mut result = self.clone();
            result &= rhs;
            result
        }
    }

    impl std::ops::BitAndAssign for BitLine {
        fn bitand_assign(&mut self, rhs: Self) {
            *self &= &rhs;
        }
    }
    impl std::ops::BitAndAssign<&BitLine> for BitLine {
        fn bitand_assign(&mut self, rhs: &Self) {
            for (lhs, rhs) in self.bits.iter_mut().zip(rhs.bits.iter()) {
                *lhs &= rhs;
            }
        }
    }

    impl std::ops::BitOr for BitLine {
        type Output = Self;

        fn bitor(mut self, rhs: Self) -> Self::Output {
            self |= rhs;
            self
        }
    }

    impl std::ops::BitOr for &BitLine {
        type Output = BitLine;

        fn bitor(self, rhs: Self) -> Self::Output {
            let mut result = self.clone();
            result |= rhs;
            result
        }
    }

    impl std::ops::BitOrAssign for BitLine {
        fn bitor_assign(&mut self, rhs: Self) {
            *self |= &rhs;
        }
    }

    impl std::ops::BitOrAssign<&BitLine> for BitLine {
        fn bitor_assign(&mut self, rhs: &Self) {
            for (lhs, rhs) in self.bits.iter_mut().zip(rhs.bits.iter()) {
                *lhs |= rhs;
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

fn hash(o: &PyAny, hash_func: &Option<PyObject>) -> PyResult<i128> {
    match hash_func {
        Some(hash_func) => {
            let hash = hash_func.call1(o.py(), (o,))?;
            Ok(hash.extract(o.py())?)
        }
        None => Ok(o.hash()? as i128),
    }
}

fn check_compatible(a: &Bloom, b: &Bloom) -> PyResult<()> {
    if a.k != b.k || a.filter.len() != b.filter.len() {
        return Err(PyValueError::new_err(
            "size and max false positive rate must be the same for both filters",
        ));
    }

    // now only the hash function can be different
    let same_hash_fn = match (&a.hash_func, &b.hash_func) {
        (Some(lhs), Some(rhs)) => lhs.is(rhs),
        (&None, &None) => true,
        _ => false,
    };

    if same_hash_fn {
        Ok(())
    } else {
        Err(PyValueError::new_err(
            "Bloom filters must have the same hash function",
        ))
    }
}

fn get_builtin_hash_func(py: Python<'_>) -> PyResult<&'_ PyAny> {
    let builtins = PyModule::import(py, "builtins")?;
    builtins.getattr("hash")
}

#[pymodule]
fn rbloom(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Bloom>()?;
    Ok(())
}
