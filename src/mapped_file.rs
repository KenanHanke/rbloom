//! Memory-maps files with Python's `mmap` module, to back filters with files.
//!
//! Files are given either as a path (`str` or `os.PathLike`), which is opened and closed here,
//! or as a file object with a `fileno()` method, which is left open. Python's `mmap` duplicates
//! the file descriptor, so the map stays valid after the file is closed.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;

/// Whether a mapped file can be changed.
#[derive(Clone, Copy)]
pub enum Access {
    Read,
    Write,
}

impl Access {
    /// Parses the `mode` argument of `Bloom.load_mmap`.
    pub fn from_mode(mode: &str) -> PyResult<Self> {
        match mode {
            "r" => Ok(Access::Read),
            "rw" => Ok(Access::Write),
            _ => Err(PyValueError::new_err(format!(
                "mode must be 'r' or 'rw', not {mode:?}"
            ))),
        }
    }

    fn open_mode(self) -> &'static str {
        match self {
            Access::Read => "rb",
            Access::Write => "r+b",
        }
    }

    fn mmap_constant(self) -> &'static str {
        match self {
            Access::Read => "ACCESS_READ",
            Access::Write => "ACCESS_WRITE",
        }
    }
}

/// Resizes `file` to exactly `len` bytes and maps it writable. A path is created if it doesn't
/// exist.
pub fn create<'py>(file: &Bound<'py, PyAny>, len: usize) -> PyResult<Bound<'py, PyAny>> {
    let py = file.py();
    with_fileno(file, "w+b", |fd| {
        PyModule::import(py, "os")?.call_method1("ftruncate", (fd, len))?;
        mmap(fd, len, Access::Write)
    })
}

/// Maps all of `file`.
pub fn open<'py>(file: &Bound<'py, PyAny>, access: Access) -> PyResult<Bound<'py, PyAny>> {
    with_fileno(file, access.open_mode(), |fd| mmap(fd, 0, access))
}

/// Calls `f` with the file descriptor of `file`, opening a path with `mode` for the duration of
/// the call.
fn with_fileno<'py, T>(
    file: &Bound<'py, PyAny>,
    mode: &str,
    f: impl FnOnce(&Bound<'py, PyAny>) -> PyResult<T>,
) -> PyResult<T> {
    let py = file.py();
    if let Ok(path) = file.extract::<PathBuf>() {
        let opened = PyModule::import(py, "io")?.call_method1("open", (path, mode))?;
        let result = opened.call_method0("fileno").and_then(|fd| f(&fd));
        let closed = opened.call_method0("close");
        let value = result?;
        closed?;
        return Ok(value);
    }
    if !file.hasattr("fileno")? {
        return Err(PyTypeError::new_err(format!(
            "expected a path or a file object with fileno(), got {}",
            file.get_type().name()?
        )));
    }
    // Writes still sitting in a Python-level buffer would be invisible to the map.
    if file.hasattr("flush")? {
        file.call_method0("flush")?;
    }
    f(&file.call_method0("fileno")?)
}

fn mmap<'py>(fd: &Bound<'py, PyAny>, len: usize, access: Access) -> PyResult<Bound<'py, PyAny>> {
    let py = fd.py();
    let module = PyModule::import(py, "mmap")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("access", module.getattr(access.mmap_constant())?)?;
    module.call_method("mmap", (fd, len), Some(&kwargs))
}
