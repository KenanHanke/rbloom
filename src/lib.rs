use pyo3::prelude::*;

#[pyclass]
struct Bloom {
    filter: Vec<u8>,
}

#[pymethods]
impl Bloom {
    #[new]
    fn new(size_in_bits: u64) -> PyResult<Self> {
        let (q, r) = (size_in_bits / 8, size_in_bits % 8);
        let size = if r == 0 { q } else { q + 1 };
        Ok(Bloom {
            filter: vec![0; size.try_into()?],
        })
    }

    fn add(&mut self, o: &PyAny) -> PyResult<()> {
        let hash = hash(o)?;
        let (q, r) = (hash / 8, hash % 8);
        let size: i64 = self.filter.len().try_into()?;
        let index = q % size;
        self.filter[index as usize] |= 1 << r;
        Ok(())
    }

    fn __contains__(&self, o: &PyAny) -> PyResult<bool> {
        let hash = hash(o)?;
        let (q, r) = (hash / 8, hash % 8);
        let size: i64 = self.filter.len().try_into()?;
        let index = q % size;
        Ok(self.filter[index as usize] & (1 << r) != 0)
    }
}

fn hash(o: &PyAny) -> PyResult<i64> {
    let hash = o.call_method0("__hash__")?;
    hash.extract()
}

#[pymodule]
fn rbloom(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Bloom>()?;
    Ok(())
}
