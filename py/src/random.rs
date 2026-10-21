//! Persistent RNG ownership; distributions, rolls, deals and gathering stay in the shared core.
use crate::{array::{Array, core, give}, convert, distribution::{self, Distribution}};
use pyo3::{exceptions::PyValueError, prelude::*};
use rand::{SeedableRng, rngs::StdRng};
use rustymath::{Arg, layout::generated_len, structure::{self, Selector}};
use rustymath_buffer::py_err;

/// One persistent generator. Numerical sampling uses the same algorithms as `rustymath.distribution`.
#[pyclass(subclass, module = "rustymath.random")]
pub struct Generator { rng: StdRng }

impl Generator {
    fn positions(&mut self, total: usize, n: usize, replace: bool) -> PyResult<Vec<u64>> {
        if (total == 0 && n != 0) || (!replace && n > total) { return Err(PyValueError::new_err("not enough items to draw this sample")); }
        let mut positions = Vec::with_capacity(n);
        if n != 0 {
            if replace { rustymath::random::roll(Arg::One(total as u64), n, &mut self.rng, &mut positions); }
            else { rustymath::random::deal(n as u64, total as u64, &mut self.rng, &mut positions); }
        }
        Ok(positions)
    }
}

fn dimensions(size: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<usize>> { size.map(crate::create::shape).transpose().map(Option::unwrap_or_default) }

#[pymethods]
impl Generator {
    #[new]
    #[pyo3(signature = (seed=None))]
    fn new(seed: Option<u64>) -> Self { Self { rng: seed.map_or_else(rand::make_rng::<StdRng>, StdRng::seed_from_u64) } }
    /// Draw from any existing distribution, advancing this generator's state.
    #[pyo3(signature = (distribution, size=None))]
    fn sample<'py>(&mut self, py: Python<'py>, distribution: &Distribution, size: Option<&Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
        distribution::sampled(py, distribution.0, &mut self.rng, size)
    }
    /// Uniform integers in [low, high), or [0, low) when `high` is omitted.
    #[pyo3(signature = (low, high=None, size=None))]
    fn integers<'py>(&mut self, py: Python<'py>, low: i64, high: Option<i64>, size: Option<&Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
        let (low, high) = high.map_or((0, low), |high| (low, high));
        if low >= high { return Err(PyValueError::new_err("integers needs low < high")); }
        let shape = dimensions(size)?;
        let n = generated_len(&shape).map_err(py_err)?;
        let mut draws = Vec::with_capacity(n);
        rustymath::random::roll(Arg::One((i128::from(high) - i128::from(low)) as u64), n, &mut self.rng, &mut draws);
        let values = draws.into_iter().map(|x| (x as i64).wrapping_add(low)).collect();
        give(py, core(values, &shape)?, size.is_none())
    }
    /// Random leading-axis cells of `x`, or integers below `x` when `x` is one integer. `replace=False` draws distinct cells.
    #[pyo3(signature = (x, size=None, replace=true))]
    fn choice<'py>(&mut self, py: Python<'py>, x: &Bound<'py, PyAny>, size: Option<&Bound<'py, PyAny>>, replace: bool) -> PyResult<Bound<'py, PyAny>> {
        let (x, total) = population(x)?;
        let shape = dimensions(size)?;
        let n = generated_len(&shape).map_err(py_err)?;
        let positions = self.positions(total, n, replace)?;
        let Some(x) = x else { return give(py, core(positions.into_iter().map(|i| i as i64).collect(), &shape)?, size.is_none()) };
        let mut layout = x.layout();
        let selector = Selector::Indices { layout: shape.into(), positions: rustymath::gather::Table::new(positions.into_iter().map(|i| i as usize).collect()) };
        if replace { layout = layout.without_keys(0); }
        let plan = structure::select(&layout, vec![selector], false).map_err(py_err)?;
        give(py, x.selected(plan)?, size.is_none())
    }
    /// A copy with the leading-axis cells in random order, or a permutation of integers below `x` when it is one integer.
    fn permutation(&mut self, x: &Bound<'_, PyAny>) -> PyResult<Array> {
        let (x, n) = population(x)?;
        generated_len(&[n]).map_err(py_err)?;
        let positions = self.positions(n, n, false)?;
        let Some(x) = x else { return Ok(Array::new(core(positions.into_iter().map(|i| i as i64).collect(), &[n])?)) };
        let layout = x.layout();
        let selector = Selector::Indices { layout: [n].into(), positions: rustymath::gather::Table::new(positions.into_iter().map(|i| i as usize).collect()) };
        Ok(Array::new(x.selected(structure::select(&layout, vec![selector], false).map_err(py_err)?)?))
    }
}

/// Reads an array, or recognizes the implicit integer population 0..n without allocating it.
fn population<'py>(x: &Bound<'py, PyAny>) -> PyResult<(Option<convert::Arg<'py>>, usize)> {
    if let Ok(n) = x.extract::<i64>() {
        if n < 0 { return Err(PyValueError::new_err("a population length can't be negative")); }
        let n = usize::try_from(n).map_err(|_| PyValueError::new_err("population length doesn't fit this platform"))?;
        return Ok((None, n));
    }
    let x = convert::one(x)?;
    if x.layout().shape().is_empty() { return Err(PyValueError::new_err("sampling needs an array with at least one axis")); }
    let n = x.layout().shape()[0];
    Ok((Some(x), n))
}

pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> { m.add_class::<Generator>() }
