//! rustymath's `polynomial` module under the names of `numpy.polynomial.polynomial`. The wrappers in `python/rustymath/polynomial.py`
//! give each argument a contiguous float or complex array. Root finding computes in complex128.
use half::f16;
use numpy::{Complex32, Complex64, PyArray1, PyArrayDescrMethods, PyReadonlyArray0, PyReadonlyArray1, PyReadwriteArrayDyn, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rustymath::polynomial;

/// Runs `$body` with `$t` as the element type of `$a`, a float or complex array.
macro_rules! numbers {
    ($a:expr, $t:ident => $body:expr) => {
        match $a.dtype().char() {
            b'F' => { type $t = Complex32; $body }
            b'D' => { type $t = Complex64; $body }
            _ => { floats!($a, $t => $body) }
        }
    };
}

/// Replaces each item of `x` with the value there of the polynomial `c`.
#[pyfunction]
pub fn polyval(x: &Bound<'_, PyUntypedArray>, c: &Bound<'_, PyUntypedArray>) -> PyResult<()> {
    if x.dtype().kind() == b'c' { numbers!(x, T => {
        let c = c.extract::<PyReadonlyArray1<T>>()?;
        let c = c.as_slice()?;
        let mut x = x.extract::<PyReadwriteArrayDyn<T>>()?;
        for v in x.as_slice_mut()? { *v = polynomial::value(c, v); }
        Ok(())
    }) } else { floats!(x, F => {
        polynomial::evaluate(c.extract::<PyReadonlyArray1<F>>()?.as_slice()?, x.extract::<PyReadwriteArrayDyn<F>>()?.as_slice_mut()?);
        Ok(())
    }) }
}

/// Replaces each item of `x` with `m` times the product of its differences from `roots`.
#[pyfunction]
pub fn polyvalfromroots(x: &Bound<'_, PyUntypedArray>, roots: &Bound<'_, PyUntypedArray>, m: &Bound<'_, PyUntypedArray>) -> PyResult<()> {
    numbers!(x, T => {
        let roots = roots.extract::<PyReadonlyArray1<T>>()?;
        let roots = roots.as_slice()?;
        let m = m.extract::<PyReadonlyArray0<T>>()?.as_array()[()];
        let mut x = x.extract::<PyReadwriteArrayDyn<T>>()?;
        for v in x.as_slice_mut()? { *v = polynomial::factored(&m, roots, v); }
        Ok(())
    })
}

/// The roots of the polynomial `c`. Where the eigenvalue solver doesn't converge, each root is NaN and NumPy reports "invalid value".
#[pyfunction]
pub fn polyroots<'py>(py: Python<'py>, c: PyReadonlyArray1<'py, Complex64>) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    let c = c.as_slice()?;
    let roots = match polynomial::roots(c) {
        Some(roots) => roots,
        None => {
            crate::report_invalid(py, c"polyroots")?;
            vec![Complex64::new(f64::NAN, f64::NAN); polynomial::trimmed(c).len() - 1]
        }
    };
    Ok(PyArray1::from_vec(py, roots))
}

/// The polynomial with roots `roots` and leading coefficient `m`.
#[pyfunction]
pub fn polyfromroots<'py>(roots: &Bound<'py, PyUntypedArray>, m: &Bound<'py, PyUntypedArray>) -> PyResult<Bound<'py, PyAny>> {
    numbers!(roots, T => {
        let r = roots.extract::<PyReadonlyArray1<T>>()?;
        let m = m.extract::<PyReadonlyArray0<T>>()?.as_array()[()];
        Ok(PyArray1::from_vec(roots.py(), polynomial::from_roots(m, r.as_slice()?)).into_any())
    })
}

/// The derivative of the polynomial `c`.
#[pyfunction]
pub fn polyder<'py>(c: &Bound<'py, PyUntypedArray>) -> PyResult<Bound<'py, PyAny>> {
    numbers!(c, T => Ok(PyArray1::from_vec(c.py(), polynomial::derivative(c.extract::<PyReadonlyArray1<T>>()?.as_slice()?)).into_any()))
}

/// Decodes consecutive cells of digits in the mixed radix `bases`. The Python wrapper checks their lengths and restores the shape.
#[pyfunction]
pub fn polydecode<'py>(bases: &Bound<'py, PyUntypedArray>, digits: &Bound<'py, PyUntypedArray>) -> PyResult<Bound<'py, PyAny>> {
    numbers!(bases, T => {
        let (bases, digits) = (bases.extract::<PyReadonlyArray1<T>>()?, digits.extract::<PyReadonlyArray1<T>>()?);
        Ok(PyArray1::from_vec(bases.py(), polynomial::decode(bases.as_slice()?, digits.as_slice()?)).into_any())
    })
}

/// The polynomial of degree `deg` that fits the points `(x, y)` by least squares.
#[pyfunction]
pub fn polyfit<'py>(x: &Bound<'py, PyUntypedArray>, y: &Bound<'py, PyUntypedArray>, deg: usize) -> PyResult<Bound<'py, PyAny>> {
    floats!(x, F => {
        let (x, y) = (x.extract::<PyReadonlyArray1<F>>()?, y.extract::<PyReadonlyArray1<F>>()?);
        Ok(PyArray1::from_vec(x.py(), polynomial::fit(x.as_slice()?, y.as_slice()?, deg)).into_any())
    })
}

/// Adds this module's functions to `m`.
pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [wrap_pyfunction!(polyval, m)?, wrap_pyfunction!(polyvalfromroots, m)?, wrap_pyfunction!(polyroots, m)?, wrap_pyfunction!(polyfromroots, m)?,
        wrap_pyfunction!(polyder, m)?, wrap_pyfunction!(polydecode, m)?, wrap_pyfunction!(polyfit, m)?] {
        m.add_function(f)?;
    }
    Ok(())
}
