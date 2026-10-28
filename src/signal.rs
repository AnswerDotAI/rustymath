//! Correlations and convolutions, which slide a kernel along an array's leading axes, and overlap-add, which adds windows back where they were read.
use crate::agreement::{LINE, Mapping};
use crate::buffer::Element;
use crate::error::{Error, Kind};
use crate::gather::{self, Steps};
use crate::layout::{Layout, count, strides};
use crate::linalg::{Inner, Mapped, Orientation, Scalar};
use crate::structure::{self, Plan, Windows};
use std::borrow::Cow;

/// Which positions a correlation gives along an axis of `n` items, for a window of `w` items.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// The `n - w + 1` positions where the window lies inside the array.
    Valid,
    /// The `n + w - 1` positions where the window overlaps the array, with `w - 1` fills on each side.
    Full,
    /// `n` positions, with `w / 2` fills before the array and `(w - 1) / 2` after it, as NumPy and SciPy centre them.
    Same,
}

/// What a correlation reads, as [`correlation`] gives it. The correlation is the matrix product of the windows, with a row for each position, and the kernel, with a column for each output.
#[derive(Clone, Debug)]
pub struct Correlation {
    /// The array with the fills that the mode adds, or `None` where it adds none.
    pub padding: Option<Plan>,
    /// The windows of the padded array: each position, then the items of one window.
    pub windows: Plan,
    /// The kernel as the product reads it, with its window axes reversed for a convolution.
    pub kernel: Plan,
    /// The product's size: the positions, the items in one window, and the outputs at each position.
    pub size: [usize; 3],
    /// The result's shape: the positions along each sliding axis, then the kernel's output axes.
    pub shape: Vec<usize>,
}

/// The reads of a correlation of an array of shape `input` with a kernel of shape `kernel`, along the array's first `slide` axes, moving `stride` positions at a time along each. The kernel's shape is a window size for each sliding axis, then the array's other axes, then the outputs at each position. The product contracts each window with the kernel over the window and the array's other axes. `flip` gives the convolution, which reverses the kernel along its window axes. A Length error where the kernel doesn't fit the array, or a window doesn't fit its axis in valid mode.
pub fn correlation(input: &[usize], kernel: &[usize], slide: usize, mode: Mode, stride: &[usize], flip: bool) -> Result<Correlation, Error> {
    let rank = input.len();
    if slide == 0 || slide > rank { return Err(Error::new(Kind::Rank, format!("a correlation slides along 1 to {rank} axes, not {slide}"))); }
    if kernel.len() < rank || kernel[slide..rank] != input[slide..] {
        return Err(Error::new(
            Kind::Length, format!("a kernel of shape {kernel:?} needs {slide} window sizes, then the array's other axes {:?}", &input[slide..])));
    }
    if stride.len() != slide || stride.contains(&0) {
        return Err(Error::new(Kind::Domain, format!("a correlation needs a positive stride for each of its {slide} sliding axes")));
    }
    let window = &kernel[..slide];
    for (&w, &n) in window.iter().zip(input) {
        if w == 0 { return Err(Error::new(Kind::Length, "a window needs at least one item")); }
        if mode == Mode::Valid && w > n { return Err(Error::new(Kind::Length, format!("a window of {w} doesn't fit an axis of {n} in valid mode"))); }
    }
    let widths: Vec<(usize, usize)> = (0..rank).map(|a| match (window.get(a), mode) {
        (None, _) | (_, Mode::Valid) => (0, 0), (Some(&w), Mode::Full) => (w - 1, w - 1), (Some(&w), Mode::Same) => (w / 2, (w - 1) / 2)
    }).collect();
    let layout = Layout::from(input.to_vec());
    let padding = widths.iter().any(|&w| w != (0, 0)).then(|| structure::pad(&layout, &widths, structure::Edge::Fill)).transpose()?;
    let padded = padding.as_ref().map_or(&layout, |p| &p.layout);
    let signed = |n: &[usize]| n.iter().map(|&n| i64::try_from(n).unwrap_or(i64::MAX)).collect::<Vec<_>>();
    let slid = Windows::new(&signed(window), &signed(stride), |_| None, padded.shape())?;
    let windows = slid.plan(padded, padded.shape())?;
    let at = strides(kernel);
    let steps = (0..kernel.len()).map(|a| match a < slide && flip {
        true => Steps::Stride { start: (kernel[a] - 1) * at[a], len: kernel[a], step: -(at[a] as isize) }, false => Steps::along(kernel[a], at[a])
    }).collect();
    let outputs = &kernel[rank..];
    let size = [count(slid.frame()), count(slid.cell()), count(outputs)];
    Ok(Correlation { padding, windows, kernel: Plan { layout: Layout::from(kernel.to_vec()), steps }, size, shape: [slid.frame(), outputs].concat() })
}

/// `pair`'s correlation of `x` with `kernel` through the reads in `c`, as a row-major matrix with a row for each position and a column for each output. `x` and `kernel` hold their arrays' items in row-major order. Fills read as zero. Float and complex [`PlusTimes`](crate::linalg::PlusTimes) read the windows a block at a time and never build them. Other pairs read them whole.
pub fn correlate<T: Element, P: Inner<T>>(pair: &P, x: &[T], kernel: &[T], c: &Correlation) -> Vec<P::Out> {
    let padded = match &c.padding {
        None => Cow::Borrowed(x),
        Some(p) => {
            let mut out = Vec::with_capacity(p.layout.count());
            gather::walk(&mut out, x, 0, &p.steps, T::FILL, gather::Same);
            Cow::Owned(out)
        }
    };
    let read = |plan: &Plan| Mapping::Steps { base: 0, steps: plan.steps.clone().into() };
    let (windows, reversed) = (read(&c.windows), read(&c.kernel));
    pair.inner_mapped(Mapped { data: &padded, mapping: &windows }, Mapped { data: kernel, mapping: &reversed }, [Orientation::Normal; 2], c.size)
}

/// Each frame of `frames` added back where its window read: the transpose of the windows of `w` items that move `step` items at a time along an axis of `len` positions. `frames` has shape `[outer, n, mid, w, inner]`, with `n` frames along its second axis and each window along its fourth. Between them lie `mid` positions of other axes, which the result puts before the windows' axis: its shape is `[outer, mid, len, inner]`. Items that several windows read add up, and positions that no window read are 0. `len` must be at least `(n - 1) × step + w` where `n` isn't 0.
pub fn overlap_add<T: Scalar>(frames: &[T], [outer, n, mid, w, inner]: [usize; 5], step: usize, len: usize) -> Vec<T> {
    assert_eq!(frames.len(), outer * n * mid * w * inner, "the frames must fill their dimensions");
    assert!(n == 0 || (n - 1) * step + w <= len, "the windows must lie inside the result");
    let (block, row) = (w * inner, len * inner);
    let mut out = vec![T::zero(); outer * mid * row];
    if n == 0 || row == 0 { return out; }
    // Row `i` of the result sums its `n` frames. Frame `f` of the row starts `f × mid × block` items after the row's first frame. It adds into the row from item `f × step × inner`. A block that fills a cache line is added whole. Slicing a shorter block costs more than adding it. Each item of a shorter block takes one strided pass across all the frames.
    for (i, out) in out.chunks_exact_mut(row).enumerate() {
        let first = &frames[(i / mid * n * mid + i % mid) * block..];
        if block * std::mem::size_of::<T>() >= LINE {
            for (f, from) in first.chunks(mid * block).take(n).enumerate() {
                for (y, &x) in out[f * step * inner..][..block].iter_mut().zip(&from[..block]) { *y = *y + x; }
            }
        } else {
            for k in 0..block {
                for (y, &x) in out[k..].iter_mut().step_by(step * inner).zip(first[k..].iter().step_by(mid * block)).take(n) { *y = *y + x; }
            }
        }
    }
    out
}
