//! Resolved numerical operations over shared [`Read`](crate::agreement::Read) plans. [`Kernel::try_resolve`] fixes the computing dtype, result dtype and character policy from an object's [`Formula`](crate::facts::Formula). The same definition supplies native execution and graph formulas.
//!
//! [`DType::computes_in`] promotes unsupported Boolean and integer inputs to i64, then to the NumPy float width. Unsupported character arithmetic is a domain error.
use crate::buffer::{Buffer, DType, Element, FloatWidth, Source, View, Width};
use crate::error::{Error, Kind};
use crate::facts::Range;
use crate::reduce::native_reduction;
use crate::slice::Arg;

/// Numeric settings rebound for each execution, outside item loops.
#[derive(Clone, Copy, Debug, Default)]
pub struct Settings { pub tolerance: Option<f64> }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction { Left, Right }

#[derive(Clone, Copy)]
pub enum Seed<'a> { Identity, Initial(crate::agreement::Read<'a>), Tail(crate::agreement::Read<'a>) }

/// Write positions: numeric indices, the addresses of a selection's root source, or flat offsets. A write skips an invalid position: one outside the array, an invalid route position, or a [`crate::gather::Steps::FILL`] offset, as in [`crate::gather::Writes`].
#[derive(Clone, Copy)]
pub enum Positions<'a> { Indices(crate::agreement::Read<'a>, DType), Route(&'a crate::agreement::ReadPlan, usize), Offsets(&'a [usize]) }

impl Positions<'_> {
    pub fn len(self) -> usize { match self { Self::Indices(read, _) => read.plan.len(), Self::Route(plan, _) => plan.len(), Self::Offsets(offsets) => offsets.len() } }
    /// The positions that `range` of these writes target in an array of `size` items. A position outside the array, an invalid route position and a `FILL` offset all give `FILL`, which writes nothing.
    pub fn chunk(self, range: std::ops::Range<usize>, size: usize, scratch: &mut Buffer) -> Result<Vec<usize>, Error> {
        use crate::{agreement::ReadAt, gather::Steps};
        match self {
            Self::Indices(read, dtype) => crate::structure::positions(read.part_view(range, dtype, false, scratch)?, size),
            Self::Route(plan, root) => Ok(range.map(|i| match plan.at(i) {
                ReadAt::Input { source, position } if source == root && position < size => position,
                _ => Steps::FILL,
            }).collect()),
            Self::Offsets(offsets) => Ok(offsets[range].iter().map(|&o| if o < size { o } else { Steps::FILL }).collect()),
        }
    }
}

/// How a kernel reads its inputs before its formula runs (`Kernel::projection`, `with_projected`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Projection {
    /// As they are.
    AsIs,
    /// Characters as their codes, under a code policy, and floats as whole numbers.
    Codes,
    /// Floats as the whole numbers they stand for, where the kernel computes at an integral dtype from float arguments.
    Wholes,
}

/// A write's logical root, or storage transferred by its final reader.
pub enum WriteRoot<'a> { Read(crate::agreement::Read<'a>), Owned(Buffer) }
impl WriteRoot<'_> {
    /// The written array's starting items, read at `dtype` as the combining kernel reads its inputs (`Kernel::projection`).
    fn initialize(self, dtype: DType, checked: bool, projection: Projection, output: &mut Buffer) -> Result<(), Error> {
        match (self, projection) {
            (Self::Read(read), Projection::AsIs) => read.read_into(0..read.plan.len(), dtype, checked, output),
            (Self::Read(read), projection) => with_projected(&[read], None, projection, |reads| reads[0].read_into(0..read.plan.len(), dtype, checked, output)),
            (Self::Owned(buffer), _) if buffer.dtype() == dtype => { *output = buffer; Ok(()) }
            (Self::Owned(buffer), Projection::AsIs) => { *output = Buffer::cast_results(buffer.view(), dtype, checked)?; Ok(()) }
            (Self::Owned(buffer), projection) => { *output = with_projected_view(buffer.view(), projection, |view| Buffer::cast_results(view, dtype, checked))??; Ok(()) }
        }
    }
}

/// A compiled fold's total over `items` items, as its 8-byte slot holds it: a float as an f64's bits, an integer extended to 64 bits, and a Boolean as 0 or 1.
#[derive(Clone, Copy, Debug)]
pub struct Total { pub items: usize, pub bits: u64 }
impl Total {
    pub fn value<T: crate::buffer::Cast>(self) -> Result<T, Error> {
        use crate::buffer::{Floats, Ints};
        match T::DTYPE {
            DType::Boolean => T::cast(View::Booleans(&[self.bits != 0]), 0),
            DType::Integer(width) => {
                let value = match width {
                    Width::U8 => self.bits as u8 as i64, Width::I16 => self.bits as i16 as i64, Width::I32 => self.bits as i32 as i64,
                    Width::I64 => self.bits as i64
                };
                T::cast(View::Integers(Ints::I64(&[value])), 0)
            }
            DType::Float(_) => T::cast(View::Floats(Floats::F64(&[f64::from_bits(self.bits)])), 0),
            _ => Err(Error::new(Kind::Domain, "compiled totals have a real machine dtype")),
        }
    }
}

pub enum Piece<'a> { Items(View<'a>), Total(Total) }

pub type Pieces<'a> = dyn Fn(std::ops::Range<usize>, &mut dyn FnMut(usize, Piece<'_>)) -> Result<(), Error> + Sync + 'a;

/// One resolved mathematical definition and its captured numerical policies.
#[derive(Clone)]
pub struct Kernel {
    pub variant: crate::facts::Variant, result: DType, boolean_inputs: bool, repair: bool, raising: bool, wholes: bool,
    policy: Option<crate::characters::Policy>, definition: std::sync::Arc<dyn Application>
}

impl std::fmt::Debug for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernel").field("variant", &self.variant).field("arity", &self.arity()).finish()
    }
}

trait Application: Send + Sync {
    fn exact(&self, args: &[num_rational::BigRational]) -> Result<Option<num_rational::BigRational>, Error>;
    fn has_exact(&self) -> bool;
    fn integer(&self, variant: crate::facts::Variant, args: &[i64]) -> Result<i64, Error>;
    fn real(&self, variant: crate::facts::Variant, args: &[f64], settings: Settings, repair: bool) -> Result<f64, Error>;
    fn complex(&self, variant: crate::facts::Variant, args: &[num_complex::Complex64], settings: Settings) -> Result<num_complex::Complex64, Error>;
    fn arity(&self) -> usize;
    fn inputs(&self) -> &[crate::facts::Range];
    fn integer_formula(&self) -> bool;
    fn ordered_kinds(&self, classes: [i32; 2]) -> Option<crate::characters::ByKinds>;
    fn may_raise(&self, variant: crate::facts::Variant, ranges: &[crate::facts::Range]) -> bool;
    fn formula(
        &self, ops: &dyn crate::recipe::Ops<f64, V = crate::facts::Slot, B = crate::facts::Slot, I = crate::facts::Slot, W = crate::facts::Slot>,
        x: &[crate::facts::Slot]) -> Option<crate::facts::Slot>;
    fn test(
        &self, ops: &dyn crate::recipe::Ops<f64, V = crate::facts::Slot, B = crate::facts::Slot, I = crate::facts::Slot, W = crate::facts::Slot>,
        x: &[crate::facts::Slot]) -> Option<crate::facts::Slot>;
    fn repair_predicate(
        &self, ops: &dyn crate::recipe::Ops<f64, V = crate::facts::Slot, B = crate::facts::Slot, I = crate::facts::Slot, W = crate::facts::Slot>,
        x: &[crate::facts::Slot]) -> Option<crate::facts::Slot>;
    fn repair_routine(&self) -> Option<crate::facts::Repair>;
    fn validity(
        &self, variant: crate::facts::Variant,
        ops: &dyn crate::recipe::Ops<f64, V = crate::facts::Slot, B = crate::facts::Slot, I = crate::facts::Slot, W = crate::facts::Slot>,
        x: &[crate::facts::Slot], result: crate::facts::Slot) -> Option<(Kind, crate::facts::Slot)>;
    fn partials(
        &self, ops: &dyn crate::recipe::Ops<f64, V = crate::facts::Slot, B = crate::facts::Slot, I = crate::facts::Slot, W = crate::facts::Slot>,
        x: &[crate::facts::Slot], result: crate::facts::Slot, needed: &[bool], output: &mut [Option<crate::facts::Slot>]);
    fn range(&self, x: &[crate::facts::Range]) -> crate::facts::Range;
    fn identity(&self) -> Option<f64>;
    fn kept(&self, constants: &[Option<f64>]) -> Option<usize>;
    fn reduction(&self, variant: crate::facts::Variant) -> Option<crate::reduce::Reduce>;
    fn inverse_fold(&self) -> Option<crate::reduce::Reduce>;
    fn truth_table(&self) -> Option<u8>;
    fn product(
        &self, variant: crate::facts::Variant, fold: Option<crate::reduce::Reduce>, boolean_inputs: bool, wanted: Option<DType>, settings: Settings
    ) -> Option<NativeProduct>;
    /// `judged`: whether each item's result is checked for errors, which a kernel skips where its arguments' types can't make one (`Kernel::may_raise`).
    fn map(
        &self, variant: crate::facts::Variant, inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, settings: Settings, repair: bool,
        judged: bool, output: &mut Buffer) -> Result<(), Error>;
    fn fold_pieces(
        &self, variant: crate::facts::Variant, dims: [usize; 3], seed: Seed<'_>, direction: Direction, settings: Settings, pieces: &Pieces<'_>,
        output: &mut Buffer) -> Result<(), Error>;
    fn fold_read(
        &self, variant: crate::facts::Variant, shape: &[usize], axis: usize, seed: Seed<'_>, direction: Direction, settings: Settings,
        input: crate::agreement::Read<'_>, output: &mut Buffer) -> Result<(), Error>;
    fn adjacent(
        &self, variant: crate::facts::Variant, dims: [usize; 3], seed: Option<crate::agreement::Read<'_>>, flip: bool, settings: Settings,
        input: crate::agreement::Read<'_>, output: &mut Buffer) -> Result<(), Error>;
    fn at(
        &self, variant: crate::facts::Variant, root: WriteRoot<'_>, values: crate::agreement::Read<'_>, ids: Positions<'_>, checked: bool, projection: Projection,
        settings: Settings, output: &mut Buffer
    ) -> Result<(), Error>;
    fn scan(
        &self, variant: crate::facts::Variant, dims: [usize; 3], seed: Seed<'_>, direction: Direction, settings: Settings, input: crate::agreement::Read<'_>,
        output: &mut Buffer) -> Result<(), Error>;
}

/// `object`, for arguments whose values lie in `inputs`, the ranges of their types.
struct Definition<K, const N: usize> { object: K, inputs: [crate::facts::Range; N] }

impl<K: crate::facts::Formula<N>, const N: usize> Application for Definition<K, N> {
    fn exact(&self, args: &[num_rational::BigRational]) -> Result<Option<num_rational::BigRational>, Error> {
        if args.len() != N { return Err(Error::new(Kind::Length, "scalar operand count changed")); }
        self.object.exact(std::array::from_fn(|i| &args[i]))
    }
    fn has_exact(&self) -> bool { K::HAS_EXACT }
    fn integer(&self, variant: crate::facts::Variant, args: &[i64]) -> Result<i64, Error> {
        let x: [i64; N] = args.try_into().map_err(|_| Error::new(Kind::Length, "scalar operand count changed"))?;
        let width = match variant.dtype {
            DType::Integer(w) => w, DType::Boolean => Width::U8,
            _ => return Err(Error::new(Kind::Domain, "an integer scalar call needs an integer computing dtype")),
        };
        with_width!(width, I => {
            let ops = crate::recipe::IntegerOps::<I>::new(variant);
            crate::error::checked(|| scalar_result(integer_value(&self.object, variant, &ops, x.map(<I as crate::recipe::Overflowing>::low)))).map(Into::into)
        })
    }
    fn real(&self, variant: crate::facts::Variant, args: &[f64], settings: Settings, repair: bool) -> Result<f64, Error> {
        let x: [f64; N] = args.try_into().map_err(|_| Error::new(Kind::Length, "scalar operand count changed"))?;
        let ops = crate::recipe::NativeOps::<{ crate::FUSED }> { tolerance: settings.tolerance };
        match variant.dtype {
            DType::Float(FloatWidth::F64) => scalar_result(real_value(&self.object, variant, &ops, x, repair)),
            DType::Float(width) => {
                let value = scalar_result(real_value(&self.object, variant, &ops, x.map(|x| x as f32), repair))?;
                Ok(if width == FloatWidth::F16 { half::f16::from_f32(value).to_f64() } else { f64::from(value) })
            }
            _ => Err(Error::new(Kind::Domain, "a real scalar call needs a floating computing dtype")),
        }
    }
    fn complex(&self, variant: crate::facts::Variant, args: &[num_complex::Complex64], settings: Settings) -> Result<num_complex::Complex64, Error> {
        let x: [num_complex::Complex64; N] = args.try_into().map_err(|_| Error::new(Kind::Length, "scalar operand count changed"))?;
        let ops = crate::recipe::NativeOps::<{ crate::FUSED }> { tolerance: settings.tolerance };
        match variant.dtype {
            DType::Complex(crate::buffer::ComplexWidth::F64) => scalar_result(complex_value(&self.object, variant, &ops, x)),
            DType::Complex(crate::buffer::ComplexWidth::F32) => {
                let value = scalar_result(complex_value(&self.object, variant, &ops, x.map(|x| num_complex::Complex32::new(x.re as f32, x.im as f32))))?;
                Ok(num_complex::Complex64::new(f64::from(value.re), f64::from(value.im)))
            }
            _ => Err(Error::new(Kind::Domain, "a complex scalar call needs a complex computing dtype")),
        }
    }
    fn arity(&self) -> usize { N }
    fn inputs(&self) -> &[crate::facts::Range] { &self.inputs }
    fn integer_formula(&self) -> bool { K::INTEGER_FORMULA }
    fn may_raise(&self, variant: crate::facts::Variant, ranges: &[crate::facts::Range]) -> bool {
        if variant.dtype == DType::Boolean && ranges.iter().all(|r| r.within(0.0, 1.0)) { return boolean_raises(&self.object, variant); }
        if matches!(variant.dtype, DType::Complex(_)) || variant.dtype.is_integral() && !K::INTEGER_FORMULA { return true; }
        let Ok(args) = ranges.try_into() else { return true; };
        // A float result raises only outside the domain that the formula declares (`Formula::domain`): range arithmetic can't show that a result isn't NaN.
        if matches!(variant.dtype, DType::Float(_)) && self.object.domain(variant.dtype, args).is_some() { return false; }
        self.object.emit_valid::<f64, _>(variant, &crate::recipe::RangeOps(variant.dtype), args, self.object.range(args))
            .is_some_and(|(_, valid)| valid != Some(true))
    }
    fn formula(
        &self, ops: &dyn crate::recipe::Ops<f64, V = crate::facts::Slot, B = crate::facts::Slot, I = crate::facts::Slot, W = crate::facts::Slot>,
        x: &[crate::facts::Slot]) -> Option<crate::facts::Slot> { self.object.emit(ops, x.try_into().ok()?) }
    fn test(
        &self, ops: &dyn crate::recipe::Ops<f64, V = crate::facts::Slot, B = crate::facts::Slot, I = crate::facts::Slot, W = crate::facts::Slot>,
        x: &[crate::facts::Slot]) -> Option<crate::facts::Slot> { self.object.emit_test(ops, x.try_into().ok()?) }
    fn repair_predicate(
        &self, ops: &dyn crate::recipe::Ops<f64, V = crate::facts::Slot, B = crate::facts::Slot, I = crate::facts::Slot, W = crate::facts::Slot>,
        x: &[crate::facts::Slot]) -> Option<crate::facts::Slot> { self.object.emit_repair(ops, x.try_into().ok()?) }
    fn repair_routine(&self) -> Option<crate::facts::Repair> { crate::facts::Formula::repair_routine(&self.object) }
    fn validity(
        &self, variant: crate::facts::Variant,
        ops: &dyn crate::recipe::Ops<f64, V = crate::facts::Slot, B = crate::facts::Slot, I = crate::facts::Slot, W = crate::facts::Slot>,
        x: &[crate::facts::Slot], result: crate::facts::Slot) -> Option<(Kind, crate::facts::Slot)> {
        self.object.emit_valid(variant, ops, x.try_into().ok()?, result)
    }
    fn partials(
        &self, ops: &dyn crate::recipe::Ops<f64, V = crate::facts::Slot, B = crate::facts::Slot, I = crate::facts::Slot, W = crate::facts::Slot>,
        x: &[crate::facts::Slot], result: crate::facts::Slot, needed: &[bool], output: &mut [Option<crate::facts::Slot>]) {
        let (Ok(x), Ok(needed)) = (x.try_into(), needed.try_into()) else { output.fill(None); return; };
        output.copy_from_slice(&self.object.emit_partials(ops, x, result, needed));
    }
    fn range(&self, x: &[crate::facts::Range]) -> crate::facts::Range { x.try_into().map_or(crate::facts::Range::ANY, |x| self.object.range(x)) }
    fn ordered_kinds(&self, classes: [i32; 2]) -> Option<crate::characters::ByKinds> {
        if N != 2 || self.object.character_rule() != Some(crate::characters::Rule::Ordered) { return None; }
        Some(crate::characters::ByKinds::ordered(classes, self.object.ordered_kinds(classes)?, K::BOOLEAN))
    }
    fn identity(&self) -> Option<f64> { self.object.identity().last().copied().flatten() }
    fn kept(&self, constants: &[Option<f64>]) -> Option<usize> {
        let identity = self.object.identity();
        if N != 2 || constants.len() != 2 { return None; }
        [1, 0].into_iter().find(|&i| constants[i].is_some() && constants[i] == identity[i]).map(|i| 1 - i)
    }
    fn reduction(&self, variant: crate::facts::Variant) -> Option<crate::reduce::Reduce> { reduction(&self.object, variant) }
    fn inverse_fold(&self) -> Option<crate::reduce::Reduce> { self.object.inverse_fold() }
    fn truth_table(&self) -> Option<u8> { self.object.truth_table() }
    fn product(
        &self, variant: crate::facts::Variant, fold: Option<crate::reduce::Reduce>, boolean_inputs: bool, wanted: Option<DType>, settings: Settings
    ) -> Option<NativeProduct> { product_for::<K, N>(&self.object, variant, fold, boolean_inputs, wanted, settings) }
    fn map(
        &self, variant: crate::facts::Variant, inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, settings: Settings, repair: bool,
        judged: bool, output: &mut Buffer) -> Result<(), Error> { map_formula::<K, N>(&self.object, variant, inputs, range, settings, repair, judged, output) }
    fn fold_pieces(
        &self, variant: crate::facts::Variant, dims: [usize; 3], seed: Seed<'_>, direction: Direction, settings: Settings, pieces: &Pieces<'_>,
        output: &mut Buffer) -> Result<(), Error> { fold_formula::<K, N>(&self.object, variant, dims, seed, direction, settings, pieces, output) }
    fn fold_read(
        &self, variant: crate::facts::Variant, shape: &[usize], axis: usize, seed: Seed<'_>, direction: Direction, settings: Settings,
        input: crate::agreement::Read<'_>, output: &mut Buffer) -> Result<(), Error> {
        fold_read_formula::<K, N>(&self.object, variant, shape, axis, seed, direction, settings, input, output)
    }
    fn adjacent(
        &self, variant: crate::facts::Variant, dims: [usize; 3], seed: Option<crate::agreement::Read<'_>>, flip: bool, settings: Settings,
        input: crate::agreement::Read<'_>, output: &mut Buffer) -> Result<(), Error> {
        adjacent_formula::<K, N>(&self.object, variant, dims, seed, flip, settings, input, output)
    }
    fn at(
        &self, variant: crate::facts::Variant, root: WriteRoot<'_>, values: crate::agreement::Read<'_>, ids: Positions<'_>, checked: bool, projection: Projection,
        settings: Settings, output: &mut Buffer
    ) -> Result<(), Error> { at_formula(&self.object, variant, root, values, ids, checked, projection, settings, output) }
    fn scan(
        &self, variant: crate::facts::Variant, dims: [usize; 3], seed: Seed<'_>, direction: Direction, settings: Settings, input: crate::agreement::Read<'_>,
        output: &mut Buffer) -> Result<(), Error> { scan_formula(&self.object, variant, dims, seed, direction, settings, input, output) }
}

impl Kernel {
    pub fn circle(code: i64, inverse: bool, real_part: bool, variant: crate::facts::Variant, types: [DType; 1]) -> Result<Option<Self>, Error> {
        macro_rules! picked {
            ($normal:expr, $real:expr) => {
                if real_part && !matches!(variant.dtype, DType::Complex(_)) {
                    Self::try_resolve($real, variant, types)
                } else {
                    Self::try_resolve($normal, variant, types)
                }
            };
        }
        let Some(mut kernel) = (match (code, inverse) {
            (0, _) => picked!(crate::Sqrt1mx2, crate::RealSqrt1mx2), (1, false) => picked!(crate::Sin, crate::Sin),
            (2, false) => picked!(crate::Cos, crate::Cos), (3, false) => picked!(crate::Tan, crate::Tan),
            (4, false) => picked!(crate::Sqrt1px2, crate::Sqrt1px2), (5, false) => picked!(crate::Sinh, crate::Sinh),
            (6, false) => picked!(crate::Cosh, crate::Cosh), (7, false) => picked!(crate::Tanh, crate::Tanh),
            (1, true) => picked!(crate::Asin, crate::RealAsin), (2, true) => picked!(crate::Acos, crate::RealAcos),
            (3, true) => picked!(crate::Atan, crate::Atan), (4, true) => picked!(crate::Sqrtx2m1, crate::RealSqrtx2m1),
            (5, true) => picked!(crate::Asinh, crate::Asinh), (6, true) => picked!(crate::Acosh, crate::RealAcosh),
            (7, true) => picked!(crate::Atanh, crate::RealAtanh), (8, inverse) => picked!(crate::sqrts::Sqrtm1mx2(inverse), crate::RealSqrtm1mx2(inverse)),
            _ => return Ok(None)
        })? else { return Ok(None); };
        if real_part && let DType::Complex(width) = kernel.result { kernel.result = DType::Float(width.into()); }
        Ok(Some(kernel))
    }
    pub fn new<K: crate::facts::Formula<N>, const N: usize>(object: K, variant: crate::facts::Variant) -> Self { Self::typed(object, variant, [variant.dtype; N]) }
    /// The kernel for arguments of `types`, whose values bound the errors it can raise (`may_raise`).
    fn typed<K: crate::facts::Formula<N>, const N: usize>(object: K, variant: crate::facts::Variant, types: [DType; N]) -> Self {
        let result = object.result_dtype(variant.dtype);
        let definition: std::sync::Arc<dyn Application> = std::sync::Arc::new(Definition::<K, N> { object, inputs: types.map(Range::of_dtype) });
        let raising = definition.may_raise(variant, definition.inputs());
        let wholes = variant.dtype.is_integral() && types.iter().any(|d| matches!(d, DType::Float(_)));
        Self { repair: true, raising, result, policy: None, boolean_inputs: types.iter().all(|&d| d == DType::Boolean), wholes, variant, definition }
    }
    pub fn try_resolve<K: crate::facts::Formula<N>, const N: usize>(object: K, mut variant: crate::facts::Variant, types: [DType; N]) -> Result<Option<Self>, Error> {
        use crate::characters::{ByKinds, Policy};
        let characters = if N == 2 {
            object.character_policy([types[0], types[1]])?
        } else if N == 1 && types[0] == DType::Character && object.character_rule() == Some(crate::characters::Rule::Identity) {
            Some(Policy::Codes { computing: DType::Integer(Width::I32), result: DType::Character })
        } else {
            if types.contains(&DType::Character) { return Err(Error::new(Kind::Domain, "function does not take character arguments")); }
            None
        };
        let result = match characters {
            Some(Policy::Codes { computing, result }) => { variant.dtype = computing; result }
            Some(Policy::Kinds(kind)) => {
                let result = match kind { ByKinds::Truth(_) => DType::Boolean, ByKinds::Left => types[0], ByKinds::Right => types[1] };
                variant.dtype = if matches!(kind, ByKinds::Truth(_)) || result == DType::Character { DType::Integer(Width::I32) } else { result };
                result
            }
            None => {
                if K::BOOLEAN && variant.dtype.is_integral() {
                    let Some(dtype) = types.into_iter().try_fold(DType::Boolean, DType::common) else { return Ok(None); };
                    variant.dtype = dtype;
                }
                let Some(dtype) = object.computing_dtype(variant.dtype) else { return Ok(None); };
                variant.dtype = dtype;
                object.result_dtype(variant.dtype)
            }
        };
        let mut kernel = Self::typed(object, variant, types);
        kernel.result = result;
        kernel.policy = characters;
        Ok(Some(kernel))
    }
    pub fn arity(&self) -> usize { self.definition.arity() }
    pub fn result_dtype(&self) -> DType { self.result }
    pub fn input_policy(&self) -> Option<crate::characters::Policy> { self.policy }
    pub fn selected_input(&self) -> Option<usize> {
        use crate::characters::{ByKinds, Policy};
        match self.policy { Some(Policy::Kinds(ByKinds::Left)) => Some(0), Some(Policy::Kinds(ByKinds::Right)) => Some(1), _ => None }
    }
    /// Resolve an order-only operation on disjoint logical domains, before numeric promotion.
    pub fn ordered(mut self, classes: &[Option<i32>], types: &[DType]) -> Self {
        let [Some(left), Some(right)] = classes else { return self; };
        if left == right { return self; }
        let Some(policy) = self.definition.ordered_kinds([*left, *right]) else { return self; };
        use crate::characters::{ByKinds, Policy};
        self.result = match policy { ByKinds::Truth(_) => DType::Boolean, ByKinds::Left => types[0], ByKinds::Right => types[1] };
        self.policy = Some(Policy::Kinds(policy));
        self
    }
    pub fn exact(&self, args: &[num_rational::BigRational]) -> Result<Option<num_rational::BigRational>, Error> { self.definition.exact(args) }
    pub fn has_exact(&self) -> bool { self.definition.has_exact() }
    /// The formula on one item of each integer argument, at the kernel's integer computing width, with its checked or wrapping variant. A Boolean computing kind reads 0 and 1.
    pub fn integers(&self, args: &[i64]) -> Result<i64, Error> { self.definition.integer(self.variant, args) }
    pub fn real(&self, args: &[f64], settings: Settings) -> Result<f64, Error> { crate::error::checked(|| self.definition.real(self.variant, args, settings, self.repair)) }
    pub fn complex(&self, args: &[num_complex::Complex64], settings: Settings) -> Result<num_complex::Complex64, Error> {
        let value = crate::error::checked(|| self.definition.complex(self.variant, args, settings))?;
        Ok(if matches!(self.result, DType::Complex(_)) { value } else { num_complex::Complex64::new(value.re, 0.0) })
    }
    fn kind_result<T: Copy>(&self, x: &[T], truth: impl FnOnce(bool) -> T) -> Option<T> {
        use crate::characters::{ByKinds, Policy};
        Some(match self.policy? {
            Policy::Kinds(ByKinds::Truth(value)) => truth(value), Policy::Kinds(ByKinds::Left) => x[0], Policy::Kinds(ByKinds::Right) => x[1],
            _ => return None,
        })
    }
    pub fn formula(
        &self, ops: &dyn crate::recipe::Ops<f64, V = crate::facts::Slot, B = crate::facts::Slot, I = crate::facts::Slot, W = crate::facts::Slot>,
        x: &[crate::facts::Slot]) -> Option<crate::facts::Slot> {
        if let Some(result) = self.kind_result(x, |value| ops.equal(ops.number(0.0), ops.number(u8::from(!value) as f64))) { return Some(result); }
        if self.variant.dtype.is_integral() && self.variant.fails_on_overflow() && !self.definition.integer_formula() { return None; }
        self.definition.formula(ops, x)
    }
    pub fn test(
        &self, ops: &dyn crate::recipe::Ops<f64, V = crate::facts::Slot, B = crate::facts::Slot, I = crate::facts::Slot, W = crate::facts::Slot>,
        x: &[crate::facts::Slot]) -> Option<crate::facts::Slot> {
        if matches!(self.policy, Some(crate::characters::Policy::Kinds(_))) { return self.formula(ops, x); }
        self.definition.test(ops, x)
    }
    pub fn repair_predicate(
        &self, ops: &dyn crate::recipe::Ops<f64, V = crate::facts::Slot, B = crate::facts::Slot, I = crate::facts::Slot, W = crate::facts::Slot>,
        x: &[crate::facts::Slot]) -> Option<crate::facts::Slot> { self.definition.repair_predicate(ops, x) }
    pub fn repair_routine(&self) -> Option<crate::facts::Repair> { if self.repair { self.definition.repair_routine() } else { None } }
    /// Whether checked execution can record a numeric data error, for arguments that hold any values of their types.
    pub fn may_raise(&self) -> bool { self.variant.checked && (self.result == DType::Character || self.wholes || self.raising) }
    /// Whether checked execution can record a numeric data error where each argument holds values of its type within its range in `narrowed`, such as a constant's own value or the nonnegative numbers.
    pub fn may_raise_on(&self, narrowed: &[Range]) -> bool {
        self.variant.checked && (self.result == DType::Character || self.wholes || self.definition.may_raise(self.variant, &self.points(narrowed)))
    }
    /// Whether every result is nonnegative where each argument holds values of its type within its range in `narrowed`. A fold by a dyadic formula keeps the fact where `nonnegative_on(&[Range::NONNEGATIVE; 2])` holds.
    pub fn nonnegative_on(&self, narrowed: &[Range]) -> bool { self.range(&self.points(narrowed)).lo >= 0.0 }
    /// Each argument's range: its type's values within its range in `narrowed`.
    fn points(&self, narrowed: &[Range]) -> Vec<Range> { self.definition.inputs().iter().zip(narrowed).map(|(&range, &n)| range.meet(n)).collect() }
    pub fn validity(
        &self, ops: &dyn crate::recipe::Ops<f64, V = crate::facts::Slot, B = crate::facts::Slot, I = crate::facts::Slot, W = crate::facts::Slot>,
        x: &[crate::facts::Slot], result: crate::facts::Slot) -> Option<(Kind, crate::facts::Slot)> {
        if !self.may_raise() { return None; }
        self.definition.validity(self.variant, ops, x, result)
    }
    pub fn partials(
        &self, ops: &dyn crate::recipe::Ops<f64, V = crate::facts::Slot, B = crate::facts::Slot, I = crate::facts::Slot, W = crate::facts::Slot>,
        x: &[crate::facts::Slot], result: crate::facts::Slot, needed: &[bool], output: &mut [Option<crate::facts::Slot>]) {
        if let Some(side) = self.kind_result(&[Some(0), Some(1)], |_| None) {
            for (i, (&needed, output)) in needed.iter().zip(output).enumerate() {
                *output = needed.then(|| ops.number(u8::from(side == Some(i)) as f64));
            }
            return;
        }
        self.definition.partials(ops, x, result, needed, output)
    }
    pub fn range(&self, x: &[crate::facts::Range]) -> crate::facts::Range {
        self.kind_result(x, |value| Range::of(Some(u8::from(value) as f64))).unwrap_or_else(|| self.definition.range(x))
    }
    pub fn identity(&self) -> Option<f64> { self.definition.identity() }
    /// The argument that a dyadic call returns unchanged where `constants` gives the arguments that are one number, 0 for the left and 1 for the right, from the formula's identities: `x+0` and `0+x` keep `x`, `x-0` keeps `x`, and `0-x` keeps nothing.
    pub fn kept(&self, constants: &[Option<f64>]) -> Option<usize> { self.definition.kept(constants) }
    /// The reduction by which a fold through this kernel combines its items, as `reduction` gives it at the kernel's variant.
    pub fn fold(&self) -> Option<crate::reduce::Reduce> { self.definition.reduction(self.variant) }
    /// The reduction that an unseeded right fold by this formula runs as an alternating one, as `-` runs as an alternating `+`.
    pub fn inverse_fold(&self) -> Option<crate::reduce::Reduce> { self.definition.inverse_fold() }
    pub fn truth_table(&self) -> Option<u8> { self.definition.truth_table() }
    pub fn total_dtype(&self) -> Option<DType> { self.fold()?.total_dtype(self.variant) }
    pub fn fold_dtype(&self, input: DType, len: usize, seed: Option<DType>) -> DType {
        self.total_dtype().unwrap_or_else(|| match (len, seed) { (0, Some(seed)) => seed, (1, None) if input != DType::Boolean => input, _ => self.result })
    }
    pub fn scan_dtype(&self) -> DType {
        let carry = match self.variant.dtype { DType::Float(FloatWidth::F16) => DType::Float(FloatWidth::F32), dtype => dtype };
        self.total_dtype().unwrap_or_else(|| carry.common(self.result).unwrap_or(self.result))
    }
    pub fn product_dtype(&self, fold: &Kernel, input: DType) -> Option<DType> {
        self.native_product(crate::facts::Variant { dtype: input, ..self.variant }, fold, input == DType::Boolean, None, Settings::default())
            .map(|p| p.result)
    }
    fn native_product(
        &self, variant: crate::facts::Variant, fold: &Kernel, boolean_inputs: bool, wanted: Option<DType>, settings: Settings
    ) -> Option<NativeProduct> {
        if matches!(self.policy, Some(crate::characters::Policy::Kinds(_))) { return None; }
        self.definition.product(variant, fold.fold(), boolean_inputs, wanted, settings)
    }
    /// How this kernel reads its inputs (`with_projected`): characters as codes under a code policy, floats as whole numbers where it computes at an integral dtype from float arguments, and otherwise as they are.
    fn projection(&self) -> Projection {
        match self.policy { Some(crate::characters::Policy::Codes { .. }) => Projection::Codes, None if self.wholes => Projection::Wholes, _ => Projection::AsIs }
    }
    /// `call` on `inputs` and `seed`'s read as this kernel reads them (`projection`): over `range`, or over the whole of each read where `None`. `range` applies to every read, so a call with a seed passes `None`.
    fn projected<R>(
        &self, inputs: &[crate::agreement::Read<'_>], range: Option<std::ops::Range<usize>>, seed: Seed<'_>,
        call: impl FnOnce(&[crate::agreement::Read<'_>], Seed<'_>) -> Result<R, Error>
    ) -> Result<R, Error> {
        let projection = self.projection();
        if projection == Projection::AsIs { return call(inputs, seed); }
        let mut all = inputs.to_vec();
        match seed { Seed::Initial(r) | Seed::Tail(r) => all.push(r), Seed::Identity => {} }
        with_projected(&all, range, projection, |projected| {
            let seed = match seed {
                Seed::Initial(_) => Seed::Initial(projected[inputs.len()]), Seed::Tail(_) => Seed::Tail(projected[inputs.len()]), Seed::Identity => Seed::Identity
            };
            call(&projected[..inputs.len()], seed)
        })
    }
    pub fn map(&self, inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, settings: Settings, output: &mut Buffer) -> Result<(), Error> {
        use crate::characters::{ByKinds, Policy};
        if inputs.len() != self.arity() { return Err(Error::new(Kind::Length, "kernel operand count changed")); }
        let wanted = output.dtype();
        if output.dtype() != self.result { *output = Buffer::with_capacity(self.result, range.len()); }
        match self.policy {
            Some(Policy::Kinds(ByKinds::Truth(value))) => *output = Buffer::Booleans(vec![value; range.len()]),
            Some(Policy::Kinds(side)) => {
                let side = usize::from(side == ByKinds::Right);
                inputs[side].read_into(range, self.result, self.variant.checked, output)?;
            }
            _ => self.projected(inputs, Some(range.clone()), Seed::Identity, |inputs, _| self.definition.map(self.variant, inputs, range, settings, self.repair, self.may_raise(), output))?,
        }
        finish(output, wanted, self.variant)
    }
    pub fn fold_pieces(
        &self, dims: [usize; 3], seed: Seed<'_>, direction: Direction, settings: Settings, pieces: &Pieces<'_>, output: &mut Buffer) -> Result<(), Error> {
        let dtype = output.dtype();
        match self.projection() {
            projection @ (Projection::Codes | Projection::Wholes) => {
                let projected = |range, f: &mut dyn FnMut(usize, Piece<'_>)| {
                    pieces(range, &mut |range, piece| match piece {
                        Piece::Items(view) => {
                            let n = view.len();
                            // A piece that isn't whole numbers records its error, and the fold reads zeros in its place.
                            if let Err(error) = with_projected_view(view, projection, |view| f(range, Piece::Items(view))) {
                                crate::error::record(error);
                                f(range, Piece::Items(View::Integers(crate::buffer::Ints::I64(&vec![0; n]))));
                            }
                        }
                        Piece::Total(total) => f(range, Piece::Total(total))
                    })
                };
                crate::error::checked(|| self.projected(&[], None, seed, |_, seed| self.definition.fold_pieces(self.variant, dims, seed, direction, settings, &projected, output)))?;
            }
            Projection::AsIs => self.definition.fold_pieces(self.variant, dims, seed, direction, settings, pieces, output)?,
        }
        finish(output, dtype, self.variant)
    }
    pub fn scan(
        &self, dims: [usize; 3], seed: Seed<'_>, direction: Direction, settings: Settings, input: crate::agreement::Read<'_>, output: &mut Buffer
    ) -> Result<(), Error> {
        let dtype = output.dtype();
        self.projected(&[input], None, seed, |inputs, seed| self.definition.scan(self.variant, dims, seed, direction, settings, inputs[0], output))?;
        finish(output, dtype, self.variant)
    }
    /// Each item of `input`, an array of shape `dims`, combined with the item a row before it along the middle axis, and each lane's first row with `seed`, or kept as it is without one, as an inverse scan runs: `flip` swaps the arguments, so the earlier item comes second. The results take `output`'s dtype, so a comparison's 0s and 1s and an unseeded first row keep the items' type where the caller asks for it.
    pub fn adjacent(
        &self, dims: [usize; 3], seed: Option<crate::agreement::Read<'_>>, flip: bool, settings: Settings, input: crate::agreement::Read<'_>, output: &mut Buffer
    ) -> Result<(), Error> {
        let dtype = output.dtype();
        self.projected(&[input], None, seed.map_or(Seed::Identity, Seed::Initial), |inputs, seed| {
            let seed = match seed { Seed::Initial(read) | Seed::Tail(read) => Some(read), Seed::Identity => None };
            self.definition.adjacent(self.variant, dims, seed, flip, settings, inputs[0], output)
        })?;
        finish(output, dtype, self.variant)
    }
    pub fn fold_read(
        &self, shape: &[usize], axis: usize, seed: Seed<'_>, direction: Direction, settings: Settings, input: crate::agreement::Read<'_>, output: &mut Buffer
    ) -> Result<(), Error> {
        let dtype = output.dtype();
        self.projected(&[input], None, seed, |inputs, seed| self.definition.fold_read(self.variant, shape, axis, seed, direction, settings, inputs[0], output))?;
        finish(output, dtype, self.variant)
    }
}

pub fn choose(
    selector: (crate::agreement::Read<'_>, DType), choices: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, dtype: DType, checked: bool,
    output: &mut Buffer) -> Result<(), Error> {
    let reads = || std::iter::once(selector.0).chain(choices.iter().copied());
    let len = reads().map(|read| read.plan.len()).find(|&n| n != 1).unwrap_or(1);
    if reads().any(|read| read.plan.len() != 1 && read.plan.len() != len) { return Err(Error::new(Kind::Length, "selection operands have different lengths")); }
    if range.start > range.end || (len != 1 && range.end > len) { return Err(Error::new(Kind::Length, "selection range exceeds its operands")); }
    if output.dtype() != dtype { *output = Buffer::with_capacity(dtype, range.len()); }
    with_dtype!(dtype, T => choose_typed::<T>(selector, choices, range, checked, T::vec_mut(output).expect("selection output dtype")))
}

fn choose_typed<T: crate::buffer::Cast>(
    selector: (crate::agreement::Read<'_>, DType), choices: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, checked: bool, output: &mut Vec<T>
) -> Result<(), Error> {
    output.clear();
    let start = range.start;
    crate::error::checked(|| {
        crate::parallel::extended(output, range.len(), crate::CHUNK, &|range, slots| {
            let part = |read: crate::agreement::Read<'_>| if read.plan.len() == 1 { 0..1 } else { start + range.start..start + range.end };
            let mut indices = Buffer::with_capacity(selector.1, 0);
            let positions = Positions::Indices(selector.0, selector.1).chunk(part(selector.0), choices.len(), &mut indices);
            let mut storage: Vec<Vec<T>> = choices.iter().map(|_| Vec::new()).collect();
            let args = choices.iter().zip(&mut storage).map(|(read, scratch)| read.part::<T>(part(*read), checked, scratch)).collect::<Result<Vec<_>, _>>();
            match positions.and_then(|positions| args.map(|args| (positions, args))) {
                Ok((positions, args)) => slots.fill(range.len(), |out| {
                    for (i, slot) in out.iter_mut().enumerate() {
                        let position = positions[if positions.len() == 1 { 0 } else { i }];
                        let value = match args.get(position) {
                            Some(Arg::One(value)) => *value, Some(Arg::Many(values)) => values[if values.len() == 1 { 0 } else { i }], None => T::FILL
                        };
                        slot.write(value);
                    }
                    true
                }),
                Err(error) => { crate::error::record(error); slots.fill(range.len(), |out| { for slot in out { slot.write(T::FILL); } false }) }
            }
        });
        Ok(())
    })
}
pub fn product(
    map: &Kernel, fold: &Kernel, inputs: [crate::agreement::Read<'_>; 2], plan: &crate::linalg::ProductPlan, settings: Settings, output: &mut Buffer
) -> Result<(), Error> {
    use crate::agreement::{Missing, Read};
    let dtype = output.dtype();
    if let Some(native) = map.native_product(map.variant, fold, map.boolean_inputs, Some(dtype), settings) {
        let ran = map.projected(&inputs, None, Seed::Identity, |inputs, _| (native.run)(map.variant, [inputs[0], inputs[1]], plan, output))?;
        if ran {
            return finish(output, dtype, crate::facts::Variant { checked: map.variant.checked || fold.variant.checked, ..map.variant });
        }
    }
    let plans = std::array::from_fn::<_, 2, _>(|i| inputs[i].plan.remapped(plan.mappings[i].clone(), plan.iteration.count(), Missing::Invalid));
    let inputs = std::array::from_fn::<_, 2, _>(|i| Read { sources: inputs[i].sources, plan: &plans[i] });
    let shape = plan.iteration.shape();
    let split = shape.len() - plan.contracted;
    let dims = [crate::layout::count(&shape[..split]), crate::layout::count(&shape[split..]), 1];
    let pieces = |range: std::ops::Range<usize>, f: &mut dyn FnMut(usize, Piece<'_>)| {
        let mut values = Buffer::with_capacity(map.result_dtype(), range.len());
        map.map(&inputs, range.clone(), settings, &mut values)?;
        f(range.start, Piece::Items(values.view()));
        Ok(())
    };
    fold.fold_pieces(dims, Seed::Identity, Direction::Right, settings, &pieces, output)
}

#[derive(Clone, Copy)]
struct NativeProduct {
    result: DType, run: fn(crate::facts::Variant, [crate::agreement::Read<'_>; 2], &crate::linalg::ProductPlan, &mut Buffer) -> Result<bool, Error>
}

fn product_for<K: crate::facts::Formula<N>, const N: usize>(
    object: &K, mut variant: crate::facts::Variant, fold: Option<crate::reduce::Reduce>, boolean_inputs: bool, wanted: Option<DType>, settings: Settings
) -> Option<NativeProduct> {
    use std::any::TypeId;
    let fold = fold?;
    if wanted == Some(DType::Boolean) && !boolean_inputs { return None; }
    if matches!(wanted, Some(DType::Boolean | DType::Float(FloatWidth::F16))) { variant.dtype = wanted.unwrap(); }
    if !variant.wrapping && matches!(variant.dtype, DType::Integer(_)) { variant.dtype = DType::Integer(Width::I64); }
    let truth = if boolean_inputs { object.truth_table() } else { None };
    macro_rules! pairs {
        (($($arg:tt)*) folds {$($folds:tt)*} booleans {$($booleans:tt)*}
            products {$($pair:ident: [$(($map:ty, $reduce:ident)),*] at [$($t:ty),*] $(wrapping [$($wt:ty),*])? => [$($names:tt)*];)*}
            boolean_products {$($bp:ident: [$(($bm:path, $bf:ident)),*] at [$($bt:ty),*] => [$($bn:tt)*];)*}
        ) => {
            $(if false $(|| (TypeId::of::<K>() == TypeId::of::<$map>() && fold == crate::reduce::Reduce::$reduce))* {
                pairs!(@run $pair at [$($t),*] $(wrapping [$($wt),*])?);
            })*
            // A Boolean row also takes a function on Booleans with its map's truth table, as `∧` and `∨` (`tolerant::Lcm`, `tolerant::Gcd`) have `BitAnd`'s and `BitOr`'s.
            $(if false $(|| ((TypeId::of::<K>() == TypeId::of::<$bm>() || truth.is_some() && truth == crate::facts::Formula::<2>::truth_table(&$bm))
                && fold == crate::reduce::Reduce::$bf))* {
                pairs!(@run $bp at [$($bt),*]);
            })*
        };
        (@run $pair:ident at [$($t:ty),*] $(wrapping [$($wt:ty),*])?) => {{
            let pair = crate::linalg::Product::$pair;
            if wanted.is_some() && pair.compares_exactly() && !variant.dtype.is_integral() && settings.tolerance.or(object.tolerance()).unwrap_or(0.0) != 0.0 { return None; }
            if !(variant.wrapping && matches!(variant.dtype, DType::Integer(_))) {
                $(if variant.dtype == <$t as Element>::DTYPE {
                    let result = product_result::<$t, crate::linalg::$pair>(variant);
                    if wanted.is_some_and(|d| d != result) { return None; }
                    return Some(NativeProduct { result, run: |v, x, p, out| product_typed::<$t, _>(crate::linalg::$pair, v, x, p, out) });
                })*
            }
            $($(if variant.wrapping && variant.dtype == <$wt as Element>::DTYPE {
                let result = variant.dtype;
                if wanted.is_some_and(|d| d != result) { return None; }
                return Some(NativeProduct { result, run: |v, x, p, out| product_typed::<$wt, _>(crate::Wrapped(crate::linalg::$pair), v, x, p, out) });
            })*)?
        }};
    }
    crate::facts!(pairs);
    None
}

fn product_result<T: Element, P: crate::linalg::Inner<T>>(variant: crate::facts::Variant) -> DType where P::Out: Element {
    if variant.wrapping && T::DTYPE.is_integral() || T::DTYPE == DType::Float(FloatWidth::F16) { T::DTYPE }
    else if matches!(P::Out::DTYPE, DType::Integer(_)) { DType::Integer(Width::I64) }
    else { P::Out::DTYPE }
}

fn product_typed<T: crate::buffer::Cast, P: crate::linalg::Inner<T>>(
    pair: P, variant: crate::facts::Variant, inputs: [crate::agreement::Read<'_>; 2], plan: &crate::linalg::ProductPlan, output: &mut Buffer
) -> Result<bool, Error> where P::Out: crate::buffer::Cast {
    use crate::agreement::{Mapping, Missing, Read};
    use crate::gather::Steps;
    use crate::linalg::{Mapped, Orientation};
    let Some(matrix) = plan.matrix_for(inputs) else { return Ok(false); };
    let [m, k, n] = matrix.size;
    if !variant.wrapping && matches!(P::Out::DTYPE, DType::Integer(w) if k > i64::MAX as usize || !w.holds(k as i64)) { return Ok(false); }
    let frames = crate::layout::count(&plan.layout.shape()[..matrix.frame_rank]);
    let expanded = std::array::from_fn::<_, 2, _>(|i| inputs[i].plan.remapped(plan.mappings[i].clone(), plan.iteration.count(), Missing::Invalid));
    let mut result = Vec::with_capacity(plan.layout.count());
    for frame in 0..frames {
        let base = frame * m * n * k;
        let left = expanded[0].remapped(Mapping::Steps { base, steps: vec![Steps::along(m, n * k), Steps::along(k, 1)].into() }, m * k, Missing::Invalid);
        let right = expanded[1].remapped(Mapping::Steps { base, steps: vec![Steps::along(k, 1), Steps::along(n, k)].into() }, k * n, Missing::Invalid);
        let reads = [Read { sources: inputs[0].sources, plan: &left }, Read { sources: inputs[1].sources, plan: &right }];
        let (mut a_items, mut b_items) = (Vec::new(), Vec::new());
        if T::DTYPE.is_integral() {
            let a = read_all::<T>(&reads[0], m * k, variant.checked, &mut a_items)?;
            let b = read_all::<T>(&reads[1], k * n, variant.checked, &mut b_items)?;
            let start = result.len();
            result.resize(start + m * n, P::Out::default());
            if !pair.checked_into(&a, &b, matrix.size, &mut result[start..]) {
                return if variant.checked { Err(Error::new(Kind::Overflow, "product does not fit its integer result")) } else { Ok(false) };
            }
        } else {
            let (a, am) = matrix_items::<T>(&reads[0], [m, k], variant.checked, &mut a_items)?;
            let (b, bm) = matrix_items::<T>(&reads[1], [k, n], variant.checked, &mut b_items)?;
            result.extend(pair.inner_mapped(Mapped { data: &a[..], mapping: &am }, Mapped { data: &b[..], mapping: &bm }, [Orientation::Normal; 2], matrix.size));
        }
    }
    *output = P::Out::buffer(result);
    Ok(true)
}

/// The `len` items that `read` reads, as one slice: `scratch` holds them where the read copies them.
fn read_all<'a, T: crate::buffer::Cast>(read: &'a crate::agreement::Read<'_>, len: usize, checked: bool, scratch: &'a mut Vec<T>) -> Result<std::borrow::Cow<'a, [T]>, Error> {
    Ok(match read.part::<T>(0..len, checked, scratch)? { Arg::One(x) => std::borrow::Cow::Owned(vec![x; len]), Arg::Many(x) => std::borrow::Cow::Borrowed(x) })
}

/// The items of `read`, a matrix of `shape`, with the walk that reaches each one: the read's source where the read is positional, and otherwise its items read once (`read_all`).
fn matrix_items<'a, T: crate::buffer::Cast>(
    read: &'a crate::agreement::Read<'_>, shape: [usize; 2], checked: bool, scratch: &'a mut Vec<T>
) -> Result<(std::borrow::Cow<'a, [T]>, crate::agreement::Walked<'static>), Error> {
    use crate::agreement::{Mapping, Walked};
    if let Some((view, mapping)) = read.positional(&shape) && let Some(data) = T::of(view) { return Ok((std::borrow::Cow::Borrowed(data), Walked::from(mapping))); }
    Ok((read_all(read, shape[0] * shape[1], checked, scratch)?, Walked::from(Mapping::Linear(1))))
}

pub fn at(
    combine: Option<&Kernel>, root: WriteRoot<'_>, values: crate::agreement::Read<'_>, ids: Positions<'_>, checked: bool, settings: Settings,
    output: &mut Buffer) -> Result<(), Error> {
    let (wanted, n) = (output.dtype(), ids.len());
    if values.plan.len() != 1 && values.plan.len() != n { return Err(Error::new(Kind::Length, "write values do not match positions")); }
    let projection = combine.map_or(Projection::AsIs, Kernel::projection);
    match combine {
        None => {
            root.initialize(wanted, checked, Projection::AsIs, output)?;
            with_dtype!(wanted, T => at_typed::<T>(values, ids, checked, None::<fn(T, T) -> (T, Option<Kind>)>, output))?;
        },
        Some(kernel) => kernel.projected(&[values], None, Seed::Identity, |inputs, _| kernel.definition.at(kernel.variant, root, inputs[0], ids, checked, projection, settings, output))?,
    }
    finish(output, wanted, crate::facts::Variant { checked, ..crate::facts::Variant::checked(wanted) })
}

/// The magnitude below which every whole float converts to an `i64`.
const I64_LIMIT: f64 = 9_223_372_036_854_775_808.0;

/// `values` as the whole numbers they equal within the default tolerance (`tolerant::wholes`), as an integer computation reads float arguments whatever the Mathish setting. A fraction, or a magnitude of 2⁶³ or more, is DOMAIN.
pub fn whole_numbers<F: crate::Float>(values: &[F]) -> Result<Vec<i64>, Error> {
    crate::tolerant::wholes(crate::tolerant::Tolerance::DEFAULT, values, I64_LIMIT).map(|(numbers, _)| numbers)
        .ok_or_else(|| Error::new(Kind::Domain, "an integer computation's float argument must be a whole number"))
}

/// `f` on `view` as an integer computation reads it (`with_projected`): characters as their codes under `Projection::Codes`, and floats as whole numbers (`whole_numbers`).
fn with_projected_view<R>(view: View<'_>, projection: Projection, f: impl FnOnce(View<'_>) -> R) -> Result<R, Error> {
    let view = if projection == Projection::Codes { crate::characters::as_codes(view) } else { view };
    if !matches!(view.dtype(), DType::Float(_)) { return Ok(f(view)); }
    let mut floats = Vec::new();
    <f64 as crate::buffer::Cast>::extend_cast(view, view.len(), true, &mut floats)?;
    Ok(f(View::Integers(crate::buffer::Ints::I64(&whole_numbers(&floats)?))))
}

/// `call` on `inputs` as an integer computation reads them: characters as their codes under `Projection::Codes`, and floats as the whole numbers they stand for (`whole_numbers`), as `i64`, over `range`, or over the whole of each input where `None`.
fn with_projected<R>(
    inputs: &[crate::agreement::Read<'_>], range: Option<std::ops::Range<usize>>, projection: Projection, call: impl FnOnce(&[crate::agreement::Read<'_>]) -> Result<R, Error>
) -> Result<R, Error> {
    use crate::{agreement::{Binding, Read, ReadPlan}, buffer::{FloatBuf, IntBuf}};
    let codes = projection == Projection::Codes;
    let sources: Vec<Vec<_>> = if codes { inputs.iter().map(|r| r.sources.iter().map(|s| s.as_codes()).collect()).collect() } else { Vec::new() };
    // Character reads keep their registered character sources, including empty buffers.
    let code_plans = if !codes { Vec::new() } else {
        inputs.iter().map(|read| {
            let mut characters = false;
            read.plan.try_for_each_source(&mut |source, _| {
                characters |= read.sources.get(source).is_some_and(|binding| binding.view.dtype() == DType::Character);
                Ok::<_, Error>(())
            })?;
            Ok(read.plan.as_codes(characters))
        }).collect::<Result<Vec<_>, Error>>()?
    };
    let mut projected: Vec<Read<'_>> = if codes { sources.iter().zip(&code_plans).map(|(sources, plan)| Read { sources, plan }).collect() } else { inputs.to_vec() };
    if projected.iter().all(|r| r.sources.iter().all(|s| !matches!(s.view.dtype(), DType::Float(_)))) { return call(&projected); }
    let converted = projected.iter().map(|read| {
        if !read.sources.iter().any(|s| matches!(s.view.dtype(), DType::Float(_))) { return Ok(None); }
        let range = range.clone().unwrap_or(0..read.plan.len());
        let mut floats = Buffer::with_capacity(DType::Float(FloatWidth::F64), range.len());
        read.read_into(range.clone(), floats.dtype(), true, &mut floats)?;
        let Buffer::Floats(FloatBuf::F64(values)) = &floats else { unreachable!("a read into 64-bit floats gives them") };
        Ok(Some((range.start, Buffer::Integers(IntBuf::I64(whole_numbers(values)?)))))
    }).collect::<Result<Vec<_>, Error>>()?;
    let bindings = converted.iter().zip(inputs)
        .map(|(buffer, read)| buffer.as_ref().map(|(start, b)| Binding::window(b.view(), *start, read.plan.len())).transpose()).collect::<Result<Vec<_>, _>>()?;
    let plans: Vec<_> = inputs.iter().map(|r| ReadPlan::input(0, r.plan.len())).collect();
    for (i, binding) in bindings.iter().enumerate() {
        if let Some(binding) = binding { projected[i] = Read { sources: std::slice::from_ref(binding), plan: &plans[i] }; }
    }
    call(&projected)
}

pub(crate) fn at_typed<T: crate::buffer::Cast>(
    values: crate::agreement::Read<'_>, ids: Positions<'_>, checked: bool, apply: Option<impl Fn(T, T) -> (T, Option<Kind>)>, output: &mut Buffer
) -> Result<(), Error> {
    let (len, n) = (output.len(), ids.len());
    let mut indices = Buffer::with_capacity(DType::Boolean, 0);
    let out = T::vec_mut(output).expect("write output has its working dtype");
    let mut scratch = Vec::new();
    for start in (0..n).step_by(crate::CHUNK) {
        let end = n.min(start + crate::CHUNK);
        let positions = ids.chunk(start..end, len, &mut indices)?;
        let source = values.part::<T>(if values.plan.len() == 1 { 0..1 } else { start..end }, checked, &mut scratch)?;
        let writes = crate::gather::Writes::Offsets(&positions);
        match &apply {
            None => match source {
                Arg::One(value) => writes.write(out, &[value], crate::gather::Same), Arg::Many(values) => writes.write(out, values, crate::gather::Same)
            },
            Some(apply) => {
                let error = std::cell::Cell::new(None);
                writes.combine(out, source, |x, y| {
                    let (value, failed) = apply(x, y);
                    if failed.is_some() { error.set(failed); }
                    value
                });
                if let Some(kind) = error.get() { return Err(Error::new(kind, "modified write is outside the function's domain")); }
            }
        }
    }
    Ok(())
}

macro_rules! with_binary_body {
    ($object:expr, $variant:expr, $settings:expr, |$T:ident, $apply:ident| $body:expr) => {{
        let (object, variant, settings) = ($object, $variant, $settings);
        let ops = crate::recipe::NativeOps::<{ crate::FUSED }> { tolerance: settings.tolerance };
        match variant.dtype {
            DType::Boolean | DType::Integer(_) => {
                let width = match variant.dtype { DType::Integer(w) => w, _ => Width::U8 };
                with_width!(width, $T => {
                    let ops = crate::recipe::IntegerOps::<$T>::new(variant);
                    let $apply = |x: $T, y: $T| integer_value(object, variant, &ops, std::array::from_fn(|i| if i == 0 { x } else { y }));
                    $body
                })
            },
            DType::Float(FloatWidth::F16 | FloatWidth::F32) => {
                type $T = f32;
                let $apply = |x: $T, y: $T| real_value(object, variant, &ops, std::array::from_fn(|i| if i == 0 { x } else { y }), true);
                $body
            },
            DType::Float(FloatWidth::F64) => {
                type $T = f64;
                let $apply = |x: $T, y: $T| real_value(object, variant, &ops, std::array::from_fn(|i| if i == 0 { x } else { y }), true);
                $body
            },
            DType::Complex(w) => with_complex_width!(w, $T => {
                let $apply = |x: $T, y: $T| complex_value(object, variant, &ops, std::array::from_fn(|i| if i == 0 { x } else { y }));
                $body
            }),
            DType::Character => Err(Error::new(Kind::Domain, "a numerical body needs code-point inputs for characters")),
        }
    }};
}

fn at_formula<K: crate::facts::Formula<N>, const N: usize>(
    object: &K, variant: crate::facts::Variant, root: WriteRoot<'_>, values: crate::agreement::Read<'_>, ids: Positions<'_>, checked: bool, projection: Projection,
    settings: Settings, output: &mut Buffer
) -> Result<(), Error> {
    if const { N != 2 } { return Err(Error::new(Kind::Length, "a modified write needs a binary function")); }
    let working = with_binary_body!(object, variant, settings, |T, _apply| Ok(T::DTYPE))?;
    if let Some(fold) = reduction(object, variant) {
        native_reduction!(fold, variant.dtype, variant.fails_on_overflow(), |R, T| {
            type S = <R as crate::reduce::Reduction<T>>::Total;
            if S::DTYPE == variant.dtype || S::DTYPE == working {
                root.initialize(S::DTYPE, checked, projection, output)?;
                return at_typed::<S>(values, ids, checked, Some(|x, y| {
                    let (value, ok) = <R as crate::reduce::Reduction<T>>::chain(x, y);
                    (value, (!ok && variant.fails_on_overflow()).then_some(Kind::Overflow))
                }), output);
            }
            None::<()>
        });
    }
    with_binary_body!(object, variant, settings, |T, apply| {
        root.initialize(T::DTYPE, checked, projection, output)?;
        at_typed::<T>(values, ids, checked, Some(apply), output)
    })
}

/// Report whether a kernel reads its operands through a non-contiguous plan.
pub fn report_maps<'a>(plans: impl IntoIterator<Item = &'a crate::agreement::ReadPlan>, range: std::ops::Range<usize>) {
    if crate::events::running() && plans.into_iter().any(|plan| plan.contiguous(range.clone()).is_none()) {
        crate::events::taken(crate::events::Route::MappedKernel, range.len());
    }
}

fn map_formula<K: crate::facts::Formula<N>, const N: usize>(
    object: &K, variant: crate::facts::Variant, inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, settings: Settings, repair: bool,
    judged: bool, output: &mut Buffer) -> Result<(), Error> {
    if inputs.len() != N { return Err(Error::new(Kind::Length, "kernel operand count changed")); }
    crate::events::taken(crate::events::Route::VectorKernel, range.len());
    report_maps(inputs.iter().map(|read| read.plan), range.clone());
    // A kernel whose arguments' types can't make an error judges no item: `judge` is `variant` unchecked. Reads and `IntegerOps` keep `variant`, so an argument that doesn't convert, or an overflow that `IntegerOps::result` reports, still fails.
    let judge = crate::facts::Variant { checked: variant.checked && judged, ..variant };
    if let Some(slices) = object.native_slices() {
        let valid = |ok: bool| if judge.checked && !ok { Err(Kind::Domain) } else { Ok(()) };
        return match variant.dtype {
            DType::Float(FloatWidth::F64) => {
                map_blocks::<f64, f64, N>(inputs, range, variant.checked, output, &|args, n, out| valid((slices.double)(object, args, n, out)))
            }
            DType::Float(FloatWidth::F32) => {
                map_blocks::<f32, f32, N>(inputs, range, variant.checked, output, &|args, n, out| valid((slices.single)(object, args, n, out)))
            }
            DType::Float(FloatWidth::F16) => map_blocks::<f32, half::f16, N>(inputs, range, variant.checked, output, &|args, n, out| {
                valid(crate::slice::buffered(n, out, |out| (slices.single)(object, args, n, out), half::f16::from_f32))
            }),
            _ => Err(Error::new(Kind::Domain, "native real body needs floating arguments")),
        };
    }
    match variant.dtype {
        DType::Float(FloatWidth::F16) => map_real::<K, f32, half::f16, N>(object, variant, inputs, range, settings, repair, judge, output),
        DType::Float(FloatWidth::F32) => map_real::<K, f32, f32, N>(object, variant, inputs, range, settings, repair, judge, output),
        DType::Float(FloatWidth::F64) => map_real::<K, f64, f64, N>(object, variant, inputs, range, settings, repair, judge, output),
        DType::Boolean => match object.truth_table() {
            Some(table) => map_truth::<N>(table, inputs, range, output),
            None => map_integer::<K, u8, N>(object, variant, inputs, range, judge, output),
        },
        DType::Integer(width) => with_width!(width, I => map_integer::<K, I, N>(object, variant, inputs, range, judge, output)),
        DType::Complex(crate::buffer::ComplexWidth::F32) => map_complex::<K, f32, N>(object, variant, inputs, range, settings, judge, output),
        DType::Complex(crate::buffer::ComplexWidth::F64) => map_complex::<K, f64, N>(object, variant, inputs, range, settings, judge, output),
        DType::Character => Err(Error::new(Kind::Domain, "characters are not numerical kernel operands")),
    }
}

#[inline(always)]
fn real_value<K, F, const N: usize>(
    object: &K, variant: crate::facts::Variant, ops: &crate::recipe::NativeOps<{ crate::FUSED }>, x: [F; N], repair: bool) -> (F, Option<Kind>) where
    K: crate::facts::Formula<N>, F: crate::float::Native {
    let result = if repair && object.emit_repair(ops, x) == Some(true) { crate::facts::Formula::repair_routine(object).map(|r| r.apply(x[0])) }
    else { object.real(ops, x) };
    let Some(result) = result else { return (F::zero(), variant.checked.then_some(Kind::Domain)); };
    let error = if variant.checked { object.emit_valid(variant, ops, x, result).and_then(|(kind, valid)| (!valid).then_some(kind)) } else { None };
    (result, error)
}

#[inline(always)]
fn integer_value<K, I, const N: usize>(object: &K, variant: crate::facts::Variant, ops: &crate::recipe::IntegerOps<I>, x: [I; N]) -> (I, Option<Kind>) where
    K: crate::facts::Formula<N>, I: crate::recipe::Overflowing {
    let Some(result) = object.integer(ops, x) else { return (I::FILL, variant.checked.then_some(Kind::Domain)); };
    let error = if variant.checked { object.emit_valid(variant, ops, x, result).and_then(|(kind, valid)| (!valid).then_some(kind)) } else { None };
    (result, error)
}

/// Whether `object`'s integer body at the Boolean `variant` raises for some Boolean arguments: it fails, or a formula that doesn't give truths gives a number other than 0 or 1, which a Boolean result can't hold. It tries all 2^N arguments.
fn boolean_raises<K: crate::facts::Formula<N>, const N: usize>(object: &K, variant: crate::facts::Variant) -> bool {
    let ops = crate::recipe::IntegerOps::<u8>::new(variant);
    (0..1usize << N).any(|bits| {
        let (value, error) = integer_value(object, variant, &ops, std::array::from_fn(|i| ((bits >> i) & 1) as u8));
        error.is_some() || !K::BOOLEAN && value > 1
    })
}

#[inline(always)]
fn complex_value<K, F, const N: usize>(
    object: &K, variant: crate::facts::Variant, ops: &crate::recipe::NativeOps<{ crate::FUSED }>, x: [num_complex::Complex<F>; N]
) -> (num_complex::Complex<F>, Option<Kind>) where K: crate::facts::Formula<N>, F: crate::float::Native {
    let Some(result) = object.complex(ops, x) else { return (num_complex::Complex::new(F::zero(), F::zero()), variant.checked.then_some(Kind::Domain)); };
    let valid = !variant.checked || !K::FAILS_NAN || crate::facts::nan_valid(ops, x.into_iter().flat_map(|x| [x.re, x.im]), [result.re, result.im]);
    (result, (!valid).then_some(Kind::Domain))
}

fn scalar_result<T>((value, error): (T, Option<Kind>)) -> Result<T, Error> {
    match error { Some(kind) => Err(Error::new(kind, "numerical argument is outside the function's domain")), None => Ok(value) }
}
fn map_real<K, F, Dst, const N: usize>(
    object: &K, variant: crate::facts::Variant, inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, settings: Settings, repair: bool,
    judge: crate::facts::Variant, output: &mut Buffer) -> Result<(), Error> where K: crate::facts::Formula<N>, F: crate::float::Native + crate::buffer::Cast + Source<Dst>,
    Dst: Element + crate::Float {
    let ops = crate::recipe::NativeOps::<{ crate::FUSED }> { tolerance: settings.tolerance };
    if !K::BOOLEAN && repair && let Some(repair) = crate::facts::Formula::repair_routine(object) {
        let body = |x| object.real(&ops, x).expect("a resolved object has a real body");
        return map_blocks::<F, Dst, N>(inputs, range, variant.checked, output, &|args, n, out| {
            let needs = |x| object.emit_repair(&ops, x).unwrap_or(false);
            let ok = crate::slice::two_pass(args, n, out, |x| (!needs(x)).then(|| body(x)), |x| if needs(x) { repair.apply(x[0]) } else { body(x) });
            if judge.checked && K::FAILS_NAN && !ok { Err(Kind::Domain) } else { Ok(()) }
        });
    }
    if const { K::BOOLEAN } {
        return map_values::<F, bool, N>(inputs, range, variant.checked, judge.checked, output, &|checked, x| {
            let (value, error) = real_value(object, crate::facts::Variant { checked, ..judge }, &ops, x, false);
            (value != F::zero(), error)
        });
    }
    map_values::<F, Dst, N>(inputs, range, variant.checked, judge.checked, output, &|checked, x| {
        let (result, error) = real_value(object, crate::facts::Variant { checked, ..judge }, &ops, x, false);
        (Source::<Dst>::read(result), error)
    })
}

fn map_integer<K, I, const N: usize>(
    object: &K, variant: crate::facts::Variant, inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, judge: crate::facts::Variant,
    output: &mut Buffer) -> Result<(), Error>
where K: crate::facts::Formula<N>, I: crate::recipe::Overflowing {
    let ops = crate::recipe::IntegerOps::<I>::new(variant);
    let apply = |checked, x| integer_value(object, crate::facts::Variant { checked, ..judge }, &ops, x);
    if const { K::BOOLEAN } {
        return map_values::<I, bool, N>(inputs, range, variant.checked, judge.checked, output, &|checked, x| {
            let (x, e) = apply(checked, x);
            (Into::<i64>::into(x) != 0, e)
        });
    }
    map_values::<I, I, N>(inputs, range, variant.checked, judge.checked, output, &apply)
}

/// A function of Booleans by its truth table (`Formula::truth_table`): bit `2x+y` of the table, or bit `x` for one argument, gives its result. A function with a table gives it for every Boolean argument, so no item checks its validity.
fn map_truth<const N: usize>(table: u8, inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, output: &mut Buffer) -> Result<(), Error> {
    map_values::<bool, bool, N>(inputs, range, false, false, output, &|_, x| (crate::facts::truth_bit(table, x), None))
}

fn map_complex<K, F, const N: usize>(
    object: &K, variant: crate::facts::Variant, inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, settings: Settings,
    judge: crate::facts::Variant, output: &mut Buffer
) -> Result<(), Error> where K: crate::facts::Formula<N>, F: crate::float::Native + crate::buffer::Cast, num_complex::Complex<F>: crate::buffer::Cast {
    let ops = crate::recipe::NativeOps::<{ crate::FUSED }> { tolerance: settings.tolerance };
    let apply = |checked, x| complex_value(object, crate::facts::Variant { checked, ..judge }, &ops, x);
    if const { K::BOOLEAN } {
        map_values::<num_complex::Complex<F>, bool, N>(inputs, range, variant.checked, judge.checked, output, &|checked, x| {
            let (x, e) = apply(checked, x);
            (x.re != F::zero(), e)
        })
    } else if output.dtype().is_real() {
        map_values::<num_complex::Complex<F>, F, N>(inputs, range, variant.checked, judge.checked, output, &|checked, x| {
            let (x, e) = apply(checked, x);
            (x.re, e)
        })
    } else { map_values::<num_complex::Complex<F>, num_complex::Complex<F>, N>(inputs, range, variant.checked, judge.checked, output, &apply) }
}

/// `apply` on each item of `inputs` over `range`, written to `output`. `apply` takes whether to judge each result's validity, and gives the result with any error kind it found. `judged` picks one of two loops for each block, and each loop passes its choice to `apply` as a constant. The unjudged loop then has no per-item test, and the judged loop gathers every item's verdict into one flag. Where an item fails, the block finds the first failing item again and gives its error.
fn map_values<F: crate::buffer::Cast, O: Element, const N: usize>(
    inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, checked: bool, judged: bool, output: &mut Buffer,
    apply: &(impl Fn(bool, [F; N]) -> (O, Option<Kind>) + Sync)) -> Result<(), Error> {
    map_blocks(inputs, range, checked, output, &|args, n, out| {
        let ok = if judged {
            crate::slice::mapped(&args, n, out, #[inline(always)] |x| { let (r, error) = apply(true, x); (r, error.is_none()) })
        } else {
            crate::slice::mapped(&args, n, out, #[inline(always)] |x| (apply(false, x).0, true))
        };
        if !ok { for i in 0..n { if let Some(kind) = apply(true, std::array::from_fn(|j| args[j].nth(i))).1 { return Err(kind); } } }
        Ok(())
    })
}

fn map_blocks<F: crate::buffer::Cast, O: Element, const N: usize>(
    inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, checked: bool, output: &mut Buffer,
    apply: &(impl Fn([Arg<'_, F>; N], usize, &mut [std::mem::MaybeUninit<O>]) -> Result<(), Kind> + Sync)) -> Result<(), Error> {
    if output.dtype() != O::DTYPE { *output = Buffer::with_capacity(O::DTYPE, range.len()); }
    let values = O::vec_mut(output).expect("kernel output has its selected dtype");
    values.clear();
    crate::error::checked(|| {
        crate::parallel::extended(values, range.len(), crate::CHUNK, &|part, slots| {
            let part = part.start + range.start..part.end + range.start;
            let n = part.len();
            let mut scratch: [Vec<F>; N] = std::array::from_fn(|_| Vec::new());
            let mut args = [Arg::Many(&[]); N];
            for ((arg, input), scratch) in args.iter_mut().zip(inputs).zip(scratch.iter_mut()) {
                match input.part::<F>(part.clone(), checked, scratch) {
                    Ok(read) => *arg = read,
                    Err(error) => { crate::error::record(error); return slots.fill(n, |out| { for slot in out { slot.write(O::FILL); } false }); }
                }
            }
            slots.fill(n, |out| match apply(args, n, out) {
                Ok(()) => true, Err(kind) => { crate::error::record(Error::new(kind, "numerical kernel argument is outside its domain")); false }
            })
        });
        Ok(())
    })
}

fn inverse_fold<K: crate::facts::Formula<N>, const N: usize>(object: &K, seed: Seed<'_>, direction: Direction) -> Option<crate::reduce::Reduce> {
    if N == 2 && direction == Direction::Right && matches!(seed, Seed::Identity) { object.inverse_fold() } else { None }
}

fn fold_formula<K: crate::facts::Formula<N>, const N: usize>(
    object: &K, variant: crate::facts::Variant, dims: [usize; 3], seed: Seed<'_>, direction: Direction, settings: Settings, pieces: &Pieces<'_>,
    output: &mut Buffer) -> Result<(), Error> {
    if let Some(fold) = reduction(object, variant) { return fold.run_pieces(variant, dims, seed, pieces, output); }
    if let Some(fold) = inverse_fold(object, seed, direction)
        && let Some(buffer) = fold.inverse_pieces(variant, dims, pieces)? { *output = buffer; return Ok(()); }
    ordered_formula(object, variant, dims, seed, direction, settings, pieces, output)
}

#[inline(never)]
fn ordered_formula<K: crate::facts::Formula<N>, const N: usize>(
    object: &K, variant: crate::facts::Variant, dims: [usize; 3], seed: Seed<'_>, direction: Direction, settings: Settings, pieces: &Pieces<'_>,
    output: &mut Buffer) -> Result<(), Error> {
    if const { N != 2 } { return Err(Error::new(Kind::Length, "a fold needs a binary function")); }
    with_binary_body!(object, variant, settings, |T, apply| ordered_fold::<T>(
        dims, seed, direction, variant.checked, object.identity()[1], pieces, apply, output
    ))
}

fn piece_values<T: crate::buffer::Cast>(range: std::ops::Range<usize>, checked: bool, pieces: &Pieces<'_>, out: &mut Vec<T>) -> Result<(), Error> {
    out.clear();
    let (mut next, mut failure) = (range.start, None);
    pieces(range.clone(), &mut |at, piece| {
        if failure.is_some() { return; }
        let Piece::Items(view) = piece else { failure = Some(Error::new(Kind::Length, "ordered fold received a precomputed total")); return; };
        if at != next || at + view.len() > range.end { failure = Some(Error::new(Kind::Length, "fold pieces do not cover the requested range")); return; }
        if let Err(error) = T::extend_cast(view, view.len(), checked, out) { failure = Some(error); return; }
        next += view.len();
    })?;
    if let Some(error) = failure { return Err(error); }
    if next != range.end { return Err(Error::new(Kind::Length, "fold pieces left an unread range")); }
    Ok(())
}

fn ordered_fold<T: crate::buffer::Cast + Source<T>>(
    [outer, len, inner]: [usize; 3], seed: Seed<'_>, direction: Direction, checked: bool, identity: Option<f64>, pieces: &Pieces<'_>,
    apply: impl Fn(T, T) -> (T, Option<Kind>), output: &mut Buffer) -> Result<(), Error> {
    let (mut result, mut items, mut seed_scratch) = (Vec::with_capacity(outer * inner), Vec::new(), Vec::new());
    if inner == 0 { *output = T::buffer(result); return Ok(()); }
    let right = direction == Direction::Right;
    for cell in 0..outer {
        let mut given = match seed {
            Seed::Identity => None,
            Seed::Initial(read) | Seed::Tail(read) => {
                let one = read.plan.len() == 1;
                let arg = read.part::<T>(if one { 0..1 } else { cell * inner..(cell + 1) * inner }, checked, &mut seed_scratch)?;
                Some((0..inner).map(|i| arg.nth(if one { 0 } else { i })).collect::<Vec<_>>())
            }
        };
        let upfront = len == 0 || match seed { Seed::Tail(_) => right, Seed::Initial(_) => !right, Seed::Identity => false };
        let mut carry = if upfront { given.take() } else { None };
        let error = std::cell::Cell::new(None);
        let step = |x, y| {
            let (value, failed) = if right { apply(x, y) } else { apply(y, x) };
            if failed.is_some() { error.set(failed); }
            (value, failed.is_none())
        };
        let rows = (crate::CHUNK / inner).max(1);
        let mut remaining = len;
        while remaining != 0 {
            let n = rows.min(remaining);
            let start = if right { remaining - n } else { len - remaining };
            let at = (cell * len + start) * inner;
            piece_values(at..at + n * inner, checked, pieces, &mut items)?;
            if !right { crate::reduce::reverse_rows(&mut items, [1, n, inner]); }
            carry = Some(if carry.is_none() && n == 1 { items.clone() } else { crate::reduce::right_fold(&items, [1, n, inner], carry.as_deref(), step).0 });
            remaining -= n;
        }
        let mut total = match carry {
            Some(total) => total,
            None => {
                let value = identity.ok_or_else(|| Error::new(Kind::Length, "an empty fold needs an identity or a seed"))?;
                vec![T::cast(View::Floats(crate::buffer::Floats::F64(&[value])), 0)?; inner]
            }
        };
        if let Some(seed) = given {
            for (total, value) in total.iter_mut().zip(seed) {
                let (result, failed) = if right { apply(value, *total) } else { apply(*total, value) };
                *total = result;
                if failed.is_some() { error.set(failed); }
            }
        }
        if let Some(kind) = error.get() { return Err(Error::new(kind, "ordered fold is outside the function's domain")); }
        result.extend(total);
    }
    *output = T::buffer(result);
    Ok(())
}

fn scan_formula<K: crate::facts::Formula<N>, const N: usize>(
    object: &K, variant: crate::facts::Variant, dims: [usize; 3], seed: Seed<'_>, direction: Direction, settings: Settings, input: crate::agreement::Read<'_>,
    output: &mut Buffer) -> Result<(), Error> {
    if let Some(fold) = reduction(object, variant) { return fold.run_scan(variant, dims, seed, direction, input, output); }
    if const { N != 2 } { return Err(Error::new(Kind::Length, "a scan needs a binary function")); }
    with_binary_body!(object, variant, settings, |T, apply| crate::reduce::scan_read::<T, T>(
        dims, seed, direction, variant.checked, input, output, |items, dims, seed, out| {
            let error = std::cell::Cell::new(None);
            let step = |x, y| {
                let (value, failed) = if direction == Direction::Right { apply(y, x) } else { apply(x, y) };
                if failed.is_some() {
                    error.set(failed);
                }
                (value, failed.is_none())
            };
            crate::reduce::left_scan(items, dims, seed, out, step);
            match error.get() {
                Some(kind) => Err(Error::new(kind, "ordered scan is outside the function's domain")),
                None => Ok(()),
            }
        },
    ))
}
fn fold_read_formula<K: crate::facts::Formula<N>, const N: usize>(
    object: &K, variant: crate::facts::Variant, shape: &[usize], axis: usize, seed: Seed<'_>, direction: Direction, settings: Settings,
    input: crate::agreement::Read<'_>, output: &mut Buffer) -> Result<(), Error> {
    if let Some(fold) = reduction(object, variant) { return fold.run_read(variant, shape, axis, seed, input, output); }
    if let Some(fold) = inverse_fold(object, seed, direction)
        && let Some(buffer) = fold.inverse_read(variant, shape, axis, input)? { *output = buffer; return Ok(()); }
    let dims = crate::layout::Lanes::new(shape, axis)?.dims();
    let pieces = |range: std::ops::Range<usize>, emit: &mut dyn FnMut(usize, Piece<'_>)| {
        let mut scratch = Buffer::with_capacity(variant.dtype, 0);
        let start = range.start;
        let view = input.part_view(range, variant.dtype, variant.checked, &mut scratch)?;
        emit(start, Piece::Items(view));
        Ok(())
    };
    ordered_formula(object, variant, dims, seed, direction, settings, &pieces, output)
}

/// `object`'s walk of adjacent items (`reduce::adjacent_read`) at `variant`: each item combined with the one a row before it, and a lane's first row with `seed`, or kept.
fn adjacent_formula<K: crate::facts::Formula<N>, const N: usize>(
    object: &K, variant: crate::facts::Variant, dims: [usize; 3], seed: Option<crate::agreement::Read<'_>>, flip: bool, settings: Settings,
    input: crate::agreement::Read<'_>, output: &mut Buffer) -> Result<(), Error> {
    if const { N != 2 } { return Err(Error::new(Kind::Length, "a walk of adjacent items needs a binary function")); }
    with_binary_body!(object, variant, settings, |T, apply| crate::reduce::adjacent_read::<T>(dims, seed, flip, variant.checked, input, output, apply))
}
/// The reduction by which a fold by `object` at `variant` combines its items: for Booleans the formula's `truth`, which counts for `+` and otherwise names the reduction of its truth table, before its `fold` for numbers.
fn reduction<K: crate::facts::Formula<N>, const N: usize>(object: &K, variant: crate::facts::Variant) -> Option<crate::reduce::Reduce> {
    if variant.dtype == DType::Boolean { object.truth().or_else(|| object.fold()) } else { object.fold() }
}

/// The float width that a Boolean or integer computes in under rustymath's own policy, NumPy's: the float that [`DType::inexact`] gives it under `f64`. `f64` for any other dtype.
pub fn numpy_float(dtype: DType) -> FloatWidth { match dtype.inexact(FloatWidth::F64) { Some((DType::Float(w), _)) => w, _ => FloatWidth::F64 } }

/// `output`, a kernel's results at `variant`, as `dtype`, which a caller can ask for narrower than the kernel's result type. An integer that doesn't fit fails where `variant.fails_on_overflow()`, and wraps otherwise.
fn finish(output: &mut Buffer, dtype: DType, variant: crate::facts::Variant) -> Result<(), Error> {
    if output.dtype() != dtype { *output = Buffer::cast_results(output.view(), dtype, variant.fails_on_overflow())?; }
    Ok(())
}
