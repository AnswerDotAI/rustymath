//! Resolved numerical operations over shared [`Read`](crate::agreement::Read) plans. [`Kernel::resolve`] fixes the computing dtype, result dtype and character policy from an object's [`Formula`](crate::facts::Formula). The same definition supplies native execution and graph formulas.
//!
//! [`DType::computes_in`] promotes unsupported Boolean and integer inputs to i64, then to the NumPy float width. Unsupported character arithmetic is a domain error.
use crate::buffer::{Buffer, DType, Element, FloatWidth, Source, View, Width};
use crate::error::{Error, Kind};
use crate::reduce::native_reduction;
use crate::slice::Arg;

/// Numeric settings rebound for each execution, outside item loops.
#[derive(Clone, Copy, Debug, Default)]
pub struct Settings { pub tolerance: Option<f64> }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction { Left, Right }

#[derive(Clone, Copy)]
pub enum Seed<'a> { Identity, Initial(crate::agreement::Read<'a>), Tail(crate::agreement::Read<'a>) }

/// Write positions from numeric indices or the addresses of a selection's root source.
#[derive(Clone, Copy)]
pub enum Positions<'a> { Indices(crate::agreement::Read<'a>, DType), Route(&'a crate::agreement::ReadPlan, usize) }

impl Positions<'_> {
    pub fn len(self) -> usize { match self { Self::Indices(read, _) => read.plan.len(), Self::Route(plan, _) => plan.len() } }
    pub fn chunk(self, range: std::ops::Range<usize>, size: usize, checked: bool, scratch: &mut Buffer) -> Result<Vec<usize>, Error> {
        use crate::{agreement::ReadAt, gather::Steps};
        match self {
            Self::Indices(read, dtype) => crate::structure::positions_mode(read.part_view(range, dtype, checked, scratch)?, size, checked),
            Self::Route(plan, root) => range.map(|i| match plan.at(i) {
                ReadAt::Input { source, position } if source == root && position < size => Ok(position),
                ReadAt::Invalid => Err(Error::new(Kind::Index, "a checked selection has an invalid position")),
                ReadAt::Input { source, .. } if source == root && checked => Err(Error::new(Kind::Index, "write position is outside the array")),
                _ => Ok(Steps::FILL),
            }).collect(),
        }
    }
}

/// A write's logical root, or storage transferred by its final reader.
pub enum WriteRoot<'a> { Read(crate::agreement::Read<'a>), Owned(Buffer) }
impl WriteRoot<'_> {
    fn len(&self) -> usize { match self { Self::Read(read) => read.plan.len(), Self::Owned(buffer) => buffer.len() } }
    fn initialize(self, dtype: DType, checked: bool, codes: bool, output: &mut Buffer) -> Result<(), Error> {
        match self {
            Self::Read(read) if codes => with_codes(&[read], None, |reads| reads[0].read_into(0..read.plan.len(), dtype, checked, output)),
            Self::Read(read) => read.read_into(0..read.plan.len(), dtype, checked, output),
            Self::Owned(buffer) if buffer.dtype() == dtype => { *output = buffer; Ok(()) }
            Self::Owned(buffer) => {
                let view = if codes { crate::characters::as_codes(buffer.view()) } else { buffer.view() };
                *output = Buffer::cast_results(view, dtype, checked)?;
                Ok(())
            }
        }
    }
}

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
            DType::Float(FloatWidth::F16) => T::cast(View::Floats(Floats::F16(&[half::f16::from_bits(self.bits as u16)])), 0),
            DType::Float(FloatWidth::F32) => T::cast(View::Floats(Floats::F32(&[f32::from_bits(self.bits as u32)])), 0),
            DType::Float(FloatWidth::F64) => T::cast(View::Floats(Floats::F64(&[f64::from_bits(self.bits)])), 0),
            _ => Err(Error::new(Kind::Domain, "compiled totals have a real machine dtype")),
        }
    }
}

pub enum Piece<'a> { Items(View<'a>), Total(Total) }

pub type Pieces<'a> = dyn Fn(std::ops::Range<usize>, &mut dyn FnMut(usize, Piece<'_>)) -> Result<(), Error> + Sync + 'a;

/// One resolved mathematical definition and its captured numerical policies.
#[derive(Clone)]
pub struct Kernel {
    pub variant: crate::facts::Variant, result: DType, boolean_inputs: bool, typed_domain: bool, repair: bool, raising: bool,
    characters: Option<crate::characters::Policy>, definition: std::sync::Arc<dyn Application>
}

impl std::fmt::Debug for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernel").field("variant", &self.variant).field("arity", &self.arity()).finish()
    }
}

trait Application: Send + Sync {
    fn exact(&self, args: &[num_rational::BigRational]) -> Result<Option<num_rational::BigRational>, Error>;
    fn has_exact(&self) -> bool;
    fn real(&self, variant: crate::facts::Variant, args: &[f64], settings: Settings, repair: bool) -> Result<f64, Error>;
    fn complex(&self, variant: crate::facts::Variant, args: &[num_complex::Complex64], settings: Settings) -> Result<num_complex::Complex64, Error>;
    fn arity(&self) -> usize;
    fn integer_formula(&self) -> bool;
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
    fn needs_repair(&self, variant: crate::facts::Variant, ranges: &[crate::facts::Range]) -> Option<bool>;
    fn validity(
        &self, variant: crate::facts::Variant,
        ops: &dyn crate::recipe::Ops<f64, V = crate::facts::Slot, B = crate::facts::Slot, I = crate::facts::Slot, W = crate::facts::Slot>,
        x: &[crate::facts::Slot], result: crate::facts::Slot) -> Option<(Kind, crate::facts::Slot)>;
    fn partials(
        &self, ops: &dyn crate::recipe::Ops<f64, V = crate::facts::Slot, B = crate::facts::Slot, I = crate::facts::Slot, W = crate::facts::Slot>,
        x: &[crate::facts::Slot], result: crate::facts::Slot, needed: &[bool], output: &mut [Option<crate::facts::Slot>]);
    fn range(&self, x: &[crate::facts::Range]) -> crate::facts::Range;
    fn domain(&self, x: &[crate::facts::Range]) -> bool;
    fn identity(&self) -> Option<f64>;
    fn fold(&self) -> Option<crate::reduce::Reduce>;
    fn truth_table(&self) -> Option<u8>;
    fn product(
        &self, variant: crate::facts::Variant, fold: Option<crate::reduce::Reduce>, boolean_inputs: bool, wanted: Option<DType>, settings: Settings
    ) -> Option<NativeProduct>;
    fn map(
        &self, variant: crate::facts::Variant, inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, settings: Settings, repair: bool,
        output: &mut Buffer) -> Result<(), Error>;
    fn fold_pieces(
        &self, variant: crate::facts::Variant, dims: [usize; 3], seed: Seed<'_>, direction: Direction, settings: Settings, pieces: &Pieces<'_>,
        output: &mut Buffer) -> Result<(), Error>;
    fn fold_read(
        &self, variant: crate::facts::Variant, shape: &[usize], axis: usize, seed: Seed<'_>, direction: Direction, settings: Settings,
        input: crate::agreement::Read<'_>, output: &mut Buffer) -> Result<(), Error>;
    fn at(
        &self, variant: crate::facts::Variant, root: WriteRoot<'_>, values: crate::agreement::Read<'_>, ids: Positions<'_>, checked: bool, codes: bool,
        settings: Settings, output: &mut Buffer
    ) -> Result<(), Error>;
    fn scan(
        &self, variant: crate::facts::Variant, dims: [usize; 3], seed: Seed<'_>, direction: Direction, settings: Settings, input: crate::agreement::Read<'_>,
        output: &mut Buffer) -> Result<(), Error>;
}

struct Definition<K, const N: usize> { object: K }

impl<K: crate::facts::Formula<N>, const N: usize> Application for Definition<K, N> {
    fn exact(&self, args: &[num_rational::BigRational]) -> Result<Option<num_rational::BigRational>, Error> {
        if args.len() != N { return Err(Error::new(Kind::Length, "scalar operand count changed")); }
        self.object.exact(std::array::from_fn(|i| &args[i]))
    }
    fn has_exact(&self) -> bool { self.object.has_exact() }
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
    fn integer_formula(&self) -> bool { self.object.integer_formula() }
    fn may_raise(&self, variant: crate::facts::Variant, ranges: &[crate::facts::Range]) -> bool {
        if matches!(variant.dtype, DType::Complex(_)) || variant.dtype.is_integral() && !self.object.integer_formula() { return true; }
        let Ok(args) = ranges.try_into() else { return true; };
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
    fn needs_repair(&self, variant: crate::facts::Variant, ranges: &[crate::facts::Range]) -> Option<bool> {
        self.object.emit_repair::<f64, _>(&crate::recipe::RangeOps(variant.dtype), ranges.try_into().ok()?)?
    }
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
    fn domain(&self, x: &[crate::facts::Range]) -> bool { x.try_into().is_ok_and(|x| !self.object.has_domain() || self.object.domain(x).is_some()) }
    fn identity(&self) -> Option<f64> { self.object.identity().last().copied().flatten() }
    fn fold(&self) -> Option<crate::reduce::Reduce> { self.object.fold() }
    fn truth_table(&self) -> Option<u8> { self.object.truth_table() }
    fn product(
        &self, variant: crate::facts::Variant, fold: Option<crate::reduce::Reduce>, boolean_inputs: bool, wanted: Option<DType>, settings: Settings
    ) -> Option<NativeProduct> { product_for::<K, N>(&self.object, variant, fold, boolean_inputs, wanted, settings) }
    fn map(
        &self, variant: crate::facts::Variant, inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, settings: Settings, repair: bool,
        output: &mut Buffer) -> Result<(), Error> { map_formula::<K, N>(&self.object, variant, inputs, range, settings, repair, output) }
    fn fold_pieces(
        &self, variant: crate::facts::Variant, dims: [usize; 3], seed: Seed<'_>, direction: Direction, settings: Settings, pieces: &Pieces<'_>,
        output: &mut Buffer) -> Result<(), Error> { fold_formula::<K, N>(&self.object, variant, dims, seed, direction, settings, pieces, output) }
    fn fold_read(
        &self, variant: crate::facts::Variant, shape: &[usize], axis: usize, seed: Seed<'_>, direction: Direction, settings: Settings,
        input: crate::agreement::Read<'_>, output: &mut Buffer) -> Result<(), Error> {
        fold_read_formula::<K, N>(&self.object, variant, shape, axis, seed, direction, settings, input, output)
    }
    fn at(
        &self, variant: crate::facts::Variant, root: WriteRoot<'_>, values: crate::agreement::Read<'_>, ids: Positions<'_>, checked: bool, codes: bool,
        settings: Settings, output: &mut Buffer
    ) -> Result<(), Error> { at_formula(&self.object, variant, root, values, ids, checked, codes, settings, output) }
    fn scan(
        &self, variant: crate::facts::Variant, dims: [usize; 3], seed: Seed<'_>, direction: Direction, settings: Settings, input: crate::agreement::Read<'_>,
        output: &mut Buffer) -> Result<(), Error> { scan_formula(&self.object, variant, dims, seed, direction, settings, input, output) }
}

impl Kernel {
    pub fn circle(code: i64, inverse: bool, real_part: bool, variant: crate::facts::Variant, types: [DType; 1]) -> Result<Option<Self>, Error> {
        macro_rules! picked {
            ($normal:expr, $real:expr) => {
                if real_part && !matches!(variant.dtype, DType::Complex(_)) {
                    Self::resolve($real, variant, types)
                } else {
                    Self::resolve($normal, variant, types)
                }
            };
        }
        let mut kernel = match (code, inverse) {
            (0, _) => picked!(crate::Sqrt1mx2, crate::RealSqrt1mx2), (1, false) => picked!(crate::Sin, crate::Sin),
            (2, false) => picked!(crate::Cos, crate::Cos), (3, false) => picked!(crate::Tan, crate::Tan),
            (4, false) => picked!(crate::Sqrt1px2, crate::Sqrt1px2), (5, false) => picked!(crate::Sinh, crate::Sinh),
            (6, false) => picked!(crate::Cosh, crate::Cosh), (7, false) => picked!(crate::Tanh, crate::Tanh),
            (1, true) => picked!(crate::Asin, crate::RealAsin), (2, true) => picked!(crate::Acos, crate::RealAcos),
            (3, true) => picked!(crate::Atan, crate::Atan), (4, true) => picked!(crate::Sqrtx2m1, crate::RealSqrtx2m1),
            (5, true) => picked!(crate::Asinh, crate::Asinh), (6, true) => picked!(crate::Acosh, crate::RealAcosh),
            (7, true) => picked!(crate::Atanh, crate::RealAtanh), (8, inverse) => Self::resolve(crate::sqrts::Sqrtm1mx2(inverse), variant, types),
            _ => return Ok(None)
        }?;
        if real_part && let DType::Complex(width) = kernel.result { kernel.result = DType::Float(width.into()); }
        Ok(Some(kernel))
    }
    pub fn unary(op: crate::recipe::Unary, variant: crate::facts::Variant) -> Option<Self> {
        use crate::recipe::Unary::*;
        macro_rules! kernel {
            ($object:expr) => {
                crate::facts::Formula::kernel(&$object, variant)
            };
        }
        match op {
            Negate => kernel!(crate::Negate), Magnitude => kernel!(crate::Abs), Sqrt => kernel!(crate::Sqrt),
            Floor => kernel!(crate::tolerant::Floor(crate::tolerant::Tolerance::EXACT)), Ln => kernel!(crate::Ln), Sin => kernel!(crate::Sin),
            Cos => kernel!(crate::Cos), Sinh => kernel!(crate::Sinh), Cosh => kernel!(crate::Cosh), Sqrt1mx2 => kernel!(crate::Sqrt1mx2),
            Sqrt1px2 => kernel!(crate::Sqrt1px2), Digamma => kernel!(crate::Digamma), Exp => kernel!(crate::Exp), Lgamma => kernel!(crate::Lgamma),
            Not => kernel!(crate::facts::Complement)
        }
    }
    pub fn binary(op: crate::recipe::Binary, variant: crate::facts::Variant) -> Option<Self> {
        use crate::recipe::Binary::*;
        macro_rules! kernel {
            ($object:expr) => {
                crate::facts::Formula::kernel(&$object, variant)
            };
        }
        match op {
            Add => kernel!(crate::Add), Subtract => kernel!(crate::Subtract), Multiply => kernel!(crate::Multiply), Divide => kernel!(crate::Divide),
            Larger => kernel!(crate::reduce::Maximum), Smaller => kernel!(crate::reduce::Minimum), Binom => kernel!(crate::Binom),
            BinomDn => kernel!(crate::BinomDn), BinomDk => kernel!(crate::BinomDk), Polygamma => kernel!(crate::Polygamma), BitAnd => kernel!(crate::BitAnd)
        }
    }
    pub fn compare(op: crate::recipe::Cmp, variant: crate::facts::Variant) -> Option<Self> {
        crate::facts::Formula::kernel(&crate::facts::Comparison(op), variant)
    }
    pub fn integer(op: crate::recipe::Integer, variant: crate::facts::Variant) -> Option<Self> {
        crate::facts::Formula::kernel(&crate::facts::IntegerOperation(op), variant)
    }
    pub fn integer_compare(op: crate::recipe::Cmp, variant: crate::facts::Variant) -> Option<Self> {
        crate::facts::Formula::kernel(&crate::facts::IntegerComparison(op), variant)
    }
    pub fn new<K: crate::facts::Formula<N>, const N: usize>(object: K, variant: crate::facts::Variant) -> Self {
        let (result, typed_domain) = (object.result_dtype(variant.dtype), object.typed_domain([variant.dtype; N]));
        let definition: std::sync::Arc<dyn Application> = std::sync::Arc::new(Definition::<K, N> { object });
        let raising = definition.may_raise(variant, &[crate::facts::Range::ANY; N]);
        Self { repair: true, raising, typed_domain, result, characters: None, boolean_inputs: variant.dtype == DType::Boolean, variant, definition }
    }
    pub fn resolve<K: crate::facts::Formula<N>, const N: usize>(object: K, mut variant: crate::facts::Variant, types: [DType; N]) -> Result<Self, Error> {
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
                variant.dtype = DType::Integer(Width::I32);
                match kind { ByKinds::Truth(_) => DType::Boolean, ByKinds::Left => types[0], ByKinds::Right => types[1] }
            }
            None => {
                variant.dtype =
                    object.computing_dtype(variant.dtype).ok_or_else(|| Error::new(Kind::Domain, "function has no numerical body for these arguments"))?;
                object.result_dtype(variant.dtype)
            }
        };
        let typed_domain = object.typed_domain(types);
        let mut kernel = Self::new(object, variant);
        kernel.result = result;
        kernel.characters = characters;
        kernel.typed_domain = typed_domain;
        kernel.boolean_inputs = types.iter().all(|&d| d == DType::Boolean);
        Ok(kernel)
    }
    pub fn arity(&self) -> usize { self.definition.arity() }
    pub fn result_dtype(&self) -> DType { self.result }
    pub fn character_policy(&self) -> Option<crate::characters::Policy> { self.characters }
    pub fn exact(&self, args: &[num_rational::BigRational]) -> Result<Option<num_rational::BigRational>, Error> { self.definition.exact(args) }
    pub fn has_exact(&self) -> bool { self.definition.has_exact() }
    pub fn real(&self, args: &[f64], settings: Settings) -> Result<f64, Error> { self.definition.real(self.variant, args, settings, self.repair) }
    pub fn complex(&self, args: &[num_complex::Complex64], settings: Settings) -> Result<num_complex::Complex64, Error> {
        let value = self.definition.complex(self.variant, args, settings)?;
        Ok(if matches!(self.result, DType::Complex(_)) { value } else { num_complex::Complex64::new(value.re, 0.0) })
    }
    pub fn formula(
        &self, ops: &dyn crate::recipe::Ops<f64, V = crate::facts::Slot, B = crate::facts::Slot, I = crate::facts::Slot, W = crate::facts::Slot>,
        x: &[crate::facts::Slot]) -> Option<crate::facts::Slot> {
        if self.variant.dtype.is_integral() && self.variant.checked && !self.variant.wrapping && !self.definition.integer_formula() { return None; }
        self.definition.formula(ops, x)
    }
    pub fn test(
        &self, ops: &dyn crate::recipe::Ops<f64, V = crate::facts::Slot, B = crate::facts::Slot, I = crate::facts::Slot, W = crate::facts::Slot>,
        x: &[crate::facts::Slot]) -> Option<crate::facts::Slot> { self.definition.test(ops, x) }
    pub fn repair_predicate(
        &self, ops: &dyn crate::recipe::Ops<f64, V = crate::facts::Slot, B = crate::facts::Slot, I = crate::facts::Slot, W = crate::facts::Slot>,
        x: &[crate::facts::Slot]) -> Option<crate::facts::Slot> { self.definition.repair_predicate(ops, x) }
    pub fn repair_routine(&self) -> Option<crate::facts::Repair> { if self.repair { self.definition.repair_routine() } else { None } }
    pub fn ranged(mut self, ranges: &[crate::facts::Range]) -> Self {
        self.repair = self.definition.needs_repair(self.variant, ranges) != Some(false);
        self.raising = self.definition.may_raise(self.variant, ranges);
        self
    }
    /// Whether checked execution can record a numeric data error.
    pub fn may_raise(&self) -> bool { self.variant.checked && (self.result == DType::Character || self.raising) }
    pub fn validity(
        &self, ops: &dyn crate::recipe::Ops<f64, V = crate::facts::Slot, B = crate::facts::Slot, I = crate::facts::Slot, W = crate::facts::Slot>,
        x: &[crate::facts::Slot], result: crate::facts::Slot) -> Option<(Kind, crate::facts::Slot)> {
        if !self.may_raise() { return None; }
        self.definition.validity(self.variant, ops, x, result)
    }
    pub fn partials(
        &self, ops: &dyn crate::recipe::Ops<f64, V = crate::facts::Slot, B = crate::facts::Slot, I = crate::facts::Slot, W = crate::facts::Slot>,
        x: &[crate::facts::Slot], result: crate::facts::Slot, needed: &[bool], output: &mut [Option<crate::facts::Slot>]) {
        self.definition.partials(ops, x, result, needed, output)
    }
    pub fn range(&self, x: &[crate::facts::Range]) -> crate::facts::Range { self.definition.range(x) }
    pub fn domain(&self, x: &[crate::facts::Range]) -> bool { self.typed_domain || self.definition.domain(x) }
    pub fn identity(&self) -> Option<f64> { self.definition.identity() }
    pub fn fold(&self) -> Option<crate::reduce::Reduce> { self.definition.fold() }
    pub fn truth_table(&self) -> Option<u8> { self.definition.truth_table() }
    pub fn total_dtype(&self) -> Option<DType> { self.fold()?.total_dtype(self.variant) }
    pub fn fold_dtype(&self, input: DType, len: usize, seed: Option<DType>) -> DType {
        self.total_dtype().unwrap_or_else(|| match (len, seed) { (0, Some(seed)) => seed, (1, None) => input, _ => self.result })
    }
    pub fn scan_dtype(&self) -> DType {
        let carry = match self.variant.dtype { DType::Float(FloatWidth::F16) => DType::Float(FloatWidth::F32), dtype => dtype };
        self.total_dtype().unwrap_or_else(|| carry.common(self.result).unwrap_or(self.result))
    }
    pub fn product_dtype(&self, fold: &Kernel, input: DType) -> Option<DType> {
        self.definition.product(crate::facts::Variant { dtype: input, ..self.variant }, fold.fold(), input == DType::Boolean, None, Settings::default())
            .map(|p| p.result)
    }
    pub fn map(&self, inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, settings: Settings, output: &mut Buffer) -> Result<(), Error> {
        use crate::characters::{ByKinds, Policy};
        if inputs.len() != self.arity() { return Err(Error::new(Kind::Length, "kernel operand count changed")); }
        let wanted = output.dtype();
        if output.dtype() != self.result { *output = Buffer::with_capacity(self.result, range.len()); }
        match self.characters {
            Some(Policy::Kinds(ByKinds::Truth(value))) => *output = Buffer::Booleans(vec![value; range.len()]),
            Some(Policy::Kinds(side)) => {
                let side = usize::from(side == ByKinds::Right);
                inputs[side].read_into(range, self.result, self.variant.checked, output)?;
            }
            Some(Policy::Codes { .. }) => {
                with_codes(inputs, Some(range.clone()), |inputs| self.definition.map(self.variant, inputs, range, settings, self.repair, output))?
            }
            None => self.definition.map(self.variant, inputs, range, settings, self.repair, output)?,
        }
        finish(output, wanted, self.variant.checked)
    }
    pub fn fold_pieces(
        &self, dims: [usize; 3], seed: Seed<'_>, direction: Direction, settings: Settings, pieces: &Pieces<'_>, output: &mut Buffer) -> Result<(), Error> {
        let dtype = output.dtype();
        if matches!(self.characters, Some(crate::characters::Policy::Codes { .. })) {
            let projected = |range, f: &mut dyn FnMut(usize, Piece<'_>)| {
                pieces(range, &mut |range, piece| match piece {
                    Piece::Items(view) => f(range, Piece::Items(crate::characters::as_codes(view))), Piece::Total(total) => f(range, Piece::Total(total))
                })
            };
            with_code_inputs(&[], seed, |_, seed| self.definition.fold_pieces(self.variant, dims, seed, direction, settings, &projected, output))?;
        } else { self.definition.fold_pieces(self.variant, dims, seed, direction, settings, pieces, output)?; }
        finish(output, dtype, self.variant.checked)
    }
    pub fn scan(
        &self, dims: [usize; 3], seed: Seed<'_>, direction: Direction, settings: Settings, input: crate::agreement::Read<'_>, output: &mut Buffer
    ) -> Result<(), Error> {
        let dtype = output.dtype();
        if matches!(self.characters, Some(crate::characters::Policy::Codes { .. })) {
            with_code_inputs(&[input], seed, |inputs, seed| self.definition.scan(self.variant, dims, seed, direction, settings, inputs[0], output))?;
        } else { self.definition.scan(self.variant, dims, seed, direction, settings, input, output)?; }
        finish(output, dtype, self.variant.checked)
    }
    pub fn fold_read(
        &self, shape: &[usize], axis: usize, seed: Seed<'_>, direction: Direction, settings: Settings, input: crate::agreement::Read<'_>, output: &mut Buffer
    ) -> Result<(), Error> {
        let dtype = output.dtype();
        if matches!(self.characters, Some(crate::characters::Policy::Codes { .. })) {
            with_code_inputs(&[input], seed, |inputs, seed| {
                self.definition.fold_read(self.variant, shape, axis, seed, direction, settings, inputs[0], output)
            })?;
        } else { self.definition.fold_read(self.variant, shape, axis, seed, direction, settings, input, output)?; }
        finish(output, dtype, self.variant.checked)
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
            let positions = Positions::Indices(selector.0, selector.1).chunk(part(selector.0), choices.len(), checked, &mut indices);
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
    if let Some(native) = map.definition.product(map.variant, fold.fold(), map.boolean_inputs, Some(dtype), settings) {
        if (native.run)(map.variant, inputs, plan, output)? {
            return finish(output, dtype, (map.variant.checked || fold.variant.checked) && !map.variant.wrapping);
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
    macro_rules! pairs {
        (($($arg:tt)*) folds {$($folds:tt)*} booleans {$($booleans:tt)*}
            products {$($pair:ident: [$(($map:ty, $reduce:ident)),*] at [$($t:ty),*] $(wrapping [$($wt:ty),*])? => [$($names:tt)*];)*}
            boolean_products {$($bp:ident: [$(($bm:ty, $bf:ident)),*] at [$($bt:ty),*] => [$($bn:tt)*];)*}
        ) => {
            pairs!(@rows $($pair: [$(($map, $reduce)),*] at [$($t),*] $(wrapping [$($wt),*])?;)* $($bp: [$(($bm, $bf)),*] at [$($bt),*];)*);
        };
        (@rows $($pair:ident: [$(($map:ty, $reduce:ident)),*] at [$($t:ty),*] $(wrapping [$($wt:ty),*])?;)*) => {$(
            if false $(|| (TypeId::of::<K>() == TypeId::of::<$map>() && fold == crate::reduce::Reduce::$reduce))* {
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
            }
        )*};
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
    use crate::agreement::{Mapping, Missing, Read, Walked};
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
        if T::DTYPE.is_integral() {
            let (mut a, mut b) = (Vec::new(), Vec::new());
            let a = reads[0].part::<T>(0..m * k, variant.checked, &mut a)?;
            let b = reads[1].part::<T>(0..k * n, variant.checked, &mut b)?;
            let a = match a { Arg::One(x) => std::borrow::Cow::Owned(vec![x; m * k]), Arg::Many(x) => std::borrow::Cow::Borrowed(x) };
            let b = match b { Arg::One(x) => std::borrow::Cow::Owned(vec![x; k * n]), Arg::Many(x) => std::borrow::Cow::Borrowed(x) };
            let start = result.len();
            result.resize(start + m * n, P::Out::default());
            if !pair.checked_into(&a, &b, matrix.size, &mut result[start..]) {
                return if variant.checked { Err(Error::new(Kind::Overflow, "product does not fit its integer result")) } else { Ok(false) };
            }
        } else {
            let (Some((av, am)), Some((bv, bm))) = (reads[0].positional(&[m, k]), reads[1].positional(&[k, n])) else { return Ok(false); };
            let (Some(a), Some(b)) = (T::of(av), T::of(bv)) else { return Ok(false); };
            let (am, bm) = (Walked::from(am), Walked::from(bm));
            result.extend(pair.inner_mapped(Mapped { data: a, mapping: &am }, Mapped { data: b, mapping: &bm }, [Orientation::Normal; 2], matrix.size));
        }
    }
    *output = P::Out::buffer(result);
    Ok(true)
}

pub fn gather_plan(
    layout: &crate::layout::Layout, routes: &crate::structure::Routes, indices: &[(crate::agreement::Read<'_>, DType)], broadcast: bool, checked: bool,
    filled: bool) -> Result<crate::agreement::ReadPlan, Error> {
    use crate::agreement::{Mapping, Missing, ReadPlan};
    let mut scratch: Vec<_> = indices.iter().map(|(_, dtype)| Buffer::with_capacity(*dtype, 0)).collect();
    let views = indices.iter().zip(&mut scratch).map(|((read, dtype), scratch)| read.part_view(0..read.plan.len(), *dtype, checked, scratch))
        .collect::<Result<Vec<_>, _>>()?;
    let plan = crate::structure::bound_routes(layout, routes, &views, broadcast, checked)?;
    Ok(ReadPlan::copy(0, layout.count(), Mapping::from(plan.steps), plan.layout.count(), if filled { Missing::Scalar(1) } else { Missing::Default }))
}
pub fn gather(
    source: crate::agreement::Read<'_>, layout: &crate::layout::Layout, routes: &crate::structure::Routes, indices: &[(crate::agreement::Read<'_>, DType)],
    broadcast: bool, checked: bool, fill: Option<crate::agreement::Read<'_>>, output: &mut Buffer) -> Result<(), Error> {
    use crate::agreement::{Binding, Read, ReadPlan};
    if source.plan.len() != layout.count() { return Err(Error::new(Kind::Length, "gather source does not fit its layout")); }
    let plan = gather_plan(layout, routes, indices, broadcast, checked, fill.is_some())?;
    let mut sources = source.sources.to_vec();
    let mut prototype = Buffer::with_capacity(output.dtype(), 1);
    let fill_source = sources.len();
    if let Some(fill) = fill {
        if fill.plan.len() != 1 { return Err(Error::new(Kind::Length, "gather fill must be scalar")); }
        fill.read_into(0..1, output.dtype(), checked, &mut prototype)?;
        sources.push(Binding::full(prototype.view()));
    }
    let read = plan.compose_inputs::<Error>(&mut |which, _| Ok(if which == 0 { source.plan.clone() } else { ReadPlan::input(fill_source, 1) }))?;
    Read { sources: &sources, plan: &read }.read_into(0..read.len(), output.dtype(), checked, output)
}

pub fn at(
    combine: Option<&Kernel>, root: WriteRoot<'_>, values: crate::agreement::Read<'_>, ids: Positions<'_>, checked: bool, settings: Settings,
    output: &mut Buffer) -> Result<(), Error> {
    let (wanted, len, n) = (output.dtype(), root.len(), ids.len());
    if values.plan.len() != 1 && values.plan.len() != n { return Err(Error::new(Kind::Length, "write values do not match positions")); }
    if checked || matches!(ids, Positions::Route(..)) {
        let mut scratch = Buffer::with_capacity(DType::Boolean, 0);
        for start in (0..n).step_by(crate::CHUNK) { ids.chunk(start..n.min(start + crate::CHUNK), len, checked, &mut scratch)?; }
    }
    let codes = combine.is_some_and(|kernel| matches!(kernel.characters, Some(crate::characters::Policy::Codes { .. })));
    match combine {
        None => {
            root.initialize(wanted, checked, false, output)?;
            with_dtype!(wanted, T => at_typed::<T>(values, ids, checked, None::<fn(T, T) -> (T, Option<Kind>)>, output))?;
        },
        Some(kernel) if codes => with_codes(&[values], None, |inputs| kernel.definition.at(kernel.variant, root, inputs[0], ids, checked, true, settings, output))?,
        Some(kernel) => kernel.definition.at(kernel.variant, root, values, ids, checked, false, settings, output)?,
    }
    finish(output, wanted, checked)
}

fn with_code_inputs<R>(
    inputs: &[crate::agreement::Read<'_>], seed: Seed<'_>, call: impl FnOnce(&[crate::agreement::Read<'_>], Seed<'_>) -> Result<R, Error>) -> Result<R, Error> {
    let mut all = inputs.to_vec();
    match seed { Seed::Initial(r) | Seed::Tail(r) => all.push(r), Seed::Identity => {} }
    with_codes(&all, None, |projected| {
        let seed = match seed {
            Seed::Initial(_) => Seed::Initial(projected[inputs.len()]), Seed::Tail(_) => Seed::Tail(projected[inputs.len()]), Seed::Identity => Seed::Identity
        };
        call(&projected[..inputs.len()], seed)
    })
}

fn with_codes<R>(
    inputs: &[crate::agreement::Read<'_>], range: Option<std::ops::Range<usize>>, call: impl FnOnce(&[crate::agreement::Read<'_>]) -> Result<R, Error>
) -> Result<R, Error> {
    use crate::{agreement::{Binding, Read, ReadPlan}, convert::{Conversion, ConversionKind}};
    let sources: Vec<Vec<_>> = inputs.iter().map(|r| r.sources.iter().map(|s| s.as_codes()).collect()).collect();
    let mut projected: Vec<_> = inputs.iter().zip(&sources).map(|(r, sources)| Read { sources, plan: r.plan }).collect();
    if projected.iter().all(|r| r.sources.iter().all(|s| !matches!(s.view.dtype(), DType::Float(_)))) { return call(&projected); }
    let converted = projected.iter().map(|read| {
        if !read.sources.iter().any(|s| matches!(s.view.dtype(), DType::Float(_))) { return Ok(None); }
        let range = range.clone().unwrap_or(0..read.plan.len());
        let mut buffer = Buffer::with_capacity(DType::Integer(Width::I64), range.len());
        Conversion { input: DType::Float(FloatWidth::F64), output: buffer.dtype(), kind: ConversionKind::Keep }.map(*read, range.clone(), true, &mut buffer)?;
        Ok(Some((range.start, buffer)))
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
        let positions = ids.chunk(start..end, len, checked, &mut indices)?;
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
    object: &K, variant: crate::facts::Variant, root: WriteRoot<'_>, values: crate::agreement::Read<'_>, ids: Positions<'_>, checked: bool, codes: bool,
    settings: Settings, output: &mut Buffer
) -> Result<(), Error> {
    if N != 2 { return Err(Error::new(Kind::Length, "a modified write needs a binary function")); }
    let working = with_binary_body!(object, variant, settings, |T, _apply| Ok(T::DTYPE))?;
    if let Some(fold) = object.fold() {
        native_reduction!(fold, variant.dtype, variant.checked && !variant.wrapping, |R, T| {
            type S = <R as crate::reduce::Reduction<T>>::Total;
            if S::DTYPE == variant.dtype || S::DTYPE == working {
                root.initialize(S::DTYPE, checked, codes, output)?;
                return at_typed::<S>(values, ids, checked, Some(|x, y| {
                    let (value, ok) = <R as crate::reduce::Reduction<T>>::chain(x, y);
                    (value, (!ok && variant.checked && !variant.wrapping).then_some(Kind::Overflow))
                }), output);
            }
            None::<()>
        });
    }
    with_binary_body!(object, variant, settings, |T, apply| {
        root.initialize(T::DTYPE, checked, codes, output)?;
        at_typed::<T>(values, ids, checked, Some(apply), output)
    })
}

fn map_formula<K: crate::facts::Formula<N>, const N: usize>(
    object: &K, variant: crate::facts::Variant, inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, settings: Settings, repair: bool,
    output: &mut Buffer) -> Result<(), Error> {
    if inputs.len() != N { return Err(Error::new(Kind::Length, "kernel operand count changed")); }
    crate::events::taken(crate::events::Route::VectorKernel, range.len());
    if inputs.iter().any(|r| r.plan.contiguous(range.clone()).is_none()) { crate::events::taken(crate::events::Route::MappedKernel, range.len()); }
    if let Some(slices) = object.native_slices() {
        let valid = |ok: bool| if variant.checked && !ok { Err(Kind::Domain) } else { Ok(()) };
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
        DType::Float(FloatWidth::F16) => map_real::<K, f32, half::f16, N>(object, variant, inputs, range, settings, repair, output),
        DType::Float(FloatWidth::F32) => map_real::<K, f32, f32, N>(object, variant, inputs, range, settings, repair, output),
        DType::Float(FloatWidth::F64) => map_real::<K, f64, f64, N>(object, variant, inputs, range, settings, repair, output),
        DType::Boolean => map_integer::<K, u8, N>(object, variant, inputs, range, output),
        DType::Integer(width) => with_width!(width, I => map_integer::<K, I, N>(object, variant, inputs, range, output)),
        DType::Complex(crate::buffer::ComplexWidth::F32) => map_complex::<K, f32, N>(object, variant, inputs, range, settings, output),
        DType::Complex(crate::buffer::ComplexWidth::F64) => map_complex::<K, f64, N>(object, variant, inputs, range, settings, output),
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

#[inline(always)]
fn complex_value<K, F, const N: usize>(
    object: &K, variant: crate::facts::Variant, ops: &crate::recipe::NativeOps<{ crate::FUSED }>, x: [num_complex::Complex<F>; N]
) -> (num_complex::Complex<F>, Option<Kind>) where K: crate::facts::Formula<N>, F: crate::float::Native {
    let Some(result) = object.complex(ops, x) else { return (num_complex::Complex::new(F::zero(), F::zero()), variant.checked.then_some(Kind::Domain)); };
    let valid = !variant.checked || !object.fails_nan() || crate::facts::nan_valid(ops, x.into_iter().flat_map(|x| [x.re, x.im]), [result.re, result.im]);
    (result, (!valid).then_some(Kind::Domain))
}

fn scalar_result<T>((value, error): (T, Option<Kind>)) -> Result<T, Error> {
    match error { Some(kind) => Err(Error::new(kind, "numerical argument is outside the function's domain")), None => Ok(value) }
}
fn map_real<K, F, Dst, const N: usize>(
    object: &K, variant: crate::facts::Variant, inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, settings: Settings, repair: bool,
    output: &mut Buffer) -> Result<(), Error> where K: crate::facts::Formula<N>, F: crate::float::Native + crate::buffer::Cast + Source<Dst>,
    Dst: Element + crate::Float {
    let ops = crate::recipe::NativeOps::<{ crate::FUSED }> { tolerance: settings.tolerance };
    if object.boolean() {
        return map_values::<F, bool, N>(inputs, range, variant.checked, output, &|x| {
            let (value, error) = real_value(object, variant, &ops, x, false);
            (value != F::zero(), error)
        });
    }
    let body = |x| object.real(&ops, x).expect("a resolved object has a real body");
    if repair && let Some(repair) = crate::facts::Formula::repair_routine(object) {
        return map_blocks::<F, Dst, N>(inputs, range, variant.checked, output, &|args, n, out| {
            let needs = |x| object.emit_repair(&ops, x).unwrap_or(false);
            let ok = crate::slice::two_pass(args, n, out, |x| (!needs(x)).then(|| body(x)), |x| if needs(x) { repair.apply(x[0]) } else { body(x) });
            if variant.checked && object.fails_nan() && !ok { Err(Kind::Domain) } else { Ok(()) }
        });
    }
    map_values::<F, Dst, N>(inputs, range, variant.checked, output, &|x| {
        let (result, error) = real_value(object, variant, &ops, x, false);
        (Source::<Dst>::read(result), error)
    })
}

fn map_integer<K, I, const N: usize>(
    object: &K, variant: crate::facts::Variant, inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, output: &mut Buffer) -> Result<(), Error>
where K: crate::facts::Formula<N>, I: crate::recipe::Overflowing {
    let ops = crate::recipe::IntegerOps::<I>::new(variant);
    let apply = |x| integer_value(object, variant, &ops, x);
    if object.boolean() {
        return map_values::<I, bool, N>(inputs, range, variant.checked, output, &|x| {
            let (x, e) = apply(x);
            (Into::<i64>::into(x) != 0, e)
        });
    }
    map_values::<I, I, N>(inputs, range, variant.checked, output, &apply)
}

fn map_complex<K, F, const N: usize>(
    object: &K, variant: crate::facts::Variant, inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, settings: Settings, output: &mut Buffer
) -> Result<(), Error> where K: crate::facts::Formula<N>, F: crate::float::Native + crate::buffer::Cast, num_complex::Complex<F>: crate::buffer::Cast {
    let ops = crate::recipe::NativeOps::<{ crate::FUSED }> { tolerance: settings.tolerance };
    let apply = |x| complex_value(object, variant, &ops, x);
    if object.boolean() {
        map_values::<num_complex::Complex<F>, bool, N>(inputs, range, variant.checked, output, &|x| {
            let (x, e) = apply(x);
            (x.re != F::zero(), e)
        })
    } else if output.dtype().is_real() {
        map_values::<num_complex::Complex<F>, F, N>(inputs, range, variant.checked, output, &|x| {
            let (x, e) = apply(x);
            (x.re, e)
        })
    } else { map_values::<num_complex::Complex<F>, num_complex::Complex<F>, N>(inputs, range, variant.checked, output, &apply) }
}

fn map_values<F: crate::buffer::Cast, O: Element, const N: usize>(
    inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, checked: bool, output: &mut Buffer,
    apply: &(impl Fn([F; N]) -> (O, Option<Kind>) + Sync)) -> Result<(), Error> {
    map_blocks(inputs, range, checked, output, &|args, n, out| {
        let ok = crate::slice::mapped(&args, n, out, #[inline(always)] |x| {
            let (r, error) = apply(x);
            (r, error.is_none())
        });
        if !ok && checked { for i in 0..n { if let Some(kind) = apply(std::array::from_fn(|j| args[j].nth(i))).1 { return Err(kind); } } }
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

fn fold_formula<K: crate::facts::Formula<N>, const N: usize>(
    object: &K, variant: crate::facts::Variant, dims: [usize; 3], seed: Seed<'_>, direction: Direction, settings: Settings, pieces: &Pieces<'_>,
    output: &mut Buffer) -> Result<(), Error> {
    if let Some(fold) = object.fold() { return fold.run_pieces(variant, dims, seed, pieces, output); }
    if N != 2 { return Err(Error::new(Kind::Length, "a fold needs a binary function")); }
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
        for i in 0..view.len() {
            match T::cast(view, i) { Ok(value) => out.push(value), Err(error) if checked => { failure = Some(error); return; } Err(_) => out.push(T::FILL) }
        }
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
    if let Some(fold) = object.fold() { return fold.run_scan(variant, dims, seed, direction, input, output); }
    if N != 2 { return Err(Error::new(Kind::Length, "a scan needs a binary function")); }
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
    if let Some(fold) = object.fold() { return fold.run_read(variant, shape, axis, seed, input, output); }
    let dims = crate::layout::Lanes::new(shape, axis)?.dims();
    let pieces = |range: std::ops::Range<usize>, emit: &mut dyn FnMut(usize, Piece<'_>)| {
        let mut scratch = Buffer::with_capacity(variant.dtype, 0);
        let start = range.start;
        let view = input.part_view(range, variant.dtype, variant.checked, &mut scratch)?;
        emit(start, Piece::Items(view));
        Ok(())
    };
    fold_formula(object, variant, dims, seed, direction, settings, &pieces, output)
}
/// The float width that a Boolean or integer computes in under rustymath's own policy, NumPy's: the float that [`DType::inexact`] gives it under `f64`. `f64` for any other dtype.
pub fn numpy_float(dtype: DType) -> FloatWidth { match dtype.inexact(FloatWidth::F64) { Some((DType::Float(w), _)) => w, _ => FloatWidth::F64 } }

fn finish(output: &mut Buffer, dtype: DType, checked: bool) -> Result<(), Error> {
    if output.dtype() != dtype { *output = Buffer::cast_results(output.view(), dtype, checked)?; }
    Ok(())
}
