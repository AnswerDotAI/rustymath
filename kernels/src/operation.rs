//! Resolved numerical operations over shared [`Read`](crate::agreement::Read) plans. [`Kernel::try_resolve`] fixes the computing dtype, result dtype and character policy from an object's [`Formula`](crate::facts::Formula). The same definition supplies native execution and graph formulas.
//!
//! [`DType::computes_in`] promotes unsupported Boolean and integer inputs to i64, then to the NumPy float width. Unsupported character arithmetic is a domain error.
use crate::buffer::{Buffer, DType, Element, FloatWidth, MapOutput, Output, Source, View, Width};
use crate::domain::Range;
use crate::error::{Error, Kind};
use crate::gather::Writes;
use crate::reduce::{Direction, Piece, Pieces, Seed, native_reduction};
use crate::slice::Arg;
use crate::{FUSED, Native, facts::Formula, recipe::NativeOps, tolerant::Banded};

/// Numeric settings rebound for each execution, outside item loops. `tolerance` is the run's comparison tolerance: tolerant objects compare within it, and a kernel reads a float argument as a whole number within it (`Projection::Wholes`). BPL sets it nonzero only under Mathish, and Python only in Mathish mode.
#[derive(Clone, Copy, Debug)]
pub struct Settings { pub tolerance: crate::tolerant::Tolerance }
impl Settings {
    /// The scalar operations that a kernel runs with: the run's tolerance in place of each tolerant object's own.
    pub fn ops(self) -> crate::recipe::NativeOps<{ crate::FUSED }> { crate::recipe::NativeOps { tolerance: Some(self.tolerance.value()) } }
}

/// How a kernel reads its inputs before its formula runs (`Kernel::projection`, `with_projected`).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Projection {
    /// As they are.
    AsIs,
    /// Characters as their codes, under a code policy, and floats as the whole numbers they equal within the tolerance.
    Codes(crate::tolerant::Tolerance),
    /// Floats as the whole numbers they equal within the tolerance, where the kernel computes at an integral dtype from float arguments.
    Wholes(crate::tolerant::Tolerance),
}
impl Projection {
    /// The tolerance within which this projection reads a float as a whole number, or `None` where it reads floats as they are.
    fn wholes(self) -> Option<crate::tolerant::Tolerance> { match self { Self::AsIs => None, Self::Codes(t) | Self::Wholes(t) => Some(t) } }
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
            (Self::Owned(buffer), projection) => {
                *output = with_projected_view(buffer.view(), projection, |view| Buffer::cast_results(view, dtype, checked))??;
                Ok(())
            }
        }
    }
}

/// One resolved mathematical definition and its captured numerical policies.
#[derive(Clone)]
pub struct Kernel {
    variant: crate::domain::Variant, result: DType, boolean_inputs: bool, repair: bool, raising: bool, wholes: bool, policy: Option<crate::characters::Policy>,
    definition: std::sync::Arc<dyn Application>
}

impl std::fmt::Debug for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernel").field("variant", &self.variant).field("arity", &self.arity()).finish()
    }
}

/// A resolved elementwise call. Kind selection is complete before this is executed.
#[derive(Clone, Debug)]
pub enum Map { Kernel(Kernel, Settings), Convert(crate::convert::Conversion, Settings, bool) }
impl Map {
    pub fn arity(&self) -> usize { match self { Self::Kernel(kernel, _) => kernel.arity(), Self::Convert(..) => 1 } }
    /// Appends every result, including initialized fallback items on a numerical error. Errors are execution failures, not a pre-write refusal to handle the call. Both variants follow [`crate::error::checked`]: an enclosing checked scope owns recorded numerical failures. Under an active error slot, `Ok` alone does not establish success. Discard private outputs when that scope fails; neither `Err` nor a recorded failure is a route refusal.
    pub fn run<'a, 'b>(
        &self, inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, scratch: &'a mut crate::buffer::Scratch, mut output: MapOutput<'a, 'b>
    ) -> Result<(), Error> {
        if inputs.len() != self.arity() { output.fill(range.len()); return Err(Error::new(Kind::Length, "map arity does not match its operands")); }
        match self {
            Self::Kernel(kernel, settings) => kernel.map_in(inputs, range, *settings, scratch, output),
            Self::Convert(conversion, settings, checked) => conversion.map_in(inputs[0], range, *checked, settings.tolerance, Some(scratch), output),
        }
    }
}

trait Application: Send + Sync {
    fn exact(&self, args: &[num_rational::BigRational]) -> Result<Option<num_rational::BigRational>, Error>;
    fn has_exact(&self) -> bool;
    fn integer(&self, variant: crate::domain::Variant, args: &[i64]) -> Result<i64, Error>;
    fn real(&self, variant: crate::domain::Variant, args: &[f64], settings: Settings, repair: bool) -> Result<f64, Error>;
    fn complex(&self, variant: crate::domain::Variant, args: &[num_complex::Complex64], settings: Settings) -> Result<num_complex::Complex64, Error>;
    fn arity(&self) -> usize;
    fn inputs(&self) -> &[crate::domain::Range];
    fn ordered_kinds(&self, classes: [i32; 2]) -> Option<crate::characters::ByKinds>;
    fn may_raise(&self, variant: crate::domain::Variant, ranges: &[crate::domain::Range]) -> bool;
    fn partials(
        &self, ops: &dyn crate::recipe::Ops<f64, V = crate::facts::Slot, B = crate::facts::Slot, I = crate::facts::Slot, W = crate::facts::Slot>,
        x: &[crate::facts::Slot], result: crate::facts::Slot, needed: &[bool], output: &mut [Option<crate::facts::Slot>]);
    fn range(&self, x: &[crate::domain::Range]) -> crate::domain::Range;
    fn same(&self, x: Range) -> Option<Range>;
    fn identity(&self) -> Option<f64>;
    fn kept(&self, constants: &[Option<f64>]) -> Option<usize>;
    fn reduction(&self, variant: crate::domain::Variant) -> Option<crate::reduce::Reduce>;
    fn inverse_fold(&self) -> Option<crate::reduce::Reduce>;
    fn truth_table(&self) -> Option<u8>;
    fn product(
        &self, variant: crate::domain::Variant, fold: Option<crate::reduce::Reduce>, boolean_inputs: bool, wanted: Option<DType>, settings: Settings
    ) -> Option<NativeProduct>;
    /// `judged`: whether each item's result is checked for errors, which a kernel skips where its arguments' types can't make one (`Kernel::may_raise`).
    fn map(
        &self, variant: crate::domain::Variant, inputs: Inputs<'_, '_>, range: std::ops::Range<usize>, settings: Settings, repair: bool, judged: bool,
        output: &mut Output<'_, '_>) -> Result<(), Error>;
    /// `route`, an ordered walk by this binary formula's per-item kernels at `variant`'s computing type (`binary_formula`).
    fn binary(&self, variant: crate::domain::Variant, settings: Settings, route: Route<'_>, output: &mut Buffer) -> Result<(), Error>;
}

/// `object`, for arguments whose values lie in `inputs`, the ranges of their types.
struct Definition<K, const N: usize> { object: K, inputs: [crate::domain::Range; N] }

impl<K: crate::facts::Formula<N>, const N: usize> Application for Definition<K, N> {
    fn exact(&self, args: &[num_rational::BigRational]) -> Result<Option<num_rational::BigRational>, Error> {
        if args.len() != N { return Err(Error::new(Kind::Length, "scalar operand count changed")); }
        self.object.exact(std::array::from_fn(|i| &args[i]))
    }
    fn has_exact(&self) -> bool { K::HAS_EXACT }
    fn integer(&self, variant: crate::domain::Variant, args: &[i64]) -> Result<i64, Error> {
        let x: [i64; N] = args.try_into().map_err(|_| Error::new(Kind::Length, "scalar operand count changed"))?;
        let width = match variant.dtype {
            DType::Integer(w) => w, DType::Boolean => Width::U8,
            _ => return Err(Error::new(Kind::Domain, "an integer scalar call needs an integer computing dtype"))
        };
        with_width!(width, I => {
            let ops = crate::recipe::IntegerOps::<I>::new(variant);
            scalar_result(integer_value(&self.object, variant, &ops, x.map(<I as crate::recipe::Overflowing>::low))).map(Into::into)
        })
    }
    fn real(&self, variant: crate::domain::Variant, args: &[f64], settings: Settings, repair: bool) -> Result<f64, Error> {
        let x: [f64; N] = args.try_into().map_err(|_| Error::new(Kind::Length, "scalar operand count changed"))?;
        let ops = settings.ops();
        match variant.dtype {
            DType::Float(FloatWidth::F64) => scalar_result(real_value(&self.object, variant, &ops, x, repair)),
            DType::Float(width) => {
                let value = scalar_result(real_value(&self.object, variant, &ops, x.map(|x| x as f32), repair))?;
                Ok(if width == FloatWidth::F16 { half::f16::from_f32(value).to_f64() } else { f64::from(value) })
            }
            _ => Err(Error::new(Kind::Domain, "a real scalar call needs a floating computing dtype")),
        }
    }
    fn complex(&self, variant: crate::domain::Variant, args: &[num_complex::Complex64], settings: Settings) -> Result<num_complex::Complex64, Error> {
        let x: [num_complex::Complex64; N] = args.try_into().map_err(|_| Error::new(Kind::Length, "scalar operand count changed"))?;
        let ops = settings.ops();
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
    fn inputs(&self) -> &[crate::domain::Range] { &self.inputs }
    fn may_raise(&self, variant: crate::domain::Variant, ranges: &[crate::domain::Range]) -> bool {
        if variant.dtype == DType::Boolean && ranges.iter().all(|r| r.within(0.0, 1.0)) { return boolean_raises(&self.object, variant); }
        if matches!(variant.dtype, DType::Complex(_)) || variant.dtype.is_integral() && !K::INTEGER_FORMULA { return true; }
        let Ok(args) = ranges.try_into() else { return true; };
        // A float result raises only outside the domain that the formula declares (`Formula::domain`): range arithmetic can't show that a result isn't NaN.
        if matches!(variant.dtype, DType::Float(_)) && self.object.domain(variant.dtype, args).is_some() { return false; }
        self.object.emit_valid::<f64, _>(variant, &crate::recipe::RangeOps(variant.dtype), args, self.object.range(args))
            .is_some_and(|(_, valid)| valid != Some(true))
    }
    fn partials(
        &self, ops: &dyn crate::recipe::Ops<f64, V = crate::facts::Slot, B = crate::facts::Slot, I = crate::facts::Slot, W = crate::facts::Slot>,
        x: &[crate::facts::Slot], result: crate::facts::Slot, needed: &[bool], output: &mut [Option<crate::facts::Slot>]) {
        let (Ok(x), Ok(needed)) = (x.try_into(), needed.try_into()) else { output.fill(None); return; };
        output.copy_from_slice(&self.object.emit_partials(ops, x, result, needed));
    }
    fn range(&self, x: &[crate::domain::Range]) -> crate::domain::Range { x.try_into().map_or(crate::domain::Range::ANY, |x| self.object.range(x)) }
    fn same(&self, x: Range) -> Option<Range> { self.object.same(x) }
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
    fn reduction(&self, variant: crate::domain::Variant) -> Option<crate::reduce::Reduce> { reduction(&self.object, variant) }
    fn inverse_fold(&self) -> Option<crate::reduce::Reduce> { self.object.inverse_fold() }
    fn truth_table(&self) -> Option<u8> { self.object.truth_table() }
    fn product(
        &self, variant: crate::domain::Variant, fold: Option<crate::reduce::Reduce>, boolean_inputs: bool, wanted: Option<DType>, settings: Settings
    ) -> Option<NativeProduct> { product_for::<K, N>(&self.object, variant, fold, boolean_inputs, wanted, settings) }
    fn map(
        &self, variant: crate::domain::Variant, inputs: Inputs<'_, '_>, range: std::ops::Range<usize>, settings: Settings, repair: bool, judged: bool,
        output: &mut Output<'_, '_>) -> Result<(), Error> { map_formula::<K, N>(&self.object, variant, inputs, range, settings, repair, judged, output) }
    fn binary(&self, variant: crate::domain::Variant, settings: Settings, route: Route<'_>, output: &mut Buffer) -> Result<(), Error> {
        binary_formula::<K, N>(&self.object, variant, settings, route, output)
    }
}

impl Kernel {
    pub fn resolve<K: crate::facts::Formula<N>, const N: usize>(object: K, variant: crate::domain::Variant, types: [DType; N]) -> Result<Self, Error> {
        Self::try_resolve(object, variant, types)?.ok_or_else(|| Error::new(Kind::Domain, "function has no numerical body for these arguments"))
    }

    pub fn circle(code: i64, inverse: bool, real_part: bool, variant: crate::domain::Variant, types: [DType; 1]) -> Result<Option<Self>, Error> {
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
            (7, true) => picked!(crate::Atanh, crate::RealAtanh), (8, inverse) => picked!(crate::Sqrtm1mx2(inverse), crate::RealSqrtm1mx2(inverse)),
            _ => return Ok(None)
        })?
        else { return Ok(None); };
        if real_part && let DType::Complex(width) = kernel.result { kernel.result = DType::Float(width.into()); }
        Ok(Some(kernel))
    }
    pub fn new<K: crate::facts::Formula<N>, const N: usize>(object: K, variant: crate::domain::Variant) -> Self {
        Self::typed(object, variant, [variant.dtype; N])
    }
    /// The kernel for arguments of `types`, whose values bound the errors it can raise (`may_raise`).
    fn typed<K: crate::facts::Formula<N>, const N: usize>(object: K, variant: crate::domain::Variant, types: [DType; N]) -> Self {
        let result = object.result_dtype(variant.dtype);
        let definition: std::sync::Arc<dyn Application> = std::sync::Arc::new(Definition::<K, N> { object, inputs: types.map(Range::of_dtype) });
        let raising = definition.may_raise(variant, definition.inputs());
        let wholes = variant.dtype.is_integral() && types.iter().any(|d| matches!(d, DType::Float(_)));
        Self { repair: true, raising, result, policy: None, boolean_inputs: types.iter().all(|&d| d == DType::Boolean), wholes, variant, definition }
    }
    pub fn try_resolve<K: crate::facts::Formula<N>, const N: usize>(
        object: K, mut variant: crate::domain::Variant, types: [DType; N]) -> Result<Option<Self>, Error> {
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
    /// `op` resolved for arguments of types `types` in `variant`, as [`Kernel::try_resolve`] resolves its object: `Ok(None)` where it has no body for them, or `types` doesn't hold one type per argument. Non-generic, so every formula's kernel code compiles in rustymath alone.
    pub fn of(op: crate::facts::Op, variant: crate::domain::Variant, types: &[DType]) -> Result<Option<Self>, Error> {
        struct Resolve<'a>(crate::domain::Variant, &'a [DType]);
        impl crate::facts::Visit for Resolve<'_> {
            type Out = Result<Option<Kernel>, Error>;
            fn visit<K: crate::facts::Formula<N>, const N: usize>(self, object: K) -> Self::Out {
                let Ok(types) = self.1.try_into() else { return Ok(None) };
                Kernel::try_resolve(object, self.0, types)
            }
        }
        op.visit(types.len(), Resolve(variant, types))
    }
    /// The kernel of `op`'s partial in its argument `side` (`derivative::Slope`), whose arguments are `op`'s and then `op`'s result, one type each in `types`, resolved as [`Kernel::of`] resolves `op`. `Ok(None)` also where `op` takes three arguments.
    pub fn slope(op: crate::facts::Op, side: usize, variant: crate::domain::Variant, types: &[DType]) -> Result<Option<Self>, Error> {
        use crate::derivative::Slope;
        struct Slopes<'a>(usize, crate::domain::Variant, &'a [DType]);
        impl crate::facts::Visit for Slopes<'_> {
            type Out = Result<Option<Kernel>, Error>;
            fn visit<K: crate::facts::Formula<N>, const N: usize>(self, _: K) -> Self::Out { Ok(None) }
            fn unary<K: crate::facts::Formula<1>>(self, object: K) -> Self::Out {
                let Ok(types) = self.2.try_into() else { return Ok(None) };
                Kernel::try_resolve(Slope(object), self.1, types)
            }
            fn binary<K: crate::facts::Formula<2>>(self, object: K) -> Self::Out {
                let Ok(types) = self.2.try_into() else { return Ok(None) };
                if self.0 == 0 { Kernel::try_resolve(Slope::<K, 0>(object), self.1, types) } else { Kernel::try_resolve(Slope::<K, 1>(object), self.1, types) }
            }
        }
        op.visit(types.len().saturating_sub(1), Slopes(side, variant, types))
    }
    pub fn arity(&self) -> usize { self.definition.arity() }
    pub fn result_dtype(&self) -> DType { self.result }
    /// The resolved computing dtype and numerical policies.
    pub fn variant(&self) -> crate::domain::Variant { self.variant }
    /// Select whether this operation reports item errors.
    pub fn with_checked(mut self, checked: bool) -> Self {
        self.variant.checked = checked;
        self.raising = self.definition.may_raise(self.variant, self.definition.inputs());
        self
    }
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
    pub fn integers(&self, args: &[i64]) -> Result<i64, Error> { crate::error::checked(|| self.definition.integer(self.variant, args)) }
    pub fn real(&self, args: &[f64], settings: Settings) -> Result<f64, Error> {
        crate::error::checked(|| self.definition.real(self.variant, args, settings, self.repair))
    }
    pub fn complex(&self, args: &[num_complex::Complex64], settings: Settings) -> Result<num_complex::Complex64, Error> {
        let value = crate::error::checked(|| self.definition.complex(self.variant, args, settings))?;
        Ok(if matches!(self.result, DType::Complex(_)) { value } else { num_complex::Complex64::new(value.re, 0.0) })
    }
    fn kind_result<T: Copy>(&self, x: &[T], truth: impl FnOnce(bool) -> T) -> Option<T> {
        use crate::characters::{ByKinds, Policy};
        Some(match self.policy? {
            Policy::Kinds(ByKinds::Truth(value)) => truth(value), Policy::Kinds(ByKinds::Left) => x[0], Policy::Kinds(ByKinds::Right) => x[1], _ => return None
        })
    }
    /// Whether checked execution can record a numeric data error, for arguments that hold any values of their types.
    pub fn may_raise(&self) -> bool { self.variant.checked && (self.result == DType::Character || self.wholes || self.raising) }
    /// Whether checked execution can record a numeric data error where each argument holds values of its type within its range in `narrowed`, such as a constant's own value or the nonnegative numbers.
    pub fn may_raise_on(&self, narrowed: &[Range]) -> bool {
        self.variant.checked && (self.result == DType::Character || self.wholes || self.definition.may_raise(self.variant, &self.points(narrowed)))
    }
    /// The result bound on arguments within `narrowed`. `same` says both arguments denote one value, enabling the formula's same-input fact.
    pub fn range_on(&self, narrowed: &[Range], same: bool) -> Range {
        let points = self.points(narrowed);
        if same && let [x, y] = points.as_slice() && let Some(range) = self.definition.same(x.meet(*y)) { return self.bound(range); }
        self.range(&points)
    }
    /// Each argument's range: its type's values within its range in `narrowed`.
    fn points(&self, narrowed: &[Range]) -> Vec<Range> { self.definition.inputs().iter().zip(narrowed).map(|(&range, &n)| range.meet(n)).collect() }
    pub fn partials(
        &self, ops: &dyn crate::recipe::Ops<f64, V = crate::facts::Slot, B = crate::facts::Slot, I = crate::facts::Slot, W = crate::facts::Slot>,
        x: &[crate::facts::Slot], result: crate::facts::Slot, needed: &[bool], output: &mut [Option<crate::facts::Slot>]) {
        if let Some(side) = self.kind_result(&[Some(0), Some(1)], |_| None) {
            for (i, (&needed, output)) in needed.iter().zip(output).enumerate() { *output = needed.then(|| ops.number(u8::from(side == Some(i)) as f64)); }
            return;
        }
        self.definition.partials(ops, x, result, needed, output)
    }
    pub fn range(&self, x: &[crate::domain::Range]) -> crate::domain::Range {
        self.bound(self.kind_result(x, |value| Range::of(Some(u8::from(value) as f64))).unwrap_or_else(|| self.definition.range(x)))
    }
    /// A mathematical result interval as this kernel stores it: rounded to its width and widened for wrapping integers.
    fn bound(&self, range: Range) -> Range {
        if matches!(self.variant.dtype, DType::Complex(_)) { return Range::ANY; }
        let range = range.at_width(self.result);
        let bounds = Range::of_dtype(self.result);
        // f64 endpoints cannot prove i64 wrapping bounds. Narrower integers are exact here until their result exceeds their type.
        if self.variant.dtype.is_integral() && !self.variant.fails_on_overflow() && self.result.is_integral()
            && (self.result == DType::Integer(Width::I64) || !range.within(bounds.lo, bounds.hi))
        { bounds }
        // Checked arithmetic can rerun wider; only a Boolean result is bounded independently of that attempt.
        else if self.result == DType::Boolean { range.meet(bounds) }
        else { range }
    }
    pub fn identity(&self) -> Option<f64> { self.definition.identity() }
    /// The argument that a dyadic call returns unchanged where `constants` gives the arguments that are one number, 0 for the left and 1 for the right, from the formula's identities: `x+0` and `0+x` keep `x`, `x-0` keeps `x`, and `0-x` keeps nothing.
    pub fn kept(&self, constants: &[Option<f64>]) -> Option<usize> { self.definition.kept(constants) }
    /// The reduction by which a fold through this kernel combines its items, as `reduction` gives it at the kernel's variant.
    pub fn fold(&self) -> Option<crate::reduce::Reduce> { self.definition.reduction(self.variant) }
    /// The variant a fold computes in. An unseeded fold of Booleans that counts its trues (`+/`) reads them as Booleans, with no cast to the integer it computes in.
    fn fold_variant(&self, seed: &Seed<'_>) -> crate::domain::Variant {
        let boolean = crate::domain::Variant { dtype: DType::Boolean, ..self.variant };
        let counts = self.boolean_inputs && matches!(seed, Seed::Identity) && matches!(self.definition.reduction(boolean), Some(crate::reduce::Reduce::Sum));
        if counts { boolean } else { self.variant }
    }
    /// The reduction that an unseeded right fold by this formula runs as an alternating one, as `-` runs as an alternating `+`.
    pub fn inverse_fold(&self) -> Option<crate::reduce::Reduce> { self.definition.inverse_fold() }
    /// The reduction that runs an unseeded right fold by this binary formula as an alternating fold (`inverse_fold`).
    fn alternated(&self, seed: Seed<'_>, direction: Direction) -> Option<crate::reduce::Reduce> {
        if self.arity() == 2 && direction == Direction::Right && matches!(seed, Seed::Identity) { self.inverse_fold() } else { None }
    }
    pub fn truth_table(&self) -> Option<u8> { self.definition.truth_table() }
    pub fn total_dtype(&self) -> Option<DType> { self.fold()?.total_dtype(self.variant) }
    pub fn fold_dtype(&self, input: DType, len: usize, seed: Option<DType>) -> DType {
        self.total_dtype().unwrap_or_else(|| match (len, seed) { (0, Some(seed)) => seed, (1, None) if input != DType::Boolean => input, _ => self.result })
    }
    /// The scan's result dtype. A seeded comparison stores only Boolean results; without a seed, each lane retains its original first item. Native reductions can accumulate at a wider type.
    pub fn scan_dtype(&self, seeded: bool) -> DType {
        if seeded && self.result == DType::Boolean { return DType::Boolean; }
        let carry = match self.variant.dtype { DType::Float(FloatWidth::F16) => DType::Float(FloatWidth::F32), dtype => dtype };
        self.total_dtype().unwrap_or_else(|| carry.common(self.result).unwrap_or(self.result))
    }
    /// The dtype of this map kernel's product with `fold` over `plan`: its native product's result where one runs (`product_for`), and otherwise `fold`'s total of this kernel's results along the contracted axes (`fold_dtype`), as a fold of them gives.
    pub fn product_dtype(&self, fold: &Kernel, plan: &crate::linalg::ProductPlan) -> DType {
        self.native_product(self.variant, fold, self.boolean_inputs, None, Settings { tolerance: crate::tolerant::Tolerance::EXACT })
            .map_or_else(|| fold.fold_dtype(self.result, plan.contracted_len(), None), |p| p.result)
    }
    fn native_product(
        &self, variant: crate::domain::Variant, fold: &Kernel, boolean_inputs: bool, wanted: Option<DType>, settings: Settings) -> Option<NativeProduct> {
        if matches!(self.policy, Some(crate::characters::Policy::Kinds(_))) { return None; }
        self.definition.product(variant, fold.fold(), boolean_inputs, wanted, settings)
    }
    /// How this kernel reads its inputs (`with_projected`) in a run with `settings`: characters as codes under a code policy, floats as the whole numbers they equal within `settings.tolerance` where it computes at an integral dtype from float arguments, and otherwise as they are.
    fn projection(&self, settings: Settings) -> Projection {
        match self.policy {
            Some(crate::characters::Policy::Codes { .. }) => Projection::Codes(settings.tolerance),
            None if self.wholes => Projection::Wholes(settings.tolerance), _ => Projection::AsIs
        }
    }
    /// `call` on `inputs` and `seed`'s read as this kernel reads them in a run with `settings` (`projection`): over `range`, or over the whole of each read where `None`. `range` applies to every read, so a call with a seed passes `None`. `call` runs in an error slot ([`crate::error::checked`]): an error that a kernel records, such as an integer overflow that a native body reports, becomes its result. Every kernel call on arrays comes through here.
    fn projected<R>(
        &self, inputs: &[crate::agreement::Read<'_>], range: Option<std::ops::Range<usize>>, seed: Seed<'_>, settings: Settings,
        call: impl FnOnce(&[crate::agreement::Read<'_>], Seed<'_>) -> Result<R, Error>) -> Result<R, Error> {
        crate::error::checked(|| {
            let projection = self.projection(settings);
            if projection == Projection::AsIs { return call(inputs, seed); }
            let mut all = inputs.to_vec();
            all.extend(seed.read());
            with_projected(&all, range, projection, |projected| {
                let seed = match seed {
                    Seed::Initial(_) => Seed::Initial(projected[inputs.len()]), Seed::Tail(_) => Seed::Tail(projected[inputs.len()]),
                    Seed::Identity => Seed::Identity
                };
                call(&projected[..inputs.len()], seed)
            })
        })
    }
    pub fn map(&self, inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, settings: Settings, output: &mut Buffer) -> Result<(), Error> {
        output.clear();
        self.mapping(Inputs(inputs), range, settings, MapOutput::Buffer(output), None)
    }

    /// Appends a mapped range using the caller's execution-local conversion buffers.
    pub fn map_in<'a, 'b>(
        &self, inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, settings: Settings, scratch: &'a mut crate::buffer::Scratch,
        output: MapOutput<'a, 'b>) -> Result<(), Error> { self.mapping(Inputs(inputs), range, settings, output, Some(scratch)) }

    fn mapping<'a, 'b>(
        &self, inputs: Inputs<'_, '_>, range: std::ops::Range<usize>, settings: Settings, mut target: MapOutput<'a, 'b>,
        scratch: Option<&'a mut crate::buffer::Scratch>) -> Result<(), Error> {
        use crate::characters::{ByKinds, Policy};
        if inputs.len() != self.arity() { target.fill(range.len()); return Err(Error::new(Kind::Length, "kernel operand count changed")); }
        if range.is_empty() { return Ok(()); }
        let mut output = Output { target, result_dtype: self.result, checked: self.variant.fails_on_overflow(), scratch };
        match self.policy {
            Some(Policy::Kinds(side)) => crate::error::checked(|| match side {
                ByKinds::Truth(value) => map::constant(value, range.len(), &mut output),
                side => map::copied(inputs.selected(usize::from(side == ByKinds::Right)), range, self.variant.checked, &mut output),
            }),
            _ => {
                let mut entered = false;
                let result = self.projected(inputs.0, Some(range.clone()), Seed::Identity, settings, |reads, _| {
                    entered = true;
                    self.definition.map(self.variant, Inputs(reads), range.clone(), settings, self.repair, self.may_raise(), &mut output)
                });
                if !entered { output.target.fill(range.len()); }
                result
            }
        }
    }
    pub fn fold_pieces(
        &self, dims: [usize; 3], seed: Seed<'_>, direction: Direction, settings: Settings, pieces: &Pieces<'_>, output: &mut Buffer) -> Result<(), Error> {
        let dtype = output.dtype();
        let variant = self.fold_variant(&seed);
        let projection = self.projection(settings);
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
                Piece::Total(total) => f(range, Piece::Total(total)),
            })
        };
        let pieces: &Pieces<'_> = if projection == Projection::AsIs { pieces } else { &projected };
        self.projected(&[], None, seed, settings, |_, seed| {
            if let Some(fold) = self.definition.reduction(variant) { return fold.run_pieces(variant, dims, seed, pieces, output); }
            if let Some(fold) = self.alternated(seed, direction) && let Some(buffer) = fold.inverse_pieces(self.variant, dims, pieces)? {
                *output = buffer;
                return Ok(());
            }
            self.definition.binary(self.variant, settings, Route::Fold { dims, seed, direction, pieces }, output)
        })?;
        finish(output, dtype, self.variant)
    }
    pub fn scan(
        &self, dims: [usize; 3], seed: Seed<'_>, direction: Direction, settings: Settings, input: crate::agreement::Read<'_>, output: &mut Buffer
    ) -> Result<(), Error> {
        let dtype = output.dtype();
        let seeded = seed.read().is_some();
        self.projected(&[input], None, seed, settings, |inputs, seed| {
            if let Some(fold) = self.fold() { return fold.run_scan(self.variant, dims, seed, direction, inputs[0], output); }
            if direction == Direction::Left && self.inverse_fold() == Some(crate::reduce::Reduce::Sum)
                && crate::reduce::negated_scan(self.variant, dims, seed, inputs[0], output)? { return Ok(()); }
            self.definition.binary(self.variant, settings, Route::Scan { dims, seed, direction, input: inputs[0] }, output)
        })?;
        // A seed makes every output a formula result; a Boolean formula already guarantees 0 or 1.
        let checked = self.variant.checked && !(seeded && dtype == DType::Boolean && self.result == DType::Boolean);
        finish(output, dtype, crate::domain::Variant { checked, ..self.variant })
    }
    /// Each item of `input`, an array of shape `dims`, combined with the item a row before it along the middle axis, and each lane's first row with `seed`, or kept as it is without one, as an inverse scan runs: `flip` swaps the arguments, so the earlier item comes second. The results take `output`'s dtype, so a comparison's 0s and 1s and an unseeded first row keep the items' type where the caller asks for it.
    pub fn adjacent(
        &self, dims: [usize; 3], seed: Option<crate::agreement::Read<'_>>, flip: bool, settings: Settings, input: crate::agreement::Read<'_>,
        output: &mut Buffer) -> Result<(), Error> {
        let dtype = output.dtype();
        self.projected(&[input], None, seed.map_or(Seed::Identity, Seed::Initial), settings, |inputs, seed| {
            let seed = seed.read();
            self.definition.binary(self.variant, settings, Route::Adjacent { dims, seed, flip, input: inputs[0] }, output)
        })?;
        finish(output, dtype, self.variant)
    }
    /// Folds along `axis` of `input`, an array of `shape`, or all its items for `None`, into `output`. Whole-array commutative folds may read a contiguous block in storage order; ordered folds retain logical order. A returned order reads `output` in the result's order, for folds that stay in their source's order (`Buffer::ordered`).
    pub fn fold_read(
        &self, shape: &[usize], axis: Option<usize>, seed: Seed<'_>, direction: Direction, settings: Settings, input: crate::agreement::Read<'_>,
        output: &mut Buffer) -> Result<Option<Vec<crate::gather::Steps>>, Error> {
        let dtype = output.dtype();
        let variant = self.fold_variant(&seed);
        let order = self.projected(&[input], None, seed, settings, |inputs, seed| {
            let input = inputs[0];
            if let Some(fold) = self.definition.reduction(variant) { return fold.run_read(variant, shape, axis, seed, input, output); }
            let flat = [crate::layout::count(shape)];
            let (shape, axis) = axis.map_or((flat.as_slice(), 0), |axis| (shape, axis));
            if let Some(fold) = self.alternated(seed, direction) && let Some(buffer) = fold.inverse_read(variant, shape, axis, input)? {
                *output = buffer;
                return Ok(None);
            }
            let pieces = |range: std::ops::Range<usize>, emit: &mut dyn FnMut(usize, Piece<'_>)| {
                let (start, mut scratch) = (range.start, Buffer::with_capacity(variant.dtype, 0));
                emit(start, Piece::Items(input.part_view(range, variant.dtype, variant.checked, &mut scratch)?));
                Ok(())
            };
            self.definition
                .binary(variant, settings, Route::Fold { dims: crate::layout::Lanes::new(shape, axis)?.dims(), seed, direction, pieces: &pieces }, output)
                .map(|()| None)
        })?;
        finish(output, dtype, self.variant).map(|()| order)
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
            let mut storage: Vec<Vec<T>> = choices.iter().map(|_| Vec::new()).collect();
            let args = choices.iter().zip(&mut storage).map(|(read, scratch)| read.part::<T>(part(*read), checked, scratch)).collect::<Result<Vec<_>, _>>();
            // A Boolean mask between two whole choices picks each item in one branch-free loop, with no positions.
            if selector.1 == DType::Boolean && let Ok([Arg::Many(no), Arg::Many(yes)]) = args.as_deref() && no.len() == range.len() && yes.len() == range.len()
            {
                let mut bits = Vec::new();
                if let Ok(Arg::Many(mask)) = selector.0.part::<bool>(part(selector.0), checked, &mut bits) && mask.len() == range.len() {
                    return unsafe {
                        slots.fill(range.len(), |out| {
                            for (((slot, &m), &n), &y) in out.iter_mut().zip(mask.iter()).zip(no.iter()).zip(yes.iter()) {
                                slot.write(core::hint::select_unpredictable(m, y, n));
                            }
                            true
                        })
                    };
                }
            }
            let mut indices = Buffer::with_capacity(selector.1, 0);
            let positions = selector.0.part_view(part(selector.0), selector.1, false, &mut indices)
                .and_then(|view| crate::structure::positions(view, choices.len(), crate::tolerant::Tolerance::EXACT));
            match positions.and_then(|positions| args.map(|args| (positions, args))) {
                Ok((positions, args)) => unsafe {
                    slots.fill(range.len(), |out| {
                        for (i, slot) in out.iter_mut().enumerate() {
                            let position = positions[if positions.len() == 1 { 0 } else { i }];
                            let value = match args.get(position) {
                                Some(Arg::One(value)) => *value, Some(Arg::Many(values)) => values[if values.len() == 1 { 0 } else { i }], None => T::FILL
                            };
                            slot.write(value);
                        }
                        true
                    })
                },
                Err(error) => {
                    crate::error::record(error);
                    unsafe { slots.fill(range.len(), |out| { for slot in out { slot.write(T::FILL); } false }) }
                }
            }
        });
        Ok(())
    })
}

/// The magnitude below which every whole float converts to an `i64`.
const I64_LIMIT: f64 = 9_223_372_036_854_775_808.0;

/// `values` as the whole numbers they equal within tolerance `t` (`tolerant::wholes`), as an integer computation reads float arguments. BPL passes the comparison tolerance, which only Mathish makes nonzero. A fraction, or a magnitude of 2⁶³ or more, is DOMAIN.
pub fn whole_numbers<F: crate::Float>(t: crate::tolerant::Tolerance, values: &[F]) -> Result<Vec<i64>, Error> {
    crate::tolerant::wholes(t, values, I64_LIMIT).map(|(numbers, _)| numbers)
        .ok_or_else(|| Error::new(Kind::Domain, "an integer computation's float argument must be a whole number"))
}

/// `f` on `view` as an integer computation reads it (`with_projected`): characters as their codes under `Projection::Codes`, and floats as the whole numbers they equal within the projection's tolerance (`whole_numbers`).
fn projected_view(view: View<'_>, projection: Projection) -> Result<crate::buffer::Items<'_>, Error> {
    use crate::buffer::{IntBuf, Items};
    let view = if matches!(projection, Projection::Codes(_)) { crate::characters::as_codes(view) } else { view };
    let Some(t) = projection.wholes().filter(|_| matches!(view.dtype(), DType::Float(_))) else { return Ok(Items::Borrowed(view)) };
    let mut floats = Vec::new();
    <f64 as crate::buffer::Cast>::extend_cast(view, view.len(), true, &mut floats)?;
    Ok(Items::Owned(Buffer::Integers(IntBuf::I64(whole_numbers(t, &floats)?))))
}

fn with_projected_view<R>(view: View<'_>, projection: Projection, f: impl FnOnce(View<'_>) -> R) -> Result<R, Error> {
    Ok(f(projected_view(view, projection)?.view()))
}

/// `call` on `inputs` as an integer computation reads them: characters as their codes under `Projection::Codes`, and floats as the whole numbers they equal within the projection's tolerance (`whole_numbers`), as `i64`, over `range`, or over the whole of each input where `None`.
fn with_projected<R>(
    inputs: &[crate::agreement::Read<'_>], range: Option<std::ops::Range<usize>>, projection: Projection,
    call: impl FnOnce(&[crate::agreement::Read<'_>]) -> Result<R, Error>) -> Result<R, Error> {
    use crate::{agreement::{Binding, Read, ReadPlan}, buffer::{Floats, IntBuf}};
    let Some(t) = projection.wholes() else { return call(inputs) };
    let codes = matches!(projection, Projection::Codes(_));
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
    let mut projected: Vec<Read<'_>> =
        if codes { sources.iter().zip(&code_plans).map(|(sources, plan)| Read { sources, plan }).collect() } else { inputs.to_vec() };
    if projected.iter().all(|r| r.sources.iter().all(|s| !matches!(s.view.dtype(), DType::Float(_)))) { return call(&projected); }
    let converted = projected.iter().map(|read| {
        if !read.sources.iter().any(|s| matches!(s.view.dtype(), DType::Float(_))) { return Ok(None); }
        let range = range.clone().unwrap_or(0..read.plan.len());
        let mut floats = Buffer::with_capacity(DType::Float(FloatWidth::F64), 0);
        let View::Floats(Floats::F64(values)) = read.part_view(range.clone(), floats.dtype(), true, &mut floats)? else {
            unreachable!("a read into 64-bit floats gives them")
        };
        Ok(Some((range.start, Buffer::Integers(IntBuf::I64(whole_numbers(t, values)?)))))
    }).collect::<Result<Vec<_>, Error>>()?;
    let bindings = converted.iter().zip(inputs)
        .map(|(buffer, read)| buffer.as_ref().map(|(start, b)| Binding::window(b.view(), *start, read.plan.len())).transpose()).collect::<Result<Vec<_>, _>>()?;
    let plans: Vec<_> = inputs.iter().map(|r| ReadPlan::input(0, r.plan.len())).collect();
    for (i, binding) in bindings.iter().enumerate() {
        if let Some(binding) = binding { projected[i] = Read { sources: std::slice::from_ref(binding), plan: &plans[i] }; }
    }
    call(&projected)
}

macro_rules! with_binary_body {
    ($object:expr, $variant:expr, $settings:expr, |$T:ident, $apply:ident, $compare:ident| $body:expr) => {{
        let (object, variant, settings) = ($object, $variant, $settings);
        let ops = settings.ops();
        match variant.dtype {
            DType::Boolean | DType::Integer(_) => {
                let width = match variant.dtype { DType::Integer(w) => w, _ => Width::U8 };
                with_width!(width, $T => {
                    let ops = crate::recipe::IntegerOps::<$T>::new(variant);
                    let $apply = { #[inline(always)] |x: $T, y: $T| integer_value(object, variant, &ops, std::array::from_fn(|i| if i == 0 { x } else { y })) };
                    let $compare = |_: $T, _: &[$T], _: bool, _: &mut Vec<bool>| None;
                    $body
                })
            },
            DType::Float(FloatWidth::F16 | FloatWidth::F32) => {
                type $T = f32;
                let $apply = { #[inline(always)] move |x: $T, y: $T| real_value(object, variant, &ops, std::array::from_fn(|i| if i == 0 { x } else { y }), true) };
                let $compare = comparison_writer(object, &ops);
                $body
            },
            DType::Float(FloatWidth::F64) => {
                type $T = f64;
                let $apply = { #[inline(always)] move |x: $T, y: $T| real_value(object, variant, &ops, std::array::from_fn(|i| if i == 0 { x } else { y }), true) };
                let $compare = comparison_writer(object, &ops);
                $body
            },
            DType::Complex(w) => with_complex_width!(w, $T => {
                let $apply = { #[inline(always)] move |x: $T, y: $T| complex_value(object, variant, &ops, std::array::from_fn(|i| if i == 0 { x } else { y })) };
                let $compare = |_: $T, _: &[$T], _: bool, _: &mut Vec<bool>| None;
                $body
            }),
            DType::Character => Err(Error::new(Kind::Domain, "a numerical body needs code-point inputs for characters")),
        }
    }};
}

/// Compares a block against a constant carry through the same tolerance bands as an elementwise map.
fn comparison_writer<K: Formula<N>, F: Native + Banded, const N: usize>(
    object: &K, ops: &NativeOps<{ FUSED }>) -> impl Fn(F, &[F], bool, &mut Vec<bool>) -> Option<bool> + Copy {
    let bands = object.bands().and_then(|(t, left, right)| Some((crate::facts::run_tolerance::<F, _>(ops, t)?, left, right)));
    move |carry, items, flipped, out| {
        let (t, left, right) = bands?;
        let (x, y) = (Arg::One(carry), Arg::Many(items));
        let (x, y) = if flipped { (y, x) } else { (x, y) };
        let (at, n) = (out.len(), items.len());
        out.reserve(n);
        let ok = crate::tolerant::against(t, x, y, n, &mut out.spare_capacity_mut()[..n], left, right, F::banded)?;
        // SAFETY: `against` writes all `n` slots when it returns `Some`.
        unsafe { out.set_len(at + n) };
        Some(ok)
    }
}

/// The results of `step` for each pair of items of `xs` and `ys`, written to `out`, in one loop that carries each result's validity as a flag with no branch. A block with a failure finds the first again and gives its kind.
fn paired<T: Copy>(step: impl Fn(T, T) -> (T, Option<Kind>)) -> impl Fn(&[T], &[T], &mut [std::mem::MaybeUninit<T>]) -> Result<(), Kind> {
    move |xs: &[T], ys: &[T], out: &mut [std::mem::MaybeUninit<T>]| {
        let mut ok = true;
        for ((&x, &y), slot) in xs.iter().zip(ys).zip(out) {
            let (value, failed) = step(x, y);
            slot.write(value);
            ok &= failed.is_none();
        }
        if ok { Ok(()) } else { xs.iter().zip(ys).find_map(|(&x, &y)| step(x, y).1).map_or(Ok(()), Err) }
    }
}

/// The running results of `step` along each lane of `len` items in `data`, appended to `out` (`reduce::lanes_scan`): each lane's first item stepped from `carry`, or kept as it is without one, and each later item stepped from the result before it. Gives the kind of the first step that failed, which a second pass finds only where one did.
fn chained<T: Copy>(step: impl Fn(T, T) -> (T, Option<Kind>) + Copy) -> impl Fn(Option<T>, &[T], usize, &mut Vec<T>) -> Option<Kind> {
    move |carry: Option<T>, data: &[T], len: usize, out: &mut Vec<T>| {
        let fits = move |x, y| { let (t, failed) = step(x, y); (t, failed.is_none()) };
        if crate::reduce::lanes_scan(data, len, out, move |x| carry.map_or((x, true), |c| fits(c, x)), fits) { return None; }
        data.chunks(len).find_map(|lane| {
            let (mut total, mut failure) = carry.map_or((lane[0], None), |c| step(c, lane[0]));
            for &x in &lane[1..] {
                if failure.is_some() { break; }
                (total, failure) = step(total, x);
            }
            failure
        })
    }
}

/// The fold of `items` from the right onto `last`: `last` stepped by each item in turn, from the last item to the first, as `step(total, item)`. Gives the kind of the first step that failed.
fn folded<T: Copy>(step: impl Fn(T, T) -> (T, Option<Kind>)) -> impl Fn(&[T], T) -> (T, Option<Kind>) {
    move |items: &[T], last: T| {
        items.iter().rev().fold((last, None), |(total, failure), &x| {
            let (t, f) = step(total, x);
            (t, failure.or(f))
        })
    }
}

/// Report whether a kernel reads its operands through a non-contiguous plan.
pub fn report_maps<'a>(plans: impl IntoIterator<Item = &'a crate::agreement::ReadPlan>, range: std::ops::Range<usize>) {
    if crate::events::running() && plans.into_iter().any(|plan| plan.contiguous(range.clone()).is_none()) {
        crate::events::taken(crate::events::Route::MappedKernel, range.len());
    }
}

/// The error that `object` finds in `result` from arguments `x` under a checked `variant`: the verdict on one item of `real_value` and `integer_value`, and of `map_values`' second loop.
#[inline(always)]
fn verdict<K: crate::facts::Formula<N>, F: crate::Float, O: crate::recipe::Ops<F, B = bool> + ?Sized, const N: usize>(
    object: &K, variant: crate::domain::Variant, ops: &O, x: [O::V; N], result: O::V) -> Option<Kind> {
    object.emit_valid(variant, ops, x, result).and_then(|(kind, valid)| (!valid).then_some(kind))
}

/// Whether a complex `result` from arguments `x` is valid: a formula that `FAILS_NAN` fails on a NaN part from arguments with no NaN part.
#[inline(always)]
fn complex_valid<K: crate::facts::Formula<N>, F: crate::Native, const N: usize>(
    ops: &crate::recipe::NativeOps<{ crate::FUSED }>, x: [num_complex::Complex<F>; N], result: num_complex::Complex<F>) -> bool {
    !K::FAILS_NAN || crate::facts::nan_valid(ops, x.into_iter().flat_map(|x| [x.re, x.im]), [result.re, result.im])
}

#[inline(always)]
fn real_value<K, F, const N: usize>(
    object: &K, variant: crate::domain::Variant, ops: &crate::recipe::NativeOps<{ crate::FUSED }>, x: [F; N], repair: bool) -> (F, Option<Kind>) where
    K: crate::facts::Formula<N>, F: crate::Native {
    let result = if repair && object.emit_repair(ops, x) == Some(true) { crate::facts::Formula::repair_routine(object).map(|r| r.apply(x[0])) }
    else { object.real(ops, x) };
    let Some(result) = result else { return (F::zero(), variant.checked.then_some(Kind::Domain)); };
    let error = if variant.checked { verdict(object, variant, ops, x, result) } else { None };
    (result, error)
}

#[inline(always)]
fn integer_value<K, I, const N: usize>(object: &K, variant: crate::domain::Variant, ops: &crate::recipe::IntegerOps<I>, x: [I; N]) -> (I, Option<Kind>) where
    K: crate::facts::Formula<N>, I: crate::recipe::Overflowing {
    let Some(result) = object.integer(ops, x) else { return (I::FILL, variant.checked.then_some(Kind::Domain)); };
    let error = if variant.checked { verdict(object, variant, ops, x, result) } else { None };
    (result, error)
}

/// Whether `object`'s integer body at the Boolean `variant` raises for some Boolean arguments: it fails, or a formula that doesn't give truths gives a number other than 0 or 1, which a Boolean result can't hold. It tries all 2^N arguments.
fn boolean_raises<K: crate::facts::Formula<N>, const N: usize>(object: &K, variant: crate::domain::Variant) -> bool {
    let ops = crate::recipe::IntegerOps::<u8>::new(variant);
    (0..1usize << N).any(|bits| {
        let (value, error) = integer_value(object, variant, &ops, std::array::from_fn(|i| ((bits >> i) & 1) as u8));
        error.is_some() || !K::BOOLEAN && value > 1
    })
}

#[inline(always)]
fn complex_value<K, F, const N: usize>(
    object: &K, variant: crate::domain::Variant, ops: &crate::recipe::NativeOps<{ crate::FUSED }>, x: [num_complex::Complex<F>; N]
) -> (num_complex::Complex<F>, Option<Kind>) where K: crate::facts::Formula<N>, F: crate::Native {
    let Some(result) = object.complex(ops, x) else { return (num_complex::Complex::new(F::zero(), F::zero()), variant.checked.then_some(Kind::Domain)); };
    let valid = !variant.checked || complex_valid::<K, F, N>(ops, x, result);
    (result, (!valid).then_some(Kind::Domain))
}

fn scalar_result<T>((value, error): (T, Option<Kind>)) -> Result<T, Error> {
    match error { Some(kind) => Err(Error::new(kind, "numerical argument is outside the function's domain")), None => Ok(value) }
}

/// The reduction by which a fold by `object` at `variant` combines its items: for Booleans the formula's `truth`, which counts for `+` and otherwise names the reduction of its truth table, before its `fold` for numbers.
fn reduction<K: crate::facts::Formula<N>, const N: usize>(object: &K, variant: crate::domain::Variant) -> Option<crate::reduce::Reduce> {
    if variant.dtype == DType::Boolean { object.truth().or_else(|| object.fold()) } else { object.fold() }
}

/// `output`, a kernel's results at `variant`, as `dtype`, which a caller can ask for narrower than the kernel's result type. An integer that doesn't fit fails where `variant.fails_on_overflow()`, and wraps otherwise.
fn finish(output: &mut Buffer, dtype: DType, variant: crate::domain::Variant) -> Result<(), Error> {
    if output.dtype() != dtype { *output = Buffer::cast_results(output.view(), dtype, variant.fails_on_overflow())?; }
    Ok(())
}

mod at;
mod binary;
mod fold;
mod map;
mod product;
pub use {at::*, product::*};
use {binary::*, fold::*, map::{Inputs, map_formula}};
