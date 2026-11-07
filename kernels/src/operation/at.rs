use super::*;

/// Writes to the addresses of `ids`, replacing or combining with the current items. Invalid positions, including fills, write nothing; repeated positions combine with the earlier result.
pub fn at(
    combine: Option<&Kernel>, root: WriteRoot<'_>, values: crate::agreement::Read<'_>, ids: Writes<'_>, checked: bool, settings: Settings, output: &mut Buffer
) -> Result<(), Error> {
    let (wanted, n) = (output.dtype(), ids.len());
    if values.plan.len() != 1 && values.plan.len() != n { return Err(Error::new(Kind::Length, "write values do not match positions")); }
    let projection = combine.map_or(Projection::AsIs, |k| k.projection(settings));
    match combine {
        None => {
            root.initialize(wanted, checked, Projection::AsIs, output)?;
            with_dtype!(wanted, T => at_typed::<T>(values, ids, checked, None, output))?;
        }
        Some(kernel) => kernel
            .projected(&[values], None, Seed::Identity, settings, |inputs, _| kernel.write(root, inputs[0], ids, checked, projection, settings, output))?,
    }
    finish(output, wanted, crate::domain::Variant { checked, ..crate::domain::Variant::checked(wanted) })
}

/// Writes `values` into `output` at `ids`, a chunk at a time: over the items there, or merged with them by `combine`, which gives any failure it finds. `combine` is one `dyn` call for each chunk, so this compiles once for each item type.
pub(crate) fn at_typed<T: crate::buffer::Cast>(
    values: crate::agreement::Read<'_>, ids: Writes<'_>, checked: bool,
    combine: Option<&dyn Fn(&mut [T], crate::gather::Writes<'_>, Arg<'_, T>) -> Option<Kind>>, output: &mut Buffer) -> Result<(), Error> {
    let out = T::vec_mut(output).expect("write output has its working dtype");
    let mut scratch = Vec::new();
    let mut result = Ok(());
    ids.runs_while(&mut |start, positions| {
        result = (|| {
            for (k, positions) in positions.chunks(crate::CHUNK).enumerate() {
                let start = start + k * crate::CHUNK;
                let source = values.part::<T>(if values.plan.len() == 1 { 0..1 } else { start..start + positions.len() }, checked, &mut scratch)?;
                let writes = Writes::Offsets(positions);
                match combine {
                    None => match source {
                        Arg::One(value) => writes.write(out, &[value], crate::gather::Same), Arg::Many(values) => writes.write(out, values, crate::gather::Same)
                    },
                    Some(combine) => {
                        if let Some(kind) = combine(&mut out[..], writes, source) {
                            return Err(Error::new(kind, "modified write is outside the function's domain"));
                        }
                    }
                }
            }
            Ok(())
        })();
        result.is_ok()
    });
    result
}

impl Kernel {
    /// `values` written into `root`'s items at `ids`, combined by this kernel: by its fold's reduction where that runs at the variant's type or the type the formula computes in (`with_binary_body!`), and otherwise item by item (`Route::At`).
    fn write(
        &self, root: WriteRoot<'_>, values: crate::agreement::Read<'_>, ids: Writes<'_>, checked: bool, projection: Projection, settings: Settings,
        output: &mut Buffer) -> Result<(), Error> {
        let variant = self.variant;
        if let Some(fold) = self.fold() {
            let working = match variant.dtype {
                DType::Boolean => DType::Integer(Width::U8), DType::Float(FloatWidth::F16) => DType::Float(FloatWidth::F32), dtype => dtype
            };
            native_reduction!(fold, variant.dtype, variant.fails_on_overflow(), |R, T| {
                type S = <R as crate::reduce::Reduction<T>>::Total;
                if S::DTYPE == variant.dtype || S::DTYPE == working {
                    root.initialize(S::DTYPE, checked, projection, output)?;
                    return at_typed::<S>(
                        values, ids, checked, Some(&|out, writes, items| {
                            writes.combine(out, items, |x, y| {
                                let (value, ok) = <R as crate::reduce::Reduction<T>>::chain(x, y);
                                (value, (!ok && variant.fails_on_overflow()).then_some(Kind::Overflow))
                            })
                        }), output,
                    );
                }
                None::<()>
            });
        }
        self.definition.binary(variant, settings, Route::At { root, values, ids, checked, projection }, output)
    }
}
