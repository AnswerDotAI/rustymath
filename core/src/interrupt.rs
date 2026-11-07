//! The process's interrupt check. Long computations call [`check`] between steps, about every [`BETWEEN_CHECKS`](crate::BETWEEN_CHECKS) items, and stop with its error. A frontend installs its check with [`install`]: basedpl's stops the running evaluation at an interrupt or a passed deadline, and rustymath's Python bindings' raise the exception of a pending signal. Until a frontend installs one, `check` passes.
use crate::error::Error;
use std::sync::OnceLock;

static HOOK: OnceLock<fn() -> Result<(), Error>> = OnceLock::new();

/// Makes `hook` the process's interrupt check. The first check installed stays, and a later call does nothing.
pub fn install(hook: fn() -> Result<(), Error>) { HOOK.get_or_init(|| hook); }

/// The installed check's result: an error of kind Interrupt or Timeout stops the computation that called it. `Ok` until a frontend installs a check.
#[inline]
pub fn check() -> Result<(), Error> { HOOK.get().map_or(Ok(()), |hook| hook()) }
