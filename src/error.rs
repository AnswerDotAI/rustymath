//! Errors from rustymath's functions on arrays. Each error has a kind, and a message for the reader that can be empty.
use std::borrow::Cow;

/// What went wrong.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Lengths that don't agree.
    Length,
    /// An argument of the wrong rank.
    Rank,
    /// A position outside an axis.
    Index,
    /// A result past an array limit, such as `layout::MAX_GENERATED_ELEMENTS`.
    Limit,
    /// An argument outside the function's domain.
    Domain,
    /// An integer result that doesn't fit its dtype.
    Overflow,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub kind: Kind,
    pub message: Cow<'static, str>,
}

impl Error {
    pub fn new(kind: Kind, message: impl Into<Cow<'static, str>>) -> Self { Self { kind, message: message.into() } }
}

/// An error of `kind` with no message.
impl From<Kind> for Error {
    fn from(kind: Kind) -> Self { Self::new(kind, "") }
}

/// The message, or the kind's name when the message is empty.
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.message.is_empty() { write!(f, "{:?}", self.kind) } else { f.write_str(&self.message) }
    }
}

impl std::error::Error for Error {}
