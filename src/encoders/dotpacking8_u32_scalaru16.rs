use crate::encoders::dotpacking8::encoder::{DotPacking8Encoder, DotPacking8QueryEvaluator};
use crate::encoders::dotpacking8::values::ScalarU16Values;

/// Bridged DotPacking8 with `u32` components and 16-bit scalar codes. Same bridge scheme and
/// code-0 invariant as [`crate::BridgedDotPacking8U32ScalarU8Encoder`]: code 0 marks a bridge,
/// so a real value must encode to a non-zero code (at least half its component's step,
/// `max / 65535`, since codes are rounded to nearest).
pub type BridgedDotPacking8U32ScalarU16Encoder = DotPacking8Encoder<u32, ScalarU16Values>;
pub type BridgedDotPacking8U32ScalarU16QueryEvaluator<'e> =
    DotPacking8QueryEvaluator<'e, u32, ScalarU16Values>;
