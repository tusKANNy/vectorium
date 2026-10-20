use crate::encoders::dotpacking8::encoder::{DotPacking8Encoder, DotPacking8QueryEvaluator};
use crate::encoders::dotpacking8::values::ScalarU16Values;

/// DotPacking8 with `u16` components and 16-bit scalar codes (one step per component,
/// `max / 65535`). Twice the value bytes of ScalarU8, near-exact scores.
pub type DotPacking8U16ScalarU16Encoder = DotPacking8Encoder<u16, ScalarU16Values>;
pub type DotPacking8U16ScalarU16QueryEvaluator<'e> =
    DotPacking8QueryEvaluator<'e, u16, ScalarU16Values>;
