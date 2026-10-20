//! Byte layout of one encoded vector (see the safety invariants in `common.rs`):
//! `[n: u16][selectors][gap payload][padding][values]`, padded to whole `u64` words.

use bytemuck::{cast_slice, from_bytes};

use crate::encoders::dotpacking8::common::{compute_safe_simd_padding, encode_blocks};
use crate::encoders::dotpacking8::values::ValueCodec;

const N: usize = 8;

pub(crate) struct EncodedView<'a> {
    pub n: usize,
    pub selectors: &'a [u8],
    pub payload: &'a [u8],
    pub values: &'a [u8],
}

impl<'a> EncodedView<'a> {
    /// # Safety
    ///
    /// `slice` must be exactly one vector written by [`write_encoded`] with the same codec `V`.
    #[inline(always)]
    pub(crate) unsafe fn new<V: ValueCodec>(slice: &'a [u64]) -> Self {
        let bytes = cast_slice::<u64, u8>(slice);
        if bytes.is_empty() {
            return Self {
                n: 0,
                selectors: &[],
                payload: &[],
                values: &[],
            };
        }
        let n = from_bytes::<u16>(&bytes[0..2]).to_le() as usize;
        let selectors_end = 2 + n.div_ceil(N).div_ceil(2);
        let values_start = bytes.len() - V::stream_bytes(n);
        debug_assert!(selectors_end <= values_start);
        Self {
            n,
            selectors: &bytes[2..selectors_end],
            payload: &bytes[selectors_end..values_start],
            values: &bytes[values_start..],
        }
    }
}

/// Writes one vector: `components` are absolute, mapped, sorted, bridged if needed, and every
/// gap fits in 16 bits; `codes` are aligned with them. Writes nothing for an empty vector.
pub(crate) fn write_encoded<V: ValueCodec>(
    components: &[u32],
    codes: &[V::Code],
    output: &mut impl Extend<u64>,
) {
    let n = components.len();
    if n == 0 {
        return;
    }
    assert!(
        n <= u16::MAX as usize,
        "DotPacking8 vectors hold at most 65535 entries (bridges included), got {n}"
    );
    let mut gaps = Vec::with_capacity(n);
    let mut last = 0u32;
    for &c in components {
        let g = c - last;
        assert!(
            g <= u16::MAX as u32,
            "gap {g} exceeds u16::MAX; bridge insertion failed"
        );
        gaps.push(g);
        last = c;
    }
    let (selectors, gap_payload) = encode_blocks(&gaps);
    let value_bytes = V::stream_bytes(n);
    let mut payload =
        Vec::with_capacity(2 + selectors.len() + gap_payload.len() + 16 + value_bytes);
    payload.extend_from_slice(&(n as u16).to_le_bytes());
    payload.extend_from_slice(&selectors);
    payload.extend_from_slice(&gap_payload);
    let alignment = (8 - (payload.len() + value_bytes) % 8) % 8;
    let padding = compute_safe_simd_padding(n, &selectors, alignment, value_bytes);
    payload.resize(payload.len() + padding, 0);
    let before = payload.len();
    V::write_stream(&mut payload, codes);
    // Cold-path checks of the layout the unchecked decoder relies on (invariants (1)-(2)).
    assert_eq!(
        payload.len() - before,
        value_bytes,
        "value stream wrote {} bytes, stream_bytes says {value_bytes}",
        payload.len() - before
    );
    assert!(
        payload.len().is_multiple_of(8),
        "encoded vector is not a whole number of u64 words"
    );
    output.extend(
        payload
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&chunk| u64::from_le_bytes(chunk)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SpaceUsage;
    use crate::encoders::dotpacking8::values::{FixedU8Values, U8Reader};

    /// A broken codec whose stream is shorter than it claims: the decoder's unchecked reads
    /// rely on `stream_bytes`, so `write_encoded` must refuse it.
    #[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
    struct ShortStream;
    impl crate::core::sealed::Sealed for ShortStream {}
    impl SpaceUsage for ShortStream {
        fn space_usage_bytes(&self) -> usize {
            0
        }
    }
    impl ValueCodec for ShortStream {
        type Code = u8;
        type Output = u8;
        type Reader<'a> = U8Reader<'a>;
        fn new_untrained(_dim: usize) -> Self {
            ShortStream
        }
        fn encode(&self, _c: usize, _v: f32) -> u8 {
            1
        }
        fn decode(&self, _c: usize, code: u8) -> f32 {
            code as f32
        }
        fn query_value(&self, _c: usize, v: f32) -> f32 {
            v
        }
        fn zero_code() -> u8 {
            0
        }
        fn stream_bytes(n: usize) -> usize {
            n
        }
        fn write_stream(_out: &mut Vec<u8>, _codes: &[u8]) {}
        fn reader(stream: &[u8]) -> U8Reader<'_> {
            FixedU8Values::reader(stream)
        }
    }

    #[test]
    #[should_panic(expected = "value stream")]
    fn write_encoded_rejects_short_value_stream() {
        let mut out: Vec<u64> = Vec::new();
        write_encoded::<ShortStream>(&[1, 2, 3], &[1, 1, 1], &mut out);
    }
}
