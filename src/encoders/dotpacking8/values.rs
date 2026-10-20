//! How DotPacking8 stores, reads and quantizes values.

use half::f16;
use num_traits::ToPrimitive;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::simd::prelude::*;

use crate::encoders::dotpacking8::common::f16_ptr_to_f32x8;
use crate::encoders::dotpacking8::nibble4bit::{NibbleReader, nibble_bytes, pack_nibbles};
use crate::utils::sparse_scalar_quantizer_steps;
use crate::{ComponentType, FixedU8Q, FromF32, SpaceUsage, ValueType};

const N: usize = 8;

/// Sequential reader over one vector's value stream: `bulk_blocks` full blocks, then a tail.
pub trait ValueReader {
    type Code: Copy;
    /// Next full block, each code converted to the `f32` the kernel multiplies by the gathered
    /// (already query-scaled) query values.
    fn next_block(&mut self) -> Simd<f32, N>;
    /// Next full block as raw codes (decode path).
    fn next_block_codes(&mut self) -> [Self::Code; N];
    /// The `rem < 8` tail codes; lanes `>= rem` are unspecified.
    fn tail(&mut self, rem: usize) -> [Self::Code; N];
    /// The kernel multiplier of one code; must equal the lane value of `next_block`.
    fn code_as_f32(code: Self::Code) -> f32;
    /// Kernel multiplier of the `i`-th tail value (`i < n % 8`), read in place without
    /// consuming it: the hot path avoids copying the tail into an array.
    fn tail_value(&self, i: usize) -> f32;

    /// Adds the tail's contribution `query[components[i]] * value_i` (`i < rem`) to `acc`, in
    /// order. Readers override it with their fastest form (measured per codec): in-place reads
    /// for byte streams, decode-then-multiply for nibbles.
    #[inline(always)]
    fn tail_dot(&mut self, mut acc: f32, query: &[f32], components: &[u32; N], rem: usize) -> f32 {
        for i in 0..rem {
            acc += query[components[i] as usize] * self.tail_value(i);
        }
        acc
    }
}

/// One byte per value. A raw cursor (like the original kernel's value pointer): reads are
/// unchecked and rely on invariant (4) in `common.rs`.
pub struct U8Reader<'a> {
    ptr: *const u8,
    #[cfg(debug_assertions)]
    end: *const u8,
    _stream: std::marker::PhantomData<&'a [u8]>,
}

impl<'a> U8Reader<'a> {
    #[inline(always)]
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self {
            ptr: bytes.as_ptr(),
            #[cfg(debug_assertions)]
            end: bytes.as_ptr_range().end,
            _stream: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    fn check(&self, len: usize) {
        #[cfg(debug_assertions)]
        assert!(
            self.ptr as usize + len <= self.end as usize,
            "value stream over-read"
        );
        let _ = len;
    }
}

impl ValueReader for U8Reader<'_> {
    type Code = u8;

    #[inline(always)]
    fn next_block(&mut self) -> Simd<f32, N> {
        self.check(N);
        // SAFETY: a full block has 8 values left in the stream, invariant (4) in `common.rs`.
        let v = unsafe { self.ptr.cast::<Simd<u8, N>>().read_unaligned() };
        self.ptr = unsafe { self.ptr.add(N) };
        v.cast()
    }

    #[inline(always)]
    fn next_block_codes(&mut self) -> [u8; N] {
        self.check(N);
        // SAFETY: as in `next_block`.
        let out = unsafe { self.ptr.cast::<[u8; N]>().read_unaligned() };
        self.ptr = unsafe { self.ptr.add(N) };
        out
    }

    #[inline(always)]
    fn tail(&mut self, rem: usize) -> [u8; N] {
        self.check(rem);
        let mut out = [0u8; N];
        // SAFETY: the tail has `rem = n % 8` values left in the stream, invariant (4).
        out[..rem].copy_from_slice(unsafe { std::slice::from_raw_parts(self.ptr, rem) });
        self.ptr = unsafe { self.ptr.add(rem) };
        out
    }

    #[inline(always)]
    fn code_as_f32(code: u8) -> f32 {
        code as f32
    }

    #[inline(always)]
    fn tail_value(&self, i: usize) -> f32 {
        self.check(i + 1);
        // SAFETY: the tail has `n % 8` values left in the stream, invariant (4) in `common.rs`.
        (unsafe { *self.ptr.add(i) }) as f32
    }
}

/// Two bytes (`f16`) per value. Raw cursor like [`U8Reader`]; relies on invariants (1) and
/// (4) in `common.rs` (the stream starts at an even offset and holds exactly `n` values).
pub struct F16Reader<'a> {
    ptr: *const f16,
    #[cfg(debug_assertions)]
    end: *const f16,
    _stream: std::marker::PhantomData<&'a [f16]>,
}

impl<'a> F16Reader<'a> {
    #[inline(always)]
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        // An empty stream (empty vector) may not be 2-byte aligned; anything else is, by
        // invariant (1).
        let values: &'a [f16] = if bytes.is_empty() {
            &[]
        } else {
            bytemuck::cast_slice(bytes)
        };
        Self {
            ptr: values.as_ptr(),
            #[cfg(debug_assertions)]
            end: values.as_ptr_range().end,
            _stream: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    fn check(&self, len: usize) {
        #[cfg(debug_assertions)]
        assert!(
            self.ptr as usize + 2 * len <= self.end as usize,
            "value stream over-read"
        );
        let _ = len;
    }
}

impl ValueReader for F16Reader<'_> {
    type Code = f16;

    #[inline(always)]
    fn next_block(&mut self) -> Simd<f32, N> {
        self.check(N);
        // SAFETY: a full block has 8 values left in the stream, invariant (4) in `common.rs`.
        let v = unsafe { f16_ptr_to_f32x8(self.ptr) };
        self.ptr = unsafe { self.ptr.add(N) };
        v
    }

    #[inline(always)]
    fn next_block_codes(&mut self) -> [f16; N] {
        self.check(N);
        // SAFETY: as in `next_block`.
        let out = unsafe { self.ptr.cast::<[f16; N]>().read_unaligned() };
        self.ptr = unsafe { self.ptr.add(N) };
        out
    }

    #[inline(always)]
    fn tail(&mut self, rem: usize) -> [f16; N] {
        self.check(rem);
        let mut out = [f16::ZERO; N];
        // SAFETY: the tail has `rem = n % 8` values left in the stream, invariant (4).
        out[..rem].copy_from_slice(unsafe { std::slice::from_raw_parts(self.ptr, rem) });
        self.ptr = unsafe { self.ptr.add(rem) };
        out
    }

    #[inline(always)]
    fn code_as_f32(code: f16) -> f32 {
        crate::utils::numeric_markers::f16_to_f32(code)
    }

    #[inline(always)]
    fn tail_value(&self, i: usize) -> f32 {
        self.check(i + 1);
        // SAFETY: the tail has `n % 8` values left in the stream, invariant (4) in `common.rs`.
        crate::utils::numeric_markers::f16_to_f32(unsafe { *self.ptr.add(i) })
    }
}

/// Two bytes (`u16` code) per value. Raw cursor like [`F16Reader`]; relies on invariants (1)
/// and (4) in `common.rs` (the stream starts at an even offset and holds exactly `n` values).
/// Reads are unaligned, so the cursor does not depend on the even offset for soundness.
pub struct U16Reader<'a> {
    ptr: *const u16,
    #[cfg(debug_assertions)]
    end: *const u16,
    _stream: std::marker::PhantomData<&'a [u8]>,
}

impl<'a> U16Reader<'a> {
    #[inline(always)]
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        let range = bytes.as_ptr_range();
        Self {
            ptr: range.start.cast(),
            #[cfg(debug_assertions)]
            end: range.end.cast(),
            _stream: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    fn check(&self, len: usize) {
        #[cfg(debug_assertions)]
        assert!(
            self.ptr as usize + 2 * len <= self.end as usize,
            "value stream over-read"
        );
        let _ = len;
    }
}

impl ValueReader for U16Reader<'_> {
    type Code = u16;

    #[inline(always)]
    fn next_block(&mut self) -> Simd<f32, N> {
        self.check(N);
        // SAFETY: a full block has 8 values left in the stream, invariant (4) in `common.rs`.
        let v = unsafe { self.ptr.cast::<Simd<u16, N>>().read_unaligned() };
        self.ptr = unsafe { self.ptr.add(N) };
        v.cast()
    }

    #[inline(always)]
    fn next_block_codes(&mut self) -> [u16; N] {
        self.check(N);
        // SAFETY: as in `next_block`.
        let out = unsafe { self.ptr.cast::<[u16; N]>().read_unaligned() };
        self.ptr = unsafe { self.ptr.add(N) };
        out
    }

    #[inline(always)]
    fn tail(&mut self, rem: usize) -> [u16; N] {
        self.check(rem);
        let mut out = [0u16; N];
        for (i, o) in out.iter_mut().enumerate().take(rem) {
            // SAFETY: the tail has `rem = n % 8` values left in the stream, invariant (4).
            *o = unsafe { self.ptr.add(i).read_unaligned() };
        }
        self.ptr = unsafe { self.ptr.add(rem) };
        out
    }

    #[inline(always)]
    fn code_as_f32(code: u16) -> f32 {
        code as f32
    }

    #[inline(always)]
    fn tail_value(&self, i: usize) -> f32 {
        self.check(i + 1);
        // SAFETY: the tail has `n % 8` values left in the stream, invariant (4) in `common.rs`.
        (unsafe { self.ptr.add(i).read_unaligned() }) as f32
    }
}

/// How one value is quantized, stored and read back. Implemented by the four value formats;
/// the encoder is generic over it.
pub trait ValueCodec:
    crate::core::sealed::Sealed
    + Clone
    + Debug
    + PartialEq
    + Send
    + Sync
    + SpaceUsage
    + Serialize
    + DeserializeOwned
{
    type Code: Copy + PartialEq + Debug + Send + Sync + 'static;
    /// `SparseDataEncoder::OutputValueType` of encoders using this codec.
    type Output: ValueType;
    type Reader<'a>: ValueReader<Code = Self::Code>;
    fn new_untrained(dim: usize) -> Self;
    /// Fits per-component parameters on `(components, values)` pairs of vectors with
    /// components in `0..dim`; no-op by default.
    fn train<'a, C, W>(&mut self, _dim: usize, _vectors: impl Iterator<Item = (&'a [C], &'a [W])>)
    where
        C: ComponentType,
        W: ValueType,
    {
    }
    fn encode(&self, component: usize, value: f32) -> Self::Code;
    fn decode(&self, component: usize, code: Self::Code) -> f32;
    /// Query value as stored in the dense query (scaled so `query * code_as_f32(code)` is the
    /// contribution).
    fn query_value(&self, component: usize, value: f32) -> f32;
    /// The code of a bridge (scores zero).
    fn zero_code() -> Self::Code;
    /// Bytes of the value stream of a vector with `n` entries.
    fn stream_bytes(n: usize) -> usize;
    /// Appends exactly `stream_bytes(codes.len())` bytes.
    fn write_stream(out: &mut Vec<u8>, codes: &[Self::Code]);
    fn reader(stream: &[u8]) -> Self::Reader<'_>;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixedU8Values;

impl SpaceUsage for FixedU8Values {
    fn space_usage_bytes(&self) -> usize {
        0
    }
}

const FIXED_U8_SCALE: f32 = 1.0 / ((1u32 << FixedU8Q::FRAC_NBITS) as f32);

impl crate::core::sealed::Sealed for FixedU8Values {}

impl ValueCodec for FixedU8Values {
    type Code = u8;
    type Output = u8;
    type Reader<'a> = U8Reader<'a>;
    fn new_untrained(_dim: usize) -> Self {
        FixedU8Values
    }
    #[inline(always)]
    fn encode(&self, _c: usize, v: f32) -> u8 {
        FixedU8Q::from_f32_saturating(v).to_bits()
    }
    #[inline(always)]
    fn decode(&self, _c: usize, code: u8) -> f32 {
        FixedU8Q::from_bits(code).to_f32().unwrap()
    }
    #[inline(always)]
    fn query_value(&self, _c: usize, v: f32) -> f32 {
        v * FIXED_U8_SCALE
    }
    fn zero_code() -> u8 {
        0
    }
    fn stream_bytes(n: usize) -> usize {
        n
    }
    fn write_stream(out: &mut Vec<u8>, codes: &[u8]) {
        out.extend_from_slice(codes)
    }
    fn reader(stream: &[u8]) -> U8Reader<'_> {
        U8Reader::new(stream)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarU8Values {
    quants: Box<[f32]>,
}

impl ScalarU8Values {
    pub fn from_steps(quants: Box<[f32]>) -> Self {
        Self { quants }
    }
    pub fn steps(&self) -> &[f32] {
        &self.quants
    }
}

impl SpaceUsage for ScalarU8Values {
    fn space_usage_bytes(&self) -> usize {
        self.quants.len() * std::mem::size_of::<f32>()
    }
}

impl crate::core::sealed::Sealed for ScalarU8Values {}

impl ValueCodec for ScalarU8Values {
    type Code = u8;
    type Output = u8;
    type Reader<'a> = U8Reader<'a>;
    fn new_untrained(dim: usize) -> Self {
        Self::from_steps(vec![0.0; dim].into_boxed_slice())
    }
    fn train<'a, C, W>(&mut self, dim: usize, vectors: impl Iterator<Item = (&'a [C], &'a [W])>)
    where
        C: ComponentType,
        W: ValueType,
    {
        self.quants =
            sparse_scalar_quantizer_steps(dim, vectors, 0.0, 1.0, 255.0).into_boxed_slice();
    }
    #[inline(always)]
    fn encode(&self, c: usize, v: f32) -> u8 {
        let q = self.quants[c];
        if q > 0.0 {
            (v / q).clamp(0.0, 255.0) as u8
        } else {
            0
        }
    }
    #[inline(always)]
    fn decode(&self, c: usize, code: u8) -> f32 {
        code as f32 * self.quants[c]
    }
    #[inline(always)]
    fn query_value(&self, c: usize, v: f32) -> f32 {
        v * self.quants[c]
    }
    fn zero_code() -> u8 {
        0
    }
    fn stream_bytes(n: usize) -> usize {
        n
    }
    fn write_stream(out: &mut Vec<u8>, codes: &[u8]) {
        out.extend_from_slice(codes)
    }
    fn reader(stream: &[u8]) -> U8Reader<'_> {
        U8Reader::new(stream)
    }
}

/// 16-bit scalar codes with one step per component (`max / 65535`), rounded to the nearest
/// code. Same per-component scheme as [`ScalarU8Values`] with 256 times finer steps.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarU16Values {
    quants: Box<[f32]>,
}

impl ScalarU16Values {
    pub fn from_steps(quants: Box<[f32]>) -> Self {
        Self { quants }
    }
    pub fn steps(&self) -> &[f32] {
        &self.quants
    }
}

impl SpaceUsage for ScalarU16Values {
    fn space_usage_bytes(&self) -> usize {
        self.quants.len() * std::mem::size_of::<f32>()
    }
}

impl crate::core::sealed::Sealed for ScalarU16Values {}

impl ValueCodec for ScalarU16Values {
    type Code = u16;
    type Output = u16;
    type Reader<'a> = U16Reader<'a>;
    fn new_untrained(dim: usize) -> Self {
        Self::from_steps(vec![0.0; dim].into_boxed_slice())
    }
    fn train<'a, C, W>(&mut self, dim: usize, vectors: impl Iterator<Item = (&'a [C], &'a [W])>)
    where
        C: ComponentType,
        W: ValueType,
    {
        self.quants =
            sparse_scalar_quantizer_steps(dim, vectors, 0.0, 1.0, 65535.0).into_boxed_slice();
    }
    #[inline(always)]
    fn encode(&self, c: usize, v: f32) -> u16 {
        let q = self.quants[c];
        if q > 0.0 {
            (v / q).round().clamp(0.0, 65535.0) as u16
        } else {
            0
        }
    }
    #[inline(always)]
    fn decode(&self, c: usize, code: u16) -> f32 {
        code as f32 * self.quants[c]
    }
    #[inline(always)]
    fn query_value(&self, c: usize, v: f32) -> f32 {
        v * self.quants[c]
    }
    fn zero_code() -> u16 {
        0
    }
    fn stream_bytes(n: usize) -> usize {
        2 * n
    }
    fn write_stream(out: &mut Vec<u8>, codes: &[u16]) {
        out.extend_from_slice(bytemuck::cast_slice(codes))
    }
    fn reader(stream: &[u8]) -> U16Reader<'_> {
        U16Reader::new(stream)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct F16Values;

impl SpaceUsage for F16Values {
    fn space_usage_bytes(&self) -> usize {
        0
    }
}

impl crate::core::sealed::Sealed for F16Values {}

impl ValueCodec for F16Values {
    type Code = f16;
    type Output = f16;
    type Reader<'a> = F16Reader<'a>;
    fn new_untrained(_dim: usize) -> Self {
        F16Values
    }
    #[inline(always)]
    fn encode(&self, _c: usize, v: f32) -> f16 {
        f16::from_f32(v)
    }
    #[inline(always)]
    fn decode(&self, _c: usize, code: f16) -> f32 {
        code.to_f32()
    }
    #[inline(always)]
    fn query_value(&self, _c: usize, v: f32) -> f32 {
        v
    }
    fn zero_code() -> f16 {
        f16::ZERO
    }
    fn stream_bytes(n: usize) -> usize {
        2 * n
    }
    fn write_stream(out: &mut Vec<u8>, codes: &[f16]) {
        out.extend_from_slice(bytemuck::cast_slice(codes))
    }
    fn reader(stream: &[u8]) -> F16Reader<'_> {
        F16Reader::new(stream)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scalar4BitValues {
    quants: Box<[f32]>,
}

/// 4-bit scalar codes span 15 quantization levels (`2^4 - 1`).
const NUM_LEVELS: f32 = 15.0;

impl Scalar4BitValues {
    pub fn from_steps(quants: Box<[f32]>) -> Self {
        Self { quants }
    }
}

impl SpaceUsage for Scalar4BitValues {
    fn space_usage_bytes(&self) -> usize {
        self.quants.len() * std::mem::size_of::<f32>()
    }
}

impl crate::core::sealed::Sealed for Scalar4BitValues {}

impl ValueCodec for Scalar4BitValues {
    type Code = u8;
    type Output = u8;
    type Reader<'a> = NibbleReader<'a>;
    fn new_untrained(dim: usize) -> Self {
        Self::from_steps(vec![0.0; dim].into_boxed_slice())
    }
    fn train<'a, C, W>(&mut self, dim: usize, vectors: impl Iterator<Item = (&'a [C], &'a [W])>)
    where
        C: ComponentType,
        W: ValueType,
    {
        self.quants =
            sparse_scalar_quantizer_steps(dim, vectors, 0.0, 1.0, NUM_LEVELS).into_boxed_slice();
    }
    #[inline(always)]
    fn encode(&self, c: usize, v: f32) -> u8 {
        let q = self.quants[c];
        if q > 0.0 {
            (v / q).clamp(0.0, NUM_LEVELS) as u8
        } else {
            0
        }
    }
    #[inline(always)]
    fn decode(&self, c: usize, code: u8) -> f32 {
        code as f32 * self.quants[c]
    }
    #[inline(always)]
    fn query_value(&self, c: usize, v: f32) -> f32 {
        v * self.quants[c]
    }
    fn zero_code() -> u8 {
        0
    }
    fn stream_bytes(n: usize) -> usize {
        nibble_bytes(n)
    }
    fn write_stream(out: &mut Vec<u8>, codes: &[u8]) {
        pack_nibbles(out, codes)
    }
    fn reader(stream: &[u8]) -> NibbleReader<'_> {
        NibbleReader::new(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn u8_reader_blocks_and_tail() {
        let bytes: Vec<u8> = (1..=11).collect();
        let mut r = U8Reader::new(&bytes);
        assert_eq!(r.next_block().to_array(), [1., 2., 3., 4., 5., 6., 7., 8.]);
        let t = r.tail(3);
        assert_eq!(&t[..3], &[9, 10, 11]);
        assert_eq!(U8Reader::code_as_f32(200), 200.0);
    }

    #[test]
    fn f16_reader_blocks_and_tail() {
        let vals: Vec<f16> = (1..=10).map(|i| f16::from_f32(i as f32 * 0.5)).collect();
        let bytes: &[u8] = bytemuck::cast_slice(&vals);
        let mut r = F16Reader::new(bytes);
        let b = r.next_block().to_array();
        assert_eq!(b, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]);
        let t = r.tail(2);
        assert_eq!(F16Reader::code_as_f32(t[0]), 4.5);
        assert_eq!(F16Reader::code_as_f32(t[1]), 5.0);
    }

    fn roundtrip<V: ValueCodec>(codec: &V, values: &[f32]) -> Vec<f32> {
        let codes: Vec<V::Code> = values
            .iter()
            .enumerate()
            .map(|(c, &v)| codec.encode(c, v))
            .collect();
        let mut stream = Vec::new();
        V::write_stream(&mut stream, &codes);
        assert_eq!(stream.len(), V::stream_bytes(codes.len()));
        let mut r = V::reader(&stream);
        let mut out = Vec::new();
        for _ in 0..codes.len() / 8 {
            out.extend(r.next_block_codes());
        }
        out.extend(&r.tail(codes.len() % 8)[..codes.len() % 8]);
        out.iter()
            .enumerate()
            .map(|(c, &code)| codec.decode(c, code))
            .collect()
    }

    #[test]
    fn codecs_roundtrip_within_one_step() {
        let values: Vec<f32> = (0..19).map(|i| 0.1 + i as f32 * 0.2).collect(); // 0.1 ..= 3.7
        let steps: Box<[f32]> = vec![4.0 / 255.0; 19].into_boxed_slice();
        let steps4: Box<[f32]> = vec![4.0 / 15.0; 19].into_boxed_slice();
        let cases: [(Vec<f32>, f32); 4] = [
            (roundtrip(&FixedU8Values, &values), 1.0 / 64.0),
            (
                roundtrip(&ScalarU8Values::from_steps(steps), &values),
                4.0 / 255.0,
            ),
            (roundtrip(&F16Values, &values), 0.002),
            (
                roundtrip(&Scalar4BitValues::from_steps(steps4), &values),
                4.0 / 15.0,
            ),
        ];
        for (decoded, tol) in cases {
            for (d, v) in decoded.iter().zip(&values) {
                assert!((d - v).abs() <= tol, "decoded {d} vs {v} (tol {tol})");
            }
        }
    }

    #[test]
    fn query_value_times_code_equals_decoded_product() {
        let s = ScalarU8Values::from_steps(vec![0.5; 3].into_boxed_slice());
        let code = s.encode(1, 2.0);
        assert_eq!(
            s.query_value(1, 3.0) * U8Reader::code_as_f32(code),
            3.0 * s.decode(1, code)
        );
    }
}
