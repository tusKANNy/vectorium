//! Nibble (4-bit) value layout shared by the 4-bit DotPacking8 value codecs.
//!
//! - the **nibble value layout** (`nibble_bytes`, `pack_nibbles`, [`NibbleReader`]) — every
//!   full block of 8 codes packs into 4 bytes as `byte[i] = code[i] | (code[i+4] << 4)`,
//!   with a trailing `rem < 8` codes stored as sequential nibble pairs;
//! - [`block_bit_width`], a selector helper for the two-tier `cdotpacking8` forks.
//!
//! The 4-bit scalar quantization itself is `values::Scalar4BitValues`. On
//! `feature/quantization` this module also holds `Centroid4BitQuantizer` (per-component
//! 16-entry codebook); it is not imported on this branch yet. [`NibbleReader`] implements
//! `values::ValueReader`, so the shared gap decoder drives 4-bit values like any other stream.

use std::simd::prelude::*;

/// Gap block size of the `dotpacking8` codec.
pub(crate) const N: usize = 8;

// ---------------------------------------------------------------------------
// Nibble value layout.
// ---------------------------------------------------------------------------

/// Number of bytes one value stream of `n` 4-bit codes occupies in the
/// half-split-bulk + sequential-tail layout.
#[inline(always)]
pub(crate) fn nibble_bytes(n: usize) -> usize {
    4 * (n / N) + (n % N).div_ceil(2)
}

/// Per-block gap bit width for `block_idx`, read directly from the selector stream.
/// Matches the packing in `encode_blocks` (4-bit selector per block, two per byte, stored
/// value is `b - 1`). Used by the two-tier `cdotpacking8` forks to advance the gap payload
/// pointer across the mapped→residual stream boundary.
#[inline(always)]
#[allow(dead_code)] // the cdotpacking8 4-bit encoders are not imported on this branch yet
pub(crate) fn block_bit_width(selectors: &[u8], block_idx: usize) -> usize {
    let sel = selectors[block_idx / 2];
    (((sel >> ((block_idx & 1) << 2)) & 0x0F) as usize) + 1
}

/// Append `codes` (each already in `0..=15`) to `out` in the half-split-bulk +
/// sequential-tail nibble layout. Appends exactly `nibble_bytes(codes.len())` bytes.
pub(crate) fn pack_nibbles(out: &mut Vec<u8>, codes: &[u8]) {
    let n = codes.len();
    let full = n / N;
    for blk in 0..full {
        let base = blk * N;
        for i in 0..4 {
            out.push((codes[base + i] & 0x0F) | ((codes[base + i + 4] & 0x0F) << 4));
        }
    }
    let rem = n - full * N;
    let base = full * N;
    let mut i = 0;
    while i < rem {
        let lo = codes[base + i] & 0x0F;
        let hi = if i + 1 < rem {
            codes[base + i + 1] & 0x0F
        } else {
            0
        };
        out.push(lo | (hi << 4));
        i += 2;
    }
}

/// Sequential cursor over one value stream's nibble-packed codes. Mirrors
/// [`pack_nibbles`]: full blocks are half-split (4 bytes -> 8 codes), the tail is
/// sequential nibble pairs.
///
/// Reads are unchecked: callers read at most `n / 8` full blocks and then one tail of
/// `n % 8` codes from a stream of exactly `nibble_bytes(n)` bytes, so every read stays in
/// bounds (`debug_assert!`ed).
pub struct NibbleReader<'a> {
    bytes: &'a [u8],
    off: usize,
}

impl<'a> NibbleReader<'a> {
    #[inline(always)]
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, off: 0 }
    }

    /// Expand the next full block (8 codes) to `Simd<f32, 8>` in natural lane order
    /// `[v0..v7]` (no cross-lane shuffle): `dup=[b0..b3,b0..b3] >> [0,0,0,0,4,4,4,4] & 0xF`.
    /// Used by the linear scalar kernels (`acc = q.mul_add(codes_f32, acc)`).
    #[inline(always)]
    pub(crate) fn next_block_f32(&mut self) -> Simd<f32, N> {
        debug_assert!(self.off + 4 <= self.bytes.len());
        // SAFETY: a full block has 4 bytes left in the stream (see the type docs).
        let b = unsafe { self.bytes.get_unchecked(self.off..self.off + 4) };
        self.off += 4;
        let dup = Simd::<u8, N>::from_array([b[0], b[1], b[2], b[3], b[0], b[1], b[2], b[3]]);
        let shifts = Simd::<u8, N>::from_array([0, 0, 0, 0, 4, 4, 4, 4]);
        ((dup >> shifts) & Simd::splat(0x0F)).cast::<f32>()
    }

    /// Expand the next full block (8 codes) to `Simd<u8, 8>` in natural lane order, ready to
    /// be widened into LUT gather indices. Used by the centroid kernels.
    #[inline(always)]
    #[allow(dead_code)] // used by the centroid 4-bit encoders, not imported on this branch yet
    pub(crate) fn next_block_codes(&mut self) -> Simd<u8, N> {
        debug_assert!(self.off + 4 <= self.bytes.len());
        // SAFETY: a full block has 4 bytes left in the stream (see the type docs).
        let b = unsafe { self.bytes.get_unchecked(self.off..self.off + 4) };
        self.off += 4;
        let dup = Simd::<u8, N>::from_array([b[0], b[1], b[2], b[3], b[0], b[1], b[2], b[3]]);
        let shifts = Simd::<u8, N>::from_array([0, 0, 0, 0, 4, 4, 4, 4]);
        (dup >> shifts) & Simd::splat(0x0F)
    }

    /// Expand the next full block (8 codes) to raw `u8` codes in natural order (decode path).
    #[inline(always)]
    pub(crate) fn next_block_u8(&mut self) -> [u8; N] {
        debug_assert!(self.off + 4 <= self.bytes.len());
        // SAFETY: a full block has 4 bytes left in the stream (see the type docs).
        let b = unsafe { self.bytes.get_unchecked(self.off..self.off + 4) };
        self.off += 4;
        [
            b[0] & 0x0F,
            b[1] & 0x0F,
            b[2] & 0x0F,
            b[3] & 0x0F,
            b[0] >> 4,
            b[1] >> 4,
            b[2] >> 4,
            b[3] >> 4,
        ]
    }

    /// Read the `rem` (`< 8`) tail codes (sequential nibble pairs). Lanes `>= rem` are 0.
    #[inline(always)]
    pub(crate) fn tail_codes(&mut self, rem: usize) -> [u8; N] {
        let mut out = [0u8; N];
        // Index loop kept on purpose: the `iter_mut().enumerate().take(rem)` form clippy
        // suggests compiles to different code and measured ~2% slower (2.38 vs 2.40 ns/call).
        #[allow(clippy::needless_range_loop)]
        for i in 0..rem {
            debug_assert!(self.off + i / 2 < self.bytes.len());
            // SAFETY: the tail has `rem.div_ceil(2)` bytes left in the stream (see the type docs).
            let byte = unsafe { *self.bytes.get_unchecked(self.off + i / 2) };
            out[i] = (byte >> ((i & 1) << 2)) & 0x0F;
        }
        self.off += rem.div_ceil(2);
        out
    }
}

impl crate::encoders::dotpacking8::values::ValueReader for NibbleReader<'_> {
    type Code = u8;

    #[inline(always)]
    fn next_block(&mut self) -> Simd<f32, N> {
        self.next_block_f32()
    }

    #[inline(always)]
    fn next_block_codes(&mut self) -> [u8; N] {
        self.next_block_u8()
    }

    #[inline(always)]
    fn tail(&mut self, rem: usize) -> [u8; N] {
        self.tail_codes(rem)
    }

    #[inline(always)]
    fn code_as_f32(code: u8) -> f32 {
        code as f32
    }

    /// Decoding the tail into an array first measured faster than in-place nibble reads
    /// (cocondenser_1M, 4 alternating rounds).
    #[inline(always)]
    fn tail_dot(&mut self, mut acc: f32, query: &[f32], components: &[u32; N], rem: usize) -> f32 {
        let codes = self.tail_codes(rem);
        for i in 0..rem {
            acc += query[components[i] as usize] * codes[i] as f32;
        }
        acc
    }

    #[inline(always)]
    fn tail_value(&self, i: usize) -> f32 {
        debug_assert!(self.off + i / 2 < self.bytes.len());
        // SAFETY: the tail has `rem.div_ceil(2)` bytes left in the stream (see the type docs).
        let byte = unsafe { *self.bytes.get_unchecked(self.off + i / 2) };
        ((byte >> ((i & 1) << 2)) & 0x0F) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nibble_pack_roundtrip_unit() {
        // Full block + odd tail. 11 codes -> 4 (bulk) + 2 (tail) = 6 bytes.
        let codes: Vec<u8> = (0..11u8).map(|i| i % 16).collect();
        let mut packed = Vec::new();
        pack_nibbles(&mut packed, &codes);
        assert_eq!(packed.len(), nibble_bytes(codes.len()));

        let mut reader = NibbleReader::new(&packed);
        assert_eq!(&reader.next_block_u8()[..], &codes[0..8]);
        let tail = reader.tail_codes(3);
        assert_eq!(&tail[0..3], &codes[8..11]);

        // The f32 and integer SIMD expansions agree with the same 8 bulk codes.
        let mut r2 = NibbleReader::new(&packed);
        let f = r2.next_block_f32().to_array();
        let expect_f: Vec<f32> = codes[0..8].iter().map(|&c| c as f32).collect();
        assert_eq!(&f[..], &expect_f[..]);

        let mut r3 = NibbleReader::new(&packed);
        assert_eq!(&r3.next_block_codes().to_array()[..], &codes[0..8]);
    }
}
