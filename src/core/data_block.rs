//! Fixed-size blocks of 16 `(u16 component, value)` pairs used by the blocked sparse encoder.
//!
//! Three block types share one layout rule — 16 components (32 bytes) followed by 16 values —
//! and differ only in the value type:
//!
//! | Block | Values | Size | Alignment |
//! |---|---|---|---|
//! | [`DataBlock`] | `f16` | 64 B | 64 (one cache line) |
//! | [`DataBlockU16`] | `u16` codes | 64 B | 64 (one cache line) |
//! | [`DataBlockU8`] | `u8` codes | 48 B | 16 (may straddle two cache lines) |
//!
//! # Invariants
//!
//! A block produced by the encoder holds 1 to 16 real entries with strictly increasing
//! components, followed by padding slots that repeat the last real component with a zero
//! value. Padding is therefore recognised by its *repeated component*, never by its value: a
//! real value may be stored as zero (an `f16` underflow, or a quantized code 0). Zero-valued
//! padding contributes nothing to any score.

use half::f16;
use num_traits::{ToPrimitive, Zero};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::ops::Add;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::{DotProduct, SpaceUsage};

pub const BLOCK_SIZE: usize = 16;

/// How a block's 16 values are widened to `f32` in the SIMD kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockValueKind {
    F16,
    U16,
    U8,
}

/// A block of [`BLOCK_SIZE`] `(u16 component, value)` pairs. Sealed: implemented by
/// [`DataBlock`], [`DataBlockU16`] and [`DataBlockU8`].
pub trait Block:
    crate::core::sealed::Sealed
    + Copy
    + Default
    + PartialEq
    + PartialOrd
    + Debug
    + Send
    + Sync
    + Serialize
    + DeserializeOwned
    + SpaceUsage
    + Zero
    + ToPrimitive
    + 'static
{
    type Value: crate::ValueType + Default + PartialEq + Debug;
    const KIND: BlockValueKind;

    fn new(components: [u16; BLOCK_SIZE], values: [Self::Value; BLOCK_SIZE]) -> Self;
    fn components(&self) -> &[u16; BLOCK_SIZE];
    fn values(&self) -> &[Self::Value; BLOCK_SIZE];
    /// The kernel multiplier of a stored value (`f16` → `f32`, code → `code as f32`).
    fn value_to_f32(value: Self::Value) -> f32;

    /// Number of real (non-padding) entries: 1 plus the number of component increases.
    #[inline]
    fn len(&self) -> usize {
        let c = self.components();
        1 + (1..BLOCK_SIZE).filter(|&i| c[i] != c[i - 1]).count()
    }

    /// Blocks produced by the encoder always hold at least one real entry.
    #[inline]
    fn is_empty(&self) -> bool {
        false
    }

    /// The last (largest) component of the block: padding repeats it.
    #[inline]
    fn last_component(&self) -> u16 {
        self.components()[BLOCK_SIZE - 1]
    }

    /// Dot product with a dense `f32` query (already scaled by the codec's steps, if any).
    /// Padding slots hold zero and contribute `0.0`.
    #[inline]
    fn dot_product_dense_query(&self, dense_query: &[f32]) -> DotProduct {
        let base = dense_query.as_ptr();
        let comps = self.components();
        let vals = self.values();

        let mut lanes0 = [0.0f32; 8];
        let mut lanes1 = [0.0f32; 8];

        // First half: components[0..8]
        for k in 0..8 {
            let idx = comps[k] as usize;
            // SAFETY: component indices are within [0, dim) and dense_query.len() == dim.
            let qk = unsafe { *base.add(idx) };
            let vk = Self::value_to_f32(vals[k]);
            lanes0[k] = qk.algebraic_mul(vk);
        }

        // Second half: components[8..16]
        for k in 0..8 {
            let idx = comps[8 + k] as usize;
            // SAFETY: same as above.
            let qk = unsafe { *base.add(idx) };
            let vk = Self::value_to_f32(vals[8 + k]);
            lanes1[k] = qk.algebraic_mul(vk);
        }

        let mut block_sum = 0.0f32;
        for &x in &lanes0 {
            block_sum = block_sum.algebraic_add(x);
        }
        for &x in &lanes1 {
            block_sum = block_sum.algebraic_add(x);
        }
        DotProduct::from(block_sum)
    }
}

macro_rules! block_type {
    ($(#[$doc:meta])* $name:ident, $value:ty, $align:literal, $kind:expr, $to_f32:expr) => {
        $(#[$doc])*
        #[repr(C, align($align))]
        #[derive(Copy, PartialEq, Default, Debug, Clone, Serialize, Deserialize)]
        pub struct $name {
            pub(crate) components: [u16; BLOCK_SIZE],
            pub(crate) values: [$value; BLOCK_SIZE],
        }

        impl crate::core::sealed::Sealed for $name {}

        impl Block for $name {
            type Value = $value;
            const KIND: BlockValueKind = $kind;

            #[inline]
            fn new(components: [u16; BLOCK_SIZE], values: [$value; BLOCK_SIZE]) -> Self {
                Self { components, values }
            }
            #[inline]
            fn components(&self) -> &[u16; BLOCK_SIZE] {
                &self.components
            }
            #[inline]
            fn values(&self) -> &[$value; BLOCK_SIZE] {
                &self.values
            }
            #[inline(always)]
            fn value_to_f32(value: $value) -> f32 {
                ($to_f32)(value)
            }
        }

        impl SpaceUsage for $name {
            fn space_usage_bytes(&self) -> usize {
                std::mem::size_of::<Self>()
            }
        }

        // Numeric trait impls required by `PackedSparseVectorEncoder::PackedDataType: ValueType`.
        impl Add for $name {
            type Output = Self;
            fn add(mut self, rhs: Self) -> Self {
                for i in 0..BLOCK_SIZE {
                    let sum = Self::value_to_f32(self.values[i]) + Self::value_to_f32(rhs.values[i]);
                    self.values[i] = <$value as FromBlockF32>::from_block_f32(sum);
                }
                self
            }
        }

        impl Zero for $name {
            fn zero() -> Self {
                Self::default()
            }
            fn is_zero(&self) -> bool {
                *self == Self::default()
            }
        }

        impl PartialOrd for $name {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                self.components.partial_cmp(&other.components)
            }
        }

        impl ToPrimitive for $name {
            fn to_i64(&self) -> Option<i64> {
                None
            }
            fn to_u64(&self) -> Option<u64> {
                None
            }
        }
    };
}

/// Conversion used only by the `Add` impls the numeric marker traits require.
trait FromBlockF32 {
    fn from_block_f32(x: f32) -> Self;
}
impl FromBlockF32 for f16 {
    fn from_block_f32(x: f32) -> Self {
        f16::from_f32(x)
    }
}
impl FromBlockF32 for u16 {
    fn from_block_f32(x: f32) -> Self {
        x.round().clamp(0.0, u16::MAX as f32) as u16
    }
}
impl FromBlockF32 for u8 {
    fn from_block_f32(x: f32) -> Self {
        x.round().clamp(0.0, u8::MAX as f32) as u8
    }
}

block_type!(
    /// A cache-line-aligned block of 16 `(u16 component, f16 value)` pairs (64 bytes).
    DataBlock, f16, 64, BlockValueKind::F16, crate::utils::numeric_markers::f16_to_f32
);
block_type!(
    /// A cache-line-aligned block of 16 `(u16 component, u16 code)` pairs (64 bytes).
    DataBlockU16, u16, 64, BlockValueKind::U16, |v: u16| v as f32
);
block_type!(
    /// A block of 16 `(u16 component, u8 code)` pairs (48 bytes, 16-byte aligned).
    DataBlockU8, u8, 16, BlockValueKind::U8, |v: u8| v as f32
);

// ---------------------------------------------------------------------------
// SIMD helper functions for the jump_dot block-skipping algorithm
// ---------------------------------------------------------------------------

/// Loads a block's 16 components and its 16 values as 16-bit lanes (`f16` bits, `u16` codes,
/// or `u8` codes zero-extended to 16 bits).
///
/// # Safety
///
/// Requires AVX2. 64-byte blocks ([`DataBlock`], [`DataBlockU16`]) use aligned loads (both
/// halves start at multiples of 32 bytes); the 48-byte [`DataBlockU8`] (16-byte aligned) uses
/// unaligned loads.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn load_block<B: Block>(block: &B) -> (__m256i, __m256i) {
    unsafe {
        let comps_ptr = block.components().as_ptr() as *const __m256i;
        let vals_ptr = block.values().as_ptr() as *const u8;
        match B::KIND {
            BlockValueKind::F16 | BlockValueKind::U16 => {
                debug_assert!(std::mem::align_of::<B>() >= 32);
                (
                    _mm256_load_si256(comps_ptr),
                    _mm256_load_si256(vals_ptr as *const __m256i),
                )
            }
            BlockValueKind::U8 => (
                _mm256_loadu_si256(comps_ptr),
                _mm256_cvtepu8_epi16(_mm_loadu_si128(vals_ptr as *const __m128i)),
            ),
        }
    }
}

/// Portable [`load_block`]: the 16 components and the 16 values as 16-bit lanes (`f16` bits,
/// `u16` codes, or `u8` codes zero-extended). Lowers to plain NEON loads (+ `uxtl` for `u8`).
#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
pub(crate) fn load_block_portable<B: Block>(
    block: &B,
) -> (
    std::simd::Simd<u16, BLOCK_SIZE>,
    std::simd::Simd<u16, BLOCK_SIZE>,
) {
    use std::simd::{Simd, num::SimdUint};
    let comps = Simd::from_array(*block.components());
    let vals_ptr = block.values().as_ptr() as *const u8;
    // SAFETY: `values()` holds 16 entries of `B::Value`, which `KIND` says is 2 bytes (`F16`,
    // `U16`) or 1 byte (`U8`) wide.
    let vals = unsafe {
        match B::KIND {
            BlockValueKind::F16 | BlockValueKind::U16 => {
                vals_ptr.cast::<Simd<u16, BLOCK_SIZE>>().read_unaligned()
            }
            BlockValueKind::U8 => vals_ptr
                .cast::<Simd<u8, BLOCK_SIZE>>()
                .read_unaligned()
                .cast(),
        }
    };
    (comps, vals)
}

/// Compares 16 components against one query component, masks the 16-bit value lanes,
/// collapses 256→128 bits with OR (at most one real match per block; zero-valued padding
/// may match too), widens the 8 lanes to `f32` according to `kind`, and FMA-accumulates
/// `value · query_value` into `acc`.
///
/// # Safety
///
/// Requires AVX2, F16C, and FMA support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "f16c")]
#[target_feature(enable = "fma")]
#[inline]
pub(crate) unsafe fn match_and_fma_16(
    kind: BlockValueKind,
    query_component: u16,
    query_value: f32,
    comps: __m256i,
    vals: __m256i,
    acc: &mut __m256,
) {
    // 16×u16 compare → 0xFFFF on match, 0 elsewhere
    let q16: __m256i = _mm256_set1_epi16(query_component as i16);
    let cmp: __m256i = _mm256_cmpeq_epi16(comps, q16);

    // Mask the value lanes by the comparison result
    let masked: __m256i = _mm256_and_si256(vals, cmp);

    // Split 256→2×128 and collapse with OR (at most one nonzero lane survives)
    let lo: __m128i = _mm256_castsi256_si128(masked);
    let hi: __m128i = _mm256_extracti128_si256(masked, 1);
    let merged: __m128i = _mm_or_si128(lo, hi);

    // Widen 8 lanes to 8×f32 and FMA accumulate
    let f: __m256 = match kind {
        BlockValueKind::F16 => _mm256_cvtph_ps(merged),
        BlockValueKind::U16 | BlockValueKind::U8 => {
            _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(merged))
        }
    };
    let scale: __m256 = _mm256_set1_ps(query_value);
    *acc = _mm256_fmadd_ps(f, scale, *acc);
}

/// Horizontal sum of an `__m256` (8×f32) → single `f32`.
///
/// # Safety
///
/// Requires AVX support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[inline]
pub(crate) unsafe fn hsum256_ps(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let sums2 = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(sums2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_sizes_and_alignment() {
        assert_eq!(std::mem::size_of::<DataBlock>(), 64);
        assert_eq!(std::mem::align_of::<DataBlock>(), 64);
        assert_eq!(std::mem::size_of::<DataBlockU16>(), 64);
        assert_eq!(std::mem::size_of::<DataBlockU8>(), 48);
        assert_eq!(std::mem::align_of::<DataBlockU8>(), 16);
    }

    #[test]
    fn len_counts_real_entries_by_component() {
        let mut c = [7u16; BLOCK_SIZE];
        c[0] = 2;
        c[1] = 5;
        // Real entries 2, 5, 7 (value 0 on the second); padding repeats 7.
        let b = DataBlockU8::new(c, [3, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(b.len(), 3);
        let full = DataBlockU16::new(std::array::from_fn(|i| i as u16), [1; BLOCK_SIZE]);
        assert_eq!(full.len(), BLOCK_SIZE);
    }
}
