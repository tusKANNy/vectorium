// This whole file exists as a workaround to Rust's deficiencies regarding trait implementation conflicts.
// It's very ugly.

use fixed::{
    FixedU8, FixedU16, FixedU32, FixedU64, FixedU128,
    traits::{FixedUnsigned, ToFixed},
};
use half::{bf16, f16};
use num_traits::AsPrimitive;
use std::any::TypeId;

use crate::ValueType;

pub(crate) trait MarkerFixedSigned: FixedUnsigned + ValueType {}
impl<T: fixed::types::extra::LeEqU8 + Send + Sync> MarkerFixedSigned for FixedU8<T> {}
impl<T: fixed::types::extra::LeEqU16 + Send + Sync> MarkerFixedSigned for FixedU16<T> {}
impl<T: fixed::types::extra::LeEqU32 + Send + Sync> MarkerFixedSigned for FixedU32<T> {}
impl<T: fixed::types::extra::LeEqU64 + Send + Sync> MarkerFixedSigned for FixedU64<T> {}
impl<T: fixed::types::extra::LeEqU128 + Send + Sync> MarkerFixedSigned for FixedU128<T> {}

pub trait FromF32 {
    fn from_f32_saturating(n: f32) -> Self;
}

impl<T> FromF32 for T
where
    T: MarkerFixedSigned,
{
    fn from_f32_saturating(n: f32) -> Self {
        n.saturating_to_fixed()
    }
}

macro_rules! impl_from_f32_saturating {
    ($($t:ty),*) => {
        $(impl FromF32 for $t {
            fn from_f32_saturating(n: f32) -> Self {
                n.as_()
            }
        })*
    }
}

impl_from_f32_saturating![f64, f32, f16, bf16];

pub trait Float {}

macro_rules! impl_from_float {
    ($($t:ty),*) => {
        $(impl Float for $t {
        })*
    }
}

impl_from_float![f64, f32, f16, bf16];

// Implement Float for fixed-point types
impl<T: fixed::types::extra::LeEqU8 + Send + Sync> Float for FixedU8<T> {}
impl<T: fixed::types::extra::LeEqU16 + Send + Sync> Float for FixedU16<T> {}

// ============================================================================
// ToF32Fast - value widening used by the distance kernels
// ============================================================================

/// The built-in `f16` (this module's `f16` is `half::f16`; nested modules do not see that import).
mod builtin {
    pub(super) type F16 = f16;
}

/// `half::f16` -> `f32` for the packed/blocked kernels. On aarch64 `half` converts out of line
/// (runtime `fp16` detection + call per value, or inline asm with `fp16` enabled), which also
/// blocks vectorization; the built-in `f16` cast inlines to `fcvt`/`fcvtl`. Elsewhere `half`.
#[inline(always)]
pub(crate) fn f16_to_f32(h: f16) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        builtin::F16::from_bits(h.to_bits()) as f32
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        h.to_f32()
    }
}

/// Widens a stored value to `f32` inside distance kernels.
///
/// `half::f16`'s conversion is inline asm (`fcvt`) on aarch64 with the `fp16` target feature, and
/// LLVM cannot vectorize across inline asm. For `half::f16` the bits are reinterpreted as the
/// built-in `f16` and widened with `as f32`, which vectorizes (`fcvtl` on ARM, `vcvtph2ps` on
/// x86). Every other type goes through `ToPrimitive::to_f32`.
///
/// `ValueType` has a blanket impl, so the `half::f16` case is selected by `TypeId` rather than by a
/// separate impl; the comparison is a constant after monomorphization.
pub(crate) trait ToF32Fast: ValueType {
    fn to_f32_fast(self) -> f32;
}

impl<T: ValueType> ToF32Fast for T {
    #[inline(always)]
    fn to_f32_fast(self) -> f32 {
        if TypeId::of::<T>() == TypeId::of::<f16>() {
            // SAFETY: `T` is `half::f16`, checked above.
            let h: f16 = unsafe { std::mem::transmute_copy(&self) };
            builtin::F16::from_bits(h.to_bits()) as f32
        } else {
            self.to_f32().unwrap()
        }
    }
}

// ============================================================================
// DenseComponent - marker type for dense vectors (no component indices)
// ============================================================================

use crate::SpaceUsage;
use num_traits::FromPrimitive;
use std::hash::Hash;

/// Marker type used as `ComponentType` for dense vectors.
/// Dense vectors don't have explicit component indices, so this is a placeholder.
///
/// We use `DenseComponent` instead of the unit type `()` because of Rust's orphan rule.
/// The orphan rule prevents implementing foreign traits (such as `ComponentType`, `ValueType`,
/// `SpaceUsage`, etc.) for foreign types. Since `()` is a built-in type, we cannot implement
/// these custom traits on it within this crate. By defining `DenseComponent` as a local marker
/// type, we can implement all required trait bounds locally without violating the orphan rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct DenseComponent;

impl SpaceUsage for DenseComponent {
    fn space_usage_bytes(&self) -> usize {
        0
    }
}

impl AsPrimitive<usize> for DenseComponent {
    fn as_(self) -> usize {
        0
    }
}

impl FromPrimitive for DenseComponent {
    fn from_i64(_: i64) -> Option<Self> {
        Some(DenseComponent)
    }
    fn from_u64(_: u64) -> Option<Self> {
        Some(DenseComponent)
    }
}

impl From<usize> for DenseComponent {
    fn from(_: usize) -> Self {
        DenseComponent
    }
}
