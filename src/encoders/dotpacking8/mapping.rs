//! Component permutation shared by all DotPacking8 encoders.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::{ComponentType, SpaceUsage};

/// Component id types a DotPacking8 encoder can store.
pub trait PackedComponent:
    crate::core::sealed::Sealed + ComponentType + Debug + PartialEq + Serialize + DeserializeOwned
{
    /// Whether gaps between (mapped) components can exceed `u16::MAX`, so that the encoder
    /// must insert and later drop bridge coordinates.
    const MAY_NEED_BRIDGES: bool;
    /// Converts a component id; `x` must fit in `Self` (debug-checked).
    fn from_usize(x: usize) -> Self;
}

impl crate::core::sealed::Sealed for u16 {}
impl crate::core::sealed::Sealed for u32 {}

impl PackedComponent for u16 {
    const MAY_NEED_BRIDGES: bool = false;
    #[inline(always)]
    fn from_usize(x: usize) -> Self {
        debug_assert!(x <= u16::MAX as usize);
        x as u16
    }
}

impl PackedComponent for u32 {
    const MAY_NEED_BRIDGES: bool = true;
    #[inline(always)]
    fn from_usize(x: usize) -> Self {
        debug_assert!(x <= u32::MAX as usize);
        x as u32
    }
}

/// A permutation of component ids: `forward[original] = mapped`, `inverse[mapped] = original`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(bound = "C: PackedComponent")]
pub struct ComponentMapping<C> {
    forward: Box<[C]>,
    inverse: Box<[C]>,
}

impl<C: PackedComponent> ComponentMapping<C> {
    /// Builds the mapping from `permutation[original] = mapped`. Panics if it is not a
    /// permutation of `0..permutation.len()`.
    pub fn from_permutation(permutation: &[usize]) -> Self {
        let dim = permutation.len();
        let mut inverse = vec![C::from_usize(0); dim];
        let mut seen = vec![false; dim];
        for (original, &mapped) in permutation.iter().enumerate() {
            assert!(
                mapped < dim,
                "component_mapping maps component {original} to out-of-bounds index {mapped} (dim={dim})"
            );
            assert!(
                !seen[mapped],
                "component_mapping is not a permutation: duplicate mapped index {mapped}"
            );
            seen[mapped] = true;
            inverse[mapped] = C::from_usize(original);
        }
        let forward: Vec<C> = permutation.iter().map(|&m| C::from_usize(m)).collect();
        Self {
            forward: forward.into_boxed_slice(),
            inverse: inverse.into_boxed_slice(),
        }
    }

    /// Orders components by decreasing document frequency over the components of every vector.
    pub fn by_decreasing_frequency<'a>(
        dim: usize,
        components: impl Iterator<Item = &'a [C]>,
    ) -> Self
    where
        C: 'a,
    {
        let permutation = crate::utils::permute_components_by_frequency(dim, components);
        Self::from_permutation(&permutation)
    }

    #[inline]
    pub fn forward(&self) -> &[C] {
        &self.forward
    }

    #[inline]
    pub fn inverse(&self) -> &[C] {
        &self.inverse
    }

    #[inline(always)]
    pub fn map(&self, c: C) -> C {
        self.forward[c.as_()]
    }

    #[inline(always)]
    pub fn unmap(&self, c: C) -> C {
        self.inverse[c.as_()]
    }
}

impl<C> SpaceUsage for ComponentMapping<C> {
    fn space_usage_bytes(&self) -> usize {
        (self.forward.len() + self.inverse.len()) * std::mem::size_of::<C>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_permutation_builds_inverse() {
        let m = ComponentMapping::<u16>::from_permutation(&[2, 0, 1]);
        assert_eq!(m.forward(), &[2, 0, 1]);
        assert_eq!(m.inverse(), &[1, 2, 0]);
        for c in 0u16..3 {
            assert_eq!(m.unmap(m.map(c)), c);
        }
    }

    #[test]
    #[should_panic(expected = "not a permutation")]
    fn from_permutation_rejects_duplicates() {
        ComponentMapping::<u32>::from_permutation(&[0, 0, 1]);
    }

    #[test]
    #[should_panic(expected = "out-of-bounds")]
    fn from_permutation_rejects_out_of_range() {
        ComponentMapping::<u16>::from_permutation(&[0, 3, 1]);
    }

    #[test]
    fn space_usage_counts_both_arrays() {
        let m = ComponentMapping::<u32>::from_permutation(&[1, 0]);
        assert_eq!(m.space_usage_bytes(), 2 * 2 * 4);
    }
}
