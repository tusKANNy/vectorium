//! The DotPacking8 encoder, generic over the component type `C` (`u16`, `u32`) and the value
//! codec `V` (see `values.rs`). The public encoders (`DotPacking8U16FixedU8Encoder`, ...) are
//! type aliases of it.

use rusty_perm::{PermApply as _, PermFromSorting as _};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::simd::{Simd, StdFloat, prelude::*};

use crate::core::sealed;
use crate::core::storage::SparseStorage;
use crate::core::vector::{PackedVectorView, SparseVectorView};
use crate::core::vector_encoder::{
    PackedSparseVectorEncoder, QueryEvaluator, SparseDataEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::datasets::sparse_dataset::SparseDatasetGeneric;
use crate::distances::DotProduct;
use crate::encoders::dotpacking8::bridges::{drop_bridges, insert_bridges};
use crate::encoders::dotpacking8::common::{GapDecoder, fast_lane_gather, simd_prefix_sum};
use crate::encoders::dotpacking8::layout::{EncodedView, write_encoded};
pub use crate::encoders::dotpacking8::mapping::{ComponentMapping, PackedComponent};
use crate::encoders::dotpacking8::values::{ValueCodec, ValueReader};
use crate::{Dataset, SpaceUsage, SparseVectorEncoder, ValueType};

const N: usize = 8;

/// Packed sparse dot-product encoder: components gap-coded in bit-packed blocks of 8 (see
/// `common.rs`), values stored by the codec `V`. `u32` components insert zero-coded bridges
/// where a gap exceeds `u16::MAX` (`PackedComponent::MAY_NEED_BRIDGES`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(bound = "C: PackedComponent, V: ValueCodec")]
pub struct DotPacking8Encoder<C, V> {
    dim: usize,
    values: V,
    mapping: Option<ComponentMapping<C>>,
}

impl<C, V> sealed::Sealed for DotPacking8Encoder<C, V> {}

impl<C: PackedComponent, V: ValueCodec> DotPacking8Encoder<C, V> {
    /// Untrained encoder: original component ids, untrained value codec.
    pub fn new(dim: usize) -> Self {
        Self::with_values(dim, V::new_untrained(dim))
    }

    pub fn with_values(dim: usize, values: V) -> Self {
        Self {
            dim,
            values,
            mapping: None,
        }
    }

    pub fn values(&self) -> &V {
        &self.values
    }

    #[inline]
    pub fn component_mapping(&self) -> Option<&[C]> {
        self.mapping.as_ref().map(|m| m.forward())
    }

    #[inline]
    pub fn inverse_component_mapping(&self) -> Option<&[C]> {
        self.mapping.as_ref().map(|m| m.inverse())
    }

    /// Installs a component permutation (e.g. one computed by another ordering).
    pub fn set_component_mapping(&mut self, mapping: ComponentMapping<C>) {
        assert_eq!(
            mapping.forward().len(),
            self.dim,
            "mapping dimension mismatch"
        );
        self.mapping = Some(mapping);
    }

    /// Reorders components by decreasing frequency in `training_data` (shorter gaps), then
    /// trains the value codec on the original component ids. Accepts any sparse dataset with
    /// this encoder's component type; values of any [`ValueType`] are widened to `f32`. The
    /// distance does not matter.
    pub fn train<E, S>(&mut self, training_data: &SparseDatasetGeneric<E, S>)
    where
        E: SparseVectorEncoder<OutputComponentType = C>,
        E::OutputValueType: ValueType,
        S: SparseStorage<E>,
    {
        self.mapping = Some(ComponentMapping::by_decreasing_frequency(
            self.dim,
            training_data.iter().map(|v| v.components()),
        ));
        self.values.train(
            training_data.output_dim(),
            training_data.iter().map(|v| (v.components(), v.values())),
        );
    }
}

pub struct DotPacking8QueryEvaluator<'e, C, V> {
    dense_query: Vec<f32>,
    _encoder: PhantomData<&'e DotPacking8Encoder<C, V>>,
}

impl<'e, C: PackedComponent, V: ValueCodec> DotPacking8QueryEvaluator<'e, C, V> {
    pub fn new(query: SparseVectorView<'_, C, f32>, encoder: &'e DotPacking8Encoder<C, V>) -> Self {
        let mut dense_query = vec![0.0f32; encoder.dim];
        for (&c, &v) in query.components().iter().zip(query.values()) {
            let idx: usize = c.as_();
            assert!(
                idx < encoder.dim,
                "Query component {idx} exceeds dimension {}",
                encoder.dim
            );
            let mapped: usize = match &encoder.mapping {
                Some(m) => m.map(c).as_(),
                None => idx,
            };
            dense_query[mapped] = encoder.values.query_value(idx, v);
        }
        Self {
            dense_query,
            _encoder: PhantomData,
        }
    }
}

impl<'e, 'v, C: PackedComponent, V: ValueCodec> QueryEvaluator<PackedVectorView<'v, u64>>
    for DotPacking8QueryEvaluator<'e, C, V>
{
    type Distance = DotProduct;

    // `inline(always)`: with plain `#[inline]` LLVM stopped inlining the f16 instance into the
    // search loop (an out-of-line call per vector, +4.3% ms/query on cocondenser_1M).
    #[inline(always)]
    fn compute_distance(&self, vector: PackedVectorView<'v, u64>) -> Self::Distance {
        // SAFETY: datasets only hold vectors written by `push_encoded` with the same codec.
        let view = unsafe { EncodedView::new::<V>(vector.data()) };
        // SAFETY: the view comes from `push_encoded` output, invariants (1)-(3) in `common.rs`.
        let mut gaps = unsafe { GapDecoder::new(view.n, view.selectors, view.payload) };
        let mut values = V::reader(view.values);
        let mut acc = Simd::<f32, N>::splat(0.0);
        let mut query_ptr = self.dense_query.as_slice();
        while let Some(block_gaps) = gaps.next_block() {
            let components = simd_prefix_sum(block_gaps);
            // SAFETY: every relative component is < query_ptr.len(), invariant (5) in `common.rs`.
            let query_values = unsafe { fast_lane_gather(query_ptr, components) };
            acc = query_values.mul_add(values.next_block(), acc);
            let last_component = components[N - 1];
            debug_assert!(last_component as usize <= query_ptr.len());
            // SAFETY: `last_component <= query_ptr.len()` by invariant (5) in `common.rs`.
            query_ptr = unsafe { query_ptr.split_at_unchecked(last_component as usize).1 };
        }
        let mut total = acc.reduce_sum();
        let (tail_gaps, rem) = gaps.tail();
        if rem > 0 {
            let components = simd_prefix_sum(tail_gaps).to_array();
            total = values.tail_dot(total, query_ptr, &components, rem);
        }
        DotProduct(total)
    }
}

impl<C: PackedComponent, V: ValueCodec> VectorEncoder for DotPacking8Encoder<C, V> {
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, C, f32>;
    type QueryVector<'q> = SparseVectorView<'q, C, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;
    type Evaluator<'e>
        = DotPacking8QueryEvaluator<'e, C, V>
    where
        Self: 'e;

    type QueryParams = ();

    fn query_evaluator<'e>(
        &'e self,
        query: Self::QueryVector<'_>,
        _params: &(),
    ) -> Self::Evaluator<'e> {
        DotPacking8QueryEvaluator::new(query, self)
    }

    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        DotPacking8QueryEvaluator::new(decoded.as_view(), self)
    }

    fn output_dim(&self) -> usize {
        self.dim
    }

    fn input_dim(&self) -> usize {
        self.dim
    }
}

impl<C: PackedComponent, V: ValueCodec> SparseDataEncoder for DotPacking8Encoder<C, V> {
    type InputComponentType = C;
    type InputValueType = f32;
    type OutputComponentType = C;
    type OutputValueType = V::Output;

    fn decode_vector<'a>(
        &self,
        encoded: Self::EncodedVector<'a>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32> {
        // SAFETY: as in `compute_distance`.
        let view = unsafe { EncodedView::new::<V>(encoded.data()) };
        // SAFETY: as in `compute_distance`.
        let mut gaps = unsafe { GapDecoder::new(view.n, view.selectors, view.payload) };
        let mut values = V::reader(view.values);
        let mut comps: Vec<u32> = Vec::with_capacity(view.n);
        let mut codes: Vec<V::Code> = Vec::with_capacity(view.n);
        let mut last_component = 0u32;
        while let Some(block_gaps) = gaps.next_block() {
            let abs = (simd_prefix_sum(block_gaps) + Simd::splat(last_component)).to_array();
            comps.extend_from_slice(&abs);
            codes.extend_from_slice(&values.next_block_codes());
            last_component = abs[N - 1];
        }
        let (tail_gaps, rem) = gaps.tail();
        let abs = (simd_prefix_sum(tail_gaps) + Simd::splat(last_component)).to_array();
        comps.extend_from_slice(&abs[..rem]);
        codes.extend_from_slice(&values.tail(rem)[..rem]);

        if C::MAY_NEED_BRIDGES {
            drop_bridges(&mut comps, &mut codes, V::zero_code());
        }

        let mut components: Vec<C> = match &self.mapping {
            Some(m) => comps
                .iter()
                .map(|&c| m.unmap(C::from_usize(c as usize)))
                .collect(),
            None => comps.iter().map(|&c| C::from_usize(c as usize)).collect(),
        };
        if self.mapping.is_some() {
            // Decode after the inverse mapping, so each code uses its original component's
            // parameters; keep codes aligned with their components.
            let permutation = rusty_perm::PermD::from_sort(components.as_slice());
            permutation.apply(codes.as_mut_slice()).unwrap();
            permutation.apply(components.as_mut_slice()).unwrap();
        }
        let decoded: Vec<f32> = components
            .iter()
            .zip(&codes)
            .map(|(&c, &code)| self.values.decode(c.as_(), code))
            .collect();
        SparseVectorOwned::new(components, decoded)
    }
}

impl<C: PackedComponent, V: ValueCodec> PackedSparseVectorEncoder for DotPacking8Encoder<C, V> {
    type PackedDataType = u64;

    fn push_encoded<'a, OutputContainer>(
        &self,
        input: SparseVectorView<'a, C, f32>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<u64>,
    {
        let mut codes: Vec<V::Code> = input
            .components()
            .iter()
            .zip(input.values())
            .map(|(&c, &v)| self.values.encode(c.as_(), v))
            .collect();
        let mut comps: Vec<u32> = match &self.mapping {
            // `map` indexes the permutation, so ids >= dim panic there.
            Some(m) => input
                .components()
                .iter()
                .map(|&c| m.map(c).as_() as u32)
                .collect(),
            None => {
                // Without a mapping the ids are stored as they are and later used as gather
                // indices into a `dim`-long query (invariant (5)): check them here.
                let mut prev: Option<usize> = None;
                for &c in input.components() {
                    let c: usize = c.as_();
                    assert!(c < self.dim, "component {c} exceeds dimension {}", self.dim);
                    assert!(
                        prev.is_none_or(|p| p < c),
                        "components must be strictly increasing (got {c} after {prev:?})"
                    );
                    prev = Some(c);
                }
                input.components().iter().map(|&c| c.as_() as u32).collect()
            }
        };
        if self.mapping.is_some() {
            let permutation = rusty_perm::PermD::from_sort(comps.as_slice());
            permutation.apply(codes.as_mut_slice()).unwrap();
            permutation.apply(comps.as_mut_slice()).unwrap();
        }
        if C::MAY_NEED_BRIDGES {
            debug_assert!(
                codes.iter().all(|&code| code != V::zero_code()),
                "a real coordinate quantized to code 0, which is reserved for bridges: every value \
                 must be at least its component's quantization step"
            );
            if let Some((bridged_comps, bridged_codes)) =
                insert_bridges(&comps, &codes, V::zero_code())
            {
                write_encoded::<V>(&bridged_comps, &bridged_codes, output);
                return;
            }
        }
        write_encoded::<V>(&comps, &codes, output);
    }
}

impl<C: PackedComponent, V: ValueCodec> SpaceUsage for DotPacking8Encoder<C, V> {
    fn space_usage_bytes(&self) -> usize {
        self.values.space_usage_bytes() + self.mapping.as_ref().map_or(0, |m| m.space_usage_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vector::{PackedVectorView, SparseVectorView};
    use crate::encoders::dotpacking8::values::*;

    fn encode<C: PackedComponent, V: ValueCodec>(
        enc: &DotPacking8Encoder<C, V>,
        comps: &[C],
        vals: &[f32],
    ) -> Vec<u64> {
        let mut out = Vec::new();
        enc.push_encoded(SparseVectorView::new(comps, vals), &mut out);
        out
    }

    fn check_roundtrip<V: ValueCodec>(enc: &DotPacking8Encoder<u16, V>, tol: f32) {
        for n in [0usize, 1, 7, 8, 9, 15, 16, 17, 40] {
            let comps: Vec<u16> = (0..n as u16).map(|i| i * 37).collect();
            let vals: Vec<f32> = (0..n).map(|i| 0.25 + (i % 13) as f32 * 0.25).collect();
            let blob = encode(enc, &comps, &vals);
            let d = enc.decode_vector(PackedVectorView::new(&blob));
            assert_eq!(d.components(), comps.as_slice(), "n={n}");
            for (x, y) in d.values().iter().zip(&vals) {
                assert!((x - y).abs() <= tol, "n={n}: {x} vs {y}");
            }
        }
    }

    #[test]
    fn roundtrip_all_codecs_all_lengths() {
        let dim = 2000;
        check_roundtrip(
            &DotPacking8Encoder::<u16, FixedU8Values>::new(dim),
            1.0 / 64.0,
        );
        check_roundtrip(&DotPacking8Encoder::<u16, F16Values>::new(dim), 0.002);
        check_roundtrip(
            &DotPacking8Encoder::with_values(
                dim,
                ScalarU8Values::from_steps(vec![4.0 / 255.0; dim].into()),
            ),
            4.0 / 255.0,
        );
        check_roundtrip(
            &DotPacking8Encoder::with_values(
                dim,
                Scalar4BitValues::from_steps(vec![4.0 / 15.0; dim].into()),
            ),
            4.0 / 15.0,
        );
        // ScalarU16 rounds to the nearest code: error within half a step (+ f32 rounding).
        check_roundtrip(
            &DotPacking8Encoder::with_values(
                dim,
                ScalarU16Values::from_steps(vec![4.0 / 65535.0; dim].into()),
            ),
            2.0 / 65535.0 + 1e-6,
        );
    }

    #[test]
    fn empty_vector_all_codecs() {
        let q = SparseVectorView::new(&[1u16, 5], &[1.0f32, 2.0]);
        let e1 = DotPacking8Encoder::<u16, FixedU8Values>::new(10);
        let e2 = DotPacking8Encoder::<u16, F16Values>::new(10);
        assert!(encode(&e1, &[], &[]).is_empty());
        assert!(encode(&e2, &[], &[]).is_empty());
        assert_eq!(
            e1.query_evaluator(q, &())
                .compute_distance(PackedVectorView::new(&[]))
                .0,
            0.0
        );
        assert_eq!(
            e2.query_evaluator(q, &())
                .compute_distance(PackedVectorView::new(&[]))
                .0,
            0.0
        );
    }

    #[test]
    fn u16_decode_keeps_zero_codes() {
        // Value 0.001 is below the 1/255 step: code 0. u16 never has bridges, so it must survive.
        let enc =
            DotPacking8Encoder::with_values(10, ScalarU8Values::from_steps(vec![1.0; 10].into()));
        let blob = encode(&enc, &[2u16, 3], &[0.001, 5.0]);
        let d = enc.decode_vector(PackedVectorView::new(&blob));
        assert_eq!(d.components(), &[2, 3]);
        assert_eq!(d.values(), &[0.0, 5.0]);
    }

    #[test]
    #[should_panic(expected = "exceeds")]
    fn query_component_out_of_range_panics() {
        let enc = DotPacking8Encoder::<u16, FixedU8Values>::new(10);
        let _ = enc.query_evaluator(SparseVectorView::new(&[10u16], &[1.0f32]), &());
    }

    /// The `n: u16` header holds 65535, so a vector of 65535 entries must round-trip
    /// (regression: the shared layout first rejected `n == u16::MAX`).
    #[test]
    fn u16_vector_with_65535_entries_roundtrips() {
        let n = u16::MAX as usize;
        let enc = DotPacking8Encoder::<u16, FixedU8Values>::new(n + 1);
        let comps: Vec<u16> = (0..n as u16).collect();
        let vals = vec![1.0f32; n];
        let blob = encode(&enc, &comps, &vals);
        let d = enc.decode_vector(PackedVectorView::new(&blob));
        assert_eq!(d.components().len(), n);
        let q = SparseVectorView::new(&comps, &vals);
        assert_eq!(
            enc.query_evaluator(q, &())
                .compute_distance(PackedVectorView::new(&blob))
                .0,
            n as f32
        );
    }

    /// Without a mapping, encoded components are used as gather indices: an id >= dim would
    /// make scoring read out of bounds, so encoding must reject it.
    #[test]
    #[should_panic(expected = "exceeds dimension")]
    fn push_encoded_rejects_component_out_of_range() {
        let enc = DotPacking8Encoder::<u16, FixedU8Values>::new(10);
        encode(&enc, &[3u16, 703], &[1.0, 1.0]);
    }

    /// Unsorted input would produce wrapped gaps / bridges far outside `dim`.
    #[test]
    #[should_panic(expected = "strictly increasing")]
    fn push_encoded_rejects_unsorted_components() {
        let enc = DotPacking8Encoder::<u32, FixedU8Values>::new(100);
        encode(&enc, &[5u32, 3], &[1.0, 1.0]);
    }

    // --- Conversion from sparse datasets (`From<SparseDataset>`) ---

    use crate::{
        Dataset, DatasetGrowable, PackedSparseDataset, PackedSparseDatasetGrowable,
        PlainSparseDataset, PlainSparseDatasetGrowable, PlainSparseQuantizer,
    };

    /// Deterministic skewed sparse vectors with ids in `0..dim` and values in (0.05, 3.9].
    fn plain_dataset<C: PackedComponent>(
        dim: usize,
        n: usize,
        seed: u64,
    ) -> PlainSparseDataset<C, f32, DotProduct> {
        let mut x = seed;
        let mut next = move |m: u64| {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            x.wrapping_mul(0x2545F4914F6CDD1D) % m
        };
        let mut g = PlainSparseDatasetGrowable::new(
            PlainSparseQuantizer::<C, f32, DotProduct>::new(dim, dim),
        );
        for i in 0..n {
            let len = [0usize, 1, 7, 8, 9, 17, 40, 130][i % 8];
            let mut c: Vec<usize> = Vec::new();
            while c.len() < len {
                let id = if next(2) == 0 {
                    next(dim as u64 / 100 + 1)
                } else {
                    next(dim as u64)
                };
                c.push(id as usize);
                c.sort_unstable();
                c.dedup();
            }
            let comps: Vec<C> = c.iter().map(|&id| C::from_usize(id)).collect();
            let vals: Vec<f32> = c
                .iter()
                .map(|_| 0.05 + next(1000) as f32 * 0.00385)
                .collect();
            g.push(SparseVectorView::new(&comps, &vals));
        }
        g.into()
    }

    /// Converting a dataset must produce exactly the bytes of `train` followed by `push`.
    fn conversion_matches_train_and_push<C, V>(dim: usize)
    where
        C: PackedComponent,
        V: ValueCodec,
        PackedSparseDataset<DotPacking8Encoder<C, V>>: From<PlainSparseDataset<C, f32, DotProduct>>,
    {
        let plain = plain_dataset::<C>(dim, 200, 0x9E3779B97F4A7C15);
        let mut enc = DotPacking8Encoder::<C, V>::new(dim);
        enc.train(&plain);
        let mut manual = PackedSparseDatasetGrowable::new(enc);
        for v in plain.iter() {
            manual.push(v);
        }
        let manual: PackedSparseDataset<DotPacking8Encoder<C, V>> = manual.into();

        let converted: PackedSparseDataset<DotPacking8Encoder<C, V>> = plain.into();

        assert_eq!(converted.offsets(), manual.offsets());
        assert_eq!(converted.data(), manual.data());
        assert_eq!(converted.nnz(), manual.nnz());
        assert_eq!(
            converted.encoder().component_mapping(),
            manual.encoder().component_mapping()
        );
    }

    #[test]
    fn conversion_matches_train_and_push_all_codecs() {
        conversion_matches_train_and_push::<u16, FixedU8Values>(3000);
        conversion_matches_train_and_push::<u16, ScalarU8Values>(3000);
        conversion_matches_train_and_push::<u16, F16Values>(3000);
        conversion_matches_train_and_push::<u16, Scalar4BitValues>(3000);
        conversion_matches_train_and_push::<u16, ScalarU16Values>(3000);
        conversion_matches_train_and_push::<u32, ScalarU16Values>(200_000);
        // Ids up to 200,000: beyond the u16 range (bridges are covered by the golden tests).
        conversion_matches_train_and_push::<u32, ScalarU8Values>(200_000);
    }

    /// `f16` input values are widened to `f32`: same bytes as converting the widened dataset.
    #[test]
    fn conversion_accepts_f16_values() {
        use half::f16;
        let dim = 3000;
        let plain32 = plain_dataset::<u16>(dim, 200, 7);
        let mut g16 = PlainSparseDatasetGrowable::new(
            PlainSparseQuantizer::<u16, f16, DotProduct>::new(dim, dim),
        );
        let mut g32 = PlainSparseDatasetGrowable::new(
            PlainSparseQuantizer::<u16, f32, DotProduct>::new(dim, dim),
        );
        for v in plain32.iter() {
            let v16: Vec<f16> = v.values().iter().map(|&x| f16::from_f32(x)).collect();
            let v32: Vec<f32> = v16.iter().map(|x| x.to_f32()).collect();
            g16.push(SparseVectorView::new(v.components(), &v16));
            g32.push(SparseVectorView::new(v.components(), &v32));
        }
        let d16: PlainSparseDataset<u16, f16, DotProduct> = g16.into();
        let d32: PlainSparseDataset<u16, f32, DotProduct> = g32.into();
        let from16: PackedSparseDataset<DotPacking8Encoder<u16, ScalarU8Values>> = d16.into();
        let from32: PackedSparseDataset<DotPacking8Encoder<u16, ScalarU8Values>> = d32.into();
        assert_eq!(from16.offsets(), from32.offsets());
        assert_eq!(from16.data(), from32.data());
    }

    #[test]
    fn conversion_of_empty_dataset() {
        let plain = plain_dataset::<u16>(100, 0, 1);
        let packed: PackedSparseDataset<DotPacking8Encoder<u16, ScalarU8Values>> = plain.into();
        assert_eq!(packed.len(), 0);
        assert_eq!(packed.nnz(), 0);
    }
}
