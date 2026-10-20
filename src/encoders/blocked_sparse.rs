use half::f16;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::marker::PhantomData;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::core::data_block::{BLOCK_SIZE, Block, DataBlock, DataBlockU8, DataBlockU16};
#[cfg(not(target_arch = "x86_64"))]
use crate::core::data_block::{BlockValueKind, load_block_portable};
#[cfg(target_arch = "x86_64")]
use crate::core::data_block::{hsum256_ps, load_block, match_and_fma_16};
use crate::core::sealed;
use crate::core::vector::{PackedVectorView, SparseVectorView};
use crate::core::vector_encoder::{
    DenseVectorOwned, PackedSparseVectorEncoder, QueryEvaluator, SparseDataEncoder,
    SparseVectorOwned, VectorEncoder,
};
use crate::distances::DotProduct;
use crate::utils::sparse_scalar_quantizer_steps;
use crate::{Distance, SpaceUsage, ValueType};

/// How a blocked encoder stores values: the block type, the input value type, and the
/// (optionally trained) mapping between `f32` values and stored values. Sealed.
///
/// - [`BlockF16Values`]: `f16` values in [`DataBlock`]s; no training; input `f16`.
/// - [`BlockScalarU8Values`] / [`BlockScalarU16Values`]: integer codes with one trained step
///   per component (`max_c / 255` or `/ 65535`, rounded to nearest) in [`DataBlockU8`] /
///   [`DataBlockU16`]; input `f32`.
pub trait BlockValues:
    sealed::Sealed + Clone + Debug + PartialEq + Send + Sync + Serialize + DeserializeOwned + SpaceUsage
{
    type Block: Block;
    /// `SparseDataEncoder::InputValueType` of the encoder.
    type Input: ValueType;
    fn new_untrained(dim: usize) -> Self;
    /// Fits per-component parameters on `(components, values)` pairs; no-op by default.
    fn train<'a, W: ValueType>(
        &mut self,
        _dim: usize,
        _vectors: impl Iterator<Item = (&'a [u16], &'a [W])>,
    ) {
    }
    fn encode(&self, component: usize, value: f32) -> <Self::Block as Block>::Value;
    fn decode(&self, component: usize, value: <Self::Block as Block>::Value) -> f32;
    /// Query value as stored in the evaluator, scaled so `query · value_to_f32(stored)` is the
    /// contribution.
    fn query_value(&self, component: usize, value: f32) -> f32;
    fn input_to_f32(value: Self::Input) -> f32;
}

/// `f16` values (the original blocked format).
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct BlockF16Values;

impl sealed::Sealed for BlockF16Values {}

impl SpaceUsage for BlockF16Values {
    fn space_usage_bytes(&self) -> usize {
        0
    }
}

impl BlockValues for BlockF16Values {
    type Block = DataBlock;
    type Input = f16;
    fn new_untrained(_dim: usize) -> Self {
        BlockF16Values
    }
    #[inline]
    fn encode(&self, _c: usize, v: f32) -> f16 {
        f16::from_f32(v)
    }
    #[inline]
    fn decode(&self, _c: usize, v: f16) -> f32 {
        v.to_f32()
    }
    #[inline]
    fn query_value(&self, _c: usize, v: f32) -> f32 {
        v
    }
    #[inline]
    fn input_to_f32(v: f16) -> f32 {
        v.to_f32()
    }
}

macro_rules! scalar_block_values {
    ($(#[$doc:meta])* $name:ident, $block:ty, $code:ty, $levels:expr) => {
        $(#[$doc])*
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        pub struct $name {
            steps: Box<[f32]>,
        }

        impl $name {
            pub fn from_steps(steps: Box<[f32]>) -> Self {
                Self { steps }
            }
            pub fn steps(&self) -> &[f32] {
                &self.steps
            }
        }

        impl sealed::Sealed for $name {}

        impl SpaceUsage for $name {
            fn space_usage_bytes(&self) -> usize {
                self.steps.len() * std::mem::size_of::<f32>()
            }
        }

        impl BlockValues for $name {
            type Block = $block;
            type Input = f32;
            fn new_untrained(dim: usize) -> Self {
                Self::from_steps(vec![0.0; dim].into_boxed_slice())
            }
            fn train<'a, W: ValueType>(&mut self, dim: usize, vectors: impl Iterator<Item = (&'a [u16], &'a [W])>) {
                self.steps = sparse_scalar_quantizer_steps(dim, vectors, 0.0, 1.0, $levels).into_boxed_slice();
            }
            #[inline]
            fn encode(&self, c: usize, v: f32) -> $code {
                let step = self.steps[c];
                if step > 0.0 { (v / step).round().clamp(0.0, $levels) as $code } else { 0 }
            }
            #[inline]
            fn decode(&self, c: usize, code: $code) -> f32 {
                code as f32 * self.steps[c]
            }
            #[inline]
            fn query_value(&self, c: usize, v: f32) -> f32 {
                v * self.steps[c]
            }
            #[inline]
            fn input_to_f32(v: f32) -> f32 {
                v
            }
        }
    };
}

scalar_block_values!(
    /// 8-bit codes with one trained step per component, in 48-byte [`DataBlockU8`]s.
    BlockScalarU8Values, DataBlockU8, u8, 255.0
);
scalar_block_values!(
    /// 16-bit codes with one trained step per component, in 64-byte [`DataBlockU16`]s.
    BlockScalarU16Values, DataBlockU16, u16, 65535.0
);

/// Encoder that packs sparse vectors with `u16` components into blocks of 16 entries.
///
/// - Input:  `SparseVectorView<u16, V::Input>` (components sorted ascending; `f16` values for
///   the default [`BlockF16Values`], `f32` for the scalar codes)
/// - Query:  `SparseVectorView<u16, f32>`
/// - Output: `PackedVectorView<V::Block>` (sequence of blocks)
///
/// The evaluator uses a runtime threshold ([`SPARSE_QUERY_THRESHOLD`]) to decide the
/// dot-product strategy: short queries (nnz < threshold) use the jump_dot block-skipping SIMD
/// algorithm; longer queries are densified into a `Vec<f32>`. Scalar codes need training
/// ([`Self::train`], or build the dataset with `From`/`convert_from`, which trains).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct BlockedSparseEncoder<V: BlockValues = BlockF16Values> {
    dim: usize,
    values: V,
}

pub type BlockedSparseScalarU8Encoder = BlockedSparseEncoder<BlockScalarU8Values>;
pub type BlockedSparseScalarU16Encoder = BlockedSparseEncoder<BlockScalarU16Values>;

impl<V: BlockValues> sealed::Sealed for BlockedSparseEncoder<V> {}

impl<V: BlockValues> BlockedSparseEncoder<V> {
    pub fn new(dim: usize) -> Self {
        assert!(
            dim <= u16::MAX as usize + 1,
            "Dimension {} exceeds u16 range",
            dim
        );
        Self {
            dim,
            values: V::new_untrained(dim),
        }
    }

    /// The value format (and, for scalar codes, the trained steps).
    pub fn values(&self) -> &V {
        &self.values
    }

    /// Fits the value codec on `training_data` (no-op for `f16`). Accepts any sparse dataset
    /// with `u16` components and any value type; the data is read in place, never copied.
    pub fn train<E, S>(
        &mut self,
        training_data: &crate::datasets::sparse_dataset::SparseDatasetGeneric<E, S>,
    ) where
        E: crate::SparseVectorEncoder<OutputComponentType = u16>,
        E::OutputValueType: ValueType,
        S: crate::core::storage::SparseStorage<E>,
    {
        use crate::Dataset;
        self.values.train(
            self.dim,
            training_data.iter().map(|v| (v.components(), v.values())),
        );
    }

    /// Appends the blocks of the vector `(components, values)`.
    pub(crate) fn push_blocks<OutputContainer>(
        &self,
        components: &[u16],
        values: impl Iterator<Item = f32>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<V::Block>,
    {
        debug_assert!(
            components.windows(2).all(|w| w[0] < w[1]),
            "Components must be sorted in strictly ascending order"
        );
        let mut values = values;
        for chunk in components.chunks(BLOCK_SIZE) {
            // Padding: component = last real component, value = zero (the default).
            let last = chunk[chunk.len() - 1];
            let comps: [u16; BLOCK_SIZE] =
                std::array::from_fn(|i| chunk.get(i).copied().unwrap_or(last));
            let mut vals = [<V::Block as Block>::Value::default(); BLOCK_SIZE];
            for (slot, &c) in vals.iter_mut().zip(chunk) {
                *slot = self
                    .values
                    .encode(c as usize, values.next().expect("one value per component"));
            }
            output.extend(std::iter::once(V::Block::new(comps, vals)));
        }
    }
}

impl<V: BlockValues> SpaceUsage for BlockedSparseEncoder<V> {
    fn space_usage_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.values.space_usage_bytes()
    }
}

impl<V: BlockValues> VectorEncoder for BlockedSparseEncoder<V> {
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, u16, V::Input>;
    type QueryVector<'q> = SparseVectorView<'q, u16, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, V::Block>;

    type Evaluator<'e>
        = BlockedSparseQueryEvaluator<V>
    where
        Self: 'e;

    type QueryParams = ();

    fn query_evaluator<'e>(
        &'e self,
        query: Self::QueryVector<'_>,
        _params: &(),
    ) -> Self::Evaluator<'e> {
        BlockedSparseQueryEvaluator::new(query, self)
    }

    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        BlockedSparseQueryEvaluator::new(decoded.as_view(), self)
    }

    #[inline]
    fn input_dim(&self) -> usize {
        self.dim
    }

    #[inline]
    fn output_dim(&self) -> usize {
        self.dim
    }
}

impl<V: BlockValues> SparseDataEncoder for BlockedSparseEncoder<V> {
    type InputComponentType = u16;
    type InputValueType = V::Input;
    type OutputComponentType = u16;
    type OutputValueType = <V::Block as Block>::Value;

    fn decode_vector<'a>(
        &self,
        encoded: PackedVectorView<'a, V::Block>,
    ) -> SparseVectorOwned<u16, f32> {
        let mut components = Vec::new();
        let mut values = Vec::new();

        for block in encoded.data() {
            // Padding is recognised by its repeated component, not by its (zero) value.
            for i in 0..block.len() {
                components.push(block.components()[i]);
                values.push(
                    self.values
                        .decode(block.components()[i] as usize, block.values()[i]),
                );
            }
        }

        SparseVectorOwned::new(components, values)
    }
}

impl<V: BlockValues> PackedSparseVectorEncoder for BlockedSparseEncoder<V> {
    type PackedDataType = V::Block;

    fn push_encoded<'a, OutputContainer>(
        &self,
        input: SparseVectorView<'a, u16, V::Input>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<V::Block>,
    {
        self.push_blocks(
            input.components(),
            input.values().iter().map(|&v| V::input_to_f32(v)),
            output,
        );
    }
}

/// If the query has fewer than this many non-zero components, the evaluator
/// keeps it sparse and uses the jump_dot block-skipping SIMD algorithm. Otherwise
/// it densifies the query and uses gather-based dot product.
pub const SPARSE_QUERY_THRESHOLD: usize = 33;

/// Evaluator for [`BlockedSparseEncoder`].
///
/// At construction time the query is either densified (when `nnz >= SPARSE_QUERY_THRESHOLD`)
/// or kept sparse (when `nnz < SPARSE_QUERY_THRESHOLD`). Query values are pre-multiplied by
/// the codec's per-component steps, so scoring reads stored values directly.
#[derive(Debug, Clone)]
pub struct BlockedSparseQueryEvaluator<V: BlockValues = BlockF16Values> {
    /// Dense path: populated when query nnz >= SPARSE_QUERY_THRESHOLD.
    dense_query: Option<DenseVectorOwned<f32>>,
    /// Sparse jump_dot path: populated when query nnz < SPARSE_QUERY_THRESHOLD.
    sparse_query: Option<SparseVectorOwned<u16, f32>>,
    _values: PhantomData<V>,
}

impl<V: BlockValues> BlockedSparseQueryEvaluator<V> {
    #[inline]
    pub fn new(query: SparseVectorView<'_, u16, f32>, encoder: &BlockedSparseEncoder<V>) -> Self {
        for (c, _) in query.iter() {
            assert!(
                (c as usize) < encoder.dim,
                "Query component {} exceeds dimension {}",
                c,
                encoder.dim
            );
        }
        let scaled = |c: u16, v: f32| encoder.values.query_value(c as usize, v);

        if query.components().len() >= SPARSE_QUERY_THRESHOLD {
            // Dense path: densify the query.
            let mut dense_query = vec![0.0f32; encoder.dim];
            for (c, v) in query.iter() {
                dense_query[c as usize] = scaled(c, v);
            }
            Self {
                dense_query: Some(DenseVectorOwned::new(dense_query)),
                sparse_query: None,
                _values: PhantomData,
            }
        } else {
            // Sparse jump_dot path: keep query as sorted component/value arrays.
            debug_assert!(
                query.components().windows(2).all(|w| w[0] < w[1]),
                "Query components must be sorted in strictly ascending order"
            );
            Self {
                dense_query: None,
                sparse_query: Some(SparseVectorOwned::new(
                    query.components().to_vec(),
                    query.iter().map(|(c, v)| scaled(c, v)).collect(),
                )),
                _values: PhantomData,
            }
        }
    }

    /// Dense gather-based dot product.
    #[inline]
    fn compute_distance_dense(
        dense_query: crate::core::vector::DenseVectorView<'_, f32>,
        vector: PackedVectorView<'_, V::Block>,
    ) -> DotProduct {
        let query = dense_query.values();
        let mut result = 0.0f32;
        for block in vector.data() {
            result = result.algebraic_add(block.dot_product_dense_query(query).distance());
        }
        DotProduct::from(result)
    }

    /// JumpDot: block-skipping SIMD dot product.
    ///
    /// Iterates over sorted query components and skips blocks whose
    /// `last_component()` is smaller than the current query component.
    /// For matching blocks, uses AVX2 SIMD to find and accumulate values.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "f16c")]
    #[target_feature(enable = "fma")]
    #[inline]
    unsafe fn compute_distance_jump_dot(
        query: SparseVectorView<'_, u16, f32>,
        vector: PackedVectorView<'_, V::Block>,
    ) -> DotProduct {
        if query.is_empty() {
            return DotProduct::from(0.0f32);
        }

        let blocks = vector.data();
        let mut cur_q = 0;
        let mut sum: __m256 = _mm256_setzero_ps();
        let query_components = query.components();
        let query_values = query.values();
        let kind = <V::Block as Block>::KIND;

        unsafe {
            'block_loop: for block in blocks {
                if query_components[cur_q] <= block.last_component() {
                    let (comps, vals) = load_block(block);
                    while query_components[cur_q] <= block.last_component() {
                        match_and_fma_16(
                            kind,
                            query_components[cur_q],
                            query_values[cur_q],
                            comps,
                            vals,
                            &mut sum,
                        );
                        cur_q += 1;
                        if cur_q >= query_components.len() {
                            break 'block_loop;
                        }
                    }
                }
            }

            DotProduct::from(hsum256_ps(sum))
        }
    }

    /// JumpDot for targets without AVX2 (NEON on aarch64), in portable SIMD: the same algorithm
    /// as the AVX2 kernel. Compare the 16 components with the query component, keep the matching
    /// 16-bit value lane, OR the two halves (at most one real match per block; zero-valued padding
    /// may match too), widen the 8 lanes to `f32` and FMA into the accumulator.
    #[cfg(not(target_arch = "x86_64"))]
    #[inline]
    fn compute_distance_jump_dot(
        query: SparseVectorView<'_, u16, f32>,
        vector: PackedVectorView<'_, V::Block>,
    ) -> DotProduct {
        use std::simd::{
            Select, Simd, StdFloat, cmp::SimdPartialEq, num::SimdFloat, num::SimdUint,
        };

        let query_components = query.components();
        let query_values = query.values();
        if query_components.is_empty() {
            return DotProduct::from(0.0f32);
        }
        let kind = <V::Block as Block>::KIND;
        let mut cur_q = 0;
        let mut sum = Simd::<f32, 8>::splat(0.0);

        'block_loop: for block in vector.data() {
            if query_components[cur_q] <= block.last_component() {
                let (comps, vals) = load_block_portable(block);
                while query_components[cur_q] <= block.last_component() {
                    let matched = comps
                        .simd_eq(Simd::splat(query_components[cur_q]))
                        .select(vals, Simd::splat(0));
                    let (lo, hi): (Simd<u16, 8>, Simd<u16, 8>) = (
                        matched.resize::<8>(0),
                        matched.rotate_elements_left::<8>().resize::<8>(0),
                    );
                    let merged = lo | hi;
                    let f: Simd<f32, 8> =
                        match kind {
                            // `f16_to_f32` vectorizes to `fcvtl` on aarch64.
                            BlockValueKind::F16 => Simd::from_array(merged.to_array().map(|b| {
                                crate::utils::numeric_markers::f16_to_f32(f16::from_bits(b))
                            })),
                            BlockValueKind::U16 | BlockValueKind::U8 => merged.cast(),
                        };
                    sum = f.mul_add(Simd::splat(query_values[cur_q]), sum);
                    cur_q += 1;
                    if cur_q >= query_components.len() {
                        break 'block_loop;
                    }
                }
            }
        }
        DotProduct::from(sum.reduce_sum())
    }
}

impl<'v, V: BlockValues> QueryEvaluator<PackedVectorView<'v, V::Block>>
    for BlockedSparseQueryEvaluator<V>
{
    type Distance = DotProduct;

    #[inline]
    fn compute_distance(&self, vector: PackedVectorView<'v, V::Block>) -> DotProduct {
        if let Some(dense_query) = &self.dense_query {
            Self::compute_distance_dense(dense_query.as_view(), vector)
        } else {
            let query = self.sparse_query.as_ref().unwrap().as_view();
            #[cfg(target_arch = "x86_64")]
            {
                // SAFETY: we build with target-cpu=native which guarantees AVX2/F16C/FMA.
                unsafe { Self::compute_distance_jump_dot(query, vector) }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                Self::compute_distance_jump_dot(query, vector)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Dataset, DatasetGrowable, PackedSparseDatasetGrowable};

    #[test]
    fn blocked_sparse_basic_push_and_get() {
        let encoder: BlockedSparseEncoder = BlockedSparseEncoder::new(100);
        let mut dataset = PackedSparseDatasetGrowable::<BlockedSparseEncoder>::new(encoder);

        let comps = [1u16, 10, 50];
        let vals = [f16::from_f32(1.5), f16::from_f32(2.0), f16::from_f32(0.5)];
        dataset.push(SparseVectorView::new(&comps, &vals));

        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset.nnz(), 3);

        // Get the vector back.
        let view = dataset.get(0);
        assert_eq!(view.data().len(), 1); // 3 elements fit in 1 block
    }

    #[test]
    fn blocked_sparse_dot_product() {
        let encoder: BlockedSparseEncoder = BlockedSparseEncoder::new(100);
        let mut dataset = PackedSparseDatasetGrowable::<BlockedSparseEncoder>::new(encoder);

        // Document: [(1, 1.0), (10, 2.0)]
        let comps = [1u16, 10];
        let vals = [f16::from_f32(1.0), f16::from_f32(2.0)];
        dataset.push(SparseVectorView::new(&comps, &vals));

        // Query: [(1, 3.0), (10, 4.0)]
        let query = SparseVectorView::new(&[1u16, 10], &[3.0f32, 4.0]);
        let evaluator = dataset.encoder().query_evaluator(query, &());
        let dist = evaluator.compute_distance(dataset.get(0));

        // Expected: 1.0 * 3.0 + 2.0 * 4.0 = 11.0
        // (with f16 quantization error)
        let expected = f16::from_f32(1.0).to_f32() * 3.0 + f16::from_f32(2.0).to_f32() * 4.0;
        assert!(
            (dist.distance() - expected).abs() < 0.01,
            "Expected {expected}, got {}",
            dist.distance()
        );
    }

    #[test]
    fn blocked_sparse_multiple_blocks() {
        let encoder: BlockedSparseEncoder = BlockedSparseEncoder::new(1000);
        let mut dataset = PackedSparseDatasetGrowable::<BlockedSparseEncoder>::new(encoder);

        // Create a vector with >16 non-zeros (needs 2+ blocks).
        let comps: Vec<u16> = (0..20).collect();
        let vals: Vec<f16> = (1..=20).map(|i| f16::from_f32(i as f32 * 0.1)).collect();
        dataset.push(SparseVectorView::new(&comps, &vals));

        let view = dataset.get(0);
        assert_eq!(view.data().len(), 2); // ceil(20/16) = 2 blocks
    }

    #[test]
    fn blocked_sparse_roundtrip_growable_immutable() {
        use crate::PackedSparseDataset;

        let encoder: BlockedSparseEncoder = BlockedSparseEncoder::new(50);
        let mut growable = PackedSparseDatasetGrowable::<BlockedSparseEncoder>::new(encoder);

        growable.push(SparseVectorView::new(
            &[5u16, 10],
            &[f16::from_f32(1.0), f16::from_f32(2.0)],
        ));

        let frozen: PackedSparseDataset<BlockedSparseEncoder> = growable.into();
        assert_eq!(frozen.len(), 1);

        let mut growable_again: PackedSparseDatasetGrowable<BlockedSparseEncoder> = frozen.into();
        growable_again.push(SparseVectorView::new(&[3u16], &[f16::from_f32(0.5)]));
        assert_eq!(growable_again.len(), 2);
    }

    #[test]
    fn convert_from_plain_sparse_dataset() {
        use crate::{
            BlockedSparseDataset, PlainSparseDatasetGrowable, PlainSparseQuantizer,
            VectorEncoder as _,
        };

        let quantizer = PlainSparseQuantizer::<u16, f16, DotProduct>::new(100, 100);
        let mut plain = PlainSparseDatasetGrowable::new(quantizer);

        plain.push(SparseVectorView::new(
            &[1u16, 10, 50],
            &[f16::from_f32(1.5), f16::from_f32(2.0), f16::from_f32(0.5)],
        ));
        plain.push(SparseVectorView::new(
            &[5u16, 20],
            &[f16::from_f32(3.0), f16::from_f32(1.0)],
        ));

        let frozen: crate::PlainSparseDataset<u16, f16, DotProduct> = plain.into();
        let blocked: BlockedSparseDataset = frozen.into();

        assert_eq!(blocked.len(), 2);
        assert_eq!(blocked.nnz(), 5);

        // Verify dot product matches.
        let query = SparseVectorView::new(&[1u16, 10], &[2.0f32, 3.0]);
        let evaluator = blocked.encoder().query_evaluator(query, &());
        let dist = evaluator.compute_distance(blocked.get(0));

        let expected = f16::from_f32(1.5).to_f32() * 2.0 + f16::from_f32(2.0).to_f32() * 3.0;
        assert!(
            (dist.distance() - expected).abs() < 0.01,
            "Expected {expected}, got {}",
            dist.distance()
        );
    }

    // -----------------------------------------------------------------------
    // jump_dot sparse-query path tests
    // -----------------------------------------------------------------------

    /// Helper: force the dense path by creating an evaluator with nnz >= SPARSE_QUERY_THRESHOLD.
    fn make_dense_evaluator(
        query_comps: &[u16],
        query_vals: &[f32],
        dim: usize,
    ) -> BlockedSparseQueryEvaluator {
        assert!(query_comps.len() >= SPARSE_QUERY_THRESHOLD);
        let encoder: BlockedSparseEncoder = BlockedSparseEncoder::new(dim);
        let query = SparseVectorView::new(query_comps, query_vals);
        let eval = BlockedSparseQueryEvaluator::new(query, &encoder);
        assert!(eval.dense_query.is_some(), "Expected dense path");
        eval
    }

    /// Helper: force the jump_dot path by creating an evaluator with nnz < SPARSE_QUERY_THRESHOLD.
    fn make_jump_dot_evaluator(
        query_comps: &[u16],
        query_vals: &[f32],
        dim: usize,
    ) -> BlockedSparseQueryEvaluator {
        assert!(query_comps.len() < SPARSE_QUERY_THRESHOLD);
        let encoder: BlockedSparseEncoder = BlockedSparseEncoder::new(dim);
        let query = SparseVectorView::new(query_comps, query_vals);
        let eval = BlockedSparseQueryEvaluator::new(query, &encoder);
        assert!(eval.sparse_query.is_some(), "Expected jump_dot path");
        eval
    }

    #[test]
    fn threshold_selects_dense_path() {
        let dim = 1000;
        let comps: Vec<u16> = (0..SPARSE_QUERY_THRESHOLD as u16).collect();
        let vals: Vec<f32> = vec![1.0; SPARSE_QUERY_THRESHOLD];
        let encoder: BlockedSparseEncoder = BlockedSparseEncoder::new(dim);
        let query = SparseVectorView::new(&comps, &vals);
        let eval = BlockedSparseQueryEvaluator::new(query, &encoder);
        assert!(eval.dense_query.is_some());
        assert!(eval.sparse_query.is_none());
    }

    #[test]
    fn threshold_selects_jump_dot_path() {
        let dim = 1000;
        let comps: Vec<u16> = (0..(SPARSE_QUERY_THRESHOLD as u16 - 1)).collect();
        let vals: Vec<f32> = vec![1.0; SPARSE_QUERY_THRESHOLD - 1];
        let encoder: BlockedSparseEncoder = BlockedSparseEncoder::new(dim);
        let query = SparseVectorView::new(&comps, &vals);
        let eval = BlockedSparseQueryEvaluator::new(query, &encoder);
        assert!(eval.sparse_query.is_some());
        assert!(eval.dense_query.is_none());
    }

    #[test]
    fn jump_dot_dot_product_basic() {
        let dim = 100;
        let encoder: BlockedSparseEncoder = BlockedSparseEncoder::new(dim);
        let mut dataset = PackedSparseDatasetGrowable::<BlockedSparseEncoder>::new(encoder);

        // Document: [(1, 1.0), (10, 2.0)]
        let doc_comps = [1u16, 10];
        let doc_vals = [f16::from_f32(1.0), f16::from_f32(2.0)];
        dataset.push(SparseVectorView::new(&doc_comps, &doc_vals));

        // Query with nnz < SPARSE_QUERY_THRESHOLD → jump_dot path
        let query_comps = [1u16, 10];
        let query_vals = [3.0f32, 4.0];
        let eval = make_jump_dot_evaluator(&query_comps, &query_vals, dim);
        let dist = eval.compute_distance(dataset.get(0));

        let expected = f16::from_f32(1.0).to_f32() * 3.0 + f16::from_f32(2.0).to_f32() * 4.0;
        assert!(
            (dist.distance() - expected).abs() < 0.01,
            "Expected {expected}, got {}",
            dist.distance()
        );
    }

    #[test]
    fn jump_dot_dot_product_multiple_blocks() {
        let dim = 1000;
        let encoder: BlockedSparseEncoder = BlockedSparseEncoder::new(dim);
        let mut dataset = PackedSparseDatasetGrowable::<BlockedSparseEncoder>::new(encoder);

        // Document with 20 nnz → 2 blocks
        let doc_comps: Vec<u16> = (0..20).collect();
        let doc_vals: Vec<f16> = (1..=20).map(|i| f16::from_f32(i as f32 * 0.1)).collect();
        dataset.push(SparseVectorView::new(&doc_comps, &doc_vals));

        // Query overlaps components in both blocks (comp 5 in block 0, comp 18 in block 1)
        let query_comps = [5u16, 18];
        let query_vals = [2.0f32, 3.0];
        let eval = make_jump_dot_evaluator(&query_comps, &query_vals, dim);
        let dist = eval.compute_distance(dataset.get(0));

        let expected =
            f16::from_f32(6.0 * 0.1).to_f32() * 2.0 + f16::from_f32(19.0 * 0.1).to_f32() * 3.0;
        assert!(
            (dist.distance() - expected).abs() < 0.05,
            "Expected {expected}, got {}",
            dist.distance()
        );
    }

    #[test]
    fn jump_dot_empty_query() {
        let dim = 100;
        let encoder: BlockedSparseEncoder = BlockedSparseEncoder::new(dim);
        let mut dataset = PackedSparseDatasetGrowable::<BlockedSparseEncoder>::new(encoder);

        let doc_comps = [1u16, 10];
        let doc_vals = [f16::from_f32(1.0), f16::from_f32(2.0)];
        dataset.push(SparseVectorView::new(&doc_comps, &doc_vals));

        let query_comps: [u16; 0] = [];
        let query_vals: [f32; 0] = [];
        let eval = make_jump_dot_evaluator(&query_comps, &query_vals, dim);
        let dist = eval.compute_distance(dataset.get(0));

        assert_eq!(dist.distance(), 0.0);
    }

    #[test]
    fn jump_dot_query_beyond_all_blocks() {
        let dim = 1000;
        let encoder: BlockedSparseEncoder = BlockedSparseEncoder::new(dim);
        let mut dataset = PackedSparseDatasetGrowable::<BlockedSparseEncoder>::new(encoder);

        // Document components: [0..3], all in block 0 with last_component = 2
        let doc_comps = [0u16, 1, 2];
        let doc_vals = [f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)];
        dataset.push(SparseVectorView::new(&doc_comps, &doc_vals));

        // Query components all > last_component of any block → no matches
        let query_comps = [500u16, 600];
        let query_vals = [1.0f32, 2.0];
        let eval = make_jump_dot_evaluator(&query_comps, &query_vals, dim);
        let dist = eval.compute_distance(dataset.get(0));

        assert_eq!(dist.distance(), 0.0);
    }

    #[test]
    fn jump_dot_disjoint_components() {
        let dim = 100;
        let encoder: BlockedSparseEncoder = BlockedSparseEncoder::new(dim);
        let mut dataset = PackedSparseDatasetGrowable::<BlockedSparseEncoder>::new(encoder);

        // Doc has even components, query has odd → no overlap
        let doc_comps = [0u16, 2, 4, 6];
        let doc_vals = [
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];
        dataset.push(SparseVectorView::new(&doc_comps, &doc_vals));

        let query_comps = [1u16, 3, 5];
        let query_vals = [1.0f32, 1.0, 1.0];
        let eval = make_jump_dot_evaluator(&query_comps, &query_vals, dim);
        let dist = eval.compute_distance(dataset.get(0));

        assert_eq!(dist.distance(), 0.0);
    }

    #[test]
    fn jump_dot_matches_dense_path() {
        use rand::rngs::StdRng;
        use rand::seq::SliceRandom;
        use rand::{Rng, SeedableRng};

        let dim = 1 << 15; // 32768
        let doc_nnz = 128;
        let query_nnz = 20; // below threshold → jump_dot

        let mut rng = StdRng::seed_from_u64(42);

        // Generate random document
        let mut doc_indices: Vec<u16> = (0..dim as u16).collect();
        doc_indices.shuffle(&mut rng);
        doc_indices.truncate(doc_nnz);
        doc_indices.sort_unstable();
        let doc_vals: Vec<f16> = (0..doc_nnz)
            .map(|_| f16::from_f32(rng.gen_range(0.01..1.0)))
            .collect();

        // Generate random query
        let mut query_indices: Vec<u16> = (0..dim as u16).collect();
        query_indices.shuffle(&mut rng);
        query_indices.truncate(query_nnz);
        query_indices.sort_unstable();
        let query_vals: Vec<f32> = (0..query_nnz).map(|_| rng.gen_range(0.01..1.0)).collect();

        let encoder: BlockedSparseEncoder = BlockedSparseEncoder::new(dim);
        let mut dataset = PackedSparseDatasetGrowable::<BlockedSparseEncoder>::new(encoder);
        dataset.push(SparseVectorView::new(&doc_indices, &doc_vals));

        // jump_dot path (nnz < threshold)
        let jump_dot_eval = make_jump_dot_evaluator(&query_indices, &query_vals, dim);
        let jump_dot_dist = jump_dot_eval.compute_distance(dataset.get(0));

        // Dense path: pad query to reach threshold
        let mut padded_comps = query_indices.clone();
        let mut padded_vals = query_vals.clone();
        // Add components that don't overlap with the document (use high indices)
        let mut extra_comp = (dim as u16) - 1;
        while padded_comps.len() < SPARSE_QUERY_THRESHOLD {
            if !padded_comps.contains(&extra_comp) {
                padded_comps.push(extra_comp);
                padded_vals.push(0.0); // zero value → no contribution
            }
            extra_comp -= 1;
        }
        padded_comps.sort_unstable();
        // Re-sort values to match sorted components
        let mut pairs: Vec<(u16, f32)> = padded_comps
            .iter()
            .copied()
            .zip(padded_vals.iter().copied())
            .collect();
        pairs.sort_by_key(|p| p.0);
        let sorted_comps: Vec<u16> = pairs.iter().map(|p| p.0).collect();
        let sorted_vals: Vec<f32> = pairs.iter().map(|p| p.1).collect();

        let dense_eval = make_dense_evaluator(&sorted_comps, &sorted_vals, dim);
        let dense_dist = dense_eval.compute_distance(dataset.get(0));

        assert!(
            (jump_dot_dist.distance() - dense_dist.distance()).abs() < 0.01,
            "jump_dot={} dense={}",
            jump_dot_dist.distance(),
            dense_dist.distance()
        );
    }

    #[test]
    fn jump_dot_fuzz_random() {
        use rand::rngs::StdRng;
        use rand::seq::SliceRandom;
        use rand::{Rng, SeedableRng};

        let dim = 1000;
        let mut rng = StdRng::seed_from_u64(0xBEEF_CAFE);

        for _ in 0..100 {
            let doc_nnz = rng.gen_range(1..100);
            let query_nnz = rng.gen_range(1..SPARSE_QUERY_THRESHOLD);

            let mut doc_indices: Vec<u16> = (0..dim as u16).collect();
            doc_indices.shuffle(&mut rng);
            doc_indices.truncate(doc_nnz);
            doc_indices.sort_unstable();
            let doc_vals: Vec<f16> = (0..doc_nnz)
                .map(|_| f16::from_f32(rng.gen_range(0.01..5.0)))
                .collect();

            let mut query_indices: Vec<u16> = (0..dim as u16).collect();
            query_indices.shuffle(&mut rng);
            query_indices.truncate(query_nnz);
            query_indices.sort_unstable();
            let query_vals: Vec<f32> = (0..query_nnz).map(|_| rng.gen_range(0.01..5.0)).collect();

            let encoder: BlockedSparseEncoder = BlockedSparseEncoder::new(dim);
            let mut dataset = PackedSparseDatasetGrowable::<BlockedSparseEncoder>::new(encoder);
            dataset.push(SparseVectorView::new(&doc_indices, &doc_vals));

            // jump_dot evaluator
            let jump_dot_eval = make_jump_dot_evaluator(&query_indices, &query_vals, dim);
            let jump_dot_dist = jump_dot_eval.compute_distance(dataset.get(0));

            // Reference: compute manually with merge
            let mut expected = 0.0f32;
            let mut qi = 0;
            let mut di = 0;
            while qi < query_indices.len() && di < doc_indices.len() {
                match query_indices[qi].cmp(&doc_indices[di]) {
                    std::cmp::Ordering::Equal => {
                        expected += query_vals[qi] * doc_vals[di].to_f32();
                        qi += 1;
                        di += 1;
                    }
                    std::cmp::Ordering::Less => qi += 1,
                    std::cmp::Ordering::Greater => di += 1,
                }
            }

            assert!(
                (jump_dot_dist.distance() - expected).abs() < 0.1 * (1.0 + expected.abs()),
                "Mismatch: jump_dot={} expected={} (doc_nnz={doc_nnz}, query_nnz={query_nnz})",
                jump_dot_dist.distance(),
                expected
            );
        }
    }

    // -----------------------------------------------------------------------
    // Golden test: freezes the f16 blocks, scores (both query paths) and decoded vectors.
    // -----------------------------------------------------------------------

    fn golden_fnv(words: impl IntoIterator<Item = u64>) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        for w in words {
            for b in w.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        h
    }

    /// Deterministic sorted vectors in `0..dim` with values in [0.05, 3.9]; lengths cover
    /// empty vectors, partial blocks and several blocks.
    fn golden_vectors(dim: u64, seed: u64, lens: &[usize]) -> Vec<(Vec<u16>, Vec<f32>)> {
        let mut x = seed;
        let mut next = move |m: u64| {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            x.wrapping_mul(0x2545F4914F6CDD1D) % m
        };
        lens.iter()
            .map(|&len| {
                let mut c: Vec<u16> = Vec::new();
                while c.len() < len {
                    let id = if next(2) == 0 {
                        next(dim / 100 + 1)
                    } else {
                        next(dim)
                    };
                    c.push(id as u16);
                    c.sort_unstable();
                    c.dedup();
                }
                let v = c
                    .iter()
                    .map(|_| 0.05 + next(1000) as f32 * 0.00385)
                    .collect();
                (c, v)
            })
            .collect()
    }

    #[test]
    fn golden_blocked_f16() {
        let dim = 30_000u64;
        let docs = golden_vectors(
            dim,
            0x9E3779B97F4A7C15,
            &[0, 1, 15, 16, 17, 31, 32, 33, 64, 130, 3, 40],
        );
        // Short queries (jump_dot) and long queries (dense path).
        let queries = golden_vectors(dim, 0xD1B54A32D192ED03, &[1, 5, 20, 32, 33, 60, 100]);

        let encoder: BlockedSparseEncoder = BlockedSparseEncoder::new(dim as usize);
        let mut words: Vec<u64> = Vec::new();
        let mut blobs: Vec<Vec<DataBlock>> = Vec::new();
        for (c, v) in &docs {
            let v16: Vec<f16> = v.iter().map(|&x| f16::from_f32(x)).collect();
            let mut blob = Vec::new();
            encoder.push_encoded(SparseVectorView::new(c, &v16), &mut blob);
            for b in &blob {
                words.extend(b.components.iter().map(|&x| x as u64));
                words.extend(b.values.iter().map(|x| x.to_bits() as u64));
            }
            words.push(u64::MAX);
            blobs.push(blob);
        }
        let mut scores: Vec<u64> = Vec::new();
        for (qc, qv) in &queries {
            let eval = encoder.query_evaluator(SparseVectorView::new(qc, qv), &());
            for blob in &blobs {
                scores.push(
                    eval.compute_distance(PackedVectorView::new(blob))
                        .distance()
                        .to_bits() as u64,
                );
            }
        }
        let mut decoded: Vec<u64> = Vec::new();
        for blob in &blobs {
            let d = encoder.decode_vector(PackedVectorView::new(blob));
            decoded.extend(d.components().iter().map(|&c| c as u64));
            decoded.extend(d.values().iter().map(|v| v.to_bits() as u64));
            decoded.push(u64::MAX);
        }
        let got = (golden_fnv(words), golden_fnv(scores), golden_fnv(decoded));
        println!(
            "golden_blocked_f16: {:#018x}, {:#018x}, {:#018x}",
            got.0, got.1, got.2
        );
        assert_eq!(
            got,
            (0x946376a33e72e409, 0x67dd56f9d19e69d3, 0x92677293522f8f3b)
        );
    }

    /// A real value can be stored as zero (an f16 underflow here; a quantized code 0 for the
    /// scalar blocks). Padding is recognised by its repeated component, so such an entry must
    /// survive decoding and count in the block length.
    #[test]
    fn decode_keeps_values_that_quantize_to_zero() {
        let encoder: BlockedSparseEncoder = BlockedSparseEncoder::new(100);
        let comps = [1u16, 2, 3];
        let vals = [f16::from_f32(1.0), f16::from_f32(1e-9), f16::from_f32(2.0)];
        assert_eq!(vals[1], f16::ZERO);
        let mut blob = Vec::new();
        encoder.push_encoded(SparseVectorView::new(&comps, &vals), &mut blob);
        assert_eq!(blob[0].len(), 3);
        let d = encoder.decode_vector(PackedVectorView::new(&blob));
        assert_eq!(d.components(), &[1u16, 2, 3]);
        assert_eq!(d.values(), &[1.0f32, 0.0, 2.0]);
    }

    // -----------------------------------------------------------------------
    // Scalar blocks (ScalarU8 / ScalarU16)
    // -----------------------------------------------------------------------

    fn scalar_plain(
        dim: usize,
        n: usize,
        seed: u64,
    ) -> crate::PlainSparseDataset<u16, f32, DotProduct> {
        use crate::{DatasetGrowable, PlainSparseDatasetGrowable, PlainSparseQuantizer};
        let mut g = PlainSparseDatasetGrowable::new(
            PlainSparseQuantizer::<u16, f32, DotProduct>::new(dim, dim),
        );
        for (c, v) in golden_vectors(
            dim as u64,
            seed,
            &[0, 1, 15, 16, 17, 33, 64, 130, 7, 40].repeat(n / 10 + 1)[..n],
        ) {
            g.push(SparseVectorView::new(&c, &v));
        }
        g.into()
    }

    /// Decoded values are within half a step; jump_dot and the dense path agree with the
    /// f32 dot product of the decoded vector; conversion equals train + push.
    fn check_scalar_blocks<V: BlockValues<Input = f32>>(step_of: impl Fn(&V, usize) -> f32)
    where
        crate::PackedSparseDataset<BlockedSparseEncoder<V>>:
            From<crate::PlainSparseDataset<u16, f32, DotProduct>>,
    {
        use crate::{Dataset, PackedSparseDataset};
        let dim = 3000;
        let data = scalar_plain(dim, 120, 21);
        let mut encoder = BlockedSparseEncoder::<V>::new(dim);
        encoder.train(&data);

        let mut manual: Vec<Vec<V::Block>> = Vec::new();
        for v in data.iter() {
            let mut blob = Vec::new();
            encoder.push_encoded(v, &mut blob);
            let d = encoder.decode_vector(PackedVectorView::new(&blob));
            assert_eq!(d.components(), v.components());
            for ((&c, &x), &y) in v.components().iter().zip(v.values()).zip(d.values()) {
                let step = step_of(encoder.values(), c as usize);
                assert!((x - y).abs() <= step / 2.0 + x.abs() * 1e-6, "{x} vs {y}");
            }
            manual.push(blob);
        }

        let queries = golden_vectors(dim as u64, 77, &[1, 3, 10, 20, 32, 33, 50, 90]);
        for (qc, qv) in &queries {
            let eval = encoder.query_evaluator(SparseVectorView::new(qc, qv), &());
            for blob in &manual {
                let got = eval
                    .compute_distance(PackedVectorView::new(blob))
                    .distance();
                let d = encoder.decode_vector(PackedVectorView::new(blob));
                let want: f32 = qc
                    .iter()
                    .zip(qv)
                    .filter_map(|(c, q)| {
                        d.components()
                            .binary_search(c)
                            .ok()
                            .map(|i| q * d.values()[i])
                    })
                    .sum();
                assert!(
                    (got - want).abs() <= 1e-4 * (1.0 + want.abs()),
                    "{got} vs {want}"
                );
            }
        }

        let converted: PackedSparseDataset<BlockedSparseEncoder<V>> = data.into();
        assert_eq!(converted.encoder().values(), encoder.values());
        for (i, blob) in manual.iter().enumerate() {
            assert_eq!(converted.get(i as u64).data(), blob.as_slice());
        }
    }

    #[test]
    fn scalar_u8_blocks_roundtrip_score_and_convert() {
        check_scalar_blocks::<BlockScalarU8Values>(|v, c| v.steps()[c]);
    }

    #[test]
    fn scalar_u16_blocks_roundtrip_score_and_convert() {
        check_scalar_blocks::<BlockScalarU16Values>(|v, c| v.steps()[c]);
    }
}
