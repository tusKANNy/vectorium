//! Sparse scalar quantization with one trained step per component (ScalarU8, ScalarU16).
//!
//! The value `v` of component `c` is stored as the code `round(v / step_c)`, clamped to the
//! code range, where `step_c = max_c / CODE_MAX` and `max_c` is the largest value of `c` in the
//! training data (`CODE_MAX` = 255 for `u8`, 65535 for `u16`). It decodes to `code · step_c`.
//! Every component uses its full code range, unlike a fixed-point type with one global step
//! (`FixedU8Q`, `FixedU16Q`), at the cost of training and `dim` `f32` steps.
//!
//! Scoring never decodes: for dot product the evaluator multiplies each query value by its
//! component's step once per query, so the per-entry work is `query[c] · code`. Squared
//! Euclidean additionally needs `‖x‖² = Σ (step_c · code)²`, one step lookup per entry.

use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::core::dataset::ConvertFrom;
use crate::core::sealed;
use crate::core::storage::SparseStorage;
use crate::core::vector::SparseVectorView;
use crate::core::vector_encoder::{
    QueryEvaluator, SparseDataEncoder, SparseVectorEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::datasets::sparse_dataset::{SparseDataset, SparseDatasetGeneric, SparseDatasetGrowable};
use crate::distances::{
    Distance, DotProduct, dot_product_dense_sparse_batch6_unchecked,
    dot_product_dense_sparse_unchecked, dot_product_sparse_with_merge_unchecked,
};
use crate::encoders::sparse_scalar::{ScalarSparseQuantizer, ScalarSparseSupportedDistance};
use crate::utils::{is_strictly_sorted, sparse_scalar_quantizer_steps};
use crate::{ComponentType, Dataset, DatasetGrowable, Float, SpaceUsage, ValueType};

/// Dimensions from this size on keep the query sparse (merge-based scoring) instead of a
/// dense query vector, as [`crate::encoders::sparse_scalar::ScalarSparseQueryEvaluator`] does.
const SPARSE_QUERY_MIN_DIM: usize = 1 << 20;

mod private {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
}

/// Integer code type of a [`ScalarStepSparseQuantizer`]: `u8` or `u16`. Sealed.
pub trait StepCode: private::Sealed + ValueType + Default + PartialEq + std::fmt::Debug {
    /// Largest code, as `f32` (the number of quantization levels).
    const MAX: f32;
    /// Rounds `x` to the nearest code, clamped to `0..=MAX` (NaN maps to 0).
    fn from_scaled(x: f32) -> Self;
    fn to_f32(self) -> f32;
}

impl StepCode for u8 {
    const MAX: f32 = u8::MAX as f32;
    #[inline]
    fn from_scaled(x: f32) -> Self {
        x.round().clamp(0.0, <Self as StepCode>::MAX) as u8
    }
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }
}

impl StepCode for u16 {
    const MAX: f32 = u16::MAX as f32;
    #[inline]
    fn from_scaled(x: f32) -> Self {
        x.round().clamp(0.0, <Self as StepCode>::MAX) as u16
    }
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }
}

/// Sparse quantizer storing each value as an integer code with a per-component step.
///
/// Train it before encoding ([`Self::train`], or build the dataset with `convert_from`, which
/// trains): untrained steps are 0 and every value encodes to code 0. Values above the
/// training maximum of their component clamp to `CODE_MAX`; negative values and values of
/// components never seen in training encode to 0. Supports [`DotProduct`] and
/// [`crate::SquaredEuclideanDistance`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ScalarStepSparseQuantizer<C, Code, D> {
    dim: usize,
    steps: Box<[f32]>,
    _phantom: PhantomData<(C, Code, D)>,
}

pub type ScalarU8SparseQuantizer<C, D> = ScalarStepSparseQuantizer<C, u8, D>;
pub type ScalarU16SparseQuantizer<C, D> = ScalarStepSparseQuantizer<C, u16, D>;
pub type ScalarU8SparseDataset<C, D> = SparseDataset<ScalarU8SparseQuantizer<C, D>>;
pub type ScalarU8SparseDatasetGrowable<C, D> = SparseDatasetGrowable<ScalarU8SparseQuantizer<C, D>>;
pub type ScalarU16SparseDataset<C, D> = SparseDataset<ScalarU16SparseQuantizer<C, D>>;
pub type ScalarU16SparseDatasetGrowable<C, D> =
    SparseDatasetGrowable<ScalarU16SparseQuantizer<C, D>>;

impl<C, Code, D> ScalarStepSparseQuantizer<C, Code, D>
where
    C: ComponentType,
    Code: StepCode,
{
    /// An untrained quantizer (all steps 0) for components in `0..dim`.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            steps: vec![0.0; dim].into_boxed_slice(),
            _phantom: PhantomData,
        }
    }

    /// Sets each component's step to its largest value in `training_data` divided by
    /// `CODE_MAX`. Accepts any sparse dataset with components `C` (growable or immutable,
    /// any value type, widened to `f32`); the data is read in place, never copied.
    pub fn train<E, S>(&mut self, training_data: &SparseDatasetGeneric<E, S>)
    where
        E: SparseVectorEncoder<OutputComponentType = C>,
        E::OutputValueType: ValueType,
        S: SparseStorage<E>,
    {
        self.steps = sparse_scalar_quantizer_steps(
            self.dim,
            training_data.iter().map(|v| (v.components(), v.values())),
            0.0,
            1.0,
            Code::MAX,
        )
        .into_boxed_slice();
    }

    /// The step of every component (0 for components never seen in training).
    #[inline]
    pub fn steps(&self) -> &[f32] {
        &self.steps
    }

    #[inline]
    fn encode_value(&self, component: C, value: f32) -> Code {
        let step = self.steps[component.as_()];
        if step > 0.0 {
            Code::from_scaled(value / step)
        } else {
            Code::default()
        }
    }
}

impl<C, Code, D> sealed::Sealed for ScalarStepSparseQuantizer<C, Code, D> {}

impl<C, Code, D> SparseDataEncoder for ScalarStepSparseQuantizer<C, Code, D>
where
    C: ComponentType,
    Code: StepCode,
    D: ScalarSparseSupportedDistance + From<f32>,
{
    type InputComponentType = C;
    type InputValueType = f32;
    type OutputComponentType = C;
    type OutputValueType = Code;

    fn decode_vector<'a>(
        &self,
        encoded: SparseVectorView<'a, C, Code>,
    ) -> SparseVectorOwned<C, f32> {
        let values = encoded
            .components()
            .iter()
            .zip(encoded.values())
            .map(|(&c, &code)| code.to_f32() * self.steps[c.as_()])
            .collect();
        SparseVectorOwned::new(encoded.components().to_vec(), values)
    }
}

impl<C, Code, D> SparseVectorEncoder for ScalarStepSparseQuantizer<C, Code, D>
where
    C: ComponentType,
    Code: StepCode,
    D: ScalarSparseSupportedDistance + From<f32>,
{
    fn push_encoded<'a, ComponentContainer, ValueContainer>(
        &self,
        input: SparseVectorView<'a, C, f32>,
        components: &mut ComponentContainer,
        values: &mut ValueContainer,
    ) where
        ComponentContainer: Extend<C>,
        ValueContainer: Extend<Code>,
    {
        components.extend(input.components().iter().copied());
        values.extend(
            input
                .components()
                .iter()
                .zip(input.values())
                .map(|(&c, &v)| self.encode_value(c, v)),
        );
    }
}

impl<C, Code, D> VectorEncoder for ScalarStepSparseQuantizer<C, Code, D>
where
    C: ComponentType,
    Code: StepCode,
    D: ScalarSparseSupportedDistance + From<f32>,
{
    type Distance = D;
    type InputVector<'a> = SparseVectorView<'a, C, f32>;
    type QueryVector<'q> = SparseVectorView<'q, C, f32>;
    type EncodedVector<'a> = SparseVectorView<'a, C, Code>;

    type Evaluator<'e>
        = ScalarStepSparseQueryEvaluator<'e, C, Code, D>
    where
        Self: 'e;

    type QueryParams = ();

    fn query_evaluator<'e>(
        &'e self,
        query: Self::QueryVector<'_>,
        _params: &(),
    ) -> Self::Evaluator<'e> {
        ScalarStepSparseQueryEvaluator::new(query, self)
    }

    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = self.decode_vector(vector);
        ScalarStepSparseQueryEvaluator::new(decoded.as_view(), self)
    }

    #[inline]
    fn output_dim(&self) -> usize {
        self.dim
    }

    #[inline]
    fn input_dim(&self) -> usize {
        self.dim
    }
}

impl<C, Code, D> SpaceUsage for ScalarStepSparseQuantizer<C, Code, D> {
    fn space_usage_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.steps.len() * std::mem::size_of::<f32>()
    }
}

/// Query evaluator of [`ScalarStepSparseQuantizer`]: the query is stored with every value
/// already multiplied by its component's step, so dot products read codes directly.
#[derive(Debug, Clone)]
pub struct ScalarStepSparseQueryEvaluator<'e, C, Code, D>
where
    C: ComponentType,
{
    /// `query[c] · step_c`, dense (for `dim < 2^20`).
    dense_query: Option<Vec<f32>>,
    /// `(c, query[c] · step_c)` sorted by component (for `dim >= 2^20`).
    sparse_query: Option<SparseVectorOwned<C, f32>>,
    /// `‖query‖²` (Euclidean only).
    query_norm: f32,
    steps: &'e [f32],
    _phantom: PhantomData<(Code, D)>,
}

impl<'e, C, Code, D> ScalarStepSparseQueryEvaluator<'e, C, Code, D>
where
    C: ComponentType,
    Code: StepCode,
    D: ScalarSparseSupportedDistance + From<f32>,
{
    pub fn new(
        query: SparseVectorView<'_, C, f32>,
        quantizer: &'e ScalarStepSparseQuantizer<C, Code, D>,
    ) -> Self {
        let dim = quantizer.dim;
        for &c in query.components() {
            assert!(
                c.as_() < dim,
                "Query vector component exceeds quantizer input dimension."
            );
        }
        let steps = &quantizer.steps[..];
        let query_norm = if D::requires_dot_query() {
            query
                .values()
                .iter()
                .fold(0.0f32, |acc, &v| acc.algebraic_add(v.algebraic_mul(v)))
        } else {
            0.0
        };
        let scaled = query
            .components()
            .iter()
            .zip(query.values())
            .map(|(&c, &v)| v * steps[c.as_()]);

        let (dense_query, sparse_query) = if dim < SPARSE_QUERY_MIN_DIM {
            let mut dense = vec![0.0f32; dim];
            for (&c, s) in query.components().iter().zip(scaled) {
                dense[c.as_()] = s;
            }
            (Some(dense), None)
        } else {
            assert!(
                is_strictly_sorted(query.components()),
                "Query components must be sorted in strictly ascending order."
            );
            (
                None,
                Some(SparseVectorOwned::new(
                    query.components().to_vec(),
                    scaled.collect(),
                )),
            )
        };

        Self {
            dense_query,
            sparse_query,
            query_norm,
            steps,
            _phantom: PhantomData,
        }
    }

    /// `Σ query[c] · step_c · code_c`.
    #[inline]
    fn dot(&self, vector: SparseVectorView<'_, C, Code>) -> f32 {
        // SAFETY: encoded components are < dim (the quantizer indexes `steps` with them when
        // encoding, which panics otherwise) and the dense query has length dim.
        unsafe {
            match &self.dense_query {
                Some(dense) => dot_product_dense_sparse_unchecked(
                    crate::core::vector::DenseVectorView::new(dense),
                    vector,
                ),
                None => dot_product_sparse_with_merge_unchecked(
                    self.sparse_query.as_ref().unwrap().as_view(),
                    vector,
                ),
            }
        }
        .distance()
    }

    /// `‖x‖² = Σ (step_c · code_c)²`.
    #[inline]
    fn vector_norm(&self, vector: SparseVectorView<'_, C, Code>) -> f32 {
        vector
            .components()
            .iter()
            .zip(vector.values())
            .fold(0.0f32, |acc, (&c, &code)| {
                let x = code.to_f32() * self.steps[c.as_()];
                acc.algebraic_add(x.algebraic_mul(x))
            })
    }

    #[inline]
    fn finish(&self, dot: f32, vector: SparseVectorView<'_, C, Code>) -> D {
        if D::requires_dot_query() {
            let dist = self
                .query_norm
                .algebraic_add(self.vector_norm(vector))
                .algebraic_sub(2.0f32.algebraic_mul(dot));
            D::from(dist)
        } else {
            D::from(dot)
        }
    }
}

impl<'e, 'v, C, Code, D> QueryEvaluator<SparseVectorView<'v, C, Code>>
    for ScalarStepSparseQueryEvaluator<'e, C, Code, D>
where
    C: ComponentType,
    Code: StepCode,
    D: ScalarSparseSupportedDistance + From<f32>,
{
    type Distance = D;

    #[inline]
    fn compute_distance(&self, vector: SparseVectorView<'v, C, Code>) -> D {
        self.finish(self.dot(vector), vector)
    }

    #[inline]
    fn compute_distances_batch6(&self, vectors: [SparseVectorView<'v, C, Code>; 6]) -> [D; 6] {
        match &self.dense_query {
            Some(dense) => {
                // SAFETY: as in `dot`.
                let dots: [DotProduct; 6] = unsafe {
                    dot_product_dense_sparse_batch6_unchecked(
                        crate::core::vector::DenseVectorView::new(dense),
                        vectors,
                    )
                };
                std::array::from_fn(|i| self.finish(dots[i].distance(), vectors[i]))
            }
            None => std::array::from_fn(|i| self.compute_distance(vectors[i])),
        }
    }
}

/// Builds a trained ScalarU8/ScalarU16 sparse dataset from a plain (or cast-quantized) sparse
/// dataset: trains the steps on `source`, then encodes every vector. Values of any float type
/// are widened to `f32` per vector; `source` is not copied.
impl<C, SrcIn, Mid, Code, D, S>
    ConvertFrom<SparseDatasetGeneric<ScalarSparseQuantizer<C, SrcIn, Mid, D>, S>>
    for SparseDataset<ScalarStepSparseQuantizer<C, Code, D>>
where
    C: ComponentType,
    SrcIn: ValueType + Float,
    Mid: ValueType + Float + crate::FromF32,
    Code: StepCode,
    D: ScalarSparseSupportedDistance + From<f32>,
    S: SparseStorage<ScalarSparseQuantizer<C, SrcIn, Mid, D>>,
{
    type Config = ();

    fn convert_from(
        source: SparseDatasetGeneric<ScalarSparseQuantizer<C, SrcIn, Mid, D>, S>,
        _config: (),
    ) -> Self {
        let dim = source.output_dim();
        let mut quantizer = ScalarStepSparseQuantizer::<C, Code, D>::new(dim);
        quantizer.train(&source);

        let mut out = SparseDatasetGrowable::with_capacity(quantizer, source.len());
        let mut values = Vec::new();
        for v in source.iter() {
            values.clear();
            values.extend(v.values().iter().map(|x| x.to_f32().unwrap()));
            out.push(SparseVectorView::new(v.components(), &values));
        }
        out.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dataset::ConvertFrom;
    use crate::{
        Dataset, DatasetGrowable, DotProduct, PlainSparseDataset, PlainSparseDatasetGrowable,
        PlainSparseQuantizer, QueryEvaluator, SquaredEuclideanDistance, VectorEncoder,
    };
    use half::f16;

    /// Deterministic skewed sparse vectors with ids in `0..dim` and values in (0.05, 3.9].
    fn plain<
        C: ComponentType + num_traits::FromPrimitive,
        D: ScalarSparseSupportedDistance + From<f32>,
    >(
        dim: usize,
        n: usize,
        seed: u64,
    ) -> PlainSparseDataset<C, f32, D> {
        let mut x = seed;
        let mut next = move |m: u64| {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            x.wrapping_mul(0x2545F4914F6CDD1D) % m
        };
        let mut g =
            PlainSparseDatasetGrowable::new(PlainSparseQuantizer::<C, f32, D>::new(dim, dim));
        for i in 0..n {
            let len = [0usize, 1, 7, 16, 17, 40, 130][i % 7];
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
            let comps: Vec<C> = c.iter().map(|&id| C::from_usize(id).unwrap()).collect();
            let vals: Vec<f32> = c
                .iter()
                .map(|_| 0.05 + next(1000) as f32 * 0.00385)
                .collect();
            g.push(SparseVectorView::new(&comps, &vals));
        }
        g.into()
    }

    fn trained<Code: StepCode, D: ScalarSparseSupportedDistance + From<f32>>(
        data: &PlainSparseDataset<u16, f32, D>,
    ) -> ScalarStepSparseQuantizer<u16, Code, D> {
        let mut q = ScalarStepSparseQuantizer::<u16, Code, D>::new(data.output_dim());
        q.train(data);
        q
    }

    fn check_error_within_half_step<Code: StepCode>() {
        let data = plain::<u16, DotProduct>(500, 300, 3);
        let q = trained::<Code, DotProduct>(&data);
        for v in data.iter() {
            let mut comps = Vec::new();
            let mut codes = Vec::new();
            q.push_encoded(v, &mut comps, &mut codes);
            let decoded = q.decode_vector(SparseVectorView::new(&comps, &codes));
            assert_eq!(decoded.components(), v.components());
            for ((&c, &x), &y) in v.components().iter().zip(v.values()).zip(decoded.values()) {
                let step = q.steps()[c as usize];
                // Half a step, plus f32 rounding of `v / step` and `code * step`.
                assert!(
                    (x - y).abs() <= step / 2.0 + x.abs() * 1e-6,
                    "c={c}: {x} vs {y}, step {step}"
                );
            }
        }
    }

    #[test]
    fn encode_decode_error_within_half_step() {
        check_error_within_half_step::<u8>();
        check_error_within_half_step::<u16>();
    }

    #[test]
    fn untrained_or_unseen_component_encodes_to_zero() {
        let untrained = ScalarStepSparseQuantizer::<u16, u8, DotProduct>::new(10);
        let mut comps = Vec::new();
        let mut codes = Vec::new();
        untrained.push_encoded(
            SparseVectorView::new(&[3u16, 7], &[1.0f32, 2.0]),
            &mut comps,
            &mut codes,
        );
        assert_eq!(codes, vec![0u8, 0]);

        // Component 9 never occurs in training: step 0, code 0, no division by zero.
        let mut g = PlainSparseDatasetGrowable::new(
            PlainSparseQuantizer::<u16, f32, DotProduct>::new(10, 10),
        );
        g.push(SparseVectorView::new(&[1u16, 2], &[1.0f32, 0.5]));
        let data: PlainSparseDataset<u16, f32, DotProduct> = g.into();
        let q = trained::<u16, DotProduct>(&data);
        assert_eq!(q.steps()[9], 0.0);
        let (mut comps, mut codes) = (Vec::new(), Vec::new());
        q.push_encoded(
            SparseVectorView::new(&[1u16, 9], &[1.0f32, 3.0]),
            &mut comps,
            &mut codes,
        );
        assert_eq!(codes, vec![u16::MAX, 0]);
    }

    /// Reference score: the f32 distance between the query and the decoded vector.
    fn reference<D: ScalarSparseSupportedDistance + From<f32>>(
        q: SparseVectorView<'_, u16, f32>,
        x: &SparseVectorOwned<u16, f32>,
    ) -> f32 {
        let mut dot = 0.0f32;
        for (c, v) in q.iter() {
            if let Ok(i) = x.components().binary_search(&c) {
                dot += v * x.values()[i];
            }
        }
        if D::requires_dot_query() {
            let qq: f32 = q.values().iter().map(|v| v * v).sum();
            let xx: f32 = x.values().iter().map(|v| v * v).sum();
            qq + xx - 2.0 * dot
        } else {
            dot
        }
    }

    fn check_scores<Code: StepCode, D: ScalarSparseSupportedDistance + From<f32>>(dim: usize) {
        let data = plain::<u16, D>(dim.min(60_000), 200, 11);
        let mut q = ScalarStepSparseQuantizer::<u16, Code, D>::new(dim);
        q.train(&data);
        let queries = plain::<u16, D>(dim.min(60_000), 30, 99);
        let encoded: Vec<(Vec<u16>, Vec<Code>)> = data
            .iter()
            .map(|v| {
                let (mut c, mut x) = (Vec::new(), Vec::new());
                q.push_encoded(v, &mut c, &mut x);
                (c, x)
            })
            .collect();
        for query in queries.iter() {
            let ev = q.query_evaluator(query, &());
            for (c, x) in &encoded {
                let view = SparseVectorView::new(c, x);
                let got = ev.compute_distance(view).distance();
                let want = reference::<D>(query, &q.decode_vector(view));
                assert!(
                    (got - want).abs() <= 1e-4 * (1.0 + want.abs()),
                    "dim {dim}: {got} vs {want}"
                );
            }
            // Six vectors at once must equal six single calls.
            for chunk in encoded.chunks_exact(6).take(5) {
                let views: [SparseVectorView<'_, u16, Code>; 6] =
                    std::array::from_fn(|i| SparseVectorView::new(&chunk[i].0, &chunk[i].1));
                let batch = ev.compute_distances_batch6(views);
                for i in 0..6 {
                    let single = ev.compute_distance(views[i]).distance();
                    assert!((batch[i].distance() - single).abs() <= 1e-5 * (1.0 + single.abs()));
                }
            }
        }
    }

    #[test]
    fn dot_product_matches_decoded_f32() {
        check_scores::<u8, DotProduct>(3000);
        check_scores::<u16, DotProduct>(3000);
        // dim >= 2^20: sparse (merge) query path.
        check_scores::<u16, DotProduct>(1 << 20);
    }

    #[test]
    fn squared_euclidean_matches_decoded_f32() {
        check_scores::<u8, SquaredEuclideanDistance>(3000);
        check_scores::<u16, SquaredEuclideanDistance>(3000);
        check_scores::<u16, SquaredEuclideanDistance>(1 << 20);
    }

    fn check_conversion<C, Code>(dim: usize)
    where
        C: ComponentType + num_traits::FromPrimitive + std::fmt::Debug,
        Code: StepCode,
    {
        let data = plain::<C, DotProduct>(dim, 200, 5);
        let mut q = ScalarStepSparseQuantizer::<C, Code, DotProduct>::new(dim);
        q.train(&data);
        let (mut comps, mut codes) = (Vec::new(), Vec::new());
        let mut offsets = vec![0usize];
        for v in data.iter() {
            q.push_encoded(v, &mut comps, &mut codes);
            offsets.push(comps.len());
        }

        let converted =
            SparseDataset::<ScalarStepSparseQuantizer<C, Code, DotProduct>>::convert_from(data, ());
        assert_eq!(converted.encoder().steps(), q.steps());
        assert_eq!(converted.len(), offsets.len() - 1);
        for i in 0..converted.len() {
            let v = converted.get(i as u64);
            assert_eq!(v.components(), &comps[offsets[i]..offsets[i + 1]]);
            assert_eq!(v.values(), &codes[offsets[i]..offsets[i + 1]]);
        }
    }

    #[test]
    fn conversion_matches_train_and_push() {
        check_conversion::<u16, u8>(3000);
        check_conversion::<u16, u16>(3000);
        check_conversion::<u32, u8>(200_000);
        check_conversion::<u32, u16>(200_000);
    }

    /// An `f16` source dataset converts exactly like its `f32` twin with the widened values.
    #[test]
    fn conversion_accepts_f16_values() {
        let dim = 3000;
        let d32 = plain::<u16, DotProduct>(dim, 200, 8);
        let mut g16 = PlainSparseDatasetGrowable::new(
            PlainSparseQuantizer::<u16, f16, DotProduct>::new(dim, dim),
        );
        let mut g32 = PlainSparseDatasetGrowable::new(
            PlainSparseQuantizer::<u16, f32, DotProduct>::new(dim, dim),
        );
        for v in d32.iter() {
            let h: Vec<f16> = v.values().iter().map(|&x| f16::from_f32(x)).collect();
            let w: Vec<f32> = h.iter().map(|x| x.to_f32()).collect();
            g16.push(SparseVectorView::new(v.components(), &h));
            g32.push(SparseVectorView::new(v.components(), &w));
        }
        let a = ScalarU16SparseDataset::<u16, DotProduct>::convert_from(
            Into::<PlainSparseDataset<u16, f16, DotProduct>>::into(g16),
            (),
        );
        let b = ScalarU16SparseDataset::<u16, DotProduct>::convert_from(
            Into::<PlainSparseDataset<u16, f32, DotProduct>>::into(g32),
            (),
        );
        assert_eq!(a.encoder().steps(), b.encoder().steps());
        for i in 0..a.len() as u64 {
            assert_eq!(a.get(i).values(), b.get(i).values());
        }
    }
}
