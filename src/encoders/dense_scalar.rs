//! Scalar dense quantizers that simply rescale floats between numeric types.
//! These modules back the dense datasets and keep distances in `f32` for accurate queries.
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::core::distances::{
    Distance, DotProduct, SquaredEuclideanDistance, dot_product_dense_batch6_unchecked,
    dot_product_dense_unchecked, squared_euclidean_distance_dense_batch6_unchecked,
    squared_euclidean_distance_dense_unchecked,
};
use crate::core::vector::DenseVectorView;
use crate::core::vector_encoder::{
    DenseVectorEncoder, DenseVectorOwned, QueryEvaluator, VectorEncoder,
};
use crate::utils::numeric_markers::ToF32Fast;
use crate::{Float, FromF32, SpaceUsage, ValueType};

/// Marker trait for distance types supported by scalar dense quantizers.
/// Provides computation methods specific to dense vectors.
pub trait ScalarDenseSupportedDistance: Distance {
    /// Compute the distance between a query and one stored dense vector.
    fn compute_dense<Q: ValueType, V: ValueType>(
        query: DenseVectorView<'_, Q>,
        vector: DenseVectorView<'_, V>,
    ) -> Self;

    /// Compute the distance between a query and six stored dense vectors in a single pass.
    ///
    /// Default falls back to six independent `compute_dense` calls. Override for a single-pass
    /// kernel with 6 independent accumulators (better ILP and cache reuse).
    fn compute_dense_batch6<Q: ValueType, V: ValueType>(
        query: DenseVectorView<'_, Q>,
        vectors: [DenseVectorView<'_, V>; 6],
    ) -> [Self; 6] {
        vectors.map(|v| Self::compute_dense(query, v))
    }
}

impl ScalarDenseSupportedDistance for SquaredEuclideanDistance {
    #[inline]
    fn compute_dense<Q: ValueType, V: ValueType>(
        query: DenseVectorView<'_, Q>,
        vector: DenseVectorView<'_, V>,
    ) -> Self {
        unsafe { squared_euclidean_distance_dense_unchecked(query, vector) }
    }

    #[inline]
    fn compute_dense_batch6<Q: ValueType, V: ValueType>(
        query: DenseVectorView<'_, Q>,
        vectors: [DenseVectorView<'_, V>; 6],
    ) -> [Self; 6] {
        unsafe { squared_euclidean_distance_dense_batch6_unchecked(query, vectors) }
    }
}

impl ScalarDenseSupportedDistance for DotProduct {
    #[inline]
    fn compute_dense<Q: ValueType, V: ValueType>(
        query: DenseVectorView<'_, Q>,
        vector: DenseVectorView<'_, V>,
    ) -> Self {
        unsafe { dot_product_dense_unchecked(query, vector) }
    }

    #[inline]
    fn compute_dense_batch6<Q: ValueType, V: ValueType>(
        query: DenseVectorView<'_, Q>,
        vectors: [DenseVectorView<'_, V>; 6],
    ) -> [Self; 6] {
        unsafe { dot_product_dense_batch6_unchecked(query, vectors) }
    }
}

/// A scalar quantizer that casts each component from `In` to `Out`.
/// The quantizer stores `d` to keep the input/output dimensionality in sync.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarDenseQuantizer<In, Out, D> {
    d: usize,
    _phantom: PhantomData<(In, Out, D)>,
}

pub type ScalarDenseQuantizerSame<V, D> = ScalarDenseQuantizer<V, V, D>;
pub type PlainDenseQuantizer<V, D> = ScalarDenseQuantizer<V, V, D>;
pub type PlainDenseQuantizerSquaredEuclidean<V> = PlainDenseQuantizer<V, SquaredEuclideanDistance>;
pub type PlainDenseQuantizerDotProduct<V> = PlainDenseQuantizer<V, DotProduct>;
pub type ScalarDenseQuantizerSquaredEuclidean<V> =
    ScalarDenseQuantizer<V, V, SquaredEuclideanDistance>;
pub type ScalarDenseQuantizerDotProduct<V> = ScalarDenseQuantizer<V, V, DotProduct>;

impl<In, Out, D> ScalarDenseQuantizer<In, Out, D> {
    /// Build a quantizer that keeps the input and output dimensionality equal.
    #[inline]
    pub fn new(d: usize) -> Self {
        Self {
            d,
            _phantom: PhantomData,
        }
    }
}

impl<In, Out, D> DenseVectorEncoder for ScalarDenseQuantizer<In, Out, D>
where
    In: ValueType + Float,
    Out: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
{
    type InputValueType = In;
    type OutputValueType = Out;

    /// Decode any encoded vector into `DenseVectorOwned<f32>` by converting each element to `f32`.
    #[inline]
    fn decode_vector<'a>(
        &self,
        encoded: DenseVectorView<'a, Self::OutputValueType>,
    ) -> DenseVectorOwned<f32> {
        let values = encoded.values().iter().map(|&v| v.to_f32_fast()).collect();
        DenseVectorOwned::new(values)
    }

    #[inline]
    fn decode_into<'a>(
        &self,
        encoded: DenseVectorView<'a, Self::OutputValueType>,
        out: &mut [f32],
    ) {
        for (o, &v) in out.iter_mut().zip(encoded.values()) {
            *o = v.to_f32_fast();
        }
    }

    /// Encode the input view into an existing buffer to avoid temporary allocations.
    #[inline]
    fn push_encoded<'a, OutputContainer>(
        &self,
        input: DenseVectorView<'a, Self::InputValueType>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<Self::OutputValueType>,
    {
        output.extend(input.values().iter().map(|&in_val| {
            let f32_val = in_val.to_f32().unwrap();
            Out::from_f32_saturating(f32_val)
        }));
    }
}

/// Evaluator that computes distances between a frozen query (owned `f32` vector) and encoded vectors.
#[derive(Debug, Clone)]
pub struct ScalarDenseQueryEvaluator<'e, In, Out, D>
where
    In: ValueType,
    Out: ValueType + FromF32,
    D: ScalarDenseSupportedDistance,
{
    _encoder: std::marker::PhantomData<&'e ScalarDenseQuantizer<In, Out, D>>,
    query: DenseVectorOwned<f32>,
}

impl<'e, In, Out, D> ScalarDenseQueryEvaluator<'e, In, Out, D>
where
    In: ValueType,
    Out: ValueType + FromF32,
    D: ScalarDenseSupportedDistance,
{
    #[inline]
    pub fn new(
        _encoder: &'e ScalarDenseQuantizer<In, Out, D>,
        query: DenseVectorOwned<f32>,
    ) -> Self {
        Self {
            _encoder: std::marker::PhantomData,
            query,
        }
    }
}

impl<'e, 'v, In, Out, D> QueryEvaluator<DenseVectorView<'v, Out>>
    for ScalarDenseQueryEvaluator<'e, In, Out, D>
where
    In: ValueType,
    Out: ValueType + FromF32,
    D: ScalarDenseSupportedDistance,
{
    type Distance = D;

    #[inline]
    fn compute_distance(&self, vector: DenseVectorView<'v, Out>) -> D {
        D::compute_dense(self.query.as_view(), vector)
    }

    #[inline]
    fn compute_distances_batch6(&self, vectors: [DenseVectorView<'v, Out>; 6]) -> [D; 6] {
        D::compute_dense_batch6(self.query.as_view(), vectors)
    }
}

/// The stored values do not depend on the metric.
impl<In, Out, D, D2> crate::core::vector_encoder::WithDistance<D2>
    for ScalarDenseQuantizer<In, Out, D>
where
    In: ValueType + Float,
    Out: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
    D2: ScalarDenseSupportedDistance,
{
    type Output = ScalarDenseQuantizer<In, Out, D2>;

    fn with_distance(&self) -> Self::Output {
        ScalarDenseQuantizer::new(self.d)
    }
}

impl<In, Out, D> VectorEncoder for ScalarDenseQuantizer<In, Out, D>
where
    In: ValueType + Float,
    Out: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
{
    type Distance = D;
    type InputVector<'a> = DenseVectorView<'a, In>;
    type QueryVector<'q> = DenseVectorView<'q, f32>;
    type EncodedVector<'a> = DenseVectorView<'a, Out>;

    type Evaluator<'e>
        = ScalarDenseQueryEvaluator<'e, In, Out, D>
    where
        Self: 'e;

    /// Build an evaluator from a dense `f32` query.
    type QueryParams = ();

    #[inline]
    fn query_evaluator<'e>(
        &'e self,
        query: Self::QueryVector<'_>,
        _params: &(),
    ) -> Self::Evaluator<'e> {
        assert_eq!(
            query.len(),
            self.input_dim(),
            "Query vector length exceeds quantizer input dimension."
        );
        ScalarDenseQueryEvaluator::new(self, query.to_owned())
    }

    /// Build an evaluator from an encoded vector stored in the dataset.
    #[inline]
    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as DenseVectorEncoder>::decode_vector(self, vector);
        ScalarDenseQueryEvaluator::new(self, decoded)
    }

    fn input_dim(&self) -> usize {
        self.d
    }

    fn output_dim(&self) -> usize {
        self.d
    }

    /// Compute distance between two stored encoded vectors without any allocation or decoding.
    ///
    /// For ScalarDenseQuantizer both vectors are already in the stored type `Out` (e.g. f32
    /// or f16). The distance kernel accepts mixed-type views, so we pass both raw slices
    /// directly. This eliminates the Vec allocation that the default via `vector_evaluator`
    /// would incur that may be expensive in hot code paths.
    #[inline]
    fn compute_distance_between(
        &self,
        v1: Self::EncodedVector<'_>,
        v2: Self::EncodedVector<'_>,
    ) -> Self::Distance {
        D::compute_dense(v1, v2)
    }
}

impl<In, Out, D> SpaceUsage for ScalarDenseQuantizer<In, Out, D>
where
    In: ValueType + Float,
    Out: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
{
    fn space_usage_bytes(&self) -> usize {
        self.d.space_usage_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vector::DenseVectorView;
    use crate::distances::DotProduct;

    #[test]
    fn scalar_dense_quantizer_encodes_and_decodes_values() {
        type Quant = ScalarDenseQuantizer<f32, f32, DotProduct>;
        let quant = Quant::new(2);

        let input = DenseVectorView::new(&[0.5f32, 1.5]);
        let mut buffer = Vec::new();
        quant.push_encoded(input, &mut buffer);
        assert_eq!(buffer, vec![0.5f32, 1.5]);

        let encoded = quant.encode_vector(input);
        assert_eq!(encoded.values(), &[0.5f32, 1.5]);

        let decoded = quant.decode_vector(encoded.as_view());
        assert_eq!(decoded.values(), &[0.5f32, 1.5]);
    }

    #[test]
    fn compute_distance_between_matches_vector_evaluator_squared_euclidean() {
        type Quant = ScalarDenseQuantizer<f32, f32, SquaredEuclideanDistance>;
        let quant = Quant::new(3);
        let v1 = DenseVectorView::new(&[1.0f32, 2.0, 3.0]);
        let v2 = DenseVectorView::new(&[4.0f32, 5.0, 6.0]);
        let via_evaluator = quant.vector_evaluator(v1).compute_distance(v2);
        let direct = quant.compute_distance_between(v1, v2);
        assert_eq!(via_evaluator, direct);
    }

    #[test]
    fn compute_distance_between_matches_vector_evaluator_dot_product() {
        type Quant = ScalarDenseQuantizer<f32, f32, DotProduct>;
        let quant = Quant::new(3);
        let v1 = DenseVectorView::new(&[1.0f32, 0.0, 0.5]);
        let v2 = DenseVectorView::new(&[2.0f32, 1.0, 3.0]);
        let via_evaluator = quant.vector_evaluator(v1).compute_distance(v2);
        let direct = quant.compute_distance_between(v1, v2);
        assert_eq!(via_evaluator, direct);
    }

    #[test]
    fn compute_distances_batch6_matches_six_compute_distance_calls() {
        type Quant = ScalarDenseQuantizer<f32, f32, SquaredEuclideanDistance>;
        let quant = Quant::new(4);
        let query = DenseVectorView::new(&[1.0f32, 2.0, 3.0, 4.0]);
        let evaluator = quant.query_evaluator(query, &());

        let v0 = DenseVectorView::new(&[0.0f32, 0.0, 0.0, 0.0]);
        let v1 = DenseVectorView::new(&[1.0f32, 1.0, 1.0, 1.0]);
        let v2 = DenseVectorView::new(&[2.0f32, 2.0, 2.0, 2.0]);
        let v3 = DenseVectorView::new(&[3.0f32, 3.0, 3.0, 3.0]);
        let v4 = DenseVectorView::new(&[0.5f32, 1.5, 2.5, 3.5]);
        let v5 = DenseVectorView::new(&[1.5f32, 1.5, 1.5, 1.5]);

        let batch = evaluator.compute_distances_batch6([v0, v1, v2, v3, v4, v5]);
        let singles = [
            evaluator.compute_distance(v0),
            evaluator.compute_distance(v1),
            evaluator.compute_distance(v2),
            evaluator.compute_distance(v3),
            evaluator.compute_distance(v4),
            evaluator.compute_distance(v5),
        ];
        assert_eq!(batch, singles);
    }

    #[test]
    fn compute_distances_batch6_dot_product_matches_singles() {
        type Quant = ScalarDenseQuantizer<f32, f32, DotProduct>;
        let quant = Quant::new(2);
        let query = DenseVectorView::new(&[1.0f32, 2.0]);
        let evaluator = quant.query_evaluator(query, &());

        let v0 = DenseVectorView::new(&[3.0f32, 4.0]);
        let v1 = DenseVectorView::new(&[0.5f32, 0.5]);
        let v2 = DenseVectorView::new(&[-1.0f32, 1.0]);
        let v3 = DenseVectorView::new(&[2.0f32, 0.0]);
        let v4 = DenseVectorView::new(&[1.0f32, 1.0]);
        let v5 = DenseVectorView::new(&[0.0f32, 3.0]);

        let batch = evaluator.compute_distances_batch6([v0, v1, v2, v3, v4, v5]);
        let singles = [
            evaluator.compute_distance(v0),
            evaluator.compute_distance(v1),
            evaluator.compute_distance(v2),
            evaluator.compute_distance(v3),
            evaluator.compute_distance(v4),
            evaluator.compute_distance(v5),
        ];
        assert_eq!(batch, singles);
    }

    #[test]
    fn scalar_dense_query_and_vector_evaluator_use_dot_product() {
        type Quant = ScalarDenseQuantizer<f32, f32, DotProduct>;
        let quant = Quant::new(2);
        let query = DenseVectorView::new(&[1.0f32, 2.0]);

        let evaluator = quant.query_evaluator(query, &());
        let vector = DenseVectorView::new(&[3.0f32, 4.0]);
        assert_eq!(evaluator.compute_distance(vector), DotProduct::from(11.0));

        let encoded = quant.encode_vector(DenseVectorView::new(&[1.0f32, 1.0]));
        let vector_eval = quant.vector_evaluator(encoded.as_view());
        assert_eq!(
            vector_eval.compute_distance(DenseVectorView::new(&[2.0f32, 0.0])),
            DotProduct::from(2.0)
        );
    }
}
