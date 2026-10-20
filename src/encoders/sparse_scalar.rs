use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::core::vector::SparseVectorView;
use crate::core::vector_encoder::{
    DenseVectorOwned, QueryEvaluator, SparseDataEncoder, SparseVectorEncoder, SparseVectorOwned,
    VectorEncoder,
};
use crate::distances::{
    Distance, DotProduct, SquaredEuclideanDistance, dot_product_dense_sparse_batch6_unchecked,
    dot_product_dense_sparse_unchecked, dot_product_sparse_with_merge,
    dot_product_sparse_with_merge_unchecked, squared_euclidean_distance_sparse_with_dense_query,
    squared_euclidean_distance_sparse_with_dense_query_batch6,
    squared_euclidean_distance_sparse_with_merge,
};
use crate::utils::is_strictly_sorted;
use crate::utils::numeric_markers::ToF32Fast;
use crate::{ComponentType, Float, FromF32, SpaceUsage, ValueType};

/// Marker trait for distance types supported by scalar sparse quantizers.
/// Provides the computation method specific to sparse vectors.
pub trait ScalarSparseSupportedDistance: Distance {
    /// Compute distance between a dense query and a sparse encoded vector
    fn requires_dot_query() -> bool {
        false
    }

    fn compute_sparse<C: ComponentType, Q: ValueType, V: ValueType + Float>(
        dense_query: &Option<DenseVectorOwned<Q>>,
        query: &Option<SparseVectorOwned<C, Q>>,
        vector_sparse: SparseVectorView<'_, C, V>,
        dot_query: Option<f32>,
    ) -> Self;

    #[inline]
    fn compute_sparse_batch6<C: ComponentType, Q: ValueType, V: ValueType + Float>(
        dense_query: &Option<DenseVectorOwned<Q>>,
        query: &Option<SparseVectorOwned<C, Q>>,
        vectors: [SparseVectorView<'_, C, V>; 6],
        dot_query: Option<f32>,
    ) -> [Self; 6]
    where
        Self: Sized,
    {
        [
            Self::compute_sparse(dense_query, query, vectors[0], dot_query),
            Self::compute_sparse(dense_query, query, vectors[1], dot_query),
            Self::compute_sparse(dense_query, query, vectors[2], dot_query),
            Self::compute_sparse(dense_query, query, vectors[3], dot_query),
            Self::compute_sparse(dense_query, query, vectors[4], dot_query),
            Self::compute_sparse(dense_query, query, vectors[5], dot_query),
        ]
    }

    /// Compute distance directly between two stored sparse encoded vectors.
    ///
    /// Bypasses evaluator creation entirely — zero allocation, zero decoding.
    /// Used in the pruning hot-path (reverse-link updates, shrink heuristic).
    fn compute_sparse_between<C: ComponentType, V: ValueType + Float>(
        v1: SparseVectorView<'_, C, V>,
        v2: SparseVectorView<'_, C, V>,
    ) -> Self;
}

fn compute_query_squared_norm(values: &[f32]) -> f32 {
    values
        .iter()
        .fold(0.0f32, |acc, &v| acc.algebraic_add(v.algebraic_mul(v)))
}

impl ScalarSparseSupportedDistance for DotProduct {
    #[inline]
    fn compute_sparse<C: ComponentType, Q: ValueType, V: ValueType + Float>(
        dense_query: &Option<DenseVectorOwned<Q>>,
        query: &Option<SparseVectorOwned<C, Q>>,
        vector_sparse: SparseVectorView<'_, C, V>,
        _dot_query: Option<f32>,
    ) -> Self {
        // If query is dense (for small dimensions), use dense-sparse dot product
        if let Some(dense_q) = dense_query {
            // Need a view of dense_q
            let dense_view = dense_q.as_view();

            unsafe { dot_product_dense_sparse_unchecked(dense_view, vector_sparse) }
        } else {
            // Otherwise use sparse-sparse dot product (merge sort style)
            // Assumes both are strictly sorted
            unsafe {
                dot_product_sparse_with_merge_unchecked(
                    query.as_ref().unwrap().as_view(),
                    vector_sparse,
                )
            }
        }
    }

    #[inline]
    fn compute_sparse_batch6<C: ComponentType, Q: ValueType, V: ValueType + Float>(
        dense_query: &Option<DenseVectorOwned<Q>>,
        query: &Option<SparseVectorOwned<C, Q>>,
        vectors: [SparseVectorView<'_, C, V>; 6],
        _dot_query: Option<f32>,
    ) -> [Self; 6] {
        if let Some(dense_q) = dense_query {
            let dense_view = dense_q.as_view();
            unsafe { dot_product_dense_sparse_batch6_unchecked(dense_view, vectors) }
        } else {
            let sparse_view = query.as_ref().unwrap().as_view();
            [
                unsafe { dot_product_sparse_with_merge_unchecked(sparse_view, vectors[0]) },
                unsafe { dot_product_sparse_with_merge_unchecked(sparse_view, vectors[1]) },
                unsafe { dot_product_sparse_with_merge_unchecked(sparse_view, vectors[2]) },
                unsafe { dot_product_sparse_with_merge_unchecked(sparse_view, vectors[3]) },
                unsafe { dot_product_sparse_with_merge_unchecked(sparse_view, vectors[4]) },
                unsafe { dot_product_sparse_with_merge_unchecked(sparse_view, vectors[5]) },
            ]
        }
    }

    #[inline]
    fn compute_sparse_between<C: ComponentType, V: ValueType + Float>(
        v1: SparseVectorView<'_, C, V>,
        v2: SparseVectorView<'_, C, V>,
    ) -> Self {
        unsafe { dot_product_sparse_with_merge_unchecked(v1, v2) }
    }
}

impl ScalarSparseSupportedDistance for SquaredEuclideanDistance {
    #[inline]
    fn requires_dot_query() -> bool {
        true
    }

    #[inline]
    fn compute_sparse<C: ComponentType, Q: ValueType, V: ValueType + Float>(
        dense_query: &Option<DenseVectorOwned<Q>>,
        query: &Option<SparseVectorOwned<C, Q>>,
        vector_sparse: SparseVectorView<'_, C, V>,
        dot_query: Option<f32>,
    ) -> Self {
        let dot_query = dot_query
            .expect("SquaredEuclideanDistance requires a precomputed query dot product value");

        if let Some(dense_q) = dense_query {
            let dense_view = dense_q.as_view();
            squared_euclidean_distance_sparse_with_dense_query(dot_query, dense_view, vector_sparse)
        } else {
            let sparse_view = query.as_ref().unwrap().as_view();
            squared_euclidean_distance_sparse_with_merge(dot_query, sparse_view, vector_sparse)
        }
    }

    #[inline]
    fn compute_sparse_batch6<C: ComponentType, Q: ValueType, V: ValueType + Float>(
        dense_query: &Option<DenseVectorOwned<Q>>,
        query: &Option<SparseVectorOwned<C, Q>>,
        vectors: [SparseVectorView<'_, C, V>; 6],
        dot_query: Option<f32>,
    ) -> [Self; 6] {
        let dot_query = dot_query
            .expect("SquaredEuclideanDistance requires a precomputed query dot product value");

        if let Some(dense_q) = dense_query {
            let dense_view = dense_q.as_view();
            squared_euclidean_distance_sparse_with_dense_query_batch6(
                dot_query, dense_view, vectors,
            )
        } else {
            let sparse_view = query.as_ref().unwrap().as_view();
            [
                squared_euclidean_distance_sparse_with_merge(dot_query, sparse_view, vectors[0]),
                squared_euclidean_distance_sparse_with_merge(dot_query, sparse_view, vectors[1]),
                squared_euclidean_distance_sparse_with_merge(dot_query, sparse_view, vectors[2]),
                squared_euclidean_distance_sparse_with_merge(dot_query, sparse_view, vectors[3]),
                squared_euclidean_distance_sparse_with_merge(dot_query, sparse_view, vectors[4]),
                squared_euclidean_distance_sparse_with_merge(dot_query, sparse_view, vectors[5]),
            ]
        }
    }

    #[inline]
    fn compute_sparse_between<C: ComponentType, V: ValueType + Float>(
        v1: SparseVectorView<'_, C, V>,
        v2: SparseVectorView<'_, C, V>,
    ) -> Self {
        // ||v1 - v2||² = dot(v1,v1) - 2*dot(v1,v2) + dot(v2,v2)
        // We use the existing kernel which takes dot(v1,v1) as the precomputed norm.
        let dot_v1_v1 = dot_product_sparse_with_merge(v1, v1).distance();
        squared_euclidean_distance_sparse_with_merge(dot_v1_v1, v1, v2)
    }
}

/// Scalar Quantizer for Sparse Vectors.
///
/// Quantizes values component-wise using a conversion function (e.g. `f32` -> `u8`).
/// Does not compress indices/components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarSparseQuantizer<C, InValue, OutValue, D> {
    dim: usize,
    _phantom: PhantomData<(C, InValue, OutValue, D)>,
}

impl<C, InValue, OutValue, D> PartialEq for ScalarSparseQuantizer<C, InValue, OutValue, D> {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim
    }
}

pub type PlainSparseQuantizer<C, V, D> = ScalarSparseQuantizer<C, V, V, D>;

/// Convenience aliases for common configurations (currently only DotProduct is supported)
pub type PlainSparseQuantizerDotProduct<C, V> = PlainSparseQuantizer<C, V, DotProduct>;
pub type ScalarSparseQuantizerDotProduct<C, V> = ScalarSparseQuantizer<C, V, V, DotProduct>;

impl<C, InValue, OutValue, D> ScalarSparseQuantizer<C, InValue, OutValue, D> {
    #[inline]
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        assert_eq!(
            input_dim, output_dim,
            "ScalarSparseQuantizer requires input_dim == output_dim"
        );

        Self {
            dim: input_dim,
            _phantom: PhantomData,
        }
    }
}

impl<C, InValue, OutValue, D> SparseDataEncoder for ScalarSparseQuantizer<C, InValue, OutValue, D>
where
    C: ComponentType,
    InValue: ValueType + Float,
    OutValue: ValueType + Float + FromF32,
    D: ScalarSparseSupportedDistance,
{
    type InputComponentType = C;
    type InputValueType = InValue;
    type OutputComponentType = C;
    type OutputValueType = OutValue;

    #[inline]
    fn decode_vector<'a>(
        &self,
        encoded: SparseVectorView<'a, Self::OutputComponentType, Self::OutputValueType>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32> {
        let components = encoded.components().to_vec();
        let values = encoded.values().iter().map(|&v| v.to_f32_fast()).collect();
        SparseVectorOwned::new(components, values)
    }
}

impl<C, InValue, OutValue, D> SparseVectorEncoder for ScalarSparseQuantizer<C, InValue, OutValue, D>
where
    C: ComponentType,
    InValue: ValueType + Float,
    OutValue: ValueType + Float + FromF32,
    D: ScalarSparseSupportedDistance,
{
    fn push_encoded<'a, ComponentContainer, ValueContainer>(
        &self,
        input: SparseVectorView<'a, C, InValue>,
        components: &mut ComponentContainer,
        values: &mut ValueContainer,
    ) where
        ComponentContainer: Extend<Self::OutputComponentType>,
        ValueContainer: Extend<Self::OutputValueType>,
    {
        components.extend(input.components().iter().cloned());
        values.extend(input.values().iter().map(|&in_val| {
            let f32_val = in_val.to_f32().unwrap();
            OutValue::from_f32_saturating(f32_val)
        }));
    }
}

/// The stored values do not depend on the metric.
impl<C, InValue, OutValue, D, D2> crate::core::vector_encoder::WithDistance<D2>
    for ScalarSparseQuantizer<C, InValue, OutValue, D>
where
    C: ComponentType,
    InValue: ValueType + Float,
    OutValue: ValueType + Float + FromF32,
    D: ScalarSparseSupportedDistance,
    D2: ScalarSparseSupportedDistance,
{
    type Output = ScalarSparseQuantizer<C, InValue, OutValue, D2>;

    fn with_distance(&self) -> Self::Output {
        ScalarSparseQuantizer::new(self.dim, self.dim)
    }
}

impl<C, InValue, OutValue, D> VectorEncoder for ScalarSparseQuantizer<C, InValue, OutValue, D>
where
    C: ComponentType,
    InValue: ValueType + Float,
    OutValue: ValueType + Float + FromF32,
    D: ScalarSparseSupportedDistance,
{
    type Distance = D;
    type InputVector<'a> = SparseVectorView<'a, C, InValue>;
    type QueryVector<'q> = SparseVectorView<'q, C, f32>;
    type EncodedVector<'a> = SparseVectorView<'a, C, OutValue>;

    type Evaluator<'e>
        = ScalarSparseQueryEvaluator<C, OutValue, D>
    where
        Self: 'e;

    type QueryParams = ();

    fn query_evaluator<'e>(
        &'e self,
        query: Self::QueryVector<'_>,
        _params: &(),
    ) -> Self::Evaluator<'e> {
        ScalarSparseQueryEvaluator::new(query, self)
    }

    fn vector_evaluator<'e>(&'e self, vector: Self::EncodedVector<'e>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        ScalarSparseQueryEvaluator::new_from_owned_query(decoded, self)
    }

    /// Compute distance between two stored sparse encoded vectors without building an evaluator.
    ///
    /// Calls the sparse distance kernel directly on the two raw sparse views,
    /// zero allocation, zero decoding.
    #[inline]
    fn compute_distance_between(
        &self,
        v1: Self::EncodedVector<'_>,
        v2: Self::EncodedVector<'_>,
    ) -> Self::Distance {
        D::compute_sparse_between(v1, v2)
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

/// Query evaluator for ScalarSparseQuantizer.
#[derive(Debug, Clone)]
pub struct ScalarSparseQueryEvaluator<C, OutValue, D>
where
    C: ComponentType,
    OutValue: ValueType + Float + FromF32,
    D: ScalarSparseSupportedDistance,
{
    dense_query: Option<DenseVectorOwned<f32>>,
    sparse_query: Option<SparseVectorOwned<C, f32>>,
    dot_query: Option<f32>,
    _phantom: PhantomData<(OutValue, D)>,
}

impl<C, OutValue, D> ScalarSparseQueryEvaluator<C, OutValue, D>
where
    C: ComponentType,
    OutValue: ValueType + Float + FromF32,
    D: ScalarSparseSupportedDistance,
{
    #[inline]
    pub fn new_from_owned_query<InValue>(
        query: SparseVectorOwned<C, f32>,
        quantizer: &ScalarSparseQuantizer<C, InValue, OutValue, D>,
    ) -> Self
    where
        InValue: ValueType + Float,
    {
        let dot_query = if D::requires_dot_query() {
            Some(compute_query_squared_norm(query.values()))
        } else {
            None
        };

        let max_c = query
            .components()
            .iter()
            .map(|c| c.as_())
            .max()
            .unwrap_or(0);

        assert!(
            max_c < quantizer.input_dim(),
            "Query vector component exceeds quantizer input dimension."
        );

        let small_dim = quantizer.input_dim() < 2_usize.pow(20);

        if small_dim {
            let mut dense_query_vec = vec![0.0f32; quantizer.input_dim()];
            for (&i, &v) in query.components().iter().zip(query.values().iter()) {
                dense_query_vec[i.as_()] = v;
            }

            Self {
                dense_query: Some(DenseVectorOwned::new(dense_query_vec)),
                sparse_query: None,
                dot_query,
                _phantom: PhantomData,
            }
        } else {
            assert!(
                is_strictly_sorted(query.components()),
                "Query components must be sorted in strictly ascending order."
            );

            Self {
                dense_query: None,
                sparse_query: Some(query),
                dot_query,
                _phantom: PhantomData,
            }
        }
    }

    pub fn new<InValue>(
        query: SparseVectorView<'_, C, f32>,
        quantizer: &ScalarSparseQuantizer<C, InValue, OutValue, D>,
    ) -> Self
    where
        InValue: ValueType + Float,
    {
        let dot_query = if D::requires_dot_query() {
            Some(compute_query_squared_norm(query.values()))
        } else {
            None
        };

        let max_c = query
            .components()
            .iter()
            .map(|c| c.as_())
            .max()
            .unwrap_or(0);

        assert!(
            max_c < quantizer.input_dim(),
            "Query vector component exceeds quantizer input dimension."
        );

        assert_eq!(
            query.components().len(),
            query.values().len(),
            "Query vector components and values length mismatch."
        );

        let small_dim = quantizer.input_dim() < 2_usize.pow(20);

        // Build a dense query only for small dimensionalities.
        // For large dimensionalities we keep a sparse representation and use merge-based computation.
        let (dense_query, sparse_query) = if small_dim {
            let values_f32 = query.values();

            let mut dense_query_vec = vec![0.0f32; quantizer.input_dim()];
            for (&i, &v) in query.components().iter().zip(values_f32.iter()) {
                dense_query_vec[i.as_()] = v;
            }

            (Some(DenseVectorOwned::new(dense_query_vec)), None)
        } else {
            assert!(
                is_strictly_sorted(query.components()),
                "Query components must be sorted in strictly ascending order."
            );

            (
                None,
                Some(SparseVectorOwned::new(
                    query.components().to_vec(),
                    query.values().to_vec(),
                )),
            )
        };

        Self {
            dense_query,
            sparse_query,
            dot_query,
            _phantom: PhantomData,
        }
    }
}

impl<'v, C, OutValue, D> QueryEvaluator<SparseVectorView<'v, C, OutValue>>
    for ScalarSparseQueryEvaluator<C, OutValue, D>
where
    C: ComponentType,
    OutValue: ValueType + Float + FromF32,
    D: ScalarSparseSupportedDistance,
{
    type Distance = D;

    #[inline]
    fn compute_distance(&self, vector: SparseVectorView<'v, C, OutValue>) -> D {
        debug_assert!(self.dense_query.is_some() || self.sparse_query.is_some());

        D::compute_sparse(
            &self.dense_query,
            &self.sparse_query,
            vector,
            self.dot_query,
        )
    }

    #[inline]
    fn compute_distances_batch6(&self, vectors: [SparseVectorView<'v, C, OutValue>; 6]) -> [D; 6] {
        debug_assert!(self.dense_query.is_some() || self.sparse_query.is_some());

        D::compute_sparse_batch6(
            &self.dense_query,
            &self.sparse_query,
            vectors,
            self.dot_query,
        )
    }
}

impl<C, InValue, OutValue, D> SpaceUsage for ScalarSparseQuantizer<C, InValue, OutValue, D>
where
    C: ComponentType,
    InValue: ValueType,
    OutValue: ValueType,
    D: ScalarSparseSupportedDistance,
{
    fn space_usage_bytes(&self) -> usize {
        self.dim.space_usage_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vector::SparseVectorView;
    use crate::distances::DotProduct;

    #[test]
    fn scalar_sparse_quantizer_encode_vector_default() {
        type Quant = PlainSparseQuantizer<u16, f32, DotProduct>;
        let quant = Quant::new(3, 3);
        let input = SparseVectorView::new(&[0_u16, 2], &[1.0_f32, 2.5]);
        let encoded = quant.encode_vector(input);
        assert_eq!(encoded.components(), &[0_u16, 2]);
        assert_eq!(encoded.values(), &[1.0_f32, 2.5]);
    }

    #[test]
    fn compute_distance_between_matches_vector_evaluator() {
        type Quant = PlainSparseQuantizer<u16, f32, DotProduct>;
        let quant = Quant::new(4, 4);
        let v1 = SparseVectorView::new(&[0_u16, 2], &[1.0_f32, 3.0]);
        let v2 = SparseVectorView::new(&[0_u16, 2], &[2.0_f32, 1.0]);
        // Default path: vector_evaluator decodes v1 into a dense f32 query.
        let via_evaluator = quant.vector_evaluator(v1).compute_distance(v2);
        // Override path: calls D::compute_sparse_between directly, zero allocation.
        let direct = quant.compute_distance_between(v1, v2);
        assert_eq!(via_evaluator, direct);
    }

    #[test]
    fn compute_distance_between_disjoint_components_is_zero() {
        type Quant = PlainSparseQuantizer<u16, f32, DotProduct>;
        let quant = Quant::new(4, 4);
        let v1 = SparseVectorView::new(&[0_u16], &[5.0_f32]);
        let v2 = SparseVectorView::new(&[3_u16], &[7.0_f32]);
        let result = quant.compute_distance_between(v1, v2);
        assert_eq!(result, DotProduct::from(0.0));
    }
}
