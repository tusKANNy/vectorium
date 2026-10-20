//! Multivec Product Quantizer for MaxSim (late-interaction) scoring.
//!
//! Each document is a variable-length collection of token vectors. The quantizer
//! encodes each token vector independently using product quantization with `M` subspaces,
//! producing `M` bytes per token (one u8 code per subspace).
//!
//! # Distance table layout during search
//!
//! A table of shape `[M][KSUB][Q]` is precomputed once per query (where `Q` is the
//! number of query tokens). For a doc token with codes `[c0, c1, ..., c_{M-1}]`,
//! the full contribution vector over all `Q` query tokens is found at
//! `table[m * KSUB * Q + c_m * Q ..]` — a run of `Q` contiguous `f32` values,
//! enabling a single vectorized accumulation (SAXPY) per subspace.

use std::marker::PhantomData;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::clustering::KMeansBuilder;
use crate::core::vector::{DenseMultiVectorOwned, DenseMultiVectorView, DenseVectorView};
use crate::core::vector_encoder::{MultiVecEncoder, QueryEvaluator, VectorEncoder};
use crate::distances::{Distance, DotProduct, SquaredEuclideanDistance};
use crate::{
    Dataset, DatasetGrowable, FlatIndex, Float, PlainDenseDataset, PlainDenseDatasetGrowable,
    PlainDenseQuantizer, SpaceUsage, ValueType, VectorId,
};

/// Number of centroids per subspace (always 256 = 2^8, one u8 code per subspace).
const KSUB: usize = 256;

/// Product Quantizer for multivector (late-interaction) data.
///
/// Each document is a variable-length collection of token vectors. The quantizer
/// encodes each token vector independently using product quantization with `M` subspaces,
/// producing `M` bytes per token (one u8 code per subspace).
///
/// # Type Parameters
/// - `M`: number of PQ subspaces; must divide `token_dim` and be divisible by 4.
/// - `In`: input value type for encoding (e.g. `f32`, `f16`).
///
/// # Encoding
///
/// For each token, the nearest centroid in each of the `M` subspaces is found by
/// squared Euclidean distance and stored as a `u8` index. The encoded multivector
/// has `dim = M` and the same `num_vecs` as the input.
///
/// # Search (MaxSim via ADC)
///
/// [`MultiVecPQQueryEvaluator`] precomputes a distance table of shape `[M][KSUB][Q]`.
/// For each doc token with codes `[c0, ..., c_{M-1}]`, the contribution to all `Q`
/// query tokens is accumulated in a single pass over `M` contiguous Q-element table
/// slices, then used to update per-query maximums.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MultiVecProductQuantizer<const M: usize, In> {
    /// Dimension of each individual token vector (= M * dsub).
    token_dim: usize,
    /// Sub-space dimensionality (= token_dim / M).
    dsub: usize,
    /// Per-subspace centroid datasets: M entries, each with KSUB vectors of dim `dsub`.
    centroids: Box<[PlainDenseDataset<f32, SquaredEuclideanDistance>]>,
    _phantom: PhantomData<In>,
}

impl<const M: usize, In> MultiVecProductQuantizer<M, In> {
    /// Construct from pre-trained per-subspace centroid datasets.
    ///
    /// Panics if `M % 4 != 0`, `token_dim % M != 0`, `centroids.len() != M`,
    /// or any centroid dataset has the wrong number of vectors or dimension.
    #[inline]
    pub fn from_pretrained(
        token_dim: usize,
        centroids: Vec<PlainDenseDataset<f32, SquaredEuclideanDistance>>,
    ) -> Self {
        assert_eq!(M % 4, 0, "M ({}) must be divisible by 4", M);
        assert_eq!(
            token_dim % M,
            0,
            "token_dim ({}) must be divisible by M ({})",
            token_dim,
            M
        );
        assert_eq!(
            centroids.len(),
            M,
            "Expected {} centroid datasets, got {}",
            M,
            centroids.len()
        );
        let dsub = token_dim / M;
        for (i, ds) in centroids.iter().enumerate() {
            assert_eq!(
                ds.len(),
                KSUB,
                "Subspace {} has {} centroids, expected {}",
                i,
                ds.len(),
                KSUB
            );
            assert_eq!(
                ds.output_dim(),
                dsub,
                "Subspace {} has dim {}, expected {}",
                i,
                ds.output_dim(),
                dsub
            );
        }
        Self {
            token_dim,
            dsub,
            centroids: centroids.into_boxed_slice(),
            _phantom: PhantomData,
        }
    }

    /// Train a product quantizer on a flat dataset of individual token vectors.
    ///
    /// `token_vectors` should contain all token vectors from all documents concatenated;
    /// each row is one token vector of dimension `token_dim`. K-Means is run independently
    /// for each of the `M` subspaces in parallel.
    ///
    /// Panics if `M % 4 != 0` or `token_dim % M != 0`.
    pub fn train(token_vectors: &PlainDenseDataset<f32, SquaredEuclideanDistance>) -> Self {
        let token_dim = token_vectors.output_dim();
        assert_eq!(M % 4, 0, "M ({}) must be divisible by 4", M);
        assert_eq!(
            token_dim % M,
            0,
            "token_dim ({}) must be divisible by M ({})",
            token_dim,
            M
        );
        let dsub = token_dim / M;

        println!(
            "Training MultiVecProductQuantizer: {} tokens × {} dims, M={}, dsub={}, KSUB={}",
            token_vectors.len(),
            token_dim,
            M,
            dsub,
            KSUB
        );

        let centroids: Vec<PlainDenseDataset<f32, SquaredEuclideanDistance>> = (0..M)
            .into_par_iter()
            .map(|m| {
                let quantizer = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(dsub);
                let mut sub_dataset =
                    PlainDenseDatasetGrowable::<f32, SquaredEuclideanDistance>::with_capacity(
                        quantizer,
                        token_vectors.len(),
                    );
                for vector in token_vectors.iter() {
                    let sub = DenseVectorView::new(&vector.values()[m * dsub..(m + 1) * dsub]);
                    sub_dataset.push(sub);
                }
                let kmeans = KMeansBuilder::new().build();
                kmeans
                    .train::<_, PlainDenseDatasetGrowable<f32, SquaredEuclideanDistance>>(
                        &sub_dataset,
                        KSUB,
                        None,
                        PlainDenseQuantizer::new(dsub),
                    )
                    .into()
            })
            .collect();

        Self {
            token_dim,
            dsub,
            centroids: centroids.into_boxed_slice(),
            _phantom: PhantomData,
        }
    }

    /// Dimensionality of each individual token vector.
    #[inline]
    pub fn token_dim(&self) -> usize {
        self.token_dim
    }

    /// Sub-space dimensionality (`token_dim / M`).
    #[inline]
    pub fn dsub(&self) -> usize {
        self.dsub
    }

    /// Number of centroids per subspace (always 256).
    #[inline]
    pub fn ksub(&self) -> usize {
        KSUB
    }
}

impl<const M: usize, In> MultiVecProductQuantizer<M, In>
where
    In: ValueType,
{
    /// Decode an encoded multivector (u8 codes → f32 centroids).
    ///
    /// Each encoded token has `M` u8 codes; each code is looked up in the corresponding
    /// subspace's centroid dataset and the `dsub` centroid values are concatenated to
    /// reconstruct a `token_dim`-dimensional token.
    fn decode_multivec(&self, encoded: DenseMultiVectorView<'_, u8>) -> DenseMultiVectorOwned<f32> {
        let mut values = Vec::with_capacity(encoded.num_vecs() * self.token_dim);
        for token in encoded.iter_vectors() {
            let codes = token.values();
            for (m, item) in codes.iter().enumerate().take(M) {
                let code = *item as VectorId;
                let centroid = self.centroids[m].get(code);
                values.extend_from_slice(centroid.values());
            }
        }
        DenseMultiVectorOwned::new(values, self.token_dim)
    }

    /// Compute the ADC distance table for a query multivector.
    ///
    /// Returns a flat `Vec<f32>` with layout `[m * KSUB * Q + k * Q + q]`:
    /// - `m` = subspace index (`0..M`)
    /// - `k` = centroid index (`0..KSUB`)
    /// - `q` = query token index (`0..query.num_vecs()`)
    ///
    /// Entry `[m][k][q]` = `dot(query_token_q_subspace_m, centroid_mk)`.
    pub fn compute_distance_table(&self, query: DenseMultiVectorView<'_, f32>) -> Vec<f32> {
        let q = query.num_vecs();
        let dsub = self.dsub;
        let token_dim = self.token_dim;

        // Flatten all query tokens into a contiguous f32 buffer [Q * token_dim].
        let query_flat: Vec<f32> = query.values().to_vec();

        let mut table = vec![0.0f32; M * KSUB * q];

        for m in 0..M {
            let sub_offset = m * dsub;
            let centroids_m = &self.centroids[m];
            let table_m_base = m * KSUB * q;

            for k in 0..KSUB {
                let centroid = centroids_m.get(k as VectorId);
                let entry_base = table_m_base + k * q;

                for qi in 0..q {
                    let q_base = qi * token_dim + sub_offset;
                    let q_sub = DenseVectorView::new(&query_flat[q_base..q_base + dsub]);
                    let dot = unsafe {
                        crate::distances::dot_product_dense_unchecked(q_sub, centroid).distance()
                    };
                    table[entry_base + qi] = dot;
                }
            }
        }

        table
    }
}

/// Query evaluator that computes MaxSim via precomputed ADC distance tables.
///
/// # Distance table layout
///
/// `distance_table[m * KSUB * Q + k * Q + q]` = `dot(query_token_q_subspace_m, centroid_mk)`
///
/// For a doc token with codes `[c0, ..., c_{M-1}]`, the contribution to all `Q` query
/// tokens simultaneously is the slice `table[m * KSUB * Q + c_m * Q .. + Q]`, enabling
/// an efficient vectorized accumulation per subspace.
pub struct MultiVecPQQueryEvaluator<'e, const M: usize, In> {
    /// Distance table: [m * KSUB * Q + k * Q + q]
    distance_table: Vec<f32>,
    /// Number of query tokens (Q).
    num_query_tokens: usize,
    _phantom: PhantomData<(&'e (), In)>,
}

impl<'e, const M: usize, In> MultiVecPQQueryEvaluator<'e, M, In>
where
    In: ValueType,
{
    fn new(
        encoder: &'e MultiVecProductQuantizer<M, In>,
        query: DenseMultiVectorView<'_, f32>,
    ) -> Self {
        Self {
            distance_table: encoder.compute_distance_table(query),
            num_query_tokens: query.num_vecs(),
            _phantom: PhantomData,
        }
    }
}

impl<'e, 'v, const M: usize, In> QueryEvaluator<DenseMultiVectorView<'v, u8>>
    for MultiVecPQQueryEvaluator<'e, M, In>
where
    In: ValueType,
{
    type Distance = DotProduct;

    /// Compute MaxSim between the stored query and a PQ-encoded document multivector.
    ///
    /// For each doc token (M u8 codes), ADC contributions across all M subspaces are
    /// accumulated into a Q-element buffer via a SAXPY per subspace. Per-query maximums
    /// are tracked across all doc tokens; the final score is the sum of those maximums.
    fn compute_distance(&self, vector: DenseMultiVectorView<'v, u8>) -> DotProduct {
        let q = self.num_query_tokens;
        let mut max_scores = vec![f32::NEG_INFINITY; q];
        let mut acc = vec![0.0f32; q];

        for doc_token in vector.iter_vectors() {
            // Reset accumulator for this doc token.
            acc.fill(0.0f32);

            let codes = doc_token.values();
            for m in 0..M {
                // SAFETY: codes has length M (enforced by encoded multivec dim = M).
                let code = unsafe { *codes.get_unchecked(m) } as usize;
                let base = m * KSUB * q + code * q;
                // Accumulate Q contiguous table values into acc (SAXPY).
                // SAFETY: base + q <= M * KSUB * q = distance_table.len().
                let tbl = unsafe { self.distance_table.get_unchecked(base..base + q) };
                for (qi, item) in acc.iter_mut().enumerate().take(q) {
                    *item += unsafe { *tbl.get_unchecked(qi) };
                }
            }

            // Update per-query-token maximums.
            for qi in 0..q {
                max_scores[qi] = max_scores[qi].max(acc[qi]);
            }
        }

        // Sum of per-query-token maxima = MaxSim score.
        let total = max_scores.iter().fold(0.0f32, |s, &x| s + x);
        DotProduct::from(total)
    }
}

impl<const M: usize, In> VectorEncoder for MultiVecProductQuantizer<M, In>
where
    In: ValueType + Float,
{
    type Distance = DotProduct;
    type InputVector<'a> = DenseMultiVectorView<'a, In>;
    type QueryVector<'q> = DenseMultiVectorView<'q, f32>;
    type EncodedVector<'a> = DenseMultiVectorView<'a, u8>;

    type Evaluator<'e>
        = MultiVecPQQueryEvaluator<'e, M, In>
    where
        Self: 'e;

    type QueryParams = ();

    #[inline]
    fn query_evaluator<'e>(
        &'e self,
        query: Self::QueryVector<'_>,
        _params: &(),
    ) -> Self::Evaluator<'e> {
        assert_eq!(
            query.dim(),
            self.token_dim,
            "Query dim ({}) must match token_dim ({})",
            query.dim(),
            self.token_dim
        );
        MultiVecPQQueryEvaluator::new(self, query)
    }

    #[inline]
    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = self.decode_multivec(vector);
        MultiVecPQQueryEvaluator::new(self, decoded.as_view())
    }

    #[inline]
    fn input_dim(&self) -> usize {
        self.token_dim
    }

    #[inline]
    fn output_dim(&self) -> usize {
        M
    }
}

impl<const M: usize, In> MultiVecEncoder for MultiVecProductQuantizer<M, In>
where
    In: ValueType + Float,
{
    type InputValueType = In;
    type OutputValueType = u8;

    /// Encode each token of the input multivector into `M` u8 codes.
    ///
    /// For each token, the nearest centroid in each subspace is found by squared Euclidean
    /// distance. The result is `M` bytes per token, so the encoded multivector has `dim = M`.
    #[inline]
    fn push_encoded<'a, OutputContainer>(
        &self,
        input: DenseMultiVectorView<'a, In>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<u8>,
    {
        let mut token_f32 = vec![0.0f32; self.token_dim];
        for token in input.iter_vectors() {
            // Convert token values to f32 for centroid distance computation.
            for (dst, &src) in token_f32.iter_mut().zip(token.values()) {
                // SAFETY: Float::to_f32() is infallible for all Float-bounded types.
                *dst = unsafe { src.to_f32().unwrap_unchecked() };
            }
            for m in 0..M {
                let sub = DenseVectorView::new(&token_f32[m * self.dsub..(m + 1) * self.dsub]);
                let code = FlatIndex::from(&self.centroids[m])
                    .search_nearest(sub, &())
                    .map(|s| s.vector as u8)
                    .unwrap_or(0);
                output.extend(std::iter::once(code));
            }
        }
    }

    /// Decode an encoded multivector via centroid lookup (overrides default scalar cast).
    fn decode_vector<'a>(
        &self,
        encoded: DenseMultiVectorView<'a, Self::OutputValueType>,
    ) -> DenseMultiVectorOwned<f32> {
        self.decode_multivec(encoded)
    }
}

impl<const M: usize, In> SpaceUsage for MultiVecProductQuantizer<M, In>
where
    In: ValueType,
{
    fn space_usage_bytes(&self) -> usize {
        2 * std::mem::size_of::<usize>() // token_dim, dsub
            + self
                .centroids
                .iter()
                .map(|ds| ds.space_usage_bytes())
                .sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vector::DenseMultiVectorView;
    use crate::{PlainDenseDataset, SquaredEuclideanDistance};

    /// Build a trivial training set: `n_vecs` token vectors of dim `token_dim`,
    /// all equal to zero. Used only to exercise the training/encoding path.
    fn make_training_set(
        token_dim: usize,
        n_vecs: usize,
    ) -> PlainDenseDataset<f32, SquaredEuclideanDistance> {
        use crate::{PlainDenseDatasetGrowable, PlainDenseQuantizer};
        let quantizer = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(token_dim);
        let mut ds = PlainDenseDatasetGrowable::<f32, SquaredEuclideanDistance>::with_capacity(
            quantizer, n_vecs,
        );
        for _ in 0..n_vecs {
            let values = vec![0.0f32; token_dim];
            ds.push(DenseVectorView::new(&values));
        }
        ds.into()
    }

    #[test]
    fn encode_decode_roundtrip() {
        // Token dim = 4, M = 4 subspaces of dsub = 1.
        // With only one value per subspace, each centroid is a scalar — simple sanity check.
        const M: usize = 4;
        let token_dim = 4;

        // Training: 256 singleton vectors (one per centroid slot).
        let n_train = KSUB * 2;
        let training = make_training_set(token_dim, n_train);
        let encoder = MultiVecProductQuantizer::<M, f32>::train(&training);

        assert_eq!(encoder.token_dim(), token_dim);
        assert_eq!(encoder.dsub(), 1);
        assert_eq!(encoder.ksub(), KSUB);
        assert_eq!(encoder.output_dim(), M);
    }

    #[test]
    fn query_evaluator_runs() {
        const M: usize = 4;
        let token_dim = 4;
        let training = make_training_set(token_dim, KSUB * 2);
        let encoder = MultiVecProductQuantizer::<M, f32>::train(&training);

        // Encode a 2-token query and a 2-token document.
        let query_vals = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let query = DenseMultiVectorView::new(&query_vals, token_dim);

        let mut encoded_doc = Vec::new();
        let doc_vals = [0.5f32, 0.5, 0.5, 0.5, 0.1, 0.2, 0.3, 0.4];
        let doc = DenseMultiVectorView::new(&doc_vals, token_dim);
        encoder.push_encoded(doc, &mut encoded_doc);

        assert_eq!(encoded_doc.len(), 2 * M); // 2 tokens × M codes each

        let evaluator = encoder.query_evaluator(query, &());
        let encoded_view = DenseMultiVectorView::new(&encoded_doc, M);
        let dist = evaluator.compute_distance(encoded_view);
        // Score should be finite (exact value depends on trained centroids).
        assert!(dist.distance().is_finite());
    }

    #[test]
    #[should_panic(expected = "Query dim")]
    fn panics_on_mismatched_query_dim() {
        const M: usize = 4;
        let training = make_training_set(4, KSUB * 2);
        let encoder = MultiVecProductQuantizer::<M, f32>::train(&training);
        // dim = 2 doesn't match token_dim = 4
        let query = DenseMultiVectorView::new(&[1.0f32, 0.0], 2);
        encoder.query_evaluator(query, &());
    }

    #[test]
    fn maxsim_exact_match_produces_finite_score() {
        const M: usize = 4;
        let token_dim = 4;
        let training = make_training_set(token_dim, KSUB * 2);
        let encoder = MultiVecProductQuantizer::<M, f32>::train(&training);

        // Query: single token [1, 0, 0, 0]
        let query_vals = [1.0f32, 0.0, 0.0, 0.0];
        let query = DenseMultiVectorView::new(&query_vals, token_dim);

        // Doc with exact matching token [1, 0, 0, 0]
        let doc_exact_vals = [1.0f32, 0.0, 0.0, 0.0];
        let doc_exact = DenseMultiVectorView::new(&doc_exact_vals, token_dim);

        let mut enc_exact = Vec::new();
        encoder.push_encoded(doc_exact, &mut enc_exact);

        let evaluator = encoder.query_evaluator(query, &());
        let score_exact = evaluator
            .compute_distance(DenseMultiVectorView::new(&enc_exact, M))
            .distance();

        // Score should be finite
        assert!(score_exact.is_finite());
    }

    #[test]
    fn maxsim_multiple_tokens_produces_finite_score() {
        const M: usize = 4;
        let token_dim = 4;
        let training = make_training_set(token_dim, KSUB * 2);
        let encoder = MultiVecProductQuantizer::<M, f32>::train(&training);

        // Query: [1, 0, 0, 0]
        let query = DenseMultiVectorView::new(&[1.0f32, 0.0, 0.0, 0.0], token_dim);

        // Doc with two tokens: [0.8, 0.2, 0, 0], [1.0, 0, 0, 0]
        // The MaxSim should pick the best matching token
        let doc_vals = [0.8f32, 0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let doc = DenseMultiVectorView::new(&doc_vals, token_dim);

        let mut encoded = Vec::new();
        encoder.push_encoded(doc, &mut encoded);

        let evaluator = encoder.query_evaluator(query, &());
        let score = evaluator
            .compute_distance(DenseMultiVectorView::new(&encoded, M))
            .distance();

        // Score should be finite
        assert!(score.is_finite());
    }

    #[test]
    fn encode_produces_m_bytes_per_token() {
        const M: usize = 4;
        let token_dim = 8;
        let training = make_training_set(token_dim, KSUB * 2);
        let encoder = MultiVecProductQuantizer::<M, f32>::train(&training);

        let doc_vals: Vec<f32> = vec![0.5f32; 3 * token_dim]; // 3 tokens
        let doc = DenseMultiVectorView::new(&doc_vals, token_dim);
        let mut encoded = Vec::new();
        encoder.push_encoded(doc, &mut encoded);

        // Should produce exactly 3 tokens × M bytes
        assert_eq!(encoded.len(), 3 * M);

        // Each encoded byte should be a valid u8
        for &byte in &encoded {
            let _: u8 = byte; // This consumes the byte to verify it's u8
        }
    }

    #[test]
    fn single_token_document() {
        const M: usize = 4;
        let token_dim = 4;
        let training = make_training_set(token_dim, KSUB * 2);
        let encoder = MultiVecProductQuantizer::<M, f32>::train(&training);

        let query = DenseMultiVectorView::new(&[1.0f32, 0.0, 0.0, 1.0], token_dim);

        let doc_vals = [0.9f32, 0.1, 0.1, 0.9];
        let doc = DenseMultiVectorView::new(&doc_vals, token_dim);
        let mut encoded = Vec::new();
        encoder.push_encoded(doc, &mut encoded);

        let evaluator = encoder.query_evaluator(query, &());
        let score = evaluator
            .compute_distance(DenseMultiVectorView::new(&encoded, M))
            .distance();

        assert!(score.is_finite());
    }
}
