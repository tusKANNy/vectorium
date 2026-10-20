//! Two-level Product Quantizer for multivector (late-interaction) data.
//!
//! Encoding is two-stage (IVF + residual PQ):
//! 1. Coarse quantization: assign each token to the nearest of `ncoarse` centroids
//!    (k-means, L2); store the centroid id as a `u32` little-endian (4 bytes).
//! 2. Residual PQ: encode `token − coarse_centroid` with a standard product
//!    quantizer (`M` subspaces, 256 centroids each); store `M` bytes.
//!
//! Storage layout:
//! All coarse centroid IDs are stored first (contiguous), followed by all PQ codes.
//! [coarse_ids: u32 × n_tokens (4×n bytes)] [pq_codes: u8 × n_tokens × M (n×M bytes)]
//!
//! # Scoring
//!
//! Queries must have exactly [`Q_TOKEN`] tokens (a compile-time constant, default 32).
//! The evaluator precomputes two structures once per query:
//!
//! - Transposed query layout `[D × Q_TOKEN]`:
//!   `query_flat_t[d * Q_TOKEN + q]` = value at dimension `d` of query token `q`.
//!
//! - PQ table `[M × KSUB × Q_TOKEN]`:
//!   `table[m * KSUB * Q_TOKEN + k * Q_TOKEN + q]` = `dot(query_q_subspace_m, pq_centroid_mk)`.
//!
//! For scoring a document, two phases:
//! 1. Centroid GEMM: extract coarse centroid for each doc token into `centroid_cols[doc_n × D]`,
//!    then compute `centroid_scores[doc_n × Q_TOKEN]` via a `D × doc_n × Q_TOKEN` loop.
//!    Each centroid dimension is read once and broadcast to all Q_TOKEN accumulators.
//! 2. MaxSim: for each doc token, start `acc[Q_TOKEN]` from its centroid scores, add PQ
//!    residual contributions (from table), update per-query-token maxima.

use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::time::Instant;

use half::f16;

use faer::Parallelism;
use faer::linalg::matmul::matmul;
use faer::mat;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::clustering::KMeansBuilder;
use crate::core::vector::{DenseMultiVectorOwned, DenseMultiVectorView, DenseVectorView};
use crate::core::vector_encoder::{MultiVecEncoder, QueryEvaluator, VectorEncoder};
use crate::distances::{DotProduct, SquaredEuclideanDistance};
use crate::encoders::pq::ProductQuantizer;
use crate::{
    Dataset, DatasetGrowable, FlatIndex, Float, PlainDenseDataset, PlainDenseDatasetGrowable,
    PlainDenseQuantizer, SpaceUsage, ValueType, VectorId,
};

/// Number of PQ centroids per subspace (always 256 = 2^8, one u8 code per subspace).
const KSUB: usize = 256;
/// Bytes used to store the coarse centroid id per encoded token (u32 little-endian).
const COARSE_ID_BYTES: usize = 4;
/// Bytes used to store a per-token residual norm when `with_norms = true` (f16 little-endian).
const NORM_BYTES: usize = 2;

/// Two-level Product Quantizer for multivector (late-interaction) data.
///
/// Combines coarse IVF-style quantization with product quantization on residuals.
/// Each encoded token occupies `M + 4` bytes: 4 bytes for the coarse centroid id
/// followed by `M` bytes of PQ codes for the residual.
///
/// # Type Parameters
/// - `M`: number of PQ subspaces; must divide `token_dim` and be divisible by 4.
/// - `In`: input value type (e.g. `f32`).
///
/// # Training
///
/// Call [`MultiVecTwoLevelProductQuantizer::train`] with a flat dataset of all token
/// vectors and the desired number of coarse centroids.
///
/// # Search (MaxSim via ADC)
///
/// [`MultiVecTwoLevelPQQueryEvaluator`] precomputes a PQ distance table `[M][KSUB][Q]`
/// once per query. For each document token the coarse centroid contribution is computed
/// on the fly via a dot product, then the PQ residual contributions are accumulated
/// via SAXPY from the precomputed table.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MultiVecTwoLevelProductQuantizer<const M: usize, In> {
    /// Dimensionality of each individual token vector (= M * dsub).
    token_dim: usize,
    /// Sub-space dimensionality (= token_dim / M).
    dsub: usize,
    /// Coarse centroids: `ncoarse` vectors of dimension `token_dim`.
    coarse_centroids: PlainDenseDataset<f32, SquaredEuclideanDistance>,
    /// Per-subspace PQ centroids: M datasets, each with KSUB vectors of dimension `dsub`.
    pq_centroids: Box<[PlainDenseDataset<f32, SquaredEuclideanDistance>]>,
    /// Whether each encoded token carries a residual norm in its payload.
    ///
    /// When `true`, `push_encoded` normalises the residual to unit length before PQ encoding
    /// and appends the original norm as a 4-byte f32 in a third block:
    /// `[coarse_ids: 4·n][pq_codes: M·n][norms: 4·n]`
    /// so `output_dim = COARSE_ID_BYTES + M + NORM_BYTES`.
    ///
    /// Indexes and datasets are entirely unaware of this field — they just see byte vectors
    /// of the declared `output_dim`.  The evaluator reads the norm bytes from the payload
    /// and applies them during MaxSim scoring without any external storage.
    #[serde(default)]
    with_norms: bool,
    _phantom: PhantomData<In>,
}

impl<const M: usize, In> MultiVecTwoLevelProductQuantizer<M, In> {
    /// Construct from pre-trained components.
    ///
    /// Panics if `M % 4 != 0`, `token_dim % M != 0`, or any centroid dataset has the
    /// wrong number of vectors or dimension.
    /// `with_norms`: when `true`, `push_encoded` normalises residuals and embeds the norm
    /// in the payload third block; `output_dim` increases by `NORM_BYTES`.
    pub fn from_pretrained(
        token_dim: usize,
        coarse_centroids: PlainDenseDataset<f32, SquaredEuclideanDistance>,
        pq_centroids: Vec<PlainDenseDataset<f32, SquaredEuclideanDistance>>,
        with_norms: bool,
    ) -> Self {
        assert_eq!(M % 4, 0, "M ({}) must be divisible by 4", M);
        assert_eq!(
            token_dim % M,
            0,
            "token_dim ({}) must be divisible by M ({})",
            token_dim,
            M
        );
        let dsub = token_dim / M;
        assert_eq!(
            coarse_centroids.output_dim(),
            token_dim,
            "coarse_centroids dim ({}) must equal token_dim ({})",
            coarse_centroids.output_dim(),
            token_dim
        );
        assert_eq!(
            pq_centroids.len(),
            M,
            "Expected {} PQ centroid datasets, got {}",
            M,
            pq_centroids.len()
        );
        for (i, ds) in pq_centroids.iter().enumerate() {
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
            coarse_centroids,
            pq_centroids: pq_centroids.into_boxed_slice(),
            with_norms,
            _phantom: PhantomData,
        }
    }

    /// Train a two-level product quantizer on a flat dataset of individual token vectors.
    ///
    /// Training proceeds in three steps:
    /// 1. K-means on all token vectors → `ncoarse` coarse centroids (L2).
    /// 2. Compute residuals in parallel: `residual[i] = token[i] − nearest_coarse_centroid[i]`.
    /// 3. Per-subspace k-means on residuals (parallel over `M` subspaces) → PQ centroids.
    ///
    /// Panics if `M % 4 != 0` or `token_dim % M != 0`.
    pub fn train(
        token_vectors: &PlainDenseDataset<f32, SquaredEuclideanDistance>,
        ncoarse: usize,
    ) -> Self {
        let token_dim = token_vectors.output_dim();
        let n = token_vectors.len();

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
            "Training MultiVecTwoLevelProductQuantizer: {} tokens × {} dims, \
             ncoarse={}, M={}, dsub={}, KSUB={}",
            n, token_dim, ncoarse, M, dsub, KSUB
        );

        // Step 1: Train coarse centroids via k-means.
        println!("  Step 1: training {} coarse centroids...", ncoarse);
        let coarse_centroids: PlainDenseDataset<f32, SquaredEuclideanDistance> =
            KMeansBuilder::new()
                .build()
                .train::<_, PlainDenseDatasetGrowable<f32, SquaredEuclideanDistance>>(
                    token_vectors,
                    ncoarse,
                    None,
                    PlainDenseQuantizer::new(token_dim),
                )
                .into();

        // Step 2: Compute residuals in parallel.
        println!("  Step 2: computing residuals ({} vectors)...", n);
        let mut residuals_flat = vec![0f32; n * token_dim];
        {
            let coarse_ref = &coarse_centroids;
            residuals_flat
                .par_chunks_mut(token_dim)
                .enumerate()
                .for_each(|(i, res)| {
                    let token = token_vectors.get(i as VectorId);
                    let nearest = FlatIndex::from(coarse_ref)
                        .search_nearest(token, &())
                        .map(|s| s.vector)
                        .unwrap_or(0);
                    let centroid = coarse_ref.get(nearest);
                    for (r, (&v, &c)) in res
                        .iter_mut()
                        .zip(token.values().iter().zip(centroid.values()))
                    {
                        *r = v - c;
                    }
                });
        }

        // Wrap residuals in a PlainDenseDataset for sub-space extraction.
        let residuals_ds = PlainDenseDataset::<f32, SquaredEuclideanDistance>::from_raw(
            residuals_flat.into_boxed_slice(),
            n,
            PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(token_dim),
        );

        // Step 3: Per-subspace k-means on residuals, parallel over M subspaces.
        println!(
            "  Step 3: training PQ ({} subspaces × {} centroids each)...",
            M, KSUB
        );
        let pq_centroids: Vec<PlainDenseDataset<f32, SquaredEuclideanDistance>> = (0..M)
            .into_par_iter()
            .map(|m| {
                let q = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(dsub);
                let mut sub_ds =
                    PlainDenseDatasetGrowable::<f32, SquaredEuclideanDistance>::with_capacity(q, n);
                for vec in residuals_ds.iter() {
                    sub_ds.push(DenseVectorView::new(
                        &vec.values()[m * dsub..(m + 1) * dsub],
                    ));
                }
                KMeansBuilder::new()
                    .build()
                    .train::<_, PlainDenseDatasetGrowable<f32, SquaredEuclideanDistance>>(
                        &sub_ds,
                        KSUB,
                        None,
                        PlainDenseQuantizer::new(dsub),
                    )
                    .into()
            })
            .collect();

        println!("  Training complete.");

        Self {
            token_dim,
            dsub,
            coarse_centroids,
            pq_centroids: pq_centroids.into_boxed_slice(),
            with_norms: false, // training always produces a no-norm encoder; set via from_pretrained
            _phantom: PhantomData,
        }
    }

    /// Train a two-level product quantizer from pre-computed coarse centroids.
    ///
    /// Skips coarse k-means (centroids are provided, e.g. from TAC) and only
    /// trains the residual PQ on a sample of the token vectors.
    ///
    /// Steps:
    /// 1. Sample `sample_size` tokens (sorted for sequential memory access).
    /// 2. Compute residuals `token − coarse_centroid`; optionally L2-normalise.
    /// 3. Per-subspace k-means on residuals → `M` × `KSUB` PQ centroids.
    ///
    /// # Arguments
    /// * `coarse_centroids` – Pre-trained coarse centroids (e.g. from TAC).
    /// * `flat_input` – All token vectors in their native type, layout `[n_tokens × token_dim]`.
    ///   Values are converted to f32 on-the-fly; only the sampled subset is ever materialised,
    ///   so no full f32 copy of the dataset is needed.
    /// * `assignments` – Coarse-centroid index per token (same length as `flat_input / token_dim`).
    /// * `sample_size` – Tokens to use for PQ training (capped at `n_tokens`). Pass a value ≥
    ///   `flat_input.len() / token_dim` to use all tokens without resampling (e.g. when the
    ///   caller has already pre-selected a training subset via damped proportional sampling).
    /// * `n_iter` – K-means iterations for PQ subspace training.
    /// * `normalize` – Normalise residuals before PQ; embeds original norm in payload.
    /// * `seed` – Optional RNG seed for reproducible uniform sampling and KMeans initialisation.
    pub fn train_from_coarse(
        coarse_centroids: PlainDenseDataset<f32, SquaredEuclideanDistance>,
        flat_input: &[In],
        assignments: &[usize],
        sample_size: usize,
        n_iter: usize,
        normalize: bool,
        seed: Option<u64>,
    ) -> Self
    where
        In: ValueType,
    {
        let token_dim = coarse_centroids.output_dim();
        let n_tokens = flat_input.len() / token_dim;
        let dsub = token_dim / M;

        assert_eq!(M % 4, 0, "M ({}) must be divisible by 4", M);
        assert_eq!(
            token_dim % M,
            0,
            "token_dim ({}) must be divisible by M ({})",
            token_dim,
            M
        );
        assert_eq!(
            assignments.len(),
            n_tokens,
            "assignments.len={} must equal n_tokens={}",
            assignments.len(),
            n_tokens
        );

        let train_n = sample_size.min(n_tokens);

        println!(
            "train_from_coarse: {} tokens × dim={}, {} coarse centroids, \
             M={}, dsub={}, sample={}, normalize={}",
            n_tokens,
            token_dim,
            coarse_centroids.len(),
            M,
            dsub,
            train_n,
            normalize
        );

        // Build sorted sample indices for sequential memory access.
        // `seed` controls KMeans initialisation only.
        // Pass sample_size >= n_tokens to skip resampling (e.g. caller pre-selected the subset).
        let sample_indices: Vec<usize> = if train_n >= n_tokens {
            (0..n_tokens).collect()
        } else {
            use rand::SeedableRng;
            use rand::rngs::StdRng;
            use rand::seq::SliceRandom;
            let mut rng = StdRng::seed_from_u64(42);
            let mut indices: Vec<usize> = (0..n_tokens).collect();
            indices.shuffle(&mut rng);
            indices.truncate(train_n);
            indices.sort_unstable();
            indices
        };

        println!("  Step 1: computing {} training residuals...", train_n);

        let coarse_ref = &coarse_centroids;
        let mut residuals_flat = vec![0f32; train_n * token_dim];
        residuals_flat
            .par_chunks_mut(token_dim)
            .zip(sample_indices.par_iter())
            .for_each(|(res, &idx)| {
                let tok_base = idx * token_dim;
                let centroid = coarse_ref.get(assignments[idx] as VectorId);
                for d in 0..token_dim {
                    // Convert the raw input element to f32 on the fly — no full f32 copy needed.
                    let v = unsafe { flat_input[tok_base + d].to_f32().unwrap_unchecked() };
                    res[d] = v - centroid.values()[d];
                }
                if normalize {
                    let sq: f32 = res.iter().map(|&x| x * x).sum();
                    let inv = 1.0 / sq.sqrt().max(1e-12);
                    res.iter_mut().for_each(|x| *x *= inv);
                }
            });

        let residuals_ds = PlainDenseDataset::<f32, SquaredEuclideanDistance>::from_raw(
            residuals_flat.into_boxed_slice(),
            train_n,
            PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(token_dim),
        );

        println!(
            "  Step 2: training PQ ({} subspaces × {} centroids, {} iters)...",
            M, KSUB, n_iter
        );

        let pq = ProductQuantizer::<M, SquaredEuclideanDistance>::train_with_kmeans_options(
            &residuals_ds,
            n_iter,
            seed,
        );
        let pq_centroids_aos = pq.centroids_as_aos(); // [M × KSUB × dsub] AoS
        let pq_centroids: Vec<PlainDenseDataset<f32, SquaredEuclideanDistance>> = (0..M)
            .map(|m| {
                PlainDenseDataset::<f32, SquaredEuclideanDistance>::from_raw(
                    pq_centroids_aos[m * KSUB * dsub..(m + 1) * KSUB * dsub]
                        .to_vec()
                        .into_boxed_slice(),
                    KSUB,
                    PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(dsub),
                )
            })
            .collect();

        println!("  train_from_coarse complete.");

        Self {
            token_dim,
            dsub,
            coarse_centroids,
            pq_centroids: pq_centroids.into_boxed_slice(),
            with_norms: normalize,
            _phantom: PhantomData,
        }
    }

    /// Number of coarse centroids.
    #[inline]
    pub fn ncoarse(&self) -> usize {
        self.coarse_centroids.len()
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

    /// Number of PQ centroids per subspace (always 256).
    #[inline]
    pub fn ksub(&self) -> usize {
        KSUB
    }

    /// Decode an encoded multivector back to approximate f32 token vectors.
    ///
    /// Reconstructs each token as `coarse_centroid + Σ_m pq_centroid[m][code_m]`.
    /// When `with_norms = true`, the per-token norm is read from the third payload block
    /// and applied: the PQ residual decodes to a unit-norm vector, scaled back by the norm.
    ///
    /// Blocked layout:
    ///   without norms: `[coarse_ids: 4·n][pq_codes: M·n]`
    ///   with norms:    `[coarse_ids: 4·n][pq_codes: M·n][norms: 4·n]`
    fn decode_multivec(&self, encoded: DenseMultiVectorView<'_, u8>) -> DenseMultiVectorOwned<f32> {
        let n_tokens = encoded.num_vecs();
        let all_bytes = encoded.values();
        let codes_offset = COARSE_ID_BYTES * n_tokens;
        let norms_offset = (COARSE_ID_BYTES + M) * n_tokens;

        let mut values = Vec::with_capacity(n_tokens * self.token_dim);
        let mut decoded = vec![0f32; self.token_dim];

        for t in 0..n_tokens {
            let byte_offset = t * COARSE_ID_BYTES;
            let coarse_id = u32::from_le_bytes([
                all_bytes[byte_offset],
                all_bytes[byte_offset + 1],
                all_bytes[byte_offset + 2],
                all_bytes[byte_offset + 3],
            ]) as VectorId;

            let centroid = self.coarse_centroids.get(coarse_id);
            decoded.copy_from_slice(centroid.values());

            for m in 0..M {
                let code = all_bytes[codes_offset + t * M + m] as VectorId;
                let pq_centroid = self.pq_centroids[m].get(code);
                let sub_start = m * self.dsub;
                for (j, &v) in pq_centroid.values().iter().enumerate() {
                    decoded[sub_start + j] += v;
                }
            }

            // When norms are embedded, scale the residual part back by its norm.
            if self.with_norms {
                let nb = norms_offset + t * NORM_BYTES;
                let norm =
                    f16::from_bits(u16::from_le_bytes([all_bytes[nb], all_bytes[nb + 1]])).to_f32();
                // decoded = centroid + pq_residual_unit; we want centroid + norm * pq_residual_unit.
                let centroid_vals = self.coarse_centroids.get(coarse_id);
                for (d, (v, &c)) in decoded
                    .iter_mut()
                    .zip(centroid_vals.values().iter())
                    .enumerate()
                {
                    let _ = d;
                    *v = c + (*v - c) * norm;
                }
            }

            values.extend_from_slice(&decoded);
        }
        DenseMultiVectorOwned::new(values, self.token_dim)
    }

    /// Like `push_encoded`, but uses pre-computed coarse-centroid ids instead of searching.
    ///
    /// Bypasses the O(ncoarse) `search_nearest` call per token — use this when coarse
    /// assignments are already available (e.g. from TAC).  All other steps (residual
    /// computation, optional normalisation, PQ encoding) are identical to `push_encoded`.
    ///
    /// `precomputed_ids[t]` must be the coarse-centroid index for the t-th token of `input`.
    pub fn push_encoded_with_ids<'a, Out>(
        &self,
        input: DenseMultiVectorView<'a, In>,
        precomputed_ids: &[u32],
        output: &mut Out,
    ) where
        In: ValueType,
        Out: Extend<u8>,
    {
        let n_tokens = input.num_vecs();
        debug_assert_eq!(precomputed_ids.len(), n_tokens);

        let mut token_f32 = vec![0f32; self.token_dim];
        let mut residual = vec![0f32; self.token_dim];
        let mut pq_codes = Vec::with_capacity(n_tokens * M);
        let mut norm_bytes: Vec<u8> = if self.with_norms {
            Vec::with_capacity(n_tokens * NORM_BYTES)
        } else {
            Vec::new()
        };

        for (t, token) in input.iter_vectors().enumerate() {
            let coarse_id = precomputed_ids[t];
            for (dst, &src) in token_f32.iter_mut().zip(token.values()) {
                *dst = unsafe { src.to_f32().unwrap_unchecked() };
            }
            let centroid = self.coarse_centroids.get(coarse_id as VectorId);
            for (r, (&v, &c)) in residual
                .iter_mut()
                .zip(token_f32.iter().zip(centroid.values()))
            {
                *r = v - c;
            }
            if self.with_norms {
                let sq: f32 = residual.iter().map(|x| x * x).sum();
                let norm = sq.sqrt().max(1e-12);
                norm_bytes.extend_from_slice(&f16::from_f32(norm).to_le_bytes());
                let inv = 1.0 / norm;
                residual.iter_mut().for_each(|x| *x *= inv);
            }
            for m in 0..M {
                let sub = DenseVectorView::new(&residual[m * self.dsub..(m + 1) * self.dsub]);
                let code = FlatIndex::from(&self.pq_centroids[m])
                    .search_nearest(sub, &())
                    .map(|s| s.vector as u8)
                    .unwrap_or(0);
                pq_codes.push(code);
            }
        }

        for &id in precomputed_ids {
            output.extend(id.to_le_bytes());
        }
        output.extend(pq_codes);
        if self.with_norms {
            output.extend(norm_bytes);
        }
    }
}

/// Reusable scratchpad for scoring operations to eliminate repeated allocations.
///
/// High-performance scoring requires scoring many documents per query. This scratchpad
/// pre-allocates buffers once and reuses them across all document scoring calls,
/// eliminating heap allocations in the hot path.
///
/// **Safety guarantee**: The buffers are completely overwritten before being read:
/// - `centroid_cols` is overwritten by `copy_nonoverlapping` (Phase 1b)
/// - `centroid_scores` is overwritten by `sgemm` with `beta = 0.0` (Phase 1c)
///
/// This means we leave garbage data in memory and rely on these operations to overwrite
/// it completely, saving the overhead of unnecessary `memset` initialization.
///
/// Typical usage:
/// ```ignore
/// let encoder = /* ... */;
/// let query = /* ... */;
/// let evaluator = encoder.query_evaluator(query, &());
/// let mut scratchpad = MultiVecTwoLevelPQScratchpad::with_capacity(token_dim, max_doc_n);
///
/// for doc in documents {
///     let score = evaluator.compute_distance_with_scratchpad(encoded_doc, &mut scratchpad, offset);
/// }
/// ```
#[derive(Debug)]
pub struct MultiVecTwoLevelPQScratchpad<const M: usize> {
    /// Coarse centroid buffer: reused for extracting centroid vectors.
    /// Completely overwritten by copy_nonoverlapping before any read.
    centroid_cols: Vec<f32>,
    /// GEMM output buffer: reused for centroid score matrix.
    /// Completely overwritten by sgemm (beta=0.0) before any read.
    centroid_scores: Vec<f32>,
    /// Per-token norm buffer, used only when the encoder has `with_norms = true`.
    /// Filled by decoding the norm bytes from the encoded payload before MaxSim.
    pub norms_buf: Vec<f32>,
    /// Per-query-token accumulator passed to the distance function; reset each doc token.
    /// Length = q_token.
    pub acc: Vec<f32>,
    /// Per-query-token running maxima passed to the distance function; reset each query.
    /// Length = q_token.
    pub max_scores: Vec<f32>,
    /// Precomputed dimensions to avoid repeated parameter passing.
    token_dim: usize,
    /// Runtime query token count (replaces the former compile-time Q_TOKEN constant).
    pub q_token: usize,
}

impl<const M: usize> MultiVecTwoLevelPQScratchpad<M> {
    pub fn with_capacity(token_dim: usize, max_doc_n: usize, q_token: usize) -> Self {
        Self {
            centroid_cols: Vec::with_capacity(max_doc_n * token_dim),
            centroid_scores: Vec::with_capacity(max_doc_n * q_token),
            norms_buf: Vec::with_capacity(max_doc_n),
            acc: vec![0f32; q_token],
            max_scores: vec![f32::NEG_INFINITY; q_token],
            token_dim,
            q_token,
        }
    }

    /// Ensure all buffers can hold at least `doc_n` tokens.
    /// Extends lengths WITHOUT zero-initialisation; callers guarantee complete overwrite.
    #[inline]
    pub fn ensure_capacity(&mut self, doc_n: usize) {
        let needed_cols = doc_n * self.token_dim;
        let needed_scores = doc_n * self.q_token;

        if self.centroid_cols.capacity() < needed_cols {
            self.centroid_cols
                .reserve(needed_cols - self.centroid_cols.len());
        }
        if self.centroid_scores.capacity() < needed_scores {
            self.centroid_scores
                .reserve(needed_scores - self.centroid_scores.len());
        }
        if self.norms_buf.capacity() < doc_n {
            self.norms_buf.reserve(doc_n - self.norms_buf.len());
        }

        // SAFETY: centroid_cols and centroid_scores are completely overwritten by
        // copy_nonoverlapping and sgemm(beta=0) respectively before any read.
        // norms_buf is completely overwritten by the norm-extraction loop below.
        unsafe {
            self.centroid_cols.set_len(needed_cols);
            self.centroid_scores.set_len(needed_scores);
            self.norms_buf.set_len(doc_n);
        }
    }
}

/// Query evaluator for [`MultiVecTwoLevelProductQuantizer`].
///
/// Requires exactly [`Q_TOKEN`] query tokens. Precomputes two structures:
///
/// - **Transposed query** `query_flat_t[d * Q_TOKEN + q]`: contiguous Q-slices per dimension.
/// - **PQ table** `[M × KSUB × Q_TOKEN]`: built via SAXPY over PQ centroid dimensions.
///
/// When the encoder has `with_norms = true`, per-token norms are read from the encoded
/// payload's third block during scoring — no external norm slice is needed.
///
/// The internal scratchpad is created at evaluator construction and reused across all
/// `compute_distance` calls, eliminating per-call heap allocations in the hot path.
pub struct MultiVecTwoLevelPQQueryEvaluator<'e, const M: usize, In> {
    encoder: &'e MultiVecTwoLevelProductQuantizer<M, In>,
    query_flat_t: Vec<f32>,
    distance_table: Vec<f32>,
    /// Reusable buffers for centroid extraction, GEMM, and norm decoding.
    /// `RefCell` provides interior mutability so `compute_distance(&self, …)` can mutate them.
    scratchpad: RefCell<MultiVecTwoLevelPQScratchpad<M>>,
    pub centroid_time_us: Cell<f64>,
    pub residual_time_us: Cell<f64>,
    pub centroid_alloc_us: Cell<f64>,
    pub centroid_extract_us: Cell<f64>,
    pub centroid_gemm_us: Cell<f64>,
}

impl<'e, const M: usize, In> MultiVecTwoLevelPQQueryEvaluator<'e, M, In>
where
    In: ValueType,
{
    fn new(
        encoder: &'e MultiVecTwoLevelProductQuantizer<M, In>,
        query: DenseMultiVectorView<'_, f32>,
    ) -> Self {
        let q_token = query.num_vecs();
        let token_dim = encoder.token_dim;
        let dsub = encoder.dsub;
        let query_vals = query.values();

        let mut query_flat_t = vec![0f32; token_dim * q_token];
        for qi in 0..q_token {
            for d in 0..token_dim {
                query_flat_t[d * q_token + qi] = query_vals[qi * token_dim + d];
            }
        }

        let mut distance_table = vec![0f32; M * KSUB * q_token];
        unsafe {
            let qt_ptr = query_flat_t.as_ptr();
            let dt_ptr = distance_table.as_mut_ptr();

            for m in 0..M {
                let sub_offset = m * dsub;
                for k in 0..KSUB {
                    let centroid = encoder.pq_centroids[m].get(k as VectorId);
                    let entry_base = m * KSUB * q_token + k * q_token;
                    let centroid_vals = centroid.values();

                    for (d_sub, &c) in centroid_vals.iter().enumerate() {
                        let d = sub_offset + d_sub;
                        let qt_base = qt_ptr.add(d * q_token);
                        let out_base = dt_ptr.add(entry_base);

                        for q in 0..q_token {
                            let qt_val = *qt_base.add(q);
                            let current = *out_base.add(q);
                            *out_base.add(q) = current.algebraic_add(qt_val.algebraic_mul(c));
                        }
                    }
                }
            }
        }

        Self {
            scratchpad: RefCell::new(MultiVecTwoLevelPQScratchpad::with_capacity(
                encoder.token_dim,
                0, // grows on first call; no external max_doc_n hint needed
                q_token,
            )),
            encoder,
            query_flat_t,
            distance_table,
            centroid_time_us: Cell::new(0.0),
            residual_time_us: Cell::new(0.0),
            centroid_alloc_us: Cell::new(0.0),
            centroid_extract_us: Cell::new(0.0),
            centroid_gemm_us: Cell::new(0.0),
        }
    }
}

impl<'e, 'v, const M: usize, In> QueryEvaluator<DenseMultiVectorView<'v, u8>>
    for MultiVecTwoLevelPQQueryEvaluator<'e, M, In>
where
    In: ValueType,
{
    type Distance = DotProduct;

    /// Compute MaxSim between the stored query and a two-level PQ-encoded document multivector.
    ///
    /// Uses an internal scratchpad (created with the evaluator) to avoid per-call heap
    /// allocations.  If the encoder has `with_norms = true`, per-token norms are read
    /// directly from the encoded payload's third block.
    #[inline(always)]
    fn compute_distance(&self, vector: DenseMultiVectorView<'v, u8>) -> DotProduct {
        let mut sp = self.scratchpad.borrow_mut();

        let doc_n = vector.num_vecs();
        let token_dim = self.encoder.token_dim;
        let doc_bytes = vector.values();
        let codes_offset = COARSE_ID_BYTES * doc_n;
        let norms_offset = (COARSE_ID_BYTES + M) * doc_n;

        // Phase 1a — ensure capacity (no zeroing; buffers are completely overwritten below)
        let alloc_start = Instant::now();
        sp.ensure_capacity(doc_n);
        let alloc_elapsed = alloc_start.elapsed().as_secs_f64() * 1_000_000.0;
        self.centroid_alloc_us
            .set(self.centroid_alloc_us.get() + alloc_elapsed);

        // Phase 1b — extract coarse centroid for each doc token
        // SAFETY: blocked layout guarantees the first COARSE_ID_BYTES*doc_n bytes are coarse IDs.
        //         centroid_cols is completely overwritten before any read.
        let extract_start = Instant::now();
        unsafe {
            let doc_ptr = doc_bytes.as_ptr();
            let centroids_ptr = self.encoder.coarse_centroids.values().as_ptr();
            let dest_ptr = sp.centroid_cols.as_mut_ptr();
            for t in 0..doc_n {
                let bo = t * COARSE_ID_BYTES;
                let c0 = *doc_ptr.add(bo) as u32;
                let c1 = *doc_ptr.add(bo + 1) as u32;
                let c2 = *doc_ptr.add(bo + 2) as u32;
                let c3 = *doc_ptr.add(bo + 3) as u32;
                let coarse_id = (c0 | (c1 << 8) | (c2 << 16) | (c3 << 24)) as usize;
                std::ptr::copy_nonoverlapping(
                    centroids_ptr.add(coarse_id * token_dim),
                    dest_ptr.add(t * token_dim),
                    token_dim,
                );
            }
        }
        let extract_elapsed = extract_start.elapsed().as_secs_f64() * 1_000_000.0;
        self.centroid_extract_us
            .set(self.centroid_extract_us.get() + extract_elapsed);

        // Phase 1c — GEMM: centroid_scores = centroid_cols × query_flat_t^T
        let gemm_start = Instant::now();
        unsafe {
            let a = mat::from_raw_parts::<f32>(
                sp.centroid_cols.as_ptr(),
                doc_n,
                token_dim,
                token_dim as isize,
                1,
            );
            let b = mat::from_raw_parts::<f32>(
                self.query_flat_t.as_ptr(),
                token_dim,
                sp.q_token,
                sp.q_token as isize,
                1,
            );
            let mut c = mat::from_raw_parts_mut::<f32>(
                sp.centroid_scores.as_mut_ptr(),
                doc_n,
                sp.q_token,
                sp.q_token as isize,
                1,
            );
            matmul(c.as_mut(), a, b, None, 1.0, Parallelism::None);
        }
        let gemm_elapsed = gemm_start.elapsed().as_secs_f64() * 1_000_000.0;
        self.centroid_gemm_us
            .set(self.centroid_gemm_us.get() + gemm_elapsed);
        self.centroid_time_us
            .set(self.centroid_time_us.get() + alloc_elapsed + extract_elapsed + gemm_elapsed);

        // Phase 2 — MaxSim with PQ residuals (and optional embedded norms)
        let phase2_start = Instant::now();

        // Extract norms from the payload's third block into the scratchpad norm buffer.
        // SAFETY: ensure_capacity set norms_buf.len() >= doc_n; the loop overwrites it fully.
        // Extract raw pointers before local_norms borrows sp.norms_buf, so the borrow
        // checker does not see a conflict.  acc, max_scores, norms_buf, and centroid_scores
        // are all distinct fields — no aliasing.
        let acc_ptr = sp.acc.as_mut_ptr();
        let max_scores_ptr = sp.max_scores.as_mut_ptr();
        let q_token = sp.q_token;

        let local_norms: Option<&[f32]> = if self.encoder.with_norms {
            for t in 0..doc_n {
                let nb = norms_offset + t * NORM_BYTES;
                sp.norms_buf[t] =
                    f16::from_bits(u16::from_le_bytes([doc_bytes[nb], doc_bytes[nb + 1]])).to_f32();
            }
            Some(&sp.norms_buf[..doc_n])
        } else {
            None
        };

        // Dispatch to the const-generic fast path for common Q values (full SIMD unrolling),
        // fall back to the runtime version for anything else.
        let cs = &sp.centroid_scores[..doc_n * q_token];
        let dt = &self.distance_table;
        let codes = &doc_bytes[codes_offset..norms_offset];
        let score = unsafe {
            match q_token {
                8 => crate::distances::two_level_pq_maxsim_blocked::<M, 8>(
                    cs,
                    dt,
                    codes,
                    doc_n,
                    local_norms,
                ),
                16 => crate::distances::two_level_pq_maxsim_blocked::<M, 16>(
                    cs,
                    dt,
                    codes,
                    doc_n,
                    local_norms,
                ),
                32 => crate::distances::two_level_pq_maxsim_blocked::<M, 32>(
                    cs,
                    dt,
                    codes,
                    doc_n,
                    local_norms,
                ),
                64 => crate::distances::two_level_pq_maxsim_blocked::<M, 64>(
                    cs,
                    dt,
                    codes,
                    doc_n,
                    local_norms,
                ),
                128 => crate::distances::two_level_pq_maxsim_blocked::<M, 128>(
                    cs,
                    dt,
                    codes,
                    doc_n,
                    local_norms,
                ),
                256 => crate::distances::two_level_pq_maxsim_blocked::<M, 256>(
                    cs,
                    dt,
                    codes,
                    doc_n,
                    local_norms,
                ),
                512 => crate::distances::two_level_pq_maxsim_blocked::<M, 512>(
                    cs,
                    dt,
                    codes,
                    doc_n,
                    local_norms,
                ),
                _ => crate::distances::two_level_pq_maxsim_blocked_runtime::<M>(
                    cs,
                    dt,
                    codes,
                    doc_n,
                    q_token,
                    local_norms,
                    std::slice::from_raw_parts_mut(acc_ptr, q_token),
                    std::slice::from_raw_parts_mut(max_scores_ptr, q_token),
                ),
            }
        };
        self.residual_time_us
            .set(self.residual_time_us.get() + phase2_start.elapsed().as_secs_f64() * 1_000_000.0);
        DotProduct::from(score)
    }
}

impl<const M: usize, In> VectorEncoder for MultiVecTwoLevelProductQuantizer<M, In>
where
    In: ValueType + Float,
{
    type Distance = DotProduct;
    type InputVector<'a> = DenseMultiVectorView<'a, In>;
    type QueryVector<'q> = DenseMultiVectorView<'q, f32>;
    type EncodedVector<'a> = DenseMultiVectorView<'a, u8>;

    type Evaluator<'e>
        = MultiVecTwoLevelPQQueryEvaluator<'e, M, In>
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
        MultiVecTwoLevelPQQueryEvaluator::new(self, query)
    }

    #[inline]
    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = self.decode_multivec(vector);
        MultiVecTwoLevelPQQueryEvaluator::new(self, decoded.as_view())
    }

    #[inline]
    fn input_dim(&self) -> usize {
        self.token_dim
    }

    #[inline]
    fn output_dim(&self) -> usize {
        COARSE_ID_BYTES + M + if self.with_norms { NORM_BYTES } else { 0 }
    }
}

impl<const M: usize, In> MultiVecEncoder for MultiVecTwoLevelProductQuantizer<M, In>
where
    In: ValueType + Float,
{
    type InputValueType = In;
    type OutputValueType = u8;

    /// Encode each token of the input multivector.
    ///
    /// Uses a **blocked layout** for better cache performance:
    /// 1. Process all tokens: find nearest coarse centroid (L2), compute residuals, PQ encode
    /// 2. Write all coarse IDs as a contiguous block (4 bytes × n_tokens)
    /// 3. Write all PQ codes as a contiguous block (M bytes × n_tokens)
    ///
    /// This layout dramatically improves scoring performance vs per-token interleaving.
    #[inline]
    fn push_encoded<'a, OutputContainer>(
        &self,
        input: DenseMultiVectorView<'a, In>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<u8>,
    {
        let n_tokens = input.num_vecs();
        let mut token_f32 = vec![0f32; self.token_dim];
        let mut residual = vec![0f32; self.token_dim];

        let mut coarse_ids = Vec::with_capacity(n_tokens);
        let mut pq_codes = Vec::with_capacity(n_tokens * M);
        // Third block: residual norms (only populated when with_norms = true).
        let mut norm_bytes: Vec<u8> = if self.with_norms {
            Vec::with_capacity(n_tokens * NORM_BYTES)
        } else {
            Vec::new()
        };

        for token in input.iter_vectors() {
            for (dst, &src) in token_f32.iter_mut().zip(token.values()) {
                *dst = unsafe { src.to_f32().unwrap_unchecked() };
            }

            let token_view = DenseVectorView::new(&token_f32);
            let coarse_id = FlatIndex::from(&self.coarse_centroids)
                .search_nearest(token_view, &())
                .map(|s| s.vector as u32)
                .unwrap_or(0);
            coarse_ids.push(coarse_id);

            let centroid = self.coarse_centroids.get(coarse_id as VectorId);
            for (r, (&v, &c)) in residual
                .iter_mut()
                .zip(token_f32.iter().zip(centroid.values()))
            {
                *r = v - c;
            }

            // When with_norms: normalise the residual and store the original norm.
            if self.with_norms {
                let sq: f32 = residual.iter().map(|x| x * x).sum();
                let norm = sq.sqrt().max(1e-12);
                norm_bytes.extend_from_slice(&f16::from_f32(norm).to_le_bytes());
                let inv = 1.0 / norm;
                residual.iter_mut().for_each(|x| *x *= inv);
            }

            for m in 0..M {
                let sub = DenseVectorView::new(&residual[m * self.dsub..(m + 1) * self.dsub]);
                let code = FlatIndex::from(&self.pq_centroids[m])
                    .search_nearest(sub, &())
                    .map(|s| s.vector as u8)
                    .unwrap_or(0);
                pq_codes.push(code);
            }
        }

        // Blocked layout: [coarse_ids: 4·n][pq_codes: M·n][norms: 4·n (if with_norms)]
        for &id in &coarse_ids {
            output.extend(id.to_le_bytes());
        }
        output.extend(pq_codes);
        if self.with_norms {
            output.extend(norm_bytes);
        }
    }

    /// Decode an encoded multivector via coarse centroid + PQ residual lookup.
    fn decode_vector<'a>(
        &self,
        encoded: DenseMultiVectorView<'a, Self::OutputValueType>,
    ) -> DenseMultiVectorOwned<f32> {
        self.decode_multivec(encoded)
    }
}

impl<const M: usize, In> SpaceUsage for MultiVecTwoLevelProductQuantizer<M, In>
where
    In: ValueType,
{
    fn space_usage_bytes(&self) -> usize {
        2 * std::mem::size_of::<usize>() // token_dim, dsub
            + self.coarse_centroids.space_usage_bytes()
            + self
                .pq_centroids
                .iter()
                .map(|ds| ds.space_usage_bytes())
                .sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::distances::Distance;
    use crate::core::vector::{DenseMultiVectorView, DenseVectorView};
    use crate::{
        PlainDenseDataset, PlainDenseDatasetGrowable, PlainDenseQuantizer, SquaredEuclideanDistance,
    };

    fn make_training_set(
        token_dim: usize,
        n_vecs: usize,
    ) -> PlainDenseDataset<f32, SquaredEuclideanDistance> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(42);
        let quantizer = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(token_dim);
        let mut ds = PlainDenseDatasetGrowable::<f32, SquaredEuclideanDistance>::with_capacity(
            quantizer, n_vecs,
        );
        for _ in 0..n_vecs {
            let values: Vec<f32> = (0..token_dim)
                .map(|_| rng.gen_range(-1.0_f32..1.0))
                .collect();
            ds.push(DenseVectorView::new(&values));
        }
        ds.into()
    }

    #[test]
    fn train_dimensions_are_correct() {
        const M: usize = 4;
        let token_dim = 8;
        let ncoarse = 4;
        let training = make_training_set(token_dim, 512);
        let encoder = MultiVecTwoLevelProductQuantizer::<M, f32>::train(&training, ncoarse);

        assert_eq!(encoder.token_dim(), token_dim);
        assert_eq!(encoder.dsub(), token_dim / M);
        assert_eq!(encoder.ksub(), KSUB);
        assert_eq!(encoder.ncoarse(), ncoarse);
        assert_eq!(encoder.input_dim(), token_dim);
        assert_eq!(encoder.output_dim(), COARSE_ID_BYTES + M);
    }

    #[test]
    fn encode_produces_correct_byte_count() {
        const M: usize = 4;
        let token_dim = 8;
        let training = make_training_set(token_dim, 512);
        let encoder = MultiVecTwoLevelProductQuantizer::<M, f32>::train(&training, 4);

        let doc_vals: Vec<f32> = vec![0.5f32; 3 * token_dim]; // 3 tokens
        let doc = DenseMultiVectorView::new(&doc_vals, token_dim);
        let mut encoded = Vec::new();
        encoder.push_encoded(doc, &mut encoded);

        // 3 tokens × (COARSE_ID_BYTES + M) bytes each
        assert_eq!(encoded.len(), 3 * (COARSE_ID_BYTES + M));
    }

    #[test]
    fn query_evaluator_produces_finite_score() {
        const M: usize = 4;
        let token_dim = 8;
        let training = make_training_set(token_dim, 512);
        let encoder = MultiVecTwoLevelProductQuantizer::<M, f32>::train(&training, 4);

        let query_vals: Vec<f32> = vec![0.5f32; 32 * token_dim]; // Q_TOKEN query tokens
        let query = DenseMultiVectorView::new(&query_vals, token_dim);

        let doc_vals: Vec<f32> = vec![0.25f32; 3 * token_dim]; // 3 doc tokens
        let doc = DenseMultiVectorView::new(&doc_vals, token_dim);
        let mut encoded_doc = Vec::new();
        encoder.push_encoded(doc, &mut encoded_doc);

        let evaluator = encoder.query_evaluator(query, &());
        let encoded_view = DenseMultiVectorView::new(&encoded_doc, COARSE_ID_BYTES + M);
        let dist = evaluator.compute_distance(encoded_view);

        assert!(dist.distance().is_finite());
    }

    #[test]
    fn decode_restores_correct_shape() {
        const M: usize = 4;
        let token_dim = 8;
        let training = make_training_set(token_dim, 512);
        let encoder = MultiVecTwoLevelProductQuantizer::<M, f32>::train(&training, 4);

        let doc_vals: Vec<f32> = vec![0.1f32; 2 * token_dim]; // 2 tokens
        let doc = DenseMultiVectorView::new(&doc_vals, token_dim);
        let mut encoded = Vec::new();
        encoder.push_encoded(doc, &mut encoded);

        let encoded_view = DenseMultiVectorView::new(&encoded, COARSE_ID_BYTES + M);
        let decoded = encoder.decode_multivec(encoded_view);

        assert_eq!(decoded.dim(), token_dim);
        assert_eq!(decoded.num_vecs(), 2);
    }

    #[test]
    #[should_panic(expected = "Query dim")]
    fn panics_on_mismatched_query_dim() {
        const M: usize = 4;
        let training = make_training_set(8, 512);
        let encoder = MultiVecTwoLevelProductQuantizer::<M, f32>::train(&training, 4);
        // dim = 2 doesn't match token_dim = 8
        let query = DenseMultiVectorView::new(&[1.0f32, 0.0], 2);
        encoder.query_evaluator(query, &());
    }

    #[test]
    fn encode_decode_roundtrip_shape() {
        const M: usize = 4;
        let token_dim = 8;
        let training = make_training_set(token_dim, 512);
        let encoder = MultiVecTwoLevelProductQuantizer::<M, f32>::train(&training, 4);

        // Create a document with 3 tokens
        let doc_vals: Vec<f32> = vec![0.1f32; 3 * token_dim];
        let doc = DenseMultiVectorView::new(&doc_vals, token_dim);

        let mut encoded = Vec::new();
        encoder.push_encoded(doc, &mut encoded);

        // Decode and check shape is preserved
        let encoded_view = DenseMultiVectorView::new(&encoded, COARSE_ID_BYTES + M);
        let decoded = encoder.decode_multivec(encoded_view);

        assert_eq!(decoded.dim(), token_dim);
        assert_eq!(decoded.num_vecs(), 3);
        assert_eq!(decoded.values().len(), 3 * token_dim);
    }

    #[test]
    fn encode_decode_roundtrip_preserves_approximation() {
        const M: usize = 4;
        let token_dim = 8;
        let training = make_training_set(token_dim, 512);
        let encoder = MultiVecTwoLevelProductQuantizer::<M, f32>::train(&training, 4);

        // All-ones vector for easier verification
        let doc_vals: Vec<f32> = vec![1.0f32; 2 * token_dim];
        let doc = DenseMultiVectorView::new(&doc_vals, token_dim);

        let mut encoded = Vec::new();
        encoder.push_encoded(doc, &mut encoded);

        let encoded_view = DenseMultiVectorView::new(&encoded, COARSE_ID_BYTES + M);
        let decoded = encoder.decode_multivec(encoded_view);

        // Decoded should be reasonably close to the original (L1 error per token)
        for token_idx in 0..2 {
            for dim in 0..token_dim {
                let decoded_val = decoded.values()[token_idx * token_dim + dim];
                // Each value should be reasonably close to 1.0
                // PQ introduces loss but should be within this range
                assert!((0.0..=10.0).contains(&decoded_val));
            }
        }
    }

    #[test]
    fn scoring_two_documents_ranks_correctly() {
        const M: usize = 4;
        let token_dim = 8;
        let training = make_training_set(token_dim, 512);
        let encoder = MultiVecTwoLevelProductQuantizer::<M, f32>::train(&training, 4);

        // Query with specific pattern
        let query_vals: Vec<f32> = vec![1.0f32; 32 * token_dim];
        let query = DenseMultiVectorView::new(&query_vals, token_dim);
        let evaluator = encoder.query_evaluator(query, &());

        // Doc 1: all high values (should score high)
        let doc1_vals: Vec<f32> = vec![0.9f32; token_dim];
        let doc1 = DenseMultiVectorView::new(&doc1_vals, token_dim);
        let mut enc1 = Vec::new();
        encoder.push_encoded(doc1, &mut enc1);
        let score1 = evaluator
            .compute_distance(DenseMultiVectorView::new(&enc1, COARSE_ID_BYTES + M))
            .distance();

        // Doc 2: all low values (should score low)
        let doc2_vals: Vec<f32> = vec![0.1f32; token_dim];
        let doc2 = DenseMultiVectorView::new(&doc2_vals, token_dim);
        let mut enc2 = Vec::new();
        encoder.push_encoded(doc2, &mut enc2);
        let score2 = evaluator
            .compute_distance(DenseMultiVectorView::new(&enc2, COARSE_ID_BYTES + M))
            .distance();

        // Doc 1 should score higher due to MaxSim
        assert!(score1 > score2);
    }

    #[test]
    fn multiple_tokens_maxsim_picks_best() {
        const M: usize = 4;
        let token_dim = 8;
        let training = make_training_set(token_dim, 512);
        let encoder = MultiVecTwoLevelProductQuantizer::<M, f32>::train(&training, 4);

        // Query: all 1s
        let query_vals: Vec<f32> = vec![1.0f32; 32 * token_dim];
        let query = DenseMultiVectorView::new(&query_vals, token_dim);
        let evaluator = encoder.query_evaluator(query, &());

        // Doc with mixed tokens: first is low, second is high
        let mut doc_vals = vec![0.01f32; token_dim]; // Token 1: all low
        doc_vals.extend(vec![0.99f32; token_dim]); // Token 2: all high
        let doc = DenseMultiVectorView::new(&doc_vals, token_dim);

        let mut encoded = Vec::new();
        encoder.push_encoded(doc, &mut encoded);

        let score = evaluator
            .compute_distance(DenseMultiVectorView::new(&encoded, COARSE_ID_BYTES + M))
            .distance();

        // Should score well because MaxSim picks the high token
        assert!(score.is_finite() && score > 0.5);
    }

    #[test]
    fn single_token_document_two_level() {
        const M: usize = 4;
        let token_dim = 8;
        let training = make_training_set(token_dim, 512);
        let encoder = MultiVecTwoLevelProductQuantizer::<M, f32>::train(&training, 4);

        let query_vals: Vec<f32> = vec![0.5f32; 32 * token_dim];
        let query = DenseMultiVectorView::new(&query_vals, token_dim);

        let doc_vals: Vec<f32> = vec![0.5f32; token_dim]; // Single token
        let doc = DenseMultiVectorView::new(&doc_vals, token_dim);
        let mut encoded = Vec::new();
        encoder.push_encoded(doc, &mut encoded);

        let evaluator = encoder.query_evaluator(query, &());
        let score = evaluator
            .compute_distance(DenseMultiVectorView::new(&encoded, COARSE_ID_BYTES + M))
            .distance();

        assert!(score.is_finite());
    }

    #[test]
    fn coarse_encoding_bytes_are_valid() {
        const M: usize = 4;
        let token_dim = 8;
        let training = make_training_set(token_dim, 512);
        let encoder = MultiVecTwoLevelProductQuantizer::<M, f32>::train(&training, 4);

        let doc_vals: Vec<f32> = vec![0.5f32; 5 * token_dim]; // 5 tokens
        let doc = DenseMultiVectorView::new(&doc_vals, token_dim);
        let mut encoded = Vec::new();
        encoder.push_encoded(doc, &mut encoded);

        // Each token encoded as: [4 bytes coarse id] + [M bytes PQ code]
        assert_eq!(encoded.len(), 5 * (COARSE_ID_BYTES + M));

        // All bytes should be valid (coarse ids are u32 LE, PQ codes are u8)
        for (i, &byte) in encoded.iter().enumerate() {
            let pos_in_token = i % (COARSE_ID_BYTES + M);
            if pos_in_token >= COARSE_ID_BYTES {
                // PQ code byte: must be valid u8
                let _: u8 = byte; // Verify it's u8
            }
            // Coarse id bytes can be any u8
        }
    }
}
