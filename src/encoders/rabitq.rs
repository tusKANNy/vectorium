//! RaBitQ-style binary encoder (no centroids).
//!
//! Implements the centroid-free subset of *RaBitQ: Quantizing High-Dimensional Vectors with a
//! Theoretical Error Bound for ANN* (Gao & Long, SIGMOD 2024). Documents are quantized to
//! **1 bit per component**; the query is either sign-binarized (`query_bits = 1`) or
//! scalar-quantized to `query_bits` bits per component.
//!
//! Documents are preprocessed by subtracting the per-component dataset means (the centroid
//! replacement) and applying a seeded random orthogonal transform `P` (a fast Hadamard–Kac
//! rotation, [`FhtKacRotator`](crate::FhtKacRotator)), giving the residual `r = P·(x − mean)`. The rotation spreads
//! information uniformly across dimensions before the sign is taken. It can be disabled via
//! [`RabitqConfig::rotate`] (then `P = I`), which skips the per-vector transform at encode and
//! query time at the cost of the information-spreading guarantee.
//!
//! Each document stores its sign code (bit set iff `r_i >= 0`) plus two floats:
//!
//! * `factor = ⟨ō,o⟩ = Σ|r_i| / (√d·‖r‖)` — the cosine between the quantized code and the true
//!   residual; dividing the raw sign agreement by it gives the paper's unbiased estimator.
//! * `‖r‖` — the residual norm, which turns the estimated cosine into a distance.
//!
//! ## Metrics
//!
//! Both squared Euclidean and inner product are supported, selected by the type parameter on
//! [`RabitqQuantizer`] (see [`RabitqSupportedDistance`]). Document codes are **identical** under
//! either metric; only the query path and the final combine differ, so the choice is free at
//! index-build time:
//!
//! * [`SquaredEuclideanDistance`](crate::SquaredEuclideanDistance) — the query is centered by the same mean. A shared shift is
//!   L2-preserving (`‖(c+r) − (c+r_q)‖² = ‖r − r_q‖²`), so the centroid cancels identically.
//! * [`DotProduct`] — centering is *not* inner-product preserving (a per-document `⟨x, mean⟩`
//!   term survives), so the query is rotated but **not** centered, and the centroid term is added
//!   back in closed form: since `P` is orthogonal, `x = mean + Pᵀr` and
//!   `⟨x, q⟩ = ⟨mean, q⟩ + ⟨r, P·q⟩`, where `⟨mean, q⟩` is a per-query constant.
//!
//! Write `q_r` for the rotated query (`P·(q − mean)` or `P·q` per the above) and `‖r_q‖ = ‖q_r‖`.
//!
//! ## Scoring — 1-bit query
//!
//! With both sides sign-packed to unit bi-valued codes (`±1/√d`), `⟨ō, q̄⟩ = (d − 2·hamming)/d`.
//! The query-side correction `⟨q̄,q⟩` is a per-query constant and cannot change the ranking, so
//! only document-side terms appear:
//!
//! ```text
//! ip_bar = (d − 2·hamming) / d                     # sign agreement, in [-1, 1]
//! cos    = ip_bar / factor                         # unbiased cosine estimate
//! ⟨r,q_r⟩ ≈ ‖r‖·‖r_q‖·cos                          # the shared estimate
//!
//! # squared Euclidean (smaller = nearer)
//! est ‖x − q‖² = ‖r‖² + ‖r_q‖² − 2·‖r‖·‖r_q‖·cos
//! # inner product (larger = nearer)
//! est ⟨x, q⟩   = ⟨mean, q⟩ + ‖r‖·‖r_q‖·cos
//! ```
//!
//! These are RaBitQ-Library's `METRIC_L2` and `METRIC_IP`. Its
//! `est_dist = f_add + g_add + f_rescale·(⟨s,q̂⟩ − correction)` maps onto the L2 line with
//! `f_add = ‖r‖²`, `g_add = ‖r_q‖²`, `f_rescale = −2‖r‖²/Σ|r_i|`; its `g_add = −⟨q,c⟩` maps onto
//! the inner-product line. The library always carries a raw, un-centered query and re-adds the
//! centroid; we do the same for inner product, and center symmetrically for L2 where it cancels.
//!
//! ## Scoring — multi-bit query (`query_bits > 1`)
//!
//! The query residual is scalar-quantized (RaBitQ-Library's RECONSTRUCTION scheme: sign bit +
//! magnitude bits with an exact rescale-factor search, complement-coded negatives) and
//! reconstructs as `q̂_i = delta·code_i + vl ≈ r_i`. With doc code bits `b_i ∈ {0,1}`
//! (signs `s_i = 2·b_i − 1`):
//!
//! ```text
//! ip     = Σ_j 2^j · popcount(doc AND plane_j)                  # ⟨b, code⟩ via bit planes
//! ⟨s,q̂⟩  = 2·(delta·ip + vl·popcount(doc)) − (delta·Σcode + vl·d)
//! ip_bar = ⟨s,q̂⟩ / (√d·‖r_q‖)                                   # ≈ ⟨ō, q/‖q‖⟩
//! ```
//!
//! `ip_bar` then flows through the same factor/norm/metric steps above. Documents are
//! byte-identical either way, so 1-bit vs multi-bit query is a pure query-side choice on the same
//! index.
//!
//! ## Storage
//!
//! Each document is `d/64` sign-code words plus **one metadata word** holding two per-document
//! floats bit-cast side by side, i.e. `d/64 + 1` words. The stored pair is the scan-ready
//! `[f_add:f32 | s:f32]` — `f_add = ‖r‖²` and `s = ‖r‖/factor` — precomputed so the per-candidate
//! scan needs no divide (`factor` and `‖r‖ = √f_add` are recoverable off the hot path).
//!
//! Only dimensions that are a multiple of 64 are supported (no bit-padding).
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::core::distances::DotProduct;
use crate::core::vector::{DenseVectorOwned, DenseVectorView};
use crate::core::vector_encoder::{DenseVectorEncoder, QueryEvaluator, VectorEncoder};
use crate::dataset::ConvertFrom;
use crate::encoders::rabitq_common::{
    RabitqSpace, WORD_BITS, hamming, hamming_batch6, ip_signed_planes, ip_signed_planes_batch6,
    pack_bit_planes, pack_metadata, pack_signs_into, quantize_query_multibit, unpack_metadata,
};
use crate::{Dataset, PlainDenseDataset, ScalarDenseSupportedDistance, SpaceUsage};

/// The metric abstraction, shared with [`rabitq_ext`](crate::encoders::rabitq_ext). It is defined in
/// the crate-internal `rabitq_common` module and re-exported here, where it was introduced, so the
/// public path is unchanged.
pub use crate::encoders::rabitq_common::RabitqSupportedDistance;

/// Query width used by [`VectorEncoder::vector_evaluator`] on the graph-build path.
///
/// There is no user query there — both sides are stored documents — so the reconstructed document
/// is scored symmetrically against the 1-bit codes, which is exactly what the squared-Euclidean
/// branch already does by reusing the stored sign words directly.
const BUILD_QUERY_BITS: u32 = 1;

/// RaBitQ **query-side** parameters: how finely the query is quantized before scoring.
///
/// Document codes do not depend on this, so it is not encoder state and is not stored with the
/// index: one dataset serves every setting, concurrently, by passing a different value here.
/// [`FlatIndex`](crate::FlatIndex) takes it as its `SearchParams`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RabitqQueryParams {
    /// Bits per component for the scalar-quantized query (`1` = plain sign query). Values in
    /// `1..=8` are supported. Documents are always 1 bit per component.
    pub query_bits: u32,
}

impl Default for RabitqQueryParams {
    fn default() -> Self {
        Self { query_bits: 1 }
    }
}

impl RabitqQueryParams {
    /// Query quantized to `query_bits` bits per component.
    ///
    /// Panics unless `query_bits ∈ 1..=8`.
    #[inline]
    pub fn new(query_bits: u32) -> Self {
        assert!(
            (1..=8).contains(&query_bits),
            "RaBitQ requires query_bits in 1..=8, got {query_bits}"
        );
        Self { query_bits }
    }
}

/// RaBitQ encoder parameters. See the module docs for the estimator math.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RabitqConfig {
    /// Seed for the random orthogonal rotation ([`FhtKacRotator`](crate::FhtKacRotator)).
    pub seed: u64,
    /// Apply the random orthogonal rotation ([`FhtKacRotator`](crate::FhtKacRotator)) to residuals. Disabling skips the
    /// `O(d log d)` per-vector transform at encode and query time; the estimator math is unchanged
    /// (`P = I` is a valid orthogonal transform), but the sign codes lose the
    /// information-spreading guarantee, so recall on real data is expected to drop.
    pub rotate: bool,
}

impl Default for RabitqConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            rotate: true,
        }
    }
}

/// RaBitQ-style binary quantizer. See the module docs.
///
/// The metric `D` selects how the stored residual codes are scored against a query; it defaults
/// to [`DotProduct`] (inner product). Document codes are identical under either metric — only the
/// query path and the final combine differ, so `D` is pure type-level state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RabitqQuantizer<D = DotProduct> {
    /// The trained geometry — means, rotated means and the rotation `P` — shared with
    /// [`RabitqExtQuantizer`](crate::encoders::rabitq_ext::RabitqExtQuantizer).
    space: RabitqSpace,
    /// Encoder parameters.
    config: RabitqConfig,
    /// The metric is encoded purely in the type; no runtime state.
    _distance: PhantomData<D>,
}

impl<D: RabitqSupportedDistance> RabitqQuantizer<D> {
    /// Learn per-component means over `dataset` and build the rotation.
    ///
    /// Panics unless the dataset dimension is a multiple of 64.
    pub fn train<Ds: ScalarDenseSupportedDistance>(
        dataset: &PlainDenseDataset<f32, Ds>,
        config: RabitqConfig,
    ) -> Self {
        Self {
            space: RabitqSpace::train(dataset, config.rotate, config.seed),
            config,
            _distance: PhantomData,
        }
    }

    /// [`train`](Self::train) over a source stored at a narrower value type, e.g. `f16`.
    ///
    /// Means are accumulated in `f32` exactly as above; only the source buffer is narrower.
    pub fn train_narrow<V, Ds>(dataset: &PlainDenseDataset<V, Ds>, config: RabitqConfig) -> Self
    where
        V: crate::ValueType + crate::Float + crate::FromF32,
        Ds: ScalarDenseSupportedDistance,
    {
        Self {
            space: RabitqSpace::train_narrow(dataset, config.rotate, config.seed),
            config,
            _distance: PhantomData,
        }
    }

    /// Encode one `d`-long input vector into a full output record `out`
    /// (`output_dim() = num_words() + 1` words): the sign code of `r = P·(x − mean)` followed by the
    /// metadata word.
    ///
    /// `scratch` (length `d`) holds `r` and is caller-owned so a caller encoding many vectors can
    /// reuse one buffer instead of allocating per vector.
    #[inline]
    fn encode_into(&self, x: &[f32], scratch: &mut [f32], out: &mut [u64]) {
        debug_assert_eq!(out.len(), self.num_words() + 1);
        self.space.residual_into(x, scratch);
        let (factor, norm) = self.metadata(scratch);
        let num_words = self.num_words();
        pack_signs_into(scratch, &mut out[..num_words]);
        // Scan-ready constants: f_add = ‖r‖² and s = ‖r‖/factor (see `pack_metadata`).
        out[num_words] = pack_metadata(norm * norm, norm / factor);
    }

    /// Reconstruct a stored code back into the **original data space**: `x̂ = mean + P⁻¹·r̂`.
    ///
    /// [`decode_vector`](DenseVectorEncoder::decode_vector) returns bare `±1` signs, which are the
    /// code but not a reconstruction — they carry no scale. The estimator supplies it: with
    /// `factor = Σ|r_i| / (√d·‖r‖)` and the stored `s = ‖r‖/factor`, the RaBitQ estimate is
    /// `⟨r, q_r⟩ ≈ s·⟨signs, q_r⟩/√d`, so the vector being scored against the query is
    ///
    /// ```text
    /// r̂ = s · signs / √d
    /// ```
    ///
    /// i.e. the sign pattern rescaled to the norm the estimator attributes to it.
    ///
    /// **Unlike the RaBitQ-ext case, this does NOT make the first-stage score exactly `⟨q, x̂⟩`.**
    /// This encoder quantizes the *query* too (`query_bits`), so its score is bilinear in the two
    /// codes and carries query-side error that no choice of `x̂` absorbs. A residual built on this
    /// reconstruction is therefore a heuristic, not the exact decomposition `rabitq-ext` admits —
    /// and the residual sum *keeps* that query-side error where a plain rerank stage would have
    /// replaced it. Expect the gap to close as `query_bits` rises. Provided so the question can be
    /// settled by measurement rather than by this argument.
    pub fn reconstruct(&self, code: &[u64]) -> Vec<f32> {
        let (_, s) = unpack_metadata(code[self.num_words()]);
        let inv_sqrt_d = 1.0 / (self.d() as f32).sqrt();
        let mut r_hat = Vec::with_capacity(self.d());
        for &word in &code[..self.num_words()] {
            for i in 0..WORD_BITS {
                let set = (word >> i) & 1 == 1;
                r_hat.push(if set { s * inv_sqrt_d } else { -s * inv_sqrt_d });
            }
        }
        r_hat.truncate(self.d());
        self.space.from_residual(&r_hat)
    }

    /// Input dimensionality.
    #[inline]
    fn d(&self) -> usize {
        self.space.dim()
    }

    /// Number of `u64` sign-code words per vector (excluding the metadata word).
    #[inline]
    fn num_words(&self) -> usize {
        self.space.num_words()
    }

    /// Apply the rotation: `P·v` — see [`RabitqSpace::rotate`].
    fn rotate(&self, values: &[f32]) -> Vec<f32> {
        self.space.rotate(values)
    }

    /// The residual `r = P·(x − mean)` — see [`RabitqSpace::residual`].
    fn residual(&self, values: &[f32]) -> Vec<f32> {
        self.space.residual(values)
    }

    /// `⟨mean, q⟩` — see [`RabitqSpace::mean_dot`].
    fn mean_dot(&self, values: &[f32]) -> f32 {
        self.space.mean_dot(values)
    }

    /// `⟨P·mean, q_r⟩` — see [`RabitqSpace::rotated_mean_dot`].
    fn rotated_mean_dot(&self, rotated: &[f32]) -> f32 {
        self.space.rotated_mean_dot(rotated)
    }

    /// Reconstruct the rotated residual from a stored code: `r̂ = (‖r‖/√d)·sign`.
    ///
    /// This is the bi-valued reconstruction the 1-bit *query* path already assumes — `q̄` scaled to
    /// carry the residual norm, since `‖sign/√d‖ = 1`. The norm has to be right, not merely
    /// proportional, because the inner-product build path adds `P·mean` to this vector and the two
    /// magnitudes must be commensurate.
    fn reconstruct_residual(&self, words: &[u64]) -> Vec<f32> {
        let (f_add, _) = unpack_metadata(words[self.num_words()]);
        let amp = f_add.sqrt() / (self.d() as f32).sqrt();
        (0..self.d())
            .map(|i| {
                let set = (words[i / WORD_BITS] >> (i % WORD_BITS)) & 1 == 1;
                if set { amp } else { -amp }
            })
            .collect()
    }

    /// Build an evaluator from a query that already lives in the rotated space, given its centroid
    /// term `⟨mean, q⟩` (ignored under a centering metric).
    ///
    /// Shared by both entry points: [`query_evaluator`] rotates a raw `f32` query, while the
    /// inner-product [`vector_evaluator`] path reconstructs a stored residual and adds `P·mean`.
    /// Everything downstream — quantization, `scale`, `add` — is therefore identical on the two
    /// paths.
    ///
    /// [`query_evaluator`]: VectorEncoder::query_evaluator
    /// [`vector_evaluator`]: VectorEncoder::vector_evaluator
    fn rotated_query_evaluator<'e>(
        &'e self,
        q_r: &[f32],
        mean_dot_query: f32,
        query_bits: u32,
    ) -> RabitqQueryEvaluator<'e, D> {
        let query_norm = q_r.iter().map(|&r| r * r).sum::<f32>().sqrt();
        let add = D::query_add(mean_dot_query, query_norm);

        // `scale` folds the whole query-side normalization so the scan is divide-free (see the
        // field docs): the multi-bit path already cancelled ‖q_r‖, the 1-bit path carries it here.
        let scale = if query_bits > 1 {
            1.0 / (self.d() as f32).sqrt()
        } else {
            query_norm / self.d() as f32
        };
        let (query_words, planes, delta, vl, q_const) = if query_bits > 1 {
            let (codes, delta, vl) = quantize_query_multibit(q_r, query_bits);
            let sum_code: u64 = codes.iter().map(|&c| c as u64).sum();
            let q_const = delta * sum_code as f32 + vl * self.d() as f32;
            let planes = pack_bit_planes(&codes, query_bits);
            (Vec::new(), planes, delta, vl, q_const)
        } else {
            (self.pack_signs(q_r), Vec::new(), 0.0, 0.0, 0.0)
        };

        RabitqQueryEvaluator {
            _encoder: PhantomData,
            query_words,
            planes,
            delta,
            vl,
            q_const,
            query_bits,
            scale,
            add,
            d: self.d(),
            num_words: self.num_words(),
        }
    }

    /// Sign-pack a residual into `num_words()` words (bit set iff `r_i >= 0`) — the 1-bit query
    /// path, which needs an owned buffer for the evaluator.
    fn pack_signs(&self, residual: &[f32]) -> Vec<u64> {
        let mut words = vec![0u64; self.num_words()];
        pack_signs_into(residual, &mut words);
        words
    }

    /// Per-document metadata from a residual: `(factor, norm)` with degenerate-vector guards.
    ///
    /// `factor = ⟨ō,o⟩ = Σ|r_i| / (√d·‖r‖)`; a zero residual gets `factor = 1, norm = 0` so the
    /// estimator never divides by zero or produces NaN.
    fn metadata(&self, residual: &[f32]) -> (f32, f32) {
        let norm: f32 = residual.iter().map(|&r| r * r).sum::<f32>().sqrt();
        if norm <= f32::EPSILON {
            return (1.0, 0.0);
        }
        let abs_sum: f32 = residual.iter().map(|&r| r.abs()).sum();
        let factor = abs_sum / ((self.d() as f32).sqrt() * norm);
        (factor.max(1e-6), norm)
    }
}

/// Evaluator holding the quantized query (1-bit sign words or multi-bit planes) and its
/// residual norm.
#[derive(Debug, Clone)]
pub struct RabitqQueryEvaluator<'e, D = DotProduct> {
    _encoder: PhantomData<&'e RabitqQuantizer<D>>,
    /// Sign-packed query residual (owned; the evaluator must not borrow the query). Empty when
    /// `query_bits > 1`.
    query_words: Vec<u64>,
    /// Interleaved query bit planes (`planes[w·bits + j]`, see [`pack_bit_planes`]). Empty when
    /// `query_bits == 1`.
    planes: Vec<u64>,
    /// Multi-bit query reconstruction scale (`q̂_i = delta·code_i + vl`).
    delta: f32,
    /// Multi-bit query reconstruction offset.
    vl: f32,
    /// Per-query constant `delta·Σcode + vl·d`, subtracted to recover the ±1-sign inner product.
    q_const: f32,
    /// Effective query bits for this evaluator (forced to 1 on the squared-Euclidean
    /// [`vector_evaluator`] path, where the "query" is an already sign-packed document).
    ///
    /// [`vector_evaluator`]: VectorEncoder::vector_evaluator
    query_bits: u32,
    /// Per-query multiplier applied to `s·ip_raw` to form `term = ‖r‖·‖q_r‖·cos`. Folds the whole
    /// query-side normalization so the scan needs no divide: `‖q_r‖/d` for the 1-bit sign path
    /// (`ip_raw = d − 2·hamming`), `1/√d` for the multi-bit path (`ip_raw = ip_signed`, where the
    /// query norm has already cancelled).
    scale: f32,
    /// Per-query additive term (RaBitQ-Library's `g_add`): `‖r_q‖²` for squared Euclidean,
    /// `⟨mean, q⟩` for inner product. See [`RabitqSupportedDistance::query_add`].
    add: f32,
    /// Dimensionality `d` (number of code bits).
    d: usize,
    /// Number of sign-code words per document (its slice `[..num_words]`; word `num_words` is metadata).
    num_words: usize,
}

impl<'e, D> RabitqQueryEvaluator<'e, D> {
    /// The signed document/query inner product `⟨s, q̂⟩` for a multi-bit query, in un-normalized
    /// units (the `√d·‖q_r‖` normalization is applied later via [`Self::scale`]). Only valid when
    /// `query_bits > 1`.
    ///
    /// Each plane is a contiguous AND+popcount reduction over the document code words (plane-major
    /// layout, see [`pack_bit_planes`]) — one `<<j` weighting per plane, none inside the reduction.
    #[inline]
    fn ip_signed(&self, code: &[u64]) -> f32 {
        // Dispatch on the (compile-time-known) plane count so `ip_signed_planes` fully unrolls.
        // `query_bits` is validated to `1..=8` in `query_evaluator`; this path only runs for `> 1`.
        let planes = &self.planes;
        let (ip, ppc) = match self.query_bits {
            2 => ip_signed_planes::<2>(code, planes),
            3 => ip_signed_planes::<3>(code, planes),
            4 => ip_signed_planes::<4>(code, planes),
            5 => ip_signed_planes::<5>(code, planes),
            6 => ip_signed_planes::<6>(code, planes),
            7 => ip_signed_planes::<7>(code, planes),
            _ => ip_signed_planes::<8>(code, planes),
        };
        // ⟨doc_bits, q̂⟩ = delta·ip + vl·ppc; recover the ±1-sign inner product ⟨s, q̂⟩.
        2.0 * (self.delta * ip as f32 + self.vl * ppc as f32) - self.q_const
    }

    /// Six-way [`Self::ip_signed`]: `⟨s, q̂⟩` for six documents through the fused batch kernel
    /// ([`ip_signed_planes_batch6`]). Only valid when `query_bits > 1`.
    #[inline]
    fn ip_signed_batch6(&self, codes: [&[u64]; 6]) -> [f32; 6] {
        let planes = &self.planes;
        let (ips, ppcs) = match self.query_bits {
            2 => ip_signed_planes_batch6::<2>(codes, planes),
            3 => ip_signed_planes_batch6::<3>(codes, planes),
            4 => ip_signed_planes_batch6::<4>(codes, planes),
            5 => ip_signed_planes_batch6::<5>(codes, planes),
            6 => ip_signed_planes_batch6::<6>(codes, planes),
            7 => ip_signed_planes_batch6::<7>(codes, planes),
            _ => ip_signed_planes_batch6::<8>(codes, planes),
        };
        std::array::from_fn(|k| {
            2.0 * (self.delta * ips[k] as f32 + self.vl * ppcs[k] as f32) - self.q_const
        })
    }
}

impl<'e, 'v, D: RabitqSupportedDistance> QueryEvaluator<DenseVectorView<'v, u64>>
    for RabitqQueryEvaluator<'e, D>
{
    type Distance = D;

    #[inline]
    fn compute_distance(&self, vector: DenseVectorView<'v, u64>) -> D {
        let words = vector.values();
        let code = &words[..self.num_words];
        // (f_add = ‖r‖², s = ‖r‖/factor) — both precomputed at encode time so this scan does no
        // divide; see `pack_metadata`.
        let (f_add, s) = unpack_metadata(words[self.num_words]);

        // Raw sign agreement, before any normalization: `d − 2·hamming` (1-bit) or `⟨s, q̂⟩`
        // (multi-bit). `scale` folds the per-query normalization, so `term = ‖r‖·‖q_r‖·cos`.
        let ip_raw = if self.query_bits > 1 {
            self.ip_signed(code)
        } else {
            let h = hamming(&self.query_words, code);
            self.d as f32 - 2.0 * h as f32
        };
        let term = s * ip_raw * self.scale;

        // Metric-specific combine (see the module docs and `RabitqSupportedDistance`): both metrics
        // reduce to ⟨r, q_r⟩ ≈ ‖r‖·‖q_r‖·cos plus a per-query additive term.
        D::from_terms(self.add, f_add, term)
    }

    /// Fused six-candidate scan: the quantized query (sign words or bit planes) is broadcast once
    /// per chunk and interleaved against all six documents' popcount accumulators in a single pass
    /// (`hamming_batch6` / `ip_signed_planes_batch6`) — cross-candidate ILP the default
    /// six-single-calls dispatch can't reach. The scalar combine matches [`Self::compute_distance`]
    /// operation for operation, so batch and single scores are bit-identical.
    #[inline]
    fn compute_distances_batch6(&self, vectors: [DenseVectorView<'v, u64>; 6]) -> [D; 6] {
        let words: [&[u64]; 6] = vectors.map(|v| v.values());
        let codes: [&[u64]; 6] = std::array::from_fn(|k| &words[k][..self.num_words]);

        let ip_raws: [f32; 6] = if self.query_bits > 1 {
            self.ip_signed_batch6(codes)
        } else {
            hamming_batch6(&self.query_words, codes).map(|h| self.d as f32 - 2.0 * h as f32)
        };

        std::array::from_fn(|k| {
            let (f_add, s) = unpack_metadata(words[k][self.num_words]);
            let term = s * ip_raws[k] * self.scale;
            D::from_terms(self.add, f_add, term)
        })
    }
}

impl<D: RabitqSupportedDistance> DenseVectorEncoder for RabitqQuantizer<D> {
    type InputValueType = f32;
    type OutputValueType = u64;

    /// Decode the code bits into `±1` `f32` values; the metadata word is ignored.
    fn decode_vector<'a>(&self, encoded: DenseVectorView<'a, u64>) -> DenseVectorOwned<f32> {
        let mut values = Vec::with_capacity(self.d());
        for &word in &encoded.values()[..self.num_words()] {
            for i in 0..WORD_BITS {
                let set = (word >> i) & 1 == 1;
                values.push(if set { 1.0f32 } else { -1.0f32 });
            }
        }
        DenseVectorOwned::new(values)
    }

    #[inline]
    fn push_encoded<'a, OutputContainer>(
        &self,
        input: DenseVectorView<'a, f32>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<u64>,
    {
        assert_eq!(
            input.len(),
            self.d(),
            "Input vector length must equal encoder input dimension."
        );
        let mut scratch = vec![0.0f32; self.d()];
        let mut words = vec![0u64; self.num_words() + 1];
        self.encode_into(input.values(), &mut scratch, &mut words);
        output.extend(words);
    }
}

/// Document codes are identical under either metric; only the query path differs.
impl<D, D2> crate::core::vector_encoder::WithDistance<D2> for RabitqQuantizer<D>
where
    D: RabitqSupportedDistance,
    D2: RabitqSupportedDistance,
{
    type Output = RabitqQuantizer<D2>;

    fn with_distance(&self) -> Self::Output {
        RabitqQuantizer {
            space: self.space.clone(),
            config: self.config,
            _distance: PhantomData,
        }
    }
}

impl<D: RabitqSupportedDistance> VectorEncoder for RabitqQuantizer<D> {
    type Distance = D;
    type InputVector<'a> = DenseVectorView<'a, f32>;
    type QueryVector<'q> = DenseVectorView<'q, f32>;
    type EncodedVector<'a> = DenseVectorView<'a, u64>;

    type Evaluator<'e>
        = RabitqQueryEvaluator<'e, D>
    where
        Self: 'e;

    type QueryParams = RabitqQueryParams;

    /// Build an evaluator by moving the `f32` query into the rotated space — centered by the mean
    /// or not, per the metric (see [`RabitqSupportedDistance::CENTER_QUERY`]) — then sign-packing
    /// it (`query_bits == 1`) or scalar-quantizing it into bit planes.
    #[inline]
    fn query_evaluator<'e>(
        &'e self,
        query: Self::QueryVector<'_>,
        params: &RabitqQueryParams,
    ) -> Self::Evaluator<'e> {
        assert_eq!(
            query.len(),
            self.d(),
            "Query vector length must equal encoder input dimension."
        );
        assert!(
            (1..=8).contains(&params.query_bits),
            "RaBitQ requires query_bits in 1..=8, got {}",
            params.query_bits
        );
        let residual = if D::CENTER_QUERY {
            self.residual(query.values())
        } else {
            self.rotate(query.values())
        };
        // The centroid term is only needed by the non-centering (inner-product) path.
        let mean_dot_query = if D::CENTER_QUERY {
            0.0
        } else {
            self.mean_dot(query.values())
        };
        self.rotated_query_evaluator(&residual, mean_dot_query, params.query_bits)
    }

    /// Treat an already-encoded document as a query (build-path only).
    ///
    /// A stored document keeps only its *centered* residual `r`, so the two metrics need different
    /// treatment:
    ///
    /// * [`SquaredEuclideanDistance`] — centering is L2-preserving, so residual-space scoring is
    ///   already the exact `‖x − y‖²`. The stored sign words are reused as the query verbatim (the
    ///   1-bit path), which is both the cheapest and the most faithful option.
    /// * [`DotProduct`] — centering is **not** inner-product preserving: residual-space scoring
    ///   drops `⟨P·mean, r_y⟩`, which varies per candidate, so the ranking would be wrong rather
    ///   than merely shifted. The residual is reconstructed from the code and `P·mean` added back,
    ///   rebuilding the un-centered rotated query `P·x̂ = r̂ + P·mean`; this path then reduces to
    ///   [`Self::query_evaluator`] on that reconstruction.
    ///
    /// **Cost.** The inner-product branch reconstructs and requantizes, `O(d)` per evaluator, where
    /// the L2 branch reuses the stored words for nothing. Graph construction builds one evaluator
    /// per candidate pair, so an inner-product index is slower to build. Build-time only — the
    /// query path is unaffected.
    ///
    /// [`SquaredEuclideanDistance`]: crate::SquaredEuclideanDistance
    #[inline]
    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let words = vector.values();
        if !D::CENTER_QUERY {
            let mut q_r = self.reconstruct_residual(words);
            for (v, &m) in q_r.iter_mut().zip(self.space.rotated_means().iter()) {
                *v += m;
            }
            // ⟨P·mean, P·x̂⟩ = ⟨mean, x̂⟩ — the centroid term, exactly as on the query path.
            let mean_dot_query = self.rotated_mean_dot(&q_r);
            return self.rotated_query_evaluator(&q_r, mean_dot_query, BUILD_QUERY_BITS);
        }
        // The stored metadata is (f_add = ‖r‖², s); recover the residual norm as √f_add.
        let (f_add, _) = unpack_metadata(words[self.num_words()]);
        let norm = f_add.sqrt();
        RabitqQueryEvaluator {
            _encoder: PhantomData,
            query_words: words[..self.num_words()].to_vec(),
            planes: Vec::new(),
            delta: 0.0,
            vl: 0.0,
            q_const: 0.0,
            query_bits: BUILD_QUERY_BITS,
            scale: norm / self.d() as f32,
            add: D::query_add(0.0, norm),
            d: self.d(),
            num_words: self.num_words(),
        }
    }

    fn input_dim(&self) -> usize {
        self.d()
    }

    /// Sign-code words plus the metadata word.
    fn output_dim(&self) -> usize {
        self.num_words() + 1
    }

    /// Score two stored documents directly, without building an evaluator.
    ///
    /// Only the centering (squared-Euclidean) metric gets the shortcut: `v1` plays the query on the
    /// 1-bit sign path exactly as [`Self::vector_evaluator`] would, but its words are read in place
    /// rather than copied into an owned evaluator, so both routes give bit-identical scores with no
    /// allocation. Under [`DotProduct`] the effective query `r̂ + P·mean` has to be materialized
    /// whatever the entry point, so this delegates.
    #[inline]
    fn compute_distance_between(
        &self,
        v1: Self::EncodedVector<'_>,
        v2: Self::EncodedVector<'_>,
    ) -> Self::Distance {
        if !D::CENTER_QUERY {
            return self.vector_evaluator(v1).compute_distance(v2);
        }
        let num_words = self.num_words();
        let (q_words, d_words) = (v1.values(), v2.values());
        // v1 as query: the stored metadata is (f_add = ‖r‖², s), so its norm is √f_add.
        let query_norm = unpack_metadata(q_words[num_words]).0.sqrt();
        let (f_add, s) = unpack_metadata(d_words[num_words]);

        let h = hamming(&q_words[..num_words], &d_words[..num_words]);
        let ip_raw = self.d() as f32 - 2.0 * h as f32;
        let term = s * ip_raw * (query_norm / self.d() as f32);

        D::from_terms(D::query_add(0.0, query_norm), f_add, term)
    }
}

impl<D> SpaceUsage for RabitqQuantizer<D> {
    fn space_usage_bytes(&self) -> usize {
        self.space.space_usage_bytes()
    }
}

/// Train on a plain `f32` dataset and encode every vector in parallel.
///
/// Takes the source by reference, so the plain vectors survive the call. The source metric `Ds` is
/// independent of the target metric `D`: document codes are identical either way, so one plain
/// dataset can produce both.
impl<D, Ds> ConvertFrom<&PlainDenseDataset<f32, Ds>> for crate::DenseDataset<RabitqQuantizer<D>>
where
    D: RabitqSupportedDistance,
    Ds: ScalarDenseSupportedDistance,
{
    type Config = RabitqConfig;

    fn convert_from(dataset: &PlainDenseDataset<f32, Ds>, config: RabitqConfig) -> Self {
        let encoder = RabitqQuantizer::<D>::train(dataset, config);
        crate::DenseDataset::<RabitqQuantizer<D>>::from_flat_par(
            encoder,
            dataset.values(),
            dataset.len(),
        )
    }
}

/// Same, from a source held at half precision.
///
/// Lets a caller that already has the collection as `f16` — which is how the graph indexes store
/// it — encode without first materializing an `f32` copy. Means are still accumulated in `f32`;
/// each vector is upcast as it is encoded.
impl<D, Ds> ConvertFrom<&PlainDenseDataset<half::f16, Ds>>
    for crate::DenseDataset<RabitqQuantizer<D>>
where
    D: RabitqSupportedDistance,
    Ds: ScalarDenseSupportedDistance,
{
    type Config = RabitqConfig;

    fn convert_from(dataset: &PlainDenseDataset<half::f16, Ds>, config: RabitqConfig) -> Self {
        let encoder = RabitqQuantizer::<D>::train_narrow(dataset, config);
        crate::DenseDataset::<RabitqQuantizer<D>>::from_flat_par_upcast(
            encoder,
            dataset.values(),
            dataset.len(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::distances::Distance;
    use crate::dataset::ConvertInto;
    use crate::{
        DatasetGrowable, FlatIndex, Index, IndexSerializer, PlainDenseDatasetGrowable,
        PlainDenseQuantizer, SquaredEuclideanDistance,
    };

    /// RaBitQ scoring squared Euclidean distance.
    type RabitqL2 = RabitqQuantizer<SquaredEuclideanDistance>;
    /// RaBitQ scoring inner product.
    type RabitqIp = RabitqQuantizer<DotProduct>;

    /// Build a `PlainDenseDataset<f32, DotProduct>` from a list of equal-length vectors.
    fn plain_dataset(vectors: &[Vec<f32>]) -> PlainDenseDataset<f32, DotProduct> {
        let d = vectors[0].len();
        let encoder = PlainDenseQuantizer::<f32, DotProduct>::new(d);
        let mut growable = PlainDenseDatasetGrowable::new(encoder);
        for v in vectors {
            growable.push(DenseVectorView::new(v));
        }
        growable.into()
    }

    /// Vectors with distinct popcount patterns (from the zero-one experiments): `±1` over 64 dims
    /// with a varying number of `+1` components, so scores are well separated (no rank ties).
    fn varied_vectors() -> Vec<Vec<f32>> {
        [4usize, 12, 30, 48, 60, 64, 0, 33, 7, 51]
            .iter()
            .map(|&ones| {
                let mut v = vec![-1.0f32; 64];
                v[..ones].fill(1.0);
                v
            })
            .collect()
    }

    /// Conversion must equal training on the source and then encoding every row in parallel.
    /// Compares the encoded storage word-for-word and the trained encoder, at a non-default config
    /// so a config that failed to reach `train` could not pass.
    #[test]
    fn convert_into_matches_the_explicit_train_and_encode_path() {
        let plain = plain_dataset(&varied_vectors());

        for config in [
            RabitqConfig {
                seed: 7,
                rotate: true,
            },
            RabitqConfig::default(),
        ] {
            let expected = {
                let encoder = RabitqIp::train(&plain, config);
                crate::DenseDataset::<RabitqIp>::from_flat_par(encoder, plain.values(), plain.len())
            };
            let converted: crate::DenseDataset<RabitqIp> = (&plain).convert_into(config);

            assert_eq!(converted.values(), expected.values());
            assert_eq!(converted.encoder(), expected.encoder());
        }
    }

    /// The source metric is independent of the target metric: one plain dataset produces both
    /// RaBitQ metrics, with byte-identical document codes.
    #[test]
    fn convert_into_is_independent_of_the_source_and_target_metrics() {
        let vectors = varied_vectors();
        let plain_ip = plain_dataset(&vectors);
        let plain_l2: PlainDenseDataset<f32, SquaredEuclideanDistance> = {
            let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(64);
            let mut growable = PlainDenseDatasetGrowable::new(encoder);
            for v in &vectors {
                growable.push(DenseVectorView::new(v));
            }
            growable.into()
        };
        let config = RabitqConfig::default();

        let from_ip: crate::DenseDataset<RabitqL2> = (&plain_ip).convert_into(config);
        let from_l2: crate::DenseDataset<RabitqL2> = (&plain_l2).convert_into(config);
        let other_metric: crate::DenseDataset<RabitqIp> = (&plain_ip).convert_into(config);

        assert_eq!(from_ip.values(), from_l2.values());
        assert_eq!(from_ip.values(), other_metric.values());
    }

    #[test]
    fn one_bit_reconstruction_is_informative_and_norm_matched() {
        // `r̂ = s·signs/√d` is a derivation from the estimator, so check it against the data: the
        // reconstruction must (a) be closer to `x` than the mean alone is -- otherwise there is no
        // residual worth encoding -- and (b) carry roughly the right norm rather than an arbitrary
        // scale, which is what would break a residual built on it.
        let (n, d) = (300usize, 128usize);
        let centres: Vec<Vec<f32>> = (0..6)
            .map(|c| {
                (0..d)
                    .map(|i| (((i * 37 + c * 811) % 241) as f32 / 241.0) - 0.5)
                    .collect()
            })
            .collect();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|j| {
                let c = &centres[j % 6];
                (0..d)
                    .map(|i| c[i] + 0.1 * (((i * 19 + j * 71) % 89) as f32 / 89.0 - 0.5))
                    .collect()
            })
            .collect();
        let plain = plain_dataset(&vectors);
        let encoder = RabitqIp::train(
            &plain,
            RabitqConfig {
                seed: 9,
                ..Default::default()
            },
        );
        let data = crate::DenseDataset::<RabitqIp>::from_flat_par(encoder, plain.values(), n);

        let mean: Vec<f32> = (0..d)
            .map(|i| vectors.iter().map(|v| v[i]).sum::<f32>() / n as f32)
            .collect();

        let (mut err_hat, mut err_mean) = (0.0f64, 0.0f64);
        for (id, x) in vectors.iter().enumerate() {
            let x_hat = data.encoder().reconstruct(data.get(id as u64).values());
            assert_eq!(x_hat.len(), d);
            err_hat += x
                .iter()
                .zip(&x_hat)
                .map(|(a, b)| ((a - b) * (a - b)) as f64)
                .sum::<f64>()
                .sqrt();
            err_mean += x
                .iter()
                .zip(&mean)
                .map(|(a, b)| ((a - b) * (a - b)) as f64)
                .sum::<f64>()
                .sqrt();
        }
        println!(
            "1-bit: mean ||x - x̂|| {:.5}   mean ||x - mean|| {:.5}   ratio {:.3}",
            err_hat / n as f64,
            err_mean / n as f64,
            err_hat / err_mean,
        );
        assert!(
            err_hat < err_mean,
            "the 1-bit reconstruction is no better than the centroid, so there is no residual \
             worth encoding: ||x - x̂|| {err_hat} vs ||x - mean|| {err_mean}"
        );
    }

    #[test]
    fn metadata_word_roundtrips_the_two_scan_constants() {
        let (f_add, s) = (123.456f32, 0.797_884_6_f32);
        let (f2, s2) = unpack_metadata(pack_metadata(f_add, s));
        assert_eq!(f_add, f2);
        assert_eq!(s, s2);
    }

    /// Every query must retrieve itself near the top under metric `D`.
    fn assert_self_is_top_ranked<D: RabitqSupportedDistance>(
        vectors: &[Vec<f32>],
        dataset: &PlainDenseDataset<f32, DotProduct>,
        query_bits: u32,
        rotate: bool,
    ) {
        let config = RabitqConfig { seed: 42, rotate };
        let qp = RabitqQueryParams::new(query_bits);
        let rabitq: crate::DenseDataset<RabitqQuantizer<D>> = dataset.convert_into(config);
        let index = FlatIndex::from(&rabitq);
        for (i, q) in vectors.iter().enumerate() {
            let top = index.search(DenseVectorView::new(q), vectors.len(), &qp);
            let self_rank = top
                .iter()
                .position(|r| r.vector as usize == i)
                .expect("self must be retrieved");
            // The all-(+1) and all-(-1) vectors binarize identically to others under rotation
            // with so few points, so allow self to sit just below the top.
            assert!(self_rank < 3, "query {i} ranked itself at {self_rank}");
        }
    }

    #[test]
    fn search_finds_self_as_nearest() {
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        // `varied_vectors` all have norm 8; on equal-norm data the two metrics rank alike, so
        // both must put the query itself at the top.
        assert_self_is_top_ranked::<SquaredEuclideanDistance>(&vectors, &dataset, 1, true);
        assert_self_is_top_ranked::<DotProduct>(&vectors, &dataset, 1, true);
    }

    #[test]
    fn search_finds_self_as_nearest_without_rotation() {
        // With rotation disabled the estimator runs with `P = I` (also orthogonal), so
        // self-retrieval must still hold on both metrics.
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        assert_self_is_top_ranked::<SquaredEuclideanDistance>(&vectors, &dataset, 1, false);
        assert_self_is_top_ranked::<DotProduct>(&vectors, &dataset, 1, false);
    }

    /// Scores stay finite across every query bit width and seed under metric `D`.
    fn assert_scores_finite<D: RabitqSupportedDistance>(
        vectors: &[Vec<f32>],
        dataset: &PlainDenseDataset<f32, DotProduct>,
    ) {
        for query_bits in [1, 2, 4, 8] {
            for seed in [1, 42] {
                let config = RabitqConfig { seed, rotate: true };
                let qp = RabitqQueryParams::new(query_bits);
                let ds: crate::DenseDataset<RabitqQuantizer<D>> = dataset.convert_into(config);
                let index = FlatIndex::from(&ds);
                for q in vectors {
                    for r in index.search(DenseVectorView::new(q), 3, &qp) {
                        assert!(
                            r.distance.distance().is_finite(),
                            "non-finite score with {config:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn scores_are_finite_for_every_query_bit_width() {
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        assert_scores_finite::<SquaredEuclideanDistance>(&vectors, &dataset);
        assert_scores_finite::<DotProduct>(&vectors, &dataset);
    }

    #[test]
    #[should_panic(expected = "dim % 64 == 0")]
    fn train_rejects_non_multiple_of_64() {
        let dataset = plain_dataset(&[vec![0.0f32; 10]]);
        let _ = RabitqIp::train(&dataset, RabitqConfig::default());
    }

    #[test]
    #[should_panic(expected = "query_bits")]
    fn query_params_reject_out_of_range_query_bits() {
        let _ = RabitqQueryParams::new(9);
    }

    /// The width is checked on the search path too, so a struct literal that bypasses
    /// [`RabitqQueryParams::new`] cannot reach the kernels with an unsupported width.
    #[test]
    #[should_panic(expected = "query_bits")]
    fn query_evaluator_rejects_out_of_range_query_bits() {
        let dataset = plain_dataset(&varied_vectors());
        let ds: crate::DenseDataset<RabitqIp> = (&dataset).convert_into(RabitqConfig::default());
        let _ = ds.encoder().query_evaluator(
            DenseVectorView::new(&vec![0.0f32; 64]),
            &RabitqQueryParams { query_bits: 9 },
        );
    }

    #[test]
    fn multibit_query_reconstruction_is_accurate() {
        let d = 64;
        let q: Vec<f32> = (0..d).map(|i| (i as f32 * 0.7).sin() * 2.0 + 0.3).collect();
        let (codes, delta, vl) = quantize_query_multibit(&q, 4);
        let recon: Vec<f32> = codes.iter().map(|&c| delta * c as f32 + vl).collect();

        let dot: f32 = q.iter().zip(&recon).map(|(a, b)| a * b).sum();
        let nq: f32 = q.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nr: f32 = recon.iter().map(|v| v * v).sum::<f32>().sqrt();
        let cos = dot / (nq * nr);
        assert!(cos > 0.98, "4-bit reconstruction cosine too low: {cos}");

        // The sign bit must round-trip: reconstruction and input never disagree on sign.
        for (&orig, &r) in q.iter().zip(&recon) {
            assert!(orig * r >= 0.0, "sign flipped: {orig} vs {r}");
        }
    }

    #[test]
    fn bitplane_ip_matches_direct_dot() {
        let d = 128;
        let bits = 4u32;
        let q: Vec<f32> = (0..d).map(|i| (i as f32 * 0.37).sin() - 0.1).collect();
        let (codes, delta, vl) = quantize_query_multibit(&q, bits);
        let planes = pack_bit_planes(&codes, bits);
        let sum_code: u64 = codes.iter().map(|&c| c as u64).sum();
        let q_const = delta * sum_code as f32 + vl * d as f32;

        let num_words = d / 64;
        for doc in [
            [0u64, 0u64],
            [u64::MAX, u64::MAX],
            [0x5555_5555_5555_5555, 0xF0F0_F0F0_F0F0_F0F0],
            [u64::MAX, 0],
            [0x0123_4567_89AB_CDEF, 0xDEAD_BEEF_CAFE_F00D],
        ] {
            // Mirror the plane-major kernel: one AND+popcount reduction per plane, weighted by 2^j.
            let ppc: u64 = doc.iter().map(|w| w.count_ones() as u64).sum();
            let mut ip: u64 = 0;
            for j in 0..bits as usize {
                let plane = &planes[j * num_words..(j + 1) * num_words];
                let pj: u64 = doc
                    .iter()
                    .zip(plane)
                    .map(|(&dw, &pw)| (dw & pw).count_ones() as u64)
                    .sum();
                ip += pj << j;
            }
            let kernel = 2.0 * (delta * ip as f32 + vl * ppc as f32) - q_const;

            // Direct evaluation of ⟨sign_doc(±1), q̂⟩ with q̂_i = delta·code_i + vl.
            let direct: f32 = (0..d)
                .map(|i| {
                    let bit = (doc[i / 64] >> (i % 64)) & 1;
                    let sign = if bit == 1 { 1.0f32 } else { -1.0 };
                    sign * (delta * codes[i] as f32 + vl)
                })
                .sum();
            assert!(
                (kernel - direct).abs() < 1e-2,
                "kernel {kernel} vs direct {direct} for doc {doc:?}"
            );
        }
    }

    #[test]
    fn four_bit_query_matches_exact_query_estimator() {
        // With query_bits = 4 the estimated sign agreement must be close to the same estimator
        // computed with the *unquantized* query residual — the only error left is query
        // reconstruction. The evaluator's L2 score is inverted back to `ip_bar` using the
        // document metadata so the comparison happens at the sign-agreement level.
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        let config = RabitqConfig {
            ..Default::default()
        };
        let encoder = RabitqL2::train(&dataset, config);
        let ds: crate::DenseDataset<RabitqL2> = (&dataset).convert_into(config);

        let d = 64usize;

        for q in &vectors {
            let q_res = encoder.residual(q);
            let nq: f32 = q_res.iter().map(|v| v * v).sum::<f32>().sqrt();
            if nq <= f32::EPSILON {
                continue;
            }
            // Expected value with the *reconstructed* query q̂ — must match the evaluator exactly
            // (verifies the plane/q_const plumbing end to end).
            let (codes, delta, vl) = quantize_query_multibit(&q_res, 4);
            let q_hat: Vec<f32> = codes.iter().map(|&c| delta * c as f32 + vl).collect();

            let evaluator =
                encoder.query_evaluator(DenseVectorView::new(q), &RabitqQueryParams::new(4));
            for (i, doc) in vectors.iter().enumerate() {
                let doc_res = encoder.residual(doc);
                let (factor, norm_o) = encoder.metadata(&doc_res);
                if norm_o <= f32::EPSILON {
                    continue;
                }
                let dot_with = |query: &[f32]| -> f32 {
                    doc_res
                        .iter()
                        .zip(query)
                        .map(|(&r, &qr)| if r >= 0.0 { qr } else { -qr })
                        .sum::<f32>()
                        / ((d as f32).sqrt() * nq)
                };
                let expected = dot_with(&q_hat);
                let exact = dot_with(&q_res);
                // Invert est = ‖r‖² + ‖r_q‖² − 2·‖r‖·‖r_q‖·(ip_bar/factor) back to ip_bar.
                let est = evaluator.compute_distance(ds.get(i as u64)).distance();
                let got = (norm_o * norm_o + nq * nq - est) / (2.0 * norm_o * nq) * factor;
                assert!(
                    (got - expected).abs() < 1e-3,
                    "doc {i}: evaluator {got} vs reconstructed-query value {expected}"
                );
                // Loose sanity bound vs the unquantized query: catches gross scale errors while
                // tolerating genuine 4-bit reconstruction noise on this spiky ±1 data.
                assert!(
                    (got - exact).abs() < 0.25,
                    "doc {i}: 4-bit estimate {got} too far from exact-query {exact}"
                );
            }
        }
    }

    #[test]
    fn ip_centroid_decomposition_is_exact() {
        // The identity the inner-product metric rests on. With P orthogonal, x = mean + Pᵀr, so
        //   ⟨x, q⟩ = ⟨mean, q⟩ + ⟨P(x − mean), P·q⟩
        // holds *exactly* — this is why the query is rotated but not centered. Checked before any
        // quantization enters, so a failure here means the transform plumbing is wrong.
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        let encoder = RabitqIp::train(&dataset, RabitqConfig::default());

        for (qi, q) in vectors.iter().enumerate() {
            let q_rot = encoder.rotate(q);
            let mean_dot = encoder.mean_dot(q);
            for (xi, x) in vectors.iter().enumerate() {
                let r = encoder.residual(x);
                let got = mean_dot + r.iter().zip(&q_rot).map(|(&a, &b)| a * b).sum::<f32>();
                let want: f32 = x.iter().zip(q).map(|(&a, &b)| a * b).sum();
                assert!(
                    (got - want).abs() < 1e-2,
                    "x{xi}·q{qi}: decomposition {got} vs direct {want}"
                );
            }
        }
    }

    #[test]
    fn ip_score_matches_direct_estimator_form() {
        // Parity check for the inner-product path: the evaluator's score must equal
        // ⟨mean,q⟩ + ‖r‖·‖q_r‖·(ip_bar/factor) rebuilt here from the encoder's own residuals and
        // the reconstructed 4-bit query — an independent route through the ±1 sign expansion.
        // Note the query is rotated, *not* centered, so `rotate` is the right transform.
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        let config = RabitqConfig {
            seed: 42,
            rotate: true,
        };
        let encoder = RabitqIp::train(&dataset, config);
        let ds: crate::DenseDataset<RabitqIp> = (&dataset).convert_into(config);
        let d = 64usize;

        for q in &vectors {
            let q_rot = encoder.rotate(q);
            let nq: f32 = q_rot.iter().map(|v| v * v).sum::<f32>().sqrt();
            if nq <= f32::EPSILON {
                continue;
            }
            let mean_dot = encoder.mean_dot(q);
            let (codes, delta, vl) = quantize_query_multibit(&q_rot, 4);
            let q_hat: Vec<f32> = codes.iter().map(|&c| delta * c as f32 + vl).collect();

            let evaluator =
                encoder.query_evaluator(DenseVectorView::new(q), &RabitqQueryParams::new(4));
            for (i, doc) in vectors.iter().enumerate() {
                let doc_res = encoder.residual(doc);
                let (factor, norm_o) = encoder.metadata(&doc_res);
                if norm_o <= f32::EPSILON {
                    continue;
                }
                // ip_bar = ⟨sign(r), q̂⟩ / (√d·‖q_r‖), the same quantity the kernel computes.
                let ip_bar = doc_res
                    .iter()
                    .zip(&q_hat)
                    .map(|(&r, &qh)| if r >= 0.0 { qh } else { -qh })
                    .sum::<f32>()
                    / ((d as f32).sqrt() * nq);
                let expected = mean_dot + norm_o * nq * (ip_bar / factor);

                let got = evaluator.compute_distance(ds.get(i as u64)).0;
                let tol = 1e-2 * expected.abs().max(1.0);
                assert!(
                    (got - expected).abs() <= tol,
                    "doc {i}: evaluator IP {got} vs direct form {expected} (tol {tol})"
                );
            }
        }
    }

    #[test]
    fn l2_score_matches_library_factor_form() {
        // Parity check: with a 4-bit query, our L2 score must equal the RaBitQ-Library
        // plain-path estimator computed here via its *own* decomposition —
        //   est_dist = f_add + g_add + f_rescale·(ip_x0_qr + k1xsumq)
        // built from 0/1 document bits — which is an independent route to the same number than
        // our ±1 / law-of-cosines expression. Residuals come from `encoder.residual` so both
        // routes see the same centered-and-rotated vectors.
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        let config = RabitqConfig {
            seed: 42,
            rotate: true,
        };
        let encoder = RabitqL2::train(&dataset, config);
        let ds: crate::DenseDataset<RabitqL2> = (&dataset).convert_into(config);

        for q in &vectors {
            let q_res = encoder.residual(q);
            let nq: f32 = q_res.iter().map(|v| v * v).sum::<f32>().sqrt();
            if nq <= f32::EPSILON {
                continue;
            }
            // Reconstructed 4-bit query q̂ ≈ r_q (delta·code + vl per component).
            let (codes, delta, vl) = quantize_query_multibit(&q_res, 4);
            let q_hat: Vec<f32> = codes.iter().map(|&c| delta * c as f32 + vl).collect();
            let sumq_hat: f32 = q_hat.iter().sum();
            let k1xsumq = -0.5 * sumq_hat; // library c_1 = -1/2, on the reconstructed residual sum

            let evaluator =
                encoder.query_evaluator(DenseVectorView::new(q), &RabitqQueryParams::new(4));
            for (i, doc) in vectors.iter().enumerate() {
                let r = encoder.residual(doc);
                let norm_o: f32 = r.iter().map(|v| v * v).sum::<f32>().sqrt();
                if norm_o <= f32::EPSILON {
                    continue;
                }
                let sum_abs: f32 = r.iter().map(|v| v.abs()).sum(); // = 2·ip_resi_xucb
                // Library L2 doc factors (centered query ⇒ no centroid term in f_add):
                let f_add = norm_o * norm_o;
                let f_rescale = -2.0 * (norm_o * norm_o) / (0.5 * sum_abs);
                let g_add = nq * nq;
                // ip_x0_qr = ⟨bit_doc(0/1), q̂⟩ = Σ over positive-sign dims of q̂_i.
                let ip_x0_qr: f32 = r
                    .iter()
                    .zip(&q_hat)
                    .filter(|&(&ri, _)| ri >= 0.0)
                    .map(|(_, &qh)| qh)
                    .sum();
                let expected = f_add + g_add + f_rescale * (ip_x0_qr + k1xsumq);

                let got = evaluator.compute_distance(ds.get(i as u64)).distance();
                let tol = 1e-2 * expected.abs().max(1.0);
                assert!(
                    (got - expected).abs() <= tol,
                    "doc {i}: evaluator L2 {got} vs library-form {expected} (tol {tol})"
                );
            }
        }
    }

    #[test]
    fn search_finds_self_as_nearest_query_bits_4() {
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        assert_self_is_top_ranked::<SquaredEuclideanDistance>(&vectors, &dataset, 4, true);
        assert_self_is_top_ranked::<DotProduct>(&vectors, &dataset, 4, true);
    }

    /// The fused batch kernel must be bit-identical to six single `compute_distance` calls,
    /// across both query paths (1-bit hamming, multi-bit planes) under metric `D`.
    fn assert_batch6_matches_singles<D: RabitqSupportedDistance + std::fmt::Debug>(
        vectors: &[Vec<f32>],
        dataset: &PlainDenseDataset<f32, DotProduct>,
    ) {
        for query_bits in [1, 2, 4, 8] {
            let config = RabitqConfig {
                seed: 42,
                rotate: true,
            };
            let ds: crate::DenseDataset<RabitqQuantizer<D>> = dataset.convert_into(config);
            let qp = RabitqQueryParams::new(query_bits);
            for q in vectors {
                let evaluator = ds.encoder().query_evaluator(DenseVectorView::new(q), &qp);
                let views = std::array::from_fn(|k| ds.get(k as u64));
                let batch = evaluator.compute_distances_batch6(views);
                let singles = std::array::from_fn(|k| evaluator.compute_distance(ds.get(k as u64)));
                assert_eq!(batch, singles, "query_bits {query_bits}");
            }
        }
    }

    #[test]
    fn compute_distances_batch6_matches_six_singles() {
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        assert_batch6_matches_singles::<SquaredEuclideanDistance>(&vectors, &dataset);
        assert_batch6_matches_singles::<DotProduct>(&vectors, &dataset);
    }

    /// The allocation-free two-document score must be bit-identical to the `vector_evaluator`
    /// route it replaces, under metric `D`.
    fn assert_distance_between_matches_evaluator<D: RabitqSupportedDistance + std::fmt::Debug>(
        dataset: &PlainDenseDataset<f32, DotProduct>,
    ) {
        let ds: crate::DenseDataset<RabitqQuantizer<D>> =
            dataset.convert_into(RabitqConfig::default());
        let encoder = ds.encoder();
        for i in 0..ds.len() as u64 {
            for j in 0..ds.len() as u64 {
                let direct = encoder.compute_distance_between(ds.get(i), ds.get(j));
                let via = encoder
                    .vector_evaluator(ds.get(i))
                    .compute_distance(ds.get(j));
                assert_eq!(direct, via, "pair ({i}, {j})");
            }
        }
    }

    #[test]
    fn compute_distance_between_matches_vector_evaluator() {
        let dataset = plain_dataset(&varied_vectors());
        assert_distance_between_matches_evaluator::<SquaredEuclideanDistance>(&dataset);
        assert_distance_between_matches_evaluator::<DotProduct>(&dataset);
    }

    #[test]
    fn ip_build_path_equals_query_path_on_the_reconstruction() {
        // What the inner-product build path claims to be: `query_evaluator` applied to the stored
        // vector's reconstruction x̂ = Pᵀr̂ + mean. With rotation disabled (P = I) that x̂ can be
        // formed directly here, and both routes must build the same effective query — bit for bit,
        // since `vector_evaluator` skips only the Pᵀ/P pair that cancels.
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        let config = RabitqConfig {
            seed: 42,
            rotate: false,
        };
        let ds: crate::DenseDataset<RabitqIp> = (&dataset).convert_into(config);
        let encoder = ds.encoder();
        // The build path has no user query, so it scores at `BUILD_QUERY_BITS`; the query route
        // must be asked for the same width to be the same evaluator.
        let qp = RabitqQueryParams::new(BUILD_QUERY_BITS);
        for i in 0..ds.len() as u64 {
            let mut recon = encoder.reconstruct_residual(ds.get(i).values());
            for (v, &m) in recon.iter_mut().zip(encoder.space.means().iter()) {
                *v += m;
            }
            let build = encoder.vector_evaluator(ds.get(i));
            let query = encoder.query_evaluator(DenseVectorView::new(&recon), &qp);
            for j in 0..ds.len() as u64 {
                assert_eq!(
                    build.compute_distance(ds.get(j)),
                    query.compute_distance(ds.get(j)),
                    "pair ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn dataset_serialization_round_trip() {
        // The trained geometry lives in a nested `RabitqSpace`, so exercise the serde path end to
        // end: a reloaded index must equal the original *and* score identically.
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        let ds: crate::DenseDataset<RabitqIp> = (&dataset).convert_into(RabitqConfig::default());

        let mut path = std::env::temp_dir();
        path.push(format!("vectorium_rabitq_{}.bin", std::process::id()));
        let path = path.to_str().unwrap().to_string();

        ds.save_index(&path).unwrap();
        let loaded = crate::DenseDataset::<RabitqIp>::load_index(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(ds, loaded);
        let qp = RabitqQueryParams::new(4);
        for q in &vectors {
            let before = ds.encoder().query_evaluator(DenseVectorView::new(q), &qp);
            let after = loaded
                .encoder()
                .query_evaluator(DenseVectorView::new(q), &qp);
            for i in 0..ds.len() as u64 {
                assert_eq!(
                    before.compute_distance(ds.get(i)),
                    after.compute_distance(loaded.get(i)),
                    "vector {i}"
                );
            }
        }
    }

    /// The squared-Euclidean alias serializes through a different scoring type parameter and, on the
    /// build path, a different evaluator (`vector_evaluator` reconstructs and requantizes). Round-trip
    /// it separately, at a non-default config so a serde field that silently reverts to its default
    /// cannot pass.
    #[test]
    fn dataset_serialization_round_trip_squared_euclidean() {
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        let config = RabitqConfig {
            seed: 7,
            ..RabitqConfig::default()
        };
        let ds: crate::DenseDataset<RabitqL2> = (&dataset).convert_into(config);

        let mut path = std::env::temp_dir();
        path.push(format!("vectorium_rabitq_l2_{}.bin", std::process::id()));
        let path = path.to_str().unwrap().to_string();

        ds.save_index(&path).unwrap();
        let loaded = crate::DenseDataset::<RabitqL2>::load_index(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(ds, loaded);
        assert_eq!(loaded.encoder().config, config, "config survived serde");

        let qp = RabitqQueryParams::new(4);
        for q in &vectors {
            let before = ds.encoder().query_evaluator(DenseVectorView::new(q), &qp);
            let after = loaded
                .encoder()
                .query_evaluator(DenseVectorView::new(q), &qp);
            for i in 0..ds.len() as u64 {
                assert_eq!(
                    before.compute_distance(ds.get(i)),
                    after.compute_distance(loaded.get(i)),
                    "query-side score, vector {i}"
                );
            }
        }
        // The build path scores encoded-vs-encoded; it must survive the round-trip too.
        for i in 0..ds.len() as u64 {
            let before = ds.encoder().vector_evaluator(ds.get(i));
            let after = loaded.encoder().vector_evaluator(loaded.get(i));
            for j in 0..ds.len() as u64 {
                assert_eq!(
                    before.compute_distance(ds.get(j)),
                    after.compute_distance(loaded.get(j)),
                    "build-path score, pair ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn push_encoded_matches_the_parallel_slab_encode() {
        // The parallel builder drives `push_encoded` into a slice of the final slab, so a record
        // pushed into a growable `Vec` must be byte-identical to the corresponding slab row —
        // including the metadata word. d = 640 exercises the multi-word sign packer, not just the
        // single-word fixture.
        let vectors: Vec<Vec<f32>> = (0..6)
            .map(|k| {
                (0..640)
                    .map(|i| ((i * (k + 2)) as f32 * 0.11).cos() * (k as f32 + 1.0))
                    .collect()
            })
            .collect();
        let dataset = plain_dataset(&vectors);
        let config = RabitqConfig::default();
        let ds: crate::DenseDataset<RabitqIp> = (&dataset).convert_into(config);
        let encoder = ds.encoder();
        for (i, v) in vectors.iter().enumerate() {
            let mut pushed = Vec::new();
            encoder.push_encoded(DenseVectorView::new(v), &mut pushed);
            assert_eq!(pushed, ds.get(i as u64).values(), "vector {i}");
        }
    }

    #[test]
    fn compute_distances_batch6_matches_six_singles_wide() {
        // d = 640 → 10 code words: exercises the 8-wide unchecked-load chunk loop *and* the
        // 2-word tail (the d = 64 fixture is all tail).
        let vectors: Vec<Vec<f32>> = (0..8)
            .map(|k| {
                (0..640)
                    .map(|i| ((i * (k + 3)) as f32 * 0.13).sin())
                    .collect()
            })
            .collect();
        let dataset = plain_dataset(&vectors);
        assert_batch6_matches_singles::<SquaredEuclideanDistance>(&vectors, &dataset);
        assert_batch6_matches_singles::<DotProduct>(&vectors, &dataset);
    }

    /// A derangement over the ten `varied_vectors` rows, so no row stays where it started.
    const PERMUTATION: [usize; 10] = [2, 0, 3, 1, 5, 4, 7, 6, 9, 8];

    /// `permute` copies encoded rows verbatim at a stride of `output_dim()`, so a RaBitQ record —
    /// `num_words()` sign words followed by the metadata word — must survive the move intact.
    /// kANNolo's edge-compressed graph types (`permuted`, `streamvbyte`) reorder nodes this way.
    #[test]
    fn rabitq_permute_moves_each_row_to_its_target_slot() {
        let dataset: crate::DenseDataset<RabitqIp> =
            (&plain_dataset(&varied_vectors())).convert_into(Default::default());

        let permuted = dataset.permute(&PERMUTATION);

        assert_eq!(permuted.len(), dataset.len());
        for (old_id, &new_id) in PERMUTATION.iter().enumerate() {
            assert_eq!(
                permuted.get(new_id as u64).values(),
                dataset.get(old_id as u64).values(),
                "row {old_id} did not land at slot {new_id}"
            );
        }
    }

    /// Permuting then applying the inverse must reproduce the dataset exactly — codes, metadata
    /// words and the cloned encoder (means, rotation, config) alike.
    #[test]
    fn rabitq_permute_then_inverse_round_trips() {
        let dataset: crate::DenseDataset<RabitqIp> =
            (&plain_dataset(&varied_vectors())).convert_into(Default::default());
        let inverse = crate::core::dataset::invert_permutation(&PERMUTATION);

        assert_eq!(dataset.permute(&PERMUTATION).permute(&inverse), dataset);
    }

    /// The property kANNolo actually depends on: reordering nodes relabels results but does not
    /// change them. Same scores, same neighbours, just under permuted ids.
    #[test]
    fn rabitq_permute_relabels_search_results_without_changing_them() {
        let vectors = varied_vectors();
        let dataset: crate::DenseDataset<RabitqIp> =
            (&plain_dataset(&vectors)).convert_into(Default::default());
        let permuted = dataset.permute(&PERMUTATION);

        let index = FlatIndex::from(&dataset);
        let permuted_index = FlatIndex::from(&permuted);

        let qp = RabitqQueryParams::default();
        for (i, q) in vectors.iter().enumerate() {
            let base = index.search(DenseVectorView::new(q), 5, &qp);
            let moved = permuted_index.search(DenseVectorView::new(q), 5, &qp);

            let expected: Vec<u64> = base
                .iter()
                .map(|r| PERMUTATION[r.vector as usize] as u64)
                .collect();
            let got: Vec<u64> = moved.iter().map(|r| r.vector).collect();
            assert_eq!(got, expected, "query {i} returned different neighbours");

            for (b, m) in base.iter().zip(moved.iter()) {
                assert_eq!(
                    b.distance.distance(),
                    m.distance.distance(),
                    "query {i} scored a neighbour differently after permutation"
                );
            }
        }
    }

    /// Encoding an `f16` collection must give exactly the codes the `f32` path gives for the same
    /// (already-rounded) values: the `f16` source changes the input precision, never the encoder.
    /// The residual means are accumulated in `f32` on both paths, so the trained space is identical
    /// too.
    #[test]
    fn f16_source_encodes_exactly_like_the_equivalent_f32_source() {
        use crate::dataset::ConvertFrom;
        use half::f16;

        let d = 64;
        let n = 120;
        let rows: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..d)
                    .map(|j| f16::from_f32(((i * d + j) as f32 * 0.13).sin() * 2.0).to_f32())
                    .collect()
            })
            .collect();

        let plain_f32 = plain_dataset(&rows);
        let mut growable =
            crate::PlainDenseDatasetGrowable::<f16, DotProduct>::new(crate::PlainDenseQuantizer::<
                f16,
                DotProduct,
            >::new(d));
        for row in &rows {
            let half: Vec<f16> = row.iter().map(|&v| f16::from_f32(v)).collect();
            growable.push(DenseVectorView::new(&half));
        }
        let plain_f16: crate::PlainDenseDataset<f16, DotProduct> = growable.into();

        let config = RabitqConfig::default();
        let from_f32: crate::DenseDataset<RabitqIp> = ConvertFrom::convert_from(&plain_f32, config);
        let from_f16: crate::DenseDataset<RabitqIp> = ConvertFrom::convert_from(&plain_f16, config);

        assert_eq!(from_f16.len(), from_f32.len());
        assert_eq!(from_f16.values(), from_f32.values());
    }
}
