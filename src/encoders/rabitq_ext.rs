//! Extended RaBitQ encoder: multi-bit document codes (no centroids).
//!
//! The extended version of [`rabitq`](crate::encoders::rabitq): documents are quantized to a
//! uniform `total_bits ∈ {2, 4, 8}` bits per component — 1 sign bit plus
//! `ex_bits = total_bits − 1` magnitude bits — following RaBitQ-Library's extended scheme
//! (*Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors*, Gao et al.).
//! `total_bits = 1` (the plain sign code, `ex_bits = 0`) is **not** handled here and is rejected by
//! [`train`](RabitqExtQuantizer::train): it degenerates to and scores identically to
//! [`RabitqQuantizer`](crate::encoders::rabitq::RabitqQuantizer), so use that 1-bit encoder for it.
//! **The query is never quantized here** — there is no `query_bits` dial (see [`Scoring`](self)).
//!
//! This file intentionally duplicates the private helpers of `rabitq.rs` (kernels, rescale-factor
//! search, query quantization) so the two implementations stay independent; they may be merged
//! later. The preprocessing is identical: residual `r = P·(x − mean)` with per-component means and
//! a seeded [`FhtKacRotator`](crate::FhtKacRotator) (disable via [`RabitqExtConfig::rotate`]).
//!
//! ## Document code
//!
//! With `o_abs_i = |r_i|/‖r‖` and `t` from the exact rescale-factor search
//! (`best_rescale_factor`), each component gets a magnitude code
//! `e_i = min(⌊t·o_abs_i⌋, 2^ex_bits − 1)` and the total code
//!
//! ```text
//! u_i = e_i | (1 << ex_bits)      if r_i ≥ 0        # sign bit on top
//! u_i = (!e_i) & (2^ex_bits − 1)  if r_i < 0        # complement-coded magnitude
//! ```
//!
//! With `cb = −(2^ex_bits − 0.5)` the shifted code is **antisymmetric**:
//! `u_i + cb = sign(r_i)·(e_i + 0.5)`, so `⟨r, u + cb⟩ = ‖r‖·ipnorm` with
//! `ipnorm = Σ (e_i + 0.5)·o_abs_i > 0` — the quantized direction always points with the residual.
//!
//! ## Scoring
//!
//! The unbiased projection estimator (any scaling of the quantized direction cancels):
//!
//! ```text
//! ⟨r, q_r⟩ ≈ (‖r‖² / ⟨r, u + cb⟩)·⟨u + cb, q_r⟩ = s_ext·⟨u + cb, q_r⟩,   s_ext = ‖r‖/ipnorm
//! ```
//!
//! against the **unquantized** rotated query `q_r`, which splits into one dot product over the
//! stored code and a per-query constant:
//!
//! ```text
//! raw  = ⟨u, q_r⟩ + cb·Σq_r          # = ⟨u + cb, q_r⟩; the second term is `q_const`
//! term = s_ext · raw                  # ≈ ⟨r, q_r⟩
//! ```
//!
//! `⟨u, q_r⟩` is `doc_code_ip_exact`: widen each `total_bits`-wide code field to f32 and FMA it
//! against the query. `term` then feeds the same metric combine as the 1-bit encoder
//! ([`RabitqSupportedDistance::from_terms`]): `f_add + g_add − 2·term` for squared Euclidean,
//! `⟨mean, q⟩ + term` for inner product.
//!
//! ## Storage
//!
//! Each document is `total_bits · d/64` code words in **component-major** layout — each `u_i`
//! occupies one aligned `total_bits`-wide field, and a 16-byte block holds `8/total_bits`
//! sub-groups of 16 consecutive components (see `byte_slot`) — plus one metadata word packing the
//! two scan-ready floats
//! `[f_add = ‖r‖² | s_ext = ‖r‖/ipnorm]`, so the per-candidate scan does no divide. Only dimensions
//! that are a multiple of 64 are supported (no bit-padding).
//!
//! ## How this differs from RaBitQ-Library
//!
//! The code this encoder produces is compatible in spirit with RaBitQ-Library but differs in
//! storage layout and in which widths are supported. Four deliberate divergences:
//!
//! 1. **One slab and one pass, against their two.** RaBitQ-Library keeps the sign bit in a `bin`
//!    slab separate from the `ex` magnitude slab, so `split_single_fulldist` scores each candidate
//!    with two kernels over two allocations, traversing the query twice. That layout serves their
//!    HNSW early-exit lower bound, which an exhaustive scan never consults. Storing all
//!    `total_bits` together reads each query element exactly once.
//! 2. **Byte-aligned widths only.** `{2,4,8}` unpack with one shift and mask on whole 8-bit lanes.
//!    Supporting every width in `1..=9`, as they do, forces cross-byte reassembly — their 7-bit
//!    kernel spends roughly 25 SSE-width integer ops per 64 components rebuilding fields before any
//!    arithmetic.
//! 3. **Four independent accumulator chains** (`doc_code_ip_exact`). Both reference kernels
//!    accumulate into a single register, so their chain is bound by FMA latency rather than
//!    throughput. GCC obtains the split for free from `-ffast-math`; rustc will not reassociate f32
//!    addition, so it is written out here.
//! 4. **The width is resolved once per query, not per candidate.** Their scan calls the ex-code
//!    kernel through a function pointer, which cannot be inlined. A `match` on `total_bits` selects
//!    a monomorphized kernel with literal loop bounds and shift amounts.
//!
//! The widths, the layout and the metadata packing are therefore load-bearing performance
//! decisions rather than incidental ones; see the kernel notes on `doc_code_ip_exact` and
//! `part_ip_direct` before changing any of them.
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::simd::num::{SimdFloat, SimdUint};
use std::simd::{Simd, StdFloat};

use crate::core::distances::DotProduct;
use crate::core::vector::{DenseVectorOwned, DenseVectorView};
use crate::core::vector_encoder::{DenseVectorEncoder, QueryEvaluator, VectorEncoder};
use crate::dataset::ConvertFrom;
use crate::encoders::rabitq_common::{
    RabitqSpace, RabitqSupportedDistance, RescaleScratch, WORD_BITS, best_rescale_factor,
    pack_metadata, unpack_metadata,
};
use crate::{Dataset, PlainDenseDataset, ScalarDenseSupportedDistance, SpaceUsage};

/// Extended RaBitQ encoder parameters. See the module docs for the estimator math.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RabitqExtConfig {
    /// Bits per component for the **document** code: 1 sign bit + `total_bits − 1` magnitude
    /// bits. **Only `2`, `4` and `8` are supported** — the byte-aligned widths, where a component's
    /// field lands on a clean boundary, so the code packs component-major and the scan costs one
    /// widen plus one FMA per 16 components.
    ///
    /// The intermediate widths are not decomposed into aligned parts (`7 = 4+2+1` and friends):
    /// a decomposed width costs one extra unpack round per part, and none of them buys a distinct
    /// operating point between `2`, `4` and `8`. Same reasoning as PQ's fixed 8-bit codes.
    /// `total_bits = 1` is [`RabitqQuantizer`](crate::RabitqQuantizer)'s job.
    pub total_bits: u32,
    /// Seed for the random orthogonal rotation ([`FhtKacRotator`](crate::FhtKacRotator)).
    pub seed: u64,
    /// Apply the random orthogonal rotation ([`FhtKacRotator`](crate::FhtKacRotator)) to residuals. Disabling skips the
    /// `O(d log d)` per-vector transform at encode and query time; the estimator math is unchanged
    /// (`P = I` is a valid orthogonal transform), but the codes lose the information-spreading
    /// guarantee, so recall on real data is expected to drop.
    pub rotate: bool,
    /// Use a **constant** rescale factor when quantizing documents instead of the exact per-vector
    /// search (`best_rescale_factor`). When set (the default), the factor is estimated once at
    /// [`train`](RabitqExtQuantizer::train) over random unit vectors and every document quantizes
    /// in a single `O(d)` pass; clearing it runs a heap sweep per vector, which is slower to encode
    /// and increasingly so at larger `total_bits`.
    ///
    /// The constant is a concentration-of-measure estimate, so what the exact search buys back in
    /// accuracy shrinks as `d` grows — it costs most where it helps least. Leave this set unless
    /// you are at large `total_bits` on low-dimensional data.
    ///
    /// Build-side only: it changes how documents are encoded, not how they are scanned.
    pub faster_quant: bool,
}

impl Default for RabitqExtConfig {
    fn default() -> Self {
        Self {
            total_bits: 4,
            seed: 42,
            rotate: true,
            faster_quant: true,
        }
    }
}

/// `⟨u, q⟩` between a document's component-major code and the **unquantized** rotated query.
///
/// One `match` on the stored width, then a single kernel: `total_bits ∈ {2,4,8}` each pack one
/// component per aligned bit-field, so there is nothing to decompose or recombine. Reads every
/// query element exactly once, and the cost does not grow with `total_bits` beyond the extra code
/// bytes fetched.
#[inline]
fn doc_code_ip_exact(code: &[u64], q: &[f32], num_words: usize, total_bits: u32) -> f32 {
    const LANES: usize = 16;
    let d = num_words * WORD_BITS;
    debug_assert!(q.len() >= d);
    let q = &q[..d];
    let mut accs = [Simd::<f32, LANES>::splat(0.0); 4];
    let bytes: &[u8] = bytemuck::cast_slice(&code[..total_bits as usize * num_words]);
    match total_bits {
        8 => part_ip_direct::<8>(bytes, q, &mut accs),
        4 => part_ip_direct::<4>(bytes, q, &mut accs),
        _ => part_ip_direct::<2>(bytes, q, &mut accs),
    }
    (accs[0] + accs[1] + accs[2] + accs[3]).reduce_sum()
}

/// Components per 16-byte block of a `p`-bit sub-array (`p ∈ {2,4,8}`): 16 bytes hold `8/p`
/// sub-groups of 16 consecutive components. `d % 64 == 0` is already required, so every block size
/// here (16, 32, 64) divides `d`.
#[inline]
const fn block_components(p: u32) -> usize {
    16 * (8 / p as usize)
}

/// Slot of component `c` in a `p`-bit sub-array: `(byte index, sub-group)`. The field occupies bits
/// `[sub·p, sub·p + p)` of that byte. Laying sub-groups out this way keeps each sub-group's 16
/// components contiguous in the *query*, so the scan reads `q` linearly.
#[inline]
const fn byte_slot(c: usize, p: u32) -> (usize, u32) {
    let bc = block_components(p);
    let (block, within) = (c / bc, c % bc);
    ((block * 16) + (within % 16), (within / 16) as u32)
}

/// Write `codes` into the component-major layout for an aligned width `total_bits ∈ {2,4,8}`.
fn pack_components_into(codes: &[u16], total_bits: u32, num_words: usize, out: &mut [u64]) {
    out.fill(0);
    debug_assert_eq!(out.len(), total_bits as usize * num_words);
    let mask = ((1u32 << total_bits) - 1) as u16;
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(out);
    for (c, &code) in codes.iter().enumerate() {
        let (byte, sub) = byte_slot(c, total_bits);
        bytes[byte] |= ((code & mask) as u8) << (sub * total_bits);
    }
}

/// Inverse of [`pack_components_into`].
fn unpack_components(words: &[u64], total_bits: u32, _num_words: usize, d: usize) -> Vec<u16> {
    let mask = ((1u32 << total_bits) - 1) as u16;
    let bytes: &[u8] = bytemuck::cast_slice(words);
    (0..d)
        .map(|c| {
            let (byte, sub) = byte_slot(c, total_bits);
            ((bytes[byte] >> (sub * total_bits)) as u16) & mask
        })
        .collect()
}

/// Single-part fast path: extract `P`-bit codes straight to f32 and FMA, with no intermediate
/// integer code. Used when `total_bits ∈ {2,4,8}` decomposes to one part, avoiding the combining
/// path's extra u32 materialisation and widen per round for a shift and OR that are both no-ops at
/// one part.
#[inline]
fn part_ip_direct<const P: usize>(bytes: &[u8], q: &[f32], accs: &mut [Simd<f32, 16>; 4]) {
    const LANES: usize = 16;
    let lo = Simd::<u8, LANES>::splat(((1u16 << P) - 1) as u8);
    macro_rules! go {
        ($blocks:literal, $subs:literal) => {
            for (bchunk, qchunk) in bytes.chunks_exact(8 * P).zip(q.chunks_exact(4 * LANES)) {
                for block in 0..$blocks {
                    let v = Simd::<u8, LANES>::from_slice(&bchunk[block * LANES..]);
                    for sub in 0..$subs {
                        let slot = (block * $subs) + sub;
                        let u = if P == 8 {
                            v.cast::<f32>()
                        } else {
                            ((v >> Simd::splat((sub * P) as u8)) & lo).cast::<f32>()
                        };
                        let qv = Simd::<f32, LANES>::from_slice(&qchunk[slot * LANES..]);
                        accs[slot] = u.mul_add(qv, accs[slot]);
                    }
                }
            }
        };
    }
    match P {
        8 => go!(4, 1),
        4 => go!(2, 2),
        _ => go!(1, 4),
    }
}

/// Number of random unit vectors averaged to estimate the constant rescale factor (RaBitQ-Library's
/// `kConstNum`).
const CONST_SCALE_SAMPLES: usize = 100;

/// Estimate a **constant** rescale factor for `dim`/`ex_bits` by averaging the exact
/// [`best_rescale_factor`] over [`CONST_SCALE_SAMPLES`] random unit vectors — RaBitQ-Library's
/// `get_const_scaling_factors`. Computed once at [`train`](RabitqExtQuantizer::train); every document
/// then quantizes in a single `O(d)` pass with this `t` instead of a per-vector heap sweep.
///
/// The vectors are standard-Gaussian (isotropic ⇒ the factor is rotation-independent), row-normalized
/// and abs'd — exactly the `o_abs` shape the search consumes. The generator is seeded, so the factor
/// is reproducible; the specific RNG need not match the C++ library since this is an average estimate.
fn get_const_scaling_factors(dim: usize, ex_bits: u32, seed: u64) -> f64 {
    use rand::{Rng, SeedableRng, rngs::StdRng};

    let mut rng = StdRng::seed_from_u64(seed);
    let mut scratch = RescaleScratch::default();
    let mut o_abs = vec![0.0f32; dim];
    let mut sum = 0.0f64;

    for _ in 0..CONST_SCALE_SAMPLES {
        // Box–Muller: draw `dim` standard-normal samples straight into `o_abs`.
        let mut k = 0;
        while k < dim {
            let u1: f64 = rng.gen_range(0.0f64..1.0).max(f64::MIN_POSITIVE);
            let u2: f64 = rng.gen_range(0.0f64..1.0);
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = std::f64::consts::TAU * u2;
            o_abs[k] = (r * theta.cos()) as f32;
            if k + 1 < dim {
                o_abs[k + 1] = (r * theta.sin()) as f32;
            }
            k += 2;
        }
        // Normalize to a unit vector, then take absolute values (the search's `o_abs` domain).
        let norm = (o_abs.iter().map(|&v| v as f64 * v as f64).sum::<f64>()).sqrt();
        let inv = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        for v in o_abs.iter_mut() {
            *v = (*v as f64 * inv).abs() as f32;
        }
        sum += best_rescale_factor(&o_abs, ex_bits, &mut scratch);
    }

    sum / CONST_SCALE_SAMPLES as f64
}

/// Extended RaBitQ quantizer: multi-bit document codes. See the module docs.
///
/// The metric `D` selects how the stored residual codes are scored against a query; it defaults
/// to [`DotProduct`] (inner product). Document codes are identical under either metric — only the
/// query path and the final combine differ, so `D` is pure type-level state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RabitqExtQuantizer<D = DotProduct> {
    /// The trained geometry — means, rotated means and the rotation `P` — shared with
    /// [`RabitqQuantizer`](crate::encoders::rabitq::RabitqQuantizer).
    space: RabitqSpace,
    /// Constant rescale factor for the fast document-quantization path
    /// ([`get_const_scaling_factors`]), computed once at [`train`](Self::train). `None` unless
    /// `config.faster_quant` is set; the exact per-vector path ignores it.
    t_const: Option<f64>,
    /// Encoder parameters.
    config: RabitqExtConfig,
    /// The metric is encoded purely in the type; no runtime state.
    _distance: PhantomData<D>,
}

impl<D: RabitqSupportedDistance> RabitqExtQuantizer<D> {
    /// The number of bits per component used by the stored *document* codes.
    ///
    /// Fixed at encode time: it determines the code layout, so changing it requires re-encoding.
    #[inline]
    pub fn total_bits(&self) -> u32 {
        self.config.total_bits
    }

    /// Learn per-component means over `dataset` and build the rotation.
    ///
    /// Panics unless the dataset dimension is a multiple of 64 and `total_bits ∈ {2, 4, 8}`.
    /// `total_bits = 1` is deliberately rejected: the plain 1-bit document code is the domain of
    /// [`RabitqQuantizer`](crate::encoders::rabitq::RabitqQuantizer) — use that encoder instead of
    /// the degenerate `ex_bits = 0` case here. See [`RabitqExtConfig::total_bits`] for why the
    /// non-byte-aligned widths are not supported.
    pub fn train<Ds: ScalarDenseSupportedDistance>(
        dataset: &PlainDenseDataset<f32, Ds>,
        config: RabitqExtConfig,
    ) -> Self {
        assert!(
            matches!(config.total_bits, 2 | 4 | 8),
            "RabitqExtQuantizer requires total_bits in {{2, 4, 8}}, got {}; \
             for total_bits = 1 use RabitqQuantizer (the 1-bit encoder) instead",
            config.total_bits
        );
        let space = RabitqSpace::train(dataset, config.rotate, config.seed);
        Self::from_space(space, config)
    }

    /// [`train`](Self::train) over a source stored at a narrower value type, e.g. `f16`.
    ///
    /// Means are accumulated in `f32` exactly as above; only the source buffer is narrower.
    pub fn train_narrow<V, Ds>(dataset: &PlainDenseDataset<V, Ds>, config: RabitqExtConfig) -> Self
    where
        V: crate::ValueType + crate::Float + crate::FromF32,
        Ds: ScalarDenseSupportedDistance,
    {
        assert!(
            matches!(config.total_bits, 2 | 4 | 8),
            "RabitqExtQuantizer requires total_bits in {{2, 4, 8}}, got {}; \
             for total_bits = 1 use RabitqQuantizer (the 1-bit encoder) instead",
            config.total_bits
        );
        let space = RabitqSpace::train_narrow(dataset, config.rotate, config.seed);
        Self::from_space(space, config)
    }

    fn from_space(space: RabitqSpace, config: RabitqExtConfig) -> Self {
        // Constant rescale factor for the fast build path, estimated once here (negligible vs. the
        // per-vector encode). `ex_bits = total_bits − 1 ≥ 1` always holds (total_bits ∈ {2,4,8}).
        let t_const = config
            .faster_quant
            .then(|| get_const_scaling_factors(space.dim(), config.total_bits - 1, config.seed));

        Self {
            space,
            t_const,
            config,
            _distance: PhantomData,
        }
    }

    /// Number of `u64` words **per bit plane** (a document stores `total_bits` planes plus the
    /// metadata word).
    #[inline]
    fn num_words(&self) -> usize {
        self.space.num_words()
    }

    /// Input dimensionality.
    #[inline]
    fn d(&self) -> usize {
        self.space.dim()
    }

    /// Magnitude bits per component (`total_bits − 1`).
    #[inline]
    fn ex_bits(&self) -> u32 {
        self.config.total_bits - 1
    }

    /// The document code's symmetric-grid offset `cb = −(2^ex_bits − 0.5)`, so that
    /// `u_i + cb = sign(r_i)·(e_i + 0.5)`.
    #[inline]
    fn cb(&self) -> f32 {
        -(((1u64 << self.ex_bits()) as f32) - 0.5)
    }

    /// Apply the rotation: `P·v` (a copy of `values` when rotation is disabled, i.e. `P = I`).
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

    /// Reconstruct the rotated residual from a stored code: `r̂_i = s_ext·(u_i + cb)`.
    ///
    /// This is the reconstruction the estimator itself uses — the stored planes enter the scan as
    /// `q̂ = delta·u + vl` with `delta = s_ext`, `vl = s_ext·cb` — so the build path stays consistent
    /// with the search path. Backs both [`decode_vector`] and the inner-product
    /// [`vector_evaluator`], which adds `P·mean` to it.
    ///
    /// [`decode_vector`]: DenseVectorEncoder::decode_vector
    /// [`vector_evaluator`]: VectorEncoder::vector_evaluator
    fn reconstruct_residual(&self, words: &[u64]) -> Vec<f32> {
        let nw = self.num_words();
        let b = self.config.total_bits as usize;
        let (_, s_ext) = unpack_metadata(words[b * nw]);
        let cb = self.cb();
        unpack_components(&words[..b * nw], b as u32, nw, self.d())
            .into_iter()
            .map(|u| s_ext * (u as f32 + cb))
            .collect()
    }

    /// Reconstruct a stored code back into the **original data space**: `x̂ = mean + P⁻¹·r̂`.
    ///
    /// [`decode_vector`](DenseVectorEncoder::decode_vector) stops at `r̂`, which lives in the
    /// rotated, centered residual space and so cannot be subtracted from an input vector. This is
    /// the form residual reranking needs: the second stage encodes `x − x̂`, and by orthogonality of
    /// `P` the estimator's own first-stage score is exactly `⟨q, x̂⟩` plus the per-query additive
    /// term — so `⟨q, x⟩ = first_stage_score + ⟨q, x − x̂⟩` is an exact decomposition under an
    /// additive metric.
    ///
    /// `code` is one stored vector's words, `total_bits · num_words` planes followed by the packed
    /// metadata word — exactly what `DenseDataset` hands back for a vector id.
    pub fn reconstruct(&self, code: &[u64]) -> Vec<f32> {
        self.space.from_residual(&self.reconstruct_residual(code))
    }

    fn rotated_query_evaluator<'e>(
        &'e self,
        q_r: &[f32],
        mean_dot_query: f32,
    ) -> RabitqExtQueryEvaluator<'e, D> {
        let query_norm = q_r.iter().map(|&r| r * r).sum::<f32>().sqrt();
        // `cb·Σq_r` takes the **exact** query sum. Accumulated in f64 because `|cb|` reaches 127.5
        // at total_bits = 8, so this term is the most error-sensitive one in the estimator.
        let sum_q = q_r.iter().map(|&v| v as f64).sum::<f64>() as f32;
        RabitqExtQueryEvaluator {
            _encoder: PhantomData,
            q_r: q_r.to_vec(),
            q_const: self.cb() * sum_q,
            add: D::query_add(mean_dot_query, query_norm),
            num_words: self.num_words(),
            total_bits: self.config.total_bits,
        }
    }

    /// Quantize a residual to per-component total codes, writing them into `codes` (length `d`).
    ///
    /// Returns the scan constants `(f_add = ‖r‖², s_ext = ‖r‖/ipnorm)`. A zero residual gets
    /// all-zero codes and `(0, 0)`, so the estimator contributes `term = 0` and never divides by
    /// zero or produces NaN. Otherwise `ipnorm = Σ (e_i + 0.5)·o_abs_i ≥ 0.5·Σ o_abs_i > 0`.
    ///
    /// The rescale factor is the constant [`Self::t_const`] (fast path) when set, else the exact
    /// per-vector [`best_rescale_factor`]. `o_abs` and `rescale` are caller-owned scratch, so a
    /// caller encoding many vectors can reuse them; the fast path folds `1/‖r‖` into the scalar and
    /// never touches `o_abs`.
    fn encode_residual_into(
        &self,
        residual: &[f32],
        codes: &mut [u16],
        o_abs: &mut Vec<f32>,
        rescale: &mut RescaleScratch,
    ) -> (f32, f32) {
        let ex_bits = self.ex_bits();
        let norm = residual
            .iter()
            .map(|&r| r as f64 * r as f64)
            .sum::<f64>()
            .sqrt();
        if norm <= f64::EPSILON {
            codes.fill(0);
            return (0.0, 0.0);
        }
        let inv_norm = 1.0 / norm;
        let mut ipnorm = 0.0f64;

        if ex_bits == 0 {
            // Plain sign code: `best_rescale_factor` must not run with max_code = 0. (Unreachable on
            // a trained encoder — `train` asserts total_bits ∈ {2,4,8} — but kept for safety.)
            for (c, &r) in codes.iter_mut().zip(residual) {
                *c = (r >= 0.0) as u16;
                ipnorm += 0.5 * (r.abs() as f64 * inv_norm);
            }
        } else {
            let max_code = (1u32 << ex_bits) - 1;
            match self.t_const {
                // Fast path: constant factor, single O(d) pass with no per-component o_abs buffer —
                // `scale·|r_i| = t·(|r_i|/‖r‖)` because `scale = t/‖r‖`.
                Some(t_const) => {
                    let scale = t_const * inv_norm;
                    for (c, &r) in codes.iter_mut().zip(residual) {
                        let ar = r.abs() as f64;
                        let e = (((scale * ar) + 1e-5) as u32).min(max_code);
                        let code = if r >= 0.0 {
                            (1u32 << ex_bits) | e
                        } else {
                            !e & max_code
                        };
                        ipnorm += (e as f64 + 0.5) * (ar * inv_norm);
                        *c = code as u16;
                    }
                }
                // Exact path: per-vector search over the normalized abs residual (reused `o_abs`).
                None => {
                    o_abs.clear();
                    o_abs.extend(residual.iter().map(|&r| (r.abs() as f64 * inv_norm) as f32));
                    let t = best_rescale_factor(o_abs, ex_bits, rescale);
                    for ((c, &r), &o) in codes.iter_mut().zip(residual).zip(o_abs.iter()) {
                        let e = (((t * o as f64) + 1e-5) as u32).min(max_code);
                        let code = if r >= 0.0 {
                            (1u32 << ex_bits) | e
                        } else {
                            !e & max_code
                        };
                        ipnorm += (e as f64 + 0.5) * o as f64;
                        *c = code as u16;
                    }
                }
            }
        }

        ((norm * norm) as f32, (norm / ipnorm) as f32)
    }

    /// [`Self::encode_residual_into`] returning the codes in a fresh `Vec` (test/inspection
    /// convenience; the hot paths reuse scratch buffers instead).
    #[cfg(test)]
    fn encode_residual(&self, residual: &[f32]) -> (Vec<u16>, f32, f32) {
        let mut codes = vec![0u16; residual.len()];
        let (f_add, s_ext) = self.encode_residual_into(
            residual,
            &mut codes,
            &mut Vec::new(),
            &mut RescaleScratch::default(),
        );
        (codes, f_add, s_ext)
    }
}

/// Evaluator holding the quantized query (1-bit sign words or multi-bit planes) and the per-query
/// constants of the extended estimator.
#[derive(Debug, Clone)]
pub struct RabitqExtQueryEvaluator<'e, D = DotProduct> {
    _encoder: PhantomData<&'e RabitqExtQuantizer<D>>,
    /// The rotated query, unquantized. Owned: the evaluator must not borrow the query.
    ///
    /// There is no query quantization in this encoder. RaBitQ-Library does not quantize the query
    /// either on its multi-bit path (`split_single_fulldist` passes the raw f32 query): quantizing
    /// it sets an error floor that no number of document bits can get past, and costs throughput
    /// proportional to the query width.
    q_r: Vec<f32>,
    /// Per-query constant `cb·Σq_r`, completing `⟨u + cb, q_r⟩`. Uses the **exact** query sum; see
    /// the module docs for why the reconstruction's sum would be wrong here.
    q_const: f32,
    /// Per-query additive term (RaBitQ-Library's `g_add`): `‖r_q‖²` for squared Euclidean,
    /// `⟨mean, q⟩` for inner product. See [`RabitqSupportedDistance::query_add`].
    add: f32,
    /// Number of `u64` words per `d` components (`d/64`).
    num_words: usize,
    /// Stored bits per component, one of `{2,4,8}`. The metadata word sits at `total_bits·num_words`.
    total_bits: u32,
}

impl<'e, 'v, D: RabitqSupportedDistance> QueryEvaluator<DenseVectorView<'v, u64>>
    for RabitqExtQueryEvaluator<'e, D>
{
    type Distance = D;

    #[inline]
    fn compute_distance(&self, vector: DenseVectorView<'v, u64>) -> D {
        let words = vector.values();
        let total_words = self.total_bits as usize * self.num_words;
        let code = &words[..total_words];
        // (f_add = ‖r‖², s_ext = ‖r‖/ipnorm) — both precomputed at encode time so this scan does
        // no divide; see `pack_metadata`.
        let (f_add, s_ext) = unpack_metadata(words[total_words]);
        // raw = ⟨u + cb, q_r⟩: the code/query dot plus the closed-form `cb·Σq_r` in `q_const`.
        let raw =
            doc_code_ip_exact(code, &self.q_r, self.num_words, self.total_bits) + self.q_const;
        // Metric-specific combine: both metrics reduce to ⟨r, q_r⟩ ≈ s_ext·⟨u + cb, q_r⟩ plus a
        // per-query additive term.
        D::from_terms(self.add, f_add, s_ext * raw)
    }

    /// Six candidates through [`Self::compute_distance`].
    ///
    /// No fused kernel: the scan is already one pass over the query per candidate, so the only
    /// thing a fused version would save is re-reading `q_r` (an L1 hit) and it would not reduce the
    /// FMA count, which is what this kernel is bound by. Left as six calls deliberately.
    #[inline]
    fn compute_distances_batch6(&self, vectors: [DenseVectorView<'v, u64>; 6]) -> [D; 6] {
        vectors.map(|v| self.compute_distance(v))
    }
}

impl<D: RabitqSupportedDistance> DenseVectorEncoder for RabitqExtQuantizer<D> {
    type InputValueType = f32;
    type OutputValueType = u64;

    /// Decode the stored planes into the residual reconstruction `r̂_i = s_ext·(u_i + cb)` (the
    /// extended analogue of the 1-bit encoder's `±1` decode; the values live in the rotated,
    /// centered residual space).
    fn decode_vector<'a>(&self, encoded: DenseVectorView<'a, u64>) -> DenseVectorOwned<f32> {
        DenseVectorOwned::new(self.reconstruct_residual(encoded.values()))
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
        let residual = self.residual(input.values());
        let mut codes = vec![0u16; self.d()];
        let (f_add, s_ext) = self.encode_residual_into(
            &residual,
            &mut codes,
            &mut Vec::new(),
            &mut RescaleScratch::default(),
        );
        let total_words = self.config.total_bits as usize * self.num_words();
        let mut words = vec![0u64; total_words + 1];
        pack_components_into(
            &codes,
            self.config.total_bits,
            self.num_words(),
            &mut words[..total_words],
        );
        // Store the scan-ready constants: f_add = ‖r‖² and s_ext = ‖r‖/ipnorm.
        words[total_words] = pack_metadata(f_add, s_ext);
        output.extend(words);
    }
}

/// Document codes are identical under either metric; only the query path differs.
impl<D, D2> crate::core::vector_encoder::WithDistance<D2> for RabitqExtQuantizer<D>
where
    D: RabitqSupportedDistance,
    D2: RabitqSupportedDistance,
{
    type Output = RabitqExtQuantizer<D2>;

    fn with_distance(&self) -> Self::Output {
        RabitqExtQuantizer {
            space: self.space.clone(),
            t_const: self.t_const,
            config: self.config,
            _distance: PhantomData,
        }
    }
}

impl<D: RabitqSupportedDistance> VectorEncoder for RabitqExtQuantizer<D> {
    type Distance = D;
    type InputVector<'a> = DenseVectorView<'a, f32>;
    type QueryVector<'q> = DenseVectorView<'q, f32>;
    type EncodedVector<'a> = DenseVectorView<'a, u64>;

    type Evaluator<'e>
        = RabitqExtQueryEvaluator<'e, D>
    where
        Self: 'e;

    /// This encoder scores against an **unquantized** query, so it has no query-side
    /// configuration.
    type QueryParams = ();

    /// Build an evaluator by moving the `f32` query into the rotated space — centered by the mean
    /// or not, per the metric (see [`RabitqSupportedDistance::CENTER_QUERY`]) — and keeping it as
    /// `f32`, with the per-query constant `cb·Σq_r` precomputed.
    #[inline]
    fn query_evaluator<'e>(
        &'e self,
        query: Self::QueryVector<'_>,
        _params: &(),
    ) -> Self::Evaluator<'e> {
        assert_eq!(
            query.len(),
            self.d(),
            "Query vector length must equal encoder input dimension."
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
        self.rotated_query_evaluator(&residual, mean_dot_query)
    }

    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let words = vector.values();
        let nw = self.num_words();
        let total_words = self.config.total_bits as usize * nw;
        // Component-major codes cannot play the part of a query directly (the scan wants f32), so
        // the build path reconstructs this document's residual and scores it as an exact query:
        // one O(d) decode per evaluator, not per candidate.
        let mut q_r = self.reconstruct_residual(words);
        if !D::CENTER_QUERY {
            // Inner product is not centering-invariant: rebuild the un-centered rotated vector
            // `P·x̂ = r̂ + P·mean` so the `⟨P·mean, r_y⟩` term survives (see the type docs).
            for (v, &m) in q_r.iter_mut().zip(self.space.rotated_means().iter()) {
                *v += m;
            }
            let mean_dot_query = self.rotated_mean_dot(&q_r);
            return self.rotated_query_evaluator(&q_r, mean_dot_query);
        }
        // Squared Euclidean: `add` must be the *stored* ‖r‖², not the reconstruction's norm.
        let (f_add, _) = unpack_metadata(words[total_words]);
        let sum_q = q_r.iter().map(|&v| v as f64).sum::<f64>() as f32;
        RabitqExtQueryEvaluator {
            _encoder: PhantomData,
            q_r,
            q_const: self.cb() * sum_q,
            add: D::query_add(0.0, f_add.sqrt()),
            num_words: nw,
            total_bits: self.config.total_bits,
        }
    }

    fn input_dim(&self) -> usize {
        self.d()
    }

    /// `total_bits` planes of code words plus the metadata word.
    fn output_dim(&self) -> usize {
        self.config.total_bits as usize * self.num_words() + 1
    }

    fn compute_distance_between(
        &self,
        v1: Self::EncodedVector<'_>,
        v2: Self::EncodedVector<'_>,
    ) -> Self::Distance {
        // Both metrics reconstruct `v1` into an exact query, so there is no allocation-free
        // shortcut to take: one evaluator, then one scan.
        self.vector_evaluator(v1).compute_distance(v2)
    }
}

impl<D> SpaceUsage for RabitqExtQuantizer<D> {
    fn space_usage_bytes(&self) -> usize {
        self.space.space_usage_bytes() + self.t_const.space_usage_bytes()
    }
}

/// Train on a plain `f32` dataset and encode every vector in parallel.
///
/// Takes the source by reference, so the plain vectors survive the call. The source metric `Ds` is
/// independent of the target metric `D`: document codes are identical either way, so one plain
/// dataset can produce both.
impl<D, Ds> ConvertFrom<&PlainDenseDataset<f32, Ds>> for crate::DenseDataset<RabitqExtQuantizer<D>>
where
    D: RabitqSupportedDistance,
    Ds: ScalarDenseSupportedDistance,
{
    type Config = RabitqExtConfig;

    fn convert_from(dataset: &PlainDenseDataset<f32, Ds>, config: RabitqExtConfig) -> Self {
        let encoder = RabitqExtQuantizer::<D>::train(dataset, config);
        crate::DenseDataset::<RabitqExtQuantizer<D>>::from_flat_par(
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
    for crate::DenseDataset<RabitqExtQuantizer<D>>
where
    D: RabitqSupportedDistance,
    Ds: ScalarDenseSupportedDistance,
{
    type Config = RabitqExtConfig;

    fn convert_from(dataset: &PlainDenseDataset<half::f16, Ds>, config: RabitqExtConfig) -> Self {
        let encoder = RabitqExtQuantizer::<D>::train_narrow(dataset, config);
        crate::DenseDataset::<RabitqExtQuantizer<D>>::from_flat_par_upcast(
            encoder,
            dataset.values(),
            dataset.len(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::distances::{Distance, SquaredEuclideanDistance};
    use crate::dataset::ConvertInto;
    use crate::{
        DatasetGrowable, FlatIndex, Index, IndexSerializer, PlainDenseDatasetGrowable,
        PlainDenseQuantizer,
    };

    /// Extended RaBitQ scoring squared Euclidean distance.
    type RabitqExtL2 = RabitqExtQuantizer<SquaredEuclideanDistance>;
    /// Extended RaBitQ scoring inner product.
    type RabitqExtIp = RabitqExtQuantizer<DotProduct>;

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

    /// A smooth, sign-varying vector of length `d` (magnitudes spread over many quantization
    /// levels, unlike the spiky ±1 fixtures).
    fn smooth_vector(d: usize) -> Vec<f32> {
        (0..d).map(|i| (i as f32 * 0.7).sin() * 2.0 + 0.3).collect()
    }

    fn ext_config(total_bits: u32, rotate: bool) -> RabitqExtConfig {
        RabitqExtConfig {
            total_bits,
            seed: 42,
            rotate,
            ..Default::default()
        }
    }

    /// Conversion must equal training on the source and then encoding every row in parallel.
    /// Swept over `total_bits`, because the row stride is `total_bits · num_words() + 1` — a config
    /// that failed to reach `train` would show up as a stride mismatch at every width but the
    /// default.
    #[test]
    fn convert_into_matches_the_explicit_train_and_encode_path() {
        let plain = plain_dataset(&varied_vectors());

        for total_bits in [2u32, 4, 8] {
            for faster_quant in [true, false] {
                let config = RabitqExtConfig {
                    total_bits,
                    seed: 7,
                    faster_quant,
                    ..Default::default()
                };
                let expected = {
                    let encoder = RabitqExtIp::train(&plain, config);
                    crate::DenseDataset::<RabitqExtIp>::from_flat_par(
                        encoder,
                        plain.values(),
                        plain.len(),
                    )
                };
                let converted: crate::DenseDataset<RabitqExtIp> = (&plain).convert_into(config);

                assert_eq!(
                    converted.values(),
                    expected.values(),
                    "total_bits {total_bits}"
                );
                assert_eq!(converted.encoder(), expected.encoder());
            }
        }
    }

    /// The premise residual reranking stands on: the estimator's score IS the inner product with
    /// the reconstruction, not a statistical estimate of the inner product with the *original*.
    ///
    /// If it were the latter, `⟨q,x⟩ = first_stage_score + ⟨q, x − x̂⟩` would double-count and a
    /// residual second stage would be worse than an independent one. Because it is the former, the
    /// decomposition is exact under dot product and 8 bits spent on `x − x̂` strictly beat 8 bits
    /// spent on `x`.
    #[test]
    fn estimator_score_equals_the_inner_product_with_the_reconstruction() {
        use crate::core::vector_encoder::VectorEncoder;

        let vectors: Vec<Vec<f32>> = (0..12)
            .map(|j| {
                (0..64)
                    .map(|i| ((i * 7 + j * 13) as f32 * 0.11).sin() * (1.0 + j as f32 * 0.05))
                    .collect()
            })
            .collect();
        let plain = plain_dataset(&vectors);

        for total_bits in [2u32, 4, 8] {
            let config = RabitqExtConfig {
                total_bits,
                seed: 11,
                ..Default::default()
            };
            let encoder = RabitqExtIp::train(&plain, config);
            let data = crate::DenseDataset::<RabitqExtIp>::from_flat_par(
                encoder.clone(),
                plain.values(),
                plain.len(),
            );

            let query: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.37).cos()).collect();
            let evaluator = data
                .encoder()
                .query_evaluator(DenseVectorView::new(&query), &());

            // Reconstruct every stored vector, then score the reconstructions through the PLAIN
            // encoder's own evaluator -- so the sign and offset conventions are the metric's, not
            // this test's guess at them.
            let reconstructions: Vec<Vec<f32>> = (0..plain.len())
                .map(|id| data.encoder().reconstruct(data.get(id as u64).values()))
                .collect();
            let plain_hat = plain_dataset(&reconstructions);
            let plain_eval = plain_hat
                .encoder()
                .query_evaluator(DenseVectorView::new(&query), &());

            for id in 0..plain.len() {
                let code = data.get(id as u64);
                let estimated = evaluator.compute_distance(code).distance();
                let direct = plain_eval
                    .compute_distance(plain_hat.get(id as u64))
                    .distance();
                let scale = estimated.abs().max(direct.abs()).max(1.0);
                assert!(
                    (estimated - direct).abs() / scale < 2e-3,
                    "total_bits {total_bits}, id {id}: estimator {estimated} vs \
                     plain score of reconstruct(code) {direct}"
                );
            }
        }
    }

    /// Is `first_stage + rerank(x − x̂)` actually a better estimate of `⟨q,x⟩` than `rerank(x)`?
    ///
    /// This is the arithmetic of residual reranking with every index and heap removed: score each
    /// vector three ways against the exact plain score and compare the ERRORS. If the residual sum
    /// is not more accurate here, no amount of plumbing above it can make it so, and the idea is
    /// wrong rather than merely mis-wired.
    #[test]
    fn residual_sum_beats_encoding_the_whole_vector() {
        use crate::core::vector_encoder::VectorEncoder;

        // Clustered, so the first stage has structure to capture and `x̂` is informative.
        let (n, d) = (400usize, 128usize);
        let centres: Vec<Vec<f32>> = (0..8)
            .map(|c| {
                (0..d)
                    .map(|i| (((i * 31 + c * 977) % 251) as f32 / 251.0) - 0.5)
                    .collect()
            })
            .collect();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|j| {
                let c = &centres[j % 8];
                (0..d)
                    .map(|i| c[i] + 0.08 * (((i * 17 + j * 53) % 97) as f32 / 97.0 - 0.5))
                    .collect()
            })
            .collect();
        let plain = plain_dataset(&vectors);
        let query: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.21).sin() * 0.5).collect();

        let exact_eval = plain
            .encoder()
            .query_evaluator(DenseVectorView::new(&query), &());

        for first_bits in [2u32, 4] {
            // Stage 1, and the reconstruction it implies.
            let cfg1 = RabitqExtConfig {
                total_bits: first_bits,
                seed: 3,
                ..Default::default()
            };
            let first = crate::DenseDataset::<RabitqExtIp>::from_flat_par(
                RabitqExtIp::train(&plain, cfg1),
                plain.values(),
                n,
            );
            let residuals: Vec<Vec<f32>> = (0..n)
                .map(|id| {
                    let x_hat = first.encoder().reconstruct(first.get(id as u64).values());
                    vectors[id].iter().zip(&x_hat).map(|(a, b)| a - b).collect()
                })
                .collect();

            // Seed sweep: kannolo gives BOTH stages `..Default::default()`, so the rerank
            // encoder re-rotates the residual by the very `P` that produced it. `seed: 3` here is
            // the first stage's seed, i.e. that same-rotation case; `seed: 5` is an independent
            // rotation. If the two differ, the shared rotation is the defect.
            for rr_seed in [3u64, 5] {
                let cfg8 = RabitqExtConfig {
                    total_bits: 8,
                    seed: rr_seed,
                    ..Default::default()
                };
                // Second stage over the residuals, and — as the control — over the whole vectors.
                let resid_plain = plain_dataset(&residuals);
                let rr_resid = crate::DenseDataset::<RabitqExtIp>::from_flat_par(
                    RabitqExtIp::train(&resid_plain, cfg8),
                    resid_plain.values(),
                    n,
                );
                let rr_full = crate::DenseDataset::<RabitqExtIp>::from_flat_par(
                    RabitqExtIp::train(&plain, cfg8),
                    plain.values(),
                    n,
                );

                let e1 = first
                    .encoder()
                    .query_evaluator(DenseVectorView::new(&query), &());
                let er = rr_resid
                    .encoder()
                    .query_evaluator(DenseVectorView::new(&query), &());
                let ef = rr_full
                    .encoder()
                    .query_evaluator(DenseVectorView::new(&query), &());

                let (mut err_sum, mut err_full, mut norm_r, mut norm_x) =
                    (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                for id in 0..n {
                    let exact = exact_eval.compute_distance(plain.get(id as u64)).distance();
                    let s1 = e1.compute_distance(first.get(id as u64)).distance();
                    let sr = er.compute_distance(rr_resid.get(id as u64)).distance();
                    let sf = ef.compute_distance(rr_full.get(id as u64)).distance();
                    err_sum += ((s1 + sr) - exact).abs() as f64;
                    err_full += (sf - exact).abs() as f64;
                    norm_r += residuals[id]
                        .iter()
                        .map(|v| (v * v) as f64)
                        .sum::<f64>()
                        .sqrt();
                    norm_x += vectors[id]
                        .iter()
                        .map(|v| (v * v) as f64)
                        .sum::<f64>()
                        .sqrt();
                }
                println!(
                    "first={first_bits}b rerank_seed={rr_seed}{}  mean|err| residual-sum {:.6}  \
                 whole-vector {:.6}  ratio {:.3}   mean||r||/||x|| {:.3}",
                    if rr_seed == 3 {
                        " (SAME rotation as stage 1)"
                    } else {
                        " (independent)      "
                    },
                    err_sum / n as f64,
                    err_full / n as f64,
                    err_sum / err_full,
                    norm_r / norm_x,
                );
            }
        }
    }

    #[test]
    fn metadata_word_roundtrips_the_two_scan_constants() {
        let (f_add, s_ext) = (123.456f32, 0.797_884_6_f32);
        let (f2, s2) = unpack_metadata(pack_metadata(f_add, s_ext));
        assert_eq!(f_add, f2);
        assert_eq!(s_ext, s2);
    }

    #[test]
    fn shifted_code_is_antisymmetric_and_ip_with_residual_positive() {
        // The identity the whole estimator rests on: u_i + cb = sign(r_i)·(e_i + 0.5), hence
        // ⟨r, u + cb⟩ = ‖r‖·ipnorm = f_add/s_ext > 0.
        let dataset = plain_dataset(&varied_vectors());
        for total_bits in [2u32, 4, 8] {
            let encoder = RabitqExtIp::train(&dataset, ext_config(total_bits, true));
            let r = smooth_vector(64);
            let (codes, f_add, s_ext) = encoder.encode_residual(&r);
            let cb = encoder.cb();
            let ex_bits = total_bits - 1;

            let mut ip = 0.0f64;
            for (&u, &ri) in codes.iter().zip(&r) {
                let xu_cb = u as f32 + cb;
                // Sign agreement (r_i is never exactly 0 for this fixture).
                assert!(
                    xu_cb * ri > 0.0,
                    "total_bits {total_bits}: u {u} + cb {cb} disagrees with r {ri}"
                );
                // The shifted code is a half-integer magnitude in (0, 2^ex_bits].
                let mag = xu_cb.abs();
                assert!(
                    (mag - (mag.floor() + 0.5)).abs() < 1e-6 && mag <= (1u64 << ex_bits) as f32,
                    "total_bits {total_bits}: |u + cb| = {mag} not a half-integer level"
                );
                ip += (ri * xu_cb) as f64;
            }
            let expected = f_add as f64 / s_ext as f64; // ‖r‖·ipnorm
            assert!(
                expected > 0.0 && (ip - expected).abs() < 1e-2 * expected,
                "total_bits {total_bits}: ⟨r, u+cb⟩ = {ip} vs f_add/s_ext = {expected}"
            );
        }
    }

    /// Every query must retrieve itself near the top under metric `D`.
    fn assert_self_is_top_ranked<D: RabitqSupportedDistance>(
        vectors: &[Vec<f32>],
        dataset: &PlainDenseDataset<f32, DotProduct>,
        config: RabitqExtConfig,
    ) {
        let rabitq: crate::DenseDataset<RabitqExtQuantizer<D>> = dataset.convert_into(config);
        let index = FlatIndex::from(&rabitq);
        for (i, q) in vectors.iter().enumerate() {
            let top = index.search(DenseVectorView::new(q), vectors.len(), &());
            let self_rank = top
                .iter()
                .position(|r| r.vector as usize == i)
                .expect("self must be retrieved");
            // The all-(+1) and all-(-1) vectors binarize identically to others under rotation
            // with so few points, so allow self to sit just below the top.
            assert!(
                self_rank < 3,
                "query {i} ranked itself at {self_rank} with {config:?}"
            );
        }
    }

    #[test]
    fn search_finds_self_as_nearest_across_bit_widths() {
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        // `varied_vectors` all have norm 8; on equal-norm data the two metrics rank alike, so
        // both must put the query itself at the top, at every bit-width combination.
        for total_bits in [2u32, 4, 8] {
            let config = ext_config(total_bits, true);
            assert_self_is_top_ranked::<SquaredEuclideanDistance>(&vectors, &dataset, config);
            assert_self_is_top_ranked::<DotProduct>(&vectors, &dataset, config);
        }
    }

    #[test]
    fn search_finds_self_as_nearest_without_rotation() {
        // With rotation disabled the estimator runs with `P = I` (also orthogonal), so
        // self-retrieval must still hold on both metrics.
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        let config = ext_config(4, false);
        assert_self_is_top_ranked::<SquaredEuclideanDistance>(&vectors, &dataset, config);
        assert_self_is_top_ranked::<DotProduct>(&vectors, &dataset, config);
    }

    /// Scores stay finite across every bit-width combination and seed under metric `D`.
    fn assert_scores_finite<D: RabitqSupportedDistance>(
        vectors: &[Vec<f32>],
        dataset: &PlainDenseDataset<f32, DotProduct>,
    ) {
        for total_bits in [2, 4, 8] {
            {
                for seed in [1, 42] {
                    let config = RabitqExtConfig {
                        total_bits,
                        seed,
                        rotate: true,
                        ..Default::default()
                    };
                    let ds: crate::DenseDataset<RabitqExtQuantizer<D>> =
                        dataset.convert_into(config);
                    let index = FlatIndex::from(&ds);
                    for q in vectors {
                        for r in index.search(DenseVectorView::new(q), 3, &()) {
                            assert!(
                                r.distance.distance().is_finite(),
                                "non-finite score with {config:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn scores_are_finite_for_every_bit_width() {
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        assert_scores_finite::<SquaredEuclideanDistance>(&vectors, &dataset);
        assert_scores_finite::<DotProduct>(&vectors, &dataset);
    }

    #[test]
    fn identical_vectors_give_finite_scores() {
        // All residuals are exactly zero (every vector equals the mean): the metadata degenerates
        // to (0, 0) and every estimate must stay finite with term = 0.
        let vectors: Vec<Vec<f32>> = vec![smooth_vector(64); 4];
        let dataset = plain_dataset(&vectors);
        {
            let config = ext_config(4, true);
            let ds: crate::DenseDataset<RabitqExtL2> = (&dataset).convert_into(config);
            let index = FlatIndex::from(&ds);
            for r in index.search(DenseVectorView::new(&vectors[0]), 4, &()) {
                assert!(r.distance.distance().is_finite(), "non-finite score");
            }
        }
    }

    #[test]
    #[should_panic(expected = "dim % 64 == 0")]
    fn train_rejects_non_multiple_of_64() {
        let dataset = plain_dataset(&[vec![0.0f32; 10]]);
        let _ = RabitqExtIp::train(&dataset, RabitqExtConfig::default());
    }

    #[test]
    #[should_panic(expected = "total_bits")]
    fn train_rejects_out_of_range_total_bits() {
        let dataset = plain_dataset(&[vec![0.0f32; 64]]);
        let config = RabitqExtConfig {
            total_bits: 10,
            ..Default::default()
        };
        let _ = RabitqExtIp::train(&dataset, config);
    }

    #[test]
    #[should_panic(expected = "total_bits")]
    fn train_rejects_zero_total_bits() {
        let dataset = plain_dataset(&[vec![0.0f32; 64]]);
        let config = RabitqExtConfig {
            total_bits: 0,
            ..Default::default()
        };
        let _ = RabitqExtIp::train(&dataset, config);
    }

    #[test]
    fn reconstruction_cosine_improves_with_total_bits() {
        let d = 64;
        let x = smooth_vector(d);
        let vectors = vec![x.clone(), x.iter().map(|&v| -v).collect()];
        let dataset = plain_dataset(&vectors);

        let mut prev_cos = 0.0f32;
        for total_bits in [2u32, 4, 8] {
            let encoder = RabitqExtL2::train(&dataset, ext_config(total_bits, true));
            let residual = encoder.residual(&x);
            let mut words: Vec<u64> = Vec::new();
            encoder.push_encoded(DenseVectorView::new(&x), &mut words);
            let decoded = encoder.decode_vector(DenseVectorView::new(&words));

            let dot: f32 = residual
                .iter()
                .zip(decoded.values())
                .map(|(a, b)| a * b)
                .sum();
            let nr: f32 = residual.iter().map(|v| v * v).sum::<f32>().sqrt();
            let nd: f32 = decoded.values().iter().map(|v| v * v).sum::<f32>().sqrt();
            let cos = dot / (nr * nd);
            assert!(
                cos >= prev_cos - 1e-4,
                "cosine dropped from {prev_cos} to {cos} at total_bits {total_bits}"
            );
            prev_cos = cos;
        }
        assert!(
            prev_cos > 0.99,
            "7-bit reconstruction cosine too low: {prev_cos}"
        );
    }

    #[test]
    fn faster_quant_matches_optimal_reconstruction() {
        // The constant-factor fast path (`faster_quant = true`) must reconstruct essentially as well
        // as the exact per-vector search: it uses an averaged rescale factor, so codes differ
        // slightly, but the reconstruction cosine must stay within a small tolerance. Assert on
        // reconstruction quality (the meaningful quantity), not bit-equality (which is expected to
        // differ by design).
        let d = 128;
        let x = smooth_vector(d);
        let vectors = vec![x.clone(), x.iter().map(|&v| -v * 0.5 + 0.2).collect()];
        let dataset = plain_dataset(&vectors);

        for total_bits in [2u32, 4, 8] {
            let cos = |faster_quant: bool| -> f32 {
                let cfg = RabitqExtConfig {
                    total_bits,
                    seed: 42,
                    rotate: true,
                    faster_quant,
                };
                let encoder = RabitqExtL2::train(&dataset, cfg);
                let residual = encoder.residual(&x);
                let mut words: Vec<u64> = Vec::new();
                encoder.push_encoded(DenseVectorView::new(&x), &mut words);
                let decoded = encoder.decode_vector(DenseVectorView::new(&words));
                let dot: f32 = residual
                    .iter()
                    .zip(decoded.values())
                    .map(|(a, b)| a * b)
                    .sum();
                let nr: f32 = residual.iter().map(|v| v * v).sum::<f32>().sqrt();
                let nd: f32 = decoded.values().iter().map(|v| v * v).sum::<f32>().sqrt();
                dot / (nr * nd)
            };
            let (fast, optimal) = (cos(true), cos(false));
            assert!(
                fast >= optimal - 0.02,
                "fast-path cosine {fast} trails optimal {optimal} by too much at total_bits {total_bits}"
            );
        }
    }

    #[test]
    fn l2_score_matches_library_factor_form() {
        // Parity check: the evaluator's L2 score must equal the extended-RaBitQ decomposition
        //   est_dist = f_add + g_add + f_rescale·⟨u + cb, q_r⟩,   f_rescale = −2·s_ext
        // rebuilt here from scalar residuals, codes and the exact rotated query — an independent
        // route through the shifted-code expansion.
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        let config = ext_config(4, true);
        let encoder = RabitqExtL2::train(&dataset, config);
        let ds: crate::DenseDataset<RabitqExtL2> = (&dataset).convert_into(config);
        let cb = encoder.cb();

        for q in &vectors {
            let q_res = encoder.residual(q);
            let nq: f32 = q_res.iter().map(|v| v * v).sum::<f32>().sqrt();
            if nq <= f32::EPSILON {
                continue;
            }

            let evaluator = encoder.query_evaluator(DenseVectorView::new(q), &());
            for (i, doc) in vectors.iter().enumerate() {
                let r = encoder.residual(doc);
                let (u, f_add, s_ext) = encoder.encode_residual(&r);
                if s_ext == 0.0 {
                    continue;
                }
                let ip_xu_q: f32 = u
                    .iter()
                    .zip(&q_res)
                    .map(|(&ui, &qh)| (ui as f32 + cb) * qh)
                    .sum();
                let expected = f_add + nq * nq - 2.0 * s_ext * ip_xu_q;

                let got = evaluator.compute_distance(ds.get(i as u64)).distance();
                let tol = 1e-2 * expected.abs().max(1.0);
                assert!(
                    (got - expected).abs() <= tol,
                    "doc {i}: evaluator L2 {got} vs factor form {expected} (tol {tol})"
                );
            }
        }
    }

    #[test]
    fn ip_score_matches_direct_estimator_form() {
        // Parity check for the inner-product path: the evaluator's score must equal
        // ⟨mean, q⟩ + s_ext·⟨u + cb, q_r⟩ rebuilt from scalars. Note the query is rotated, *not*
        // centered, so `rotate` is the right transform.
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        let config = ext_config(4, true);
        let encoder = RabitqExtIp::train(&dataset, config);
        let ds: crate::DenseDataset<RabitqExtIp> = (&dataset).convert_into(config);
        let cb = encoder.cb();

        for q in &vectors {
            let q_rot = encoder.rotate(q);
            let nq: f32 = q_rot.iter().map(|v| v * v).sum::<f32>().sqrt();
            if nq <= f32::EPSILON {
                continue;
            }
            let mean_dot = encoder.mean_dot(q);

            let evaluator = encoder.query_evaluator(DenseVectorView::new(q), &());
            for (i, doc) in vectors.iter().enumerate() {
                let r = encoder.residual(doc);
                let (u, _, s_ext) = encoder.encode_residual(&r);
                if s_ext == 0.0 {
                    continue;
                }
                let ip_xu_q: f32 = u
                    .iter()
                    .zip(&q_rot)
                    .map(|(&ui, &qh)| (ui as f32 + cb) * qh)
                    .sum();
                let expected = mean_dot + s_ext * ip_xu_q;

                let got = evaluator.compute_distance(ds.get(i as u64)).0;
                let tol = 1e-2 * expected.abs().max(1.0);
                assert!(
                    (got - expected).abs() <= tol,
                    "doc {i}: evaluator IP {got} vs direct form {expected} (tol {tol})"
                );
            }
        }
    }

    /// The fused batch kernel must be bit-identical to six single `compute_distance` calls,
    /// across the 1-bit and multi-bit query paths and every document plane count, under metric `D`.
    fn assert_batch6_matches_singles<D: RabitqSupportedDistance + std::fmt::Debug>(
        vectors: &[Vec<f32>],
        dataset: &PlainDenseDataset<f32, DotProduct>,
    ) {
        for total_bits in [2u32, 4, 8] {
            let config = ext_config(total_bits, true);
            let ds: crate::DenseDataset<RabitqExtQuantizer<D>> = dataset.convert_into(config);
            for q in vectors {
                let evaluator = ds.encoder().query_evaluator(DenseVectorView::new(q), &());
                let views = std::array::from_fn(|k| ds.get(k as u64));
                let batch = evaluator.compute_distances_batch6(views);
                let singles = std::array::from_fn(|k| evaluator.compute_distance(ds.get(k as u64)));
                assert_eq!(batch, singles, "{config:?}");
            }
            // The build path (`vector_evaluator`) dispatches on `total_bits` planes (up to 9):
            // its batch scores must also match its singles.
            let evaluator = ds.encoder().vector_evaluator(ds.get(0));
            let views = std::array::from_fn(|k| ds.get(k as u64));
            let batch = evaluator.compute_distances_batch6(views);
            let singles = std::array::from_fn(|k| evaluator.compute_distance(ds.get(k as u64)));
            assert_eq!(batch, singles, "build path, {config:?}");
        }
    }

    #[test]
    fn compute_distances_batch6_matches_six_singles() {
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        assert_batch6_matches_singles::<SquaredEuclideanDistance>(&vectors, &dataset);
        assert_batch6_matches_singles::<DotProduct>(&vectors, &dataset);
    }

    #[test]
    fn dataset_serialization_round_trip() {
        // The trained geometry lives in a nested `RabitqSpace`, so exercise the serde path end to
        // end: a reloaded index must equal the original *and* score identically.
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        let ds: crate::DenseDataset<RabitqExtIp> =
            (&dataset).convert_into(RabitqExtConfig::default());

        let mut path = std::env::temp_dir();
        path.push(format!("vectorium_rabitq_ext_{}.bin", std::process::id()));
        let path = path.to_str().unwrap().to_string();

        ds.save_index(&path).unwrap();
        let loaded = crate::DenseDataset::<RabitqExtIp>::load_index(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(ds, loaded);
        for q in &vectors {
            let before = ds.encoder().query_evaluator(DenseVectorView::new(q), &());
            let after = loaded
                .encoder()
                .query_evaluator(DenseVectorView::new(q), &());
            for i in 0..ds.len() as u64 {
                assert_eq!(
                    before.compute_distance(ds.get(i)),
                    after.compute_distance(loaded.get(i)),
                    "vector {i}"
                );
            }
        }
    }

    /// Squared-Euclidean alias, swept over `total_bits` because the stored row is
    /// `total_bits · num_words() + 1` words — a serde bug that mixes up the plane count would only
    /// show at a non-default width. `faster_quant: false` also exercises the exact per-vector
    /// rescale search, whose trained constant is *not* stored in that mode.
    #[test]
    fn dataset_serialization_round_trip_squared_euclidean() {
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);

        for (total_bits, faster_quant) in [(2u32, true), (4, false), (8, true)] {
            let config = RabitqExtConfig {
                total_bits,
                faster_quant,
                ..RabitqExtConfig::default()
            };
            let ds: crate::DenseDataset<RabitqExtL2> = (&dataset).convert_into(config);

            let mut path = std::env::temp_dir();
            path.push(format!(
                "vectorium_rabitq_ext_l2_{}_{}.bin",
                std::process::id(),
                total_bits
            ));
            let path = path.to_str().unwrap().to_string();

            ds.save_index(&path).unwrap();
            let loaded = crate::DenseDataset::<RabitqExtL2>::load_index(&path).unwrap();
            std::fs::remove_file(&path).unwrap();

            assert_eq!(ds, loaded, "total_bits {total_bits}");
            assert_eq!(loaded.encoder().config, config, "config survived serde");

            for q in &vectors {
                let before = ds.encoder().query_evaluator(DenseVectorView::new(q), &());
                let after = loaded
                    .encoder()
                    .query_evaluator(DenseVectorView::new(q), &());
                for i in 0..ds.len() as u64 {
                    assert_eq!(
                        before.compute_distance(ds.get(i)),
                        after.compute_distance(loaded.get(i)),
                        "query-side score, total_bits {total_bits}, vector {i}"
                    );
                }
            }
            // The build path dispatches on `total_bits`, so exercise it after a reload too.
            for i in 0..ds.len() as u64 {
                let before = ds.encoder().vector_evaluator(ds.get(i));
                let after = loaded.encoder().vector_evaluator(loaded.get(i));
                for j in 0..ds.len() as u64 {
                    assert_eq!(
                        before.compute_distance(ds.get(j)),
                        after.compute_distance(loaded.get(j)),
                        "build-path score, total_bits {total_bits}, pair ({i}, {j})"
                    );
                }
            }
        }
    }

    #[test]
    fn compute_distances_batch6_matches_six_singles_wide() {
        // d = 640 → 10 words per plane: exercises the 8-wide unchecked-load chunk loop *and* the
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

    /// The allocation-free two-document score must be bit-identical to the `vector_evaluator`
    /// route it replaces, across the stored plane counts, under metric `D`.
    fn assert_distance_between_matches_evaluator<D: RabitqSupportedDistance + std::fmt::Debug>(
        dataset: &PlainDenseDataset<f32, DotProduct>,
    ) {
        for total_bits in [2, 4, 8] {
            let config = ext_config(total_bits, true);
            let ds: crate::DenseDataset<RabitqExtQuantizer<D>> = dataset.convert_into(config);
            let encoder = ds.encoder();
            for i in 0..ds.len() as u64 {
                for j in 0..ds.len() as u64 {
                    let direct = encoder.compute_distance_between(ds.get(i), ds.get(j));
                    let via = encoder
                        .vector_evaluator(ds.get(i))
                        .compute_distance(ds.get(j));
                    assert_eq!(direct, via, "pair ({i}, {j}), {config:?}");
                }
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
        for total_bits in [2u32, 4, 8] {
            let config = ext_config(total_bits, false);
            let ds: crate::DenseDataset<RabitqExtIp> = (&dataset).convert_into(config);
            let encoder = ds.encoder();
            for i in 0..ds.len() as u64 {
                let mut recon = encoder.reconstruct_residual(ds.get(i).values());
                for (v, &m) in recon.iter_mut().zip(encoder.space.means().iter()) {
                    *v += m;
                }
                let build = encoder.vector_evaluator(ds.get(i));
                let query = encoder.query_evaluator(DenseVectorView::new(&recon), &());
                for j in 0..ds.len() as u64 {
                    assert_eq!(
                        build.compute_distance(ds.get(j)),
                        query.compute_distance(ds.get(j)),
                        "pair ({i}, {j}), {config:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn ip_build_path_estimates_the_true_dot_product() {
        // The build path must estimate ⟨x̂, ŷ⟩, not the residual-space ⟨r̂_x, r̂_y⟩ it used to
        // return: the latter drops ⟨P·mean, r_y⟩, which varies per candidate and so reorders
        // results. Run at high fidelity on both sides so anything left is quantization noise
        // rather than a missing term.
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        let config = ext_config(8, true);
        let ds: crate::DenseDataset<RabitqExtIp> = (&dataset).convert_into(config);
        let encoder = ds.encoder();

        // Reconstructions in rotated space: P·x̂ = r̂_x + P·mean (P orthogonal, so no inverse
        // rotation is needed and every dot product below can be taken here).
        let rotated: Vec<Vec<f32>> = (0..ds.len() as u64)
            .map(|i| {
                let mut v = encoder.reconstruct_residual(ds.get(i).values());
                for (c, &m) in v.iter_mut().zip(encoder.space.rotated_means().iter()) {
                    *c += m;
                }
                v
            })
            .collect();
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(&x, &y)| x * y).sum::<f32>();

        let mut worst_est = 0.0f32;
        let mut worst_residual_only = 0.0f32;
        for i in 0..ds.len() as u64 {
            let evaluator = encoder.vector_evaluator(ds.get(i));
            let r_x = encoder.reconstruct_residual(ds.get(i).values());
            for j in 0..ds.len() as u64 {
                let truth = dot(&rotated[i as usize], &rotated[j as usize]);
                let residual_only = dot(&r_x, &encoder.reconstruct_residual(ds.get(j).values()));
                let est = evaluator.compute_distance(ds.get(j)).0;

                // Errors are measured against ‖x̂‖·‖ŷ‖, the scale of the quantity being estimated;
                // dividing by |truth| would blow up on the pairs that happen to be near-orthogonal.
                let denom = (dot(&rotated[i as usize], &rotated[i as usize])
                    * dot(&rotated[j as usize], &rotated[j as usize]))
                .sqrt();
                worst_est = worst_est.max((est - truth).abs() / denom);
                worst_residual_only =
                    worst_residual_only.max((residual_only - truth).abs() / denom);
            }
        }
        assert!(
            worst_est < 0.02,
            "build-path IP off the true dot product by {worst_est} of ‖x̂‖·‖ŷ‖"
        );
        // Confirms the bound above has teeth: dropping the centroid terms misses by far more.
        assert!(
            worst_residual_only > 0.2,
            "residual-space scoring only off by {worst_residual_only} — fixture centroid too small \
             for this test to discriminate"
        );
    }

    #[test]
    #[should_panic(expected = "total_bits")]
    fn train_rejects_one_total_bits() {
        // total_bits = 1 (the plain sign code) is the domain of RabitqQuantizer, not this encoder;
        // the constructor must reject it rather than silently run the degenerate ex_bits = 0 path.
        let dataset = plain_dataset(&[vec![0.0f32; 64]]);
        let config = RabitqExtConfig {
            total_bits: 1,
            ..Default::default()
        };
        let _ = RabitqExtIp::train(&dataset, config);
    }

    /// A derangement over the ten `varied_vectors` rows, so no row stays where it started.
    const PERMUTATION: [usize; 10] = [2, 0, 3, 1, 5, 4, 7, 6, 9, 8];

    /// `permute` copies encoded rows verbatim at a stride of `output_dim()`. For the extended
    /// encoder that stride is `total_bits · num_words() + 1`, so it is swept over several widths:
    /// a stride bug shows up as a row that shifts by a multiple of the plane count.
    /// kANNolo's edge-compressed graph types (`permuted`, `streamvbyte`) reorder nodes this way.
    #[test]
    fn rabitq_ext_permute_moves_each_row_to_its_target_slot() {
        let plain = plain_dataset(&varied_vectors());

        for total_bits in [2u32, 4, 8] {
            let config = RabitqExtConfig {
                total_bits,
                ..Default::default()
            };
            let dataset: crate::DenseDataset<RabitqExtIp> = (&plain).convert_into(config);

            let permuted = dataset.permute(&PERMUTATION);

            assert_eq!(permuted.len(), dataset.len());
            for (old_id, &new_id) in PERMUTATION.iter().enumerate() {
                assert_eq!(
                    permuted.get(new_id as u64).values(),
                    dataset.get(old_id as u64).values(),
                    "row {old_id} did not land at slot {new_id} at total_bits={total_bits}"
                );
            }
        }
    }

    /// Permuting then applying the inverse must reproduce the dataset exactly — every bit plane,
    /// the metadata word and the cloned encoder (means, rotation, config) alike.
    #[test]
    fn rabitq_ext_permute_then_inverse_round_trips() {
        let plain = plain_dataset(&varied_vectors());
        let inverse = crate::core::dataset::invert_permutation(&PERMUTATION);

        for total_bits in [2u32, 4, 8] {
            let config = RabitqExtConfig {
                total_bits,
                ..Default::default()
            };
            let dataset: crate::DenseDataset<RabitqExtIp> = (&plain).convert_into(config);

            assert_eq!(
                dataset.permute(&PERMUTATION).permute(&inverse),
                dataset,
                "round trip failed at total_bits={total_bits}"
            );
        }
    }

    /// Same contract as the 1-bit encoder: an `f16` source must yield exactly the codes the `f32`
    /// path yields for the same (already-rounded) values.
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

        for total_bits in [2u32, 4, 8] {
            let config = ext_config(total_bits, true);
            let from_f32: crate::DenseDataset<RabitqExtIp> =
                ConvertFrom::convert_from(&plain_f32, config);
            let from_f16: crate::DenseDataset<RabitqExtIp> =
                ConvertFrom::convert_from(&plain_f16, config);

            assert_eq!(
                from_f16.values(),
                from_f32.values(),
                "codes differ at total_bits={total_bits}"
            );
        }
    }
}
