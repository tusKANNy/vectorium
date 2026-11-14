// Portions of this file are ported from faiss `contrib/e_means`.
// Copyright (c) Meta Platforms, Inc. and affiliates.
// Licensed under the MIT License; see `LICENSE-MIT` in the root of this repository.

//! E-means: mini-batch k-means with an exponential moving average and fitted scaling laws.
//!
//! A Rust port of the single-worker CPU algorithm (`NumpyEmeans`) from
//! [faiss `contrib/e_means`](https://github.com/facebookresearch/faiss/tree/main/contrib/e_means)
//! (MIT license, Copyright (c) Meta Platforms, Inc. and affiliates), which implements
//! Algorithm 1 of "e-means: outperforming k-means (in one-tenth the epochs)" (Bessa et al.,
//! <https://arxiv.org/abs/2611.09501>). Please cite the paper if you use this.
//!
//! # Algorithm
//!
//! Training makes a fixed budget of epochs over the data (no convergence test): the
//! [`KMeans`] epoch budget, 2 by default for e-means (faiss defaults to 5).
//! Each epoch visits the vectors in a fresh random order, in mini-batches of `B`. For each
//! batch, vectors are assigned to their nearest centroid against a fixed snapshot of the
//! centroids, and every centroid `j` touched by the batch (batch count `n_j`, batch mean
//! `m_j`) is moved by an EMA step `alpha`:
//!
//! ```text
//! N_j <- (1 - alpha) * N_j                      (every cluster: its mass decays)
//! N_j <- N_j + alpha * n_j                      (touched clusters)
//! c_j <- c_j + (alpha * n_j / N_j) * (m_j - c_j)
//! ```
//!
//! `alpha` follows a cosine schedule from `alpha0` down to `0.1 * alpha0`. Every 5 updates,
//! clusters whose mass is below `tau` times the mean mass are revived by splitting donors
//! sampled in proportion to their mass. After the last epoch, every centroid that was assigned
//! vectors during that epoch is replaced by the exact mean of those vectors.
//!
//! Unless overridden, the batch size `B` and the initial step `alpha0` come from the fitted
//! scaling laws in terms of `n`, `k` and `nepochs` (see [`EMeansRecipe::resolve`]). The
//! coefficients are the ones shipped in faiss, which differ from the paper's Table 2.
//!
//! # Representation and metric
//!
//! E-means is a [`Solver`] of [`KMeans`], so it shares its contract: it clusters any
//! [`MeanDataset`] (batch means are [`MeanDataset::mean`], the EMA step is
//! [`MeanDataset::lerp`]) into a growable centroid dataset `G` whose encoder fixes the
//! centroids' precision and the clustering metric. The
//! scaling laws were fitted on dense data under squared Euclidean distance; other metrics and
//! sparse data work, but are outside that fit.
//!
//! # Differences from the faiss implementation
//!
//! * Assignment uses vectorium's distance kernels (the same as the Lloyd solver), not BLAS, so
//!   a benchmark against Lloyd compares solvers, not kernels.
//! * Random streams come from `rand::StdRng`, not NumPy's `RandomState`, so results are not
//!   bit-identical to faiss for the same seed. They are reproducible for a fixed seed, across
//!   thread counts.
//! * Optional per-vector `weights` turn counts and means into weighted counts and means, and an
//!   optional `refine` hook post-processes every updated centroid (as for Lloyd).
//! * [`KMeans::train_with_index`] maintains its centroid index with the
//!   policy shared by every solver ([`IndexRefresh`]): by default the index is refreshed for
//!   every mini-batch, as in faiss's IVF e-means, and always right after a revival moved
//!   centroids; a longer cadence trades assignment accuracy for index maintenance.

#[cfg(doc)]
use super::index_assign::IndexRefresh;
use super::index_assign::UpdateCtx;
use super::kmeans::{Assign, ClusterGroups, Run, imbalance_factor};
#[cfg(doc)]
use super::kmeans::{KMeans, Solver};
use crate::core::dataset::{DatasetGrowable, MeanDataset};
use crate::core::vector_encoder::VectorEncoder;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::{Duration, Instant};

/// Minimum step relative to `alpha0` at the end of the cosine schedule (Equation 5).
const ALPHA_MIN_FACTOR: f64 = 0.1;
/// Number of updates between revival checks (Algorithm 1, line 9).
const CHECK_UNUSED_EVERY: usize = 5;
/// Initial EMA mass `N_j`. faiss uses 2^-16; the paper uses 1e-10.
const INIT_USAGE: f32 = 1.0 / 65536.0;
/// Relative coordinate perturbation used when splitting a donor (Appendix J, Eq. 8).
const SPLIT_EPSILON: f32 = 1e-4;
/// Default revival threshold `tau`: revive clusters with `N_j < tau * mean(N)`.
const UNUSED_THRESHOLD: f32 = 0.1;
/// At or below this dimension the default turns revival off (it hurts on 2-D data).
const NO_REVIVAL_MAX_D: usize = 2;
/// At or below this dimension `BatchLaw::Auto` picks the quality law.
const AUTO_QUALITY_MAX_D: usize = 16;

/// Batch law `B = C_B * k^b_k * n^b_n * T^b_T` as `(C_B, b_k, b_n, b_T)` (Equation 7).
const THROUGHPUT_LAW: (f64, f64, f64, f64) = (0.035134, 0.134749, 0.876473, 0.813344);
/// The quality law uses one eighth of the throughput coefficient.
const QUALITY_LAW: (f64, f64, f64, f64) = (0.00439175, 0.134749, 0.876473, 0.813344);
/// Step law `alpha0 = C * k^a_k * B^a_B * n^a_n * T^a_T` as `(C, a_k, a_B, a_n, a_T)`
/// (Equation 6), clamped to `[1e-4, 1]`.
const ALPHA_LAW: (f64, f64, f64, f64, f64) = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057);

/// Which fitted batch law picks the batch size when none is given.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BatchLaw {
    /// [`Quality`](BatchLaw::Quality) when `d <= 16`, [`Throughput`](BatchLaw::Throughput)
    /// otherwise.
    #[default]
    Auto,
    /// Large batches: fewer, cheaper-to-parallelize updates.
    Throughput,
    /// Batches about 8x smaller than throughput: more updates per epoch.
    Quality,
}

/// The resolved training plan for one run: batch size, step, and the per-epoch schedule.
#[derive(Clone, Debug, PartialEq)]
pub struct EMeansRecipe {
    /// Rows per update (`B`).
    pub batch_size: usize,
    /// Initial EMA step.
    pub alpha0: f64,
    /// The batch law that chose `batch_size`, or `None` if it was given explicitly.
    pub batch_law: Option<BatchLaw>,
    /// Rows visited in each epoch. A fractional `nepochs` adds a partial epoch before the last.
    pub epoch_rows: Vec<usize>,
    /// Updates in each epoch (the last batch of an epoch may be short).
    pub epoch_steps: Vec<usize>,
    /// Total number of updates.
    pub total_steps: usize,
    /// Revival threshold `tau` (`0` disables revival).
    pub unused_threshold: f32,
}

impl EMeansRecipe {
    /// Resolves the recipe for `n` rows of dimension `d` clustered into `k` centroids over
    /// `nepochs` epochs. `batch_size`, `alpha0` and `unused_threshold` override the laws and
    /// defaults when given.
    #[allow(clippy::too_many_arguments)]
    pub fn resolve(
        n: usize,
        k: usize,
        d: usize,
        nepochs: f64,
        batch_law: BatchLaw,
        batch_size: Option<usize>,
        alpha0: Option<f64>,
        unused_threshold: Option<f32>,
    ) -> Self {
        assert!(n > 0 && k > 0, "e-means needs n > 0 and k > 0");
        assert!(
            nepochs.is_finite() && nepochs > 0.0,
            "nepochs must be positive (got {nepochs})"
        );
        let (batch_size, batch_law) = match batch_size {
            Some(b) => (b.clamp(1, n), None),
            None => {
                let law = match batch_law {
                    BatchLaw::Auto if d <= AUTO_QUALITY_MAX_D => BatchLaw::Quality,
                    BatchLaw::Auto => BatchLaw::Throughput,
                    other => other,
                };
                let (c, bk, bn, bt) = match law {
                    BatchLaw::Quality => QUALITY_LAW,
                    _ => THROUGHPUT_LAW,
                };
                let raw = c * (k as f64).powf(bk) * (n as f64).powf(bn) * nepochs.powf(bt);
                let mut b = round_power_of_two(raw);
                // The two shipped recipes cap differently; reproduce both.
                b = match law {
                    BatchLaw::Throughput => b.min(round_power_of_two(n as f64 / 2.0)),
                    _ => b.min(n / 2),
                };
                (b.clamp(1, n), Some(law))
            }
        };
        let alpha0 = alpha0.unwrap_or_else(|| step_law(n, k, nepochs, batch_size));
        let epoch_rows = epoch_rows(n, nepochs);
        assert!(
            epoch_rows.iter().all(|&r| r > 0),
            "nepochs = {nepochs} rounds to an empty pass over {n} rows"
        );
        let epoch_steps: Vec<usize> = epoch_rows.iter().map(|r| r.div_ceil(batch_size)).collect();
        let total_steps = epoch_steps.iter().sum();
        let unused_threshold = match unused_threshold {
            Some(t) => {
                assert!(
                    t >= 0.0,
                    "unused_threshold must be >= 0 (0 disables revival)"
                );
                t
            }
            None if d <= NO_REVIVAL_MAX_D => 0.0,
            None => UNUSED_THRESHOLD,
        };
        Self {
            batch_size,
            alpha0,
            batch_law,
            epoch_rows,
            epoch_steps,
            total_steps,
            unused_threshold,
        }
    }
}

/// Rounds a positive value to the nearest power of two (ties go up, as in faiss).
fn round_power_of_two(value: f64) -> usize {
    if value <= 1.0 {
        return 1;
    }
    let low = 2f64.powi(value.log2().floor() as i32);
    let high = 2.0 * low;
    (if value / low < high / value {
        low
    } else {
        high
    }) as usize
}

/// Initial EMA step coupled to a batch size by the fitted step law, clamped to `[1e-4, 1]`.
fn step_law(n: usize, k: usize, nepochs: f64, batch_size: usize) -> f64 {
    let (c, ak, ab, an, at) = ALPHA_LAW;
    let alpha0 = c
        * (k as f64).powf(ak)
        * (batch_size as f64).powf(ab)
        * (n as f64).powf(an)
        * nepochs.powf(at);
    alpha0.clamp(1e-4, 1.0)
}

/// Rows per epoch: whole epochs, then a partial epoch for the fractional part (rounded half
/// to even, as Python's `round`), then a final whole epoch whose rows give the closing means.
/// Below one epoch, only the partial prefix is visited.
fn epoch_rows(n: usize, nepochs: f64) -> Vec<usize> {
    let whole = nepochs.floor() as usize;
    let remainder = ((nepochs - whole as f64) * n as f64).round_ties_even() as usize;
    if whole == 0 {
        return vec![remainder];
    }
    let mut rows = vec![n; whole - 1];
    if remainder > 0 {
        rows.push(remainder);
    }
    rows.push(n);
    rows
}

/// Per-epoch permutation seed, derived without ambient RNG state.
fn epoch_seed(seed: u64, epoch: usize) -> u64 {
    ((seed as u128 * 1_000_003 + epoch as u128) % (1u128 << 32)) as u64
}

/// Cosine-anneals the EMA step from `alpha0` towards `0.1 * alpha0`.
fn scheduled_alpha(alpha0: f64, step: usize, total_steps: usize) -> f64 {
    let fraction = step.min(total_steps) as f64 / total_steps.max(1) as f64;
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * fraction).cos());
    alpha0 * (ALPHA_MIN_FACTOR + (1.0 - ALPHA_MIN_FACTOR) * cosine)
}

/// Parameters of the e-means solver ([`Solver::EMeans`](super::Solver::EMeans)). Every option
/// defaults to the faiss `NumpyEmeans` default. The epoch budget is not here: it belongs to
/// [`KMeans`] ([`KMeansBuilder::epochs`](super::KMeansBuilder::epochs)), which defaults to 2
/// epochs for e-means where faiss uses 5.
#[derive(Clone, Debug, PartialEq)]
pub struct EMeansParams {
    /// Fixes the batch size `B` instead of using the batch law. The step law still sets
    /// `alpha0` for this `B` unless `alpha0` is also given.
    pub batch_size: Option<usize>,
    /// Fixes the initial EMA step instead of using the step law.
    pub alpha0: Option<f64>,
    /// The batch law used when no batch size is given (default [`BatchLaw::Auto`]).
    pub batch_law: BatchLaw,
    /// Revival threshold `tau` (`Some(0.0)` disables revival). `None` (default) is 0.1, or 0
    /// when `d <= 2`.
    pub unused_threshold: Option<f32>,
}

impl Default for EMeansParams {
    fn default() -> Self {
        Self {
            batch_size: None,
            alpha0: None,
            batch_law: BatchLaw::Auto,
            unused_threshold: None,
        }
    }
}

impl EMeansParams {
    /// The recipe these parameters resolve to for `n` vectors of dimension `d`, `k` clusters
    /// and an epoch budget of `nepochs` ([`KMeans::epochs`]). Training uses exactly this
    /// recipe.
    pub fn recipe(&self, n: usize, k: usize, d: usize, nepochs: f64) -> EMeansRecipe {
        EMeansRecipe::resolve(
            n,
            k,
            d,
            nepochs,
            self.batch_law,
            self.batch_size,
            self.alpha0,
            self.unused_threshold,
        )
    }
}

/// Runs e-means and returns its centroids after the closing means.
pub(super) fn fit<D, G>(
    run: &Run<'_, D, G>,
    params: &EMeansParams,
    assign: &mut Assign<'_, G>,
    refine: &(impl Fn(&[u32], &mut D::Vector) + Sync),
) -> G
where
    D: MeanDataset + Sync,
    G: DatasetGrowable,
    for<'a> &'a D::Vector: Into<<G::Encoder as VectorEncoder>::InputVector<'a>>,
{
    let dataset = run.dataset;
    let (n, k) = (dataset.len(), run.k);
    let recipe = params.recipe(n, k, dataset.input_dim(), run.epochs);
    let seed = run.seed;

    if run.verbose {
        println!(
            "E-means: {} epochs (batch {}, alpha0 {:.4}, {} updates, law {:?}, tau {}). \
             Please cite Bessa et al., arXiv:2611.09501.",
            run.epochs,
            recipe.batch_size,
            recipe.alpha0,
            recipe.total_steps,
            recipe.batch_law,
            recipe.unused_threshold,
        );
    }

    let mut centroids = run.initial_centroids(&mut StdRng::seed_from_u64(seed.wrapping_add(1)));
    let mut usage = vec![INIT_USAGE; k];
    let mut split_rng = StdRng::seed_from_u64(seed);
    let mut order: Vec<u32> = (0..n as u32).collect();
    // Cluster of every vector in the last epoch, for the closing means.
    let mut last_assignment = vec![u32::MAX; n];
    let mut pass_counts = vec![0f64; k];
    let mut n_splits = 0usize;
    let mut revived_last = false;
    let mut step = 0usize;
    let n_passes = recipe.epoch_rows.len();
    let start = Instant::now();

    for (epoch, &rows) in recipe.epoch_rows.iter().enumerate() {
        let last = epoch + 1 == n_passes;
        // Reset to the identity so each epoch's order depends only on (seed, epoch).
        order
            .iter_mut()
            .enumerate()
            .for_each(|(i, o)| *o = i as u32);
        order.shuffle(&mut StdRng::seed_from_u64(epoch_seed(seed, epoch)));
        pass_counts.iter_mut().for_each(|c| *c = 0.0);
        let mut objective = 0f64;
        let splits_before = n_splits;
        let mut assign_time = Duration::ZERO;
        let mut update_time = Duration::ZERO;

        for (b, batch) in order[..rows].chunks(recipe.batch_size).enumerate() {
            let ctx = UpdateCtx {
                update: step,
                epoch,
                epoch_start: b == 0,
                centroids_jumped: revived_last,
            };
            let t0 = Instant::now();
            // Assign against a snapshot of the centroids (their unit directions in spherical
            // mode; the moving average itself runs on unnormalized positions).
            let current: G = if run.spherical {
                let mut unit = centroids.clone();
                run.normalize_if_spherical(&mut unit);
                run.to_dataset(&unit)
            } else {
                run.to_dataset(&centroids)
            };
            let assignments = assign(&current, ctx, Some(batch));
            drop(current);
            assign_time += t0.elapsed();

            let t0 = Instant::now();
            objective += assignments.iter().map(|a| a.0 as f64).sum::<f64>();
            if last {
                for (&i, &(_, c)) in batch.iter().zip(&assignments) {
                    last_assignment[i as usize] = c as u32;
                }
            }
            let alpha = scheduled_alpha(recipe.alpha0, step, recipe.total_steps);
            apply_batch(
                dataset,
                run.weights,
                batch,
                &assignments,
                alpha,
                &mut centroids,
                &mut usage,
                &mut pass_counts,
                refine,
            );
            revived_last = false;
            if (step + 1).is_multiple_of(CHECK_UNUSED_EVERY) {
                let splits = revive_unused(
                    &mut centroids,
                    &mut usage,
                    recipe.unused_threshold,
                    &mut split_rng,
                );
                n_splits += splits;
                revived_last = splits > 0;
            }
            update_time += t0.elapsed();
            step += 1;
        }

        if run.verbose {
            println!(
                "  Epoch {}/{n_passes} ({rows} rows, {} updates, {:.2?}): objective={objective:.6e} \
                 imbalance={:.4} splits={} centroid nnz={} assign={assign_time:.2?} \
                 update={update_time:.2?}",
                epoch + 1,
                recipe.epoch_steps[epoch],
                start.elapsed(),
                imbalance_factor(pass_counts.iter().copied()),
                n_splits - splits_before,
                run.to_dataset(&centroids).nnz(),
            );
        }
    }

    // Closing means (Algorithm 1, line 12): centroids that received vectors in the last
    // epoch become the exact mean of those vectors; the others keep their EMA position.
    let (visited, visited_assignments): (Vec<u32>, Vec<(f32, usize)>) = last_assignment
        .iter()
        .enumerate()
        .filter(|(_, c)| **c != u32::MAX)
        .map(|(i, &c)| (i as u32, (0.0, c as usize)))
        .unzip();
    let groups = ClusterGroups::new(Some(&visited), &visited_assignments, k);
    centroids
        .par_iter_mut()
        .zip(run.means(&groups).into_par_iter())
        .enumerate()
        .for_each(|(c, (centroid, mean))| {
            if let Some((mean, count)) = mean
                && count > 0.0
            {
                *centroid = mean;
                refine(groups.members(c), centroid);
            }
        });
    run.normalize_if_spherical(&mut centroids);
    let centroids = run.to_dataset(&centroids);
    if run.verbose {
        println!("  Closing means: centroid nnz={}", centroids.nnz());
    }
    centroids
}

/// Applies one batch's EMA update (Algorithm 1, line 8) to `centroids` and `usage`, refines
/// every moved centroid with its batch members, and adds the batch's counts to the epoch totals.
///
/// At `alpha >= 1` the update replaces: a touched cluster's centroid becomes its batch mean and
/// its mass the batch count, and untouched clusters keep their mass (faiss `replace_usage`).
#[allow(clippy::too_many_arguments)]
fn apply_batch<D>(
    dataset: &D,
    weights: Option<&[f32]>,
    batch: &[u32],
    assignments: &[(f32, usize)],
    alpha: f64,
    centroids: &mut [D::Vector],
    usage: &mut [f32],
    pass_counts: &mut [f64],
    refine: &(impl Fn(&[u32], &mut D::Vector) + Sync),
) where
    D: MeanDataset + Sync,
{
    let groups = ClusterGroups::new(Some(batch), assignments, centroids.len());
    let replace = alpha >= 1.0;
    let alpha = alpha as f32;

    // One parallel pass computes each cluster's mean and applies its update: mini-batches can
    // be small (hundreds of rows), so a separate pass for the means costs a measurable
    // share of every update.
    let counts: Vec<f32> = centroids
        .par_iter_mut()
        .zip(usage.par_iter_mut())
        .enumerate()
        .map(|(c, (centroid, mass))| {
            let members = groups.members(c);
            let count: f32 = members
                .iter()
                .map(|&i| weights.map_or(1.0, |w| w[i as usize]))
                .sum();
            if members.is_empty() || count <= 0.0 {
                if !replace {
                    *mass *= 1.0 - alpha;
                }
                return 0.0;
            }
            let mean = dataset.mean(members, weights);
            *mass = if replace { 0.0 } else { *mass * (1.0 - alpha) } + alpha * count;
            let weight = alpha * count / *mass;
            if weight >= 1.0 {
                *centroid = mean;
            } else {
                dataset.lerp(centroid, &mean, weight);
            }
            refine(members, centroid);
            count
        })
        .collect();
    for (p, c) in pass_counts.iter_mut().zip(counts) {
        *p += c as f64;
    }
}

/// Revives clusters whose mass fell below `tau * mean(mass)` by splitting donors sampled
/// without replacement in proportion to their mass (Gumbel top-k). Donor and target each
/// take half the donor's mass and a copy of its centroid perturbed in opposite directions.
/// Returns the number of splits.
fn revive_unused<V: AsMut<[f32]> + Clone>(
    centroids: &mut [V],
    usage: &mut [f32],
    tau: f32,
    rng: &mut StdRng,
) -> usize {
    if tau <= 0.0 {
        return 0;
    }
    let k = usage.len();
    let threshold = usage.iter().sum::<f32>() * tau / k as f32;
    let is_donor = |u: f32| u >= threshold && u > 0.0;
    let unused: Vec<usize> = (0..k).filter(|&j| usage[j] < threshold).collect();
    let n_donors = usage.iter().filter(|&&u| is_donor(u)).count();
    let n_split = unused.len().min(n_donors);
    if n_split == 0 {
        return 0;
    }
    // One uniform per cluster, drawn in cluster order, so the stream does not depend on
    // which clusters are donors.
    let mut keyed: Vec<(f64, usize)> = (0..k)
        .map(|j| {
            let u: f64 = rng.r#gen::<f64>().max(f64::MIN_POSITIVE);
            let key = if is_donor(usage[j]) {
                (usage[j] as f64).ln() - (-u.ln()).ln()
            } else {
                f64::NEG_INFINITY
            };
            (key, j)
        })
        .collect();
    keyed.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));

    for (&(_, src), &dst) in keyed.iter().take(n_split).zip(&unused) {
        let mut child = centroids[src].clone();
        let donor = centroids[src].as_mut();
        let child_values = child.as_mut();
        let mut identical = true;
        for (x, (dn, ch)) in donor.iter_mut().zip(child_values.iter_mut()).enumerate() {
            let offset = if x % 2 == 0 {
                SPLIT_EPSILON
            } else {
                -SPLIT_EPSILON
            };
            *dn *= 1.0 - offset;
            *ch *= 1.0 + offset;
            identical &= *dn == *ch;
        }
        // Zero (or tiny) donors stay identical after scaling; separate them along one axis.
        if identical && let Some(first) = child_values.first_mut() {
            let up = first.next_up();
            *first = if up.is_finite() {
                up
            } else {
                first.next_down()
            };
        }
        centroids[dst] = child;
        let half = usage[src] / 2.0;
        usage[src] = half;
        usage[dst] = half;
    }
    n_split
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dataset;
    use crate::clustering::{KMeans, KMeansBuilder, Solver, sparse_centroid_pruning};
    use crate::core::flat_index::FlatIndex;
    use crate::core::vector::{DenseVectorOwned, DenseVectorView, SparseVectorView};
    use crate::encoders::sparse_scalar::PlainSparseQuantizer;
    use crate::{
        DotProduct, PlainDenseDataset, PlainDenseDatasetGrowable, PlainDenseQuantizer,
        PlainSparseDatasetGrowable, SquaredEuclideanDistance,
    };

    type Plain = PlainDenseDataset<f32, SquaredEuclideanDistance>;
    type Grow = PlainDenseDatasetGrowable<f32, SquaredEuclideanDistance>;

    fn enc(d: usize) -> PlainDenseQuantizer<f32, SquaredEuclideanDistance> {
        PlainDenseQuantizer::new(d)
    }

    /// A k-means builder with the e-means solver and these parameters.
    /// A k-means builder with this e-means solver and the 5-epoch budget these tests were
    /// written for (faiss's default; vectorium's is 2).
    fn emeans_with(params: EMeansParams) -> KMeansBuilder {
        KMeansBuilder::new()
            .solver(Solver::EMeans(params))
            .epochs(5.0)
    }

    /// A k-means builder with the default e-means solver.
    fn emeans() -> KMeansBuilder {
        emeans_with(EMeansParams::default())
    }

    // ---- recipe: values hand-derived from faiss `base.py` ---------------------

    #[test]
    fn recipe_matches_faiss_readme_example() {
        // README: NumpyEmeans(d=64, k=500) on n=200_000 with faiss's default 5 epochs.
        // raw B = 0.035134 * 500^0.1347 * 200000^0.8765 * 5^0.8133 ~= 13_332 -> 16_384;
        // the step law gives ~1.014, clamped to 1.
        let r = EMeansRecipe::resolve(200_000, 500, 64, 5.0, BatchLaw::Auto, None, None, None);
        assert_eq!(r.batch_size, 16_384);
        assert_eq!(r.batch_law, Some(BatchLaw::Throughput));
        assert_eq!(r.alpha0, 1.0);
        assert_eq!(r.epoch_rows, vec![200_000; 5]);
        assert_eq!(r.epoch_steps, vec![13; 5]);
        assert_eq!(r.total_steps, 65);
        assert_eq!(r.unused_threshold, 0.1);
    }

    #[test]
    fn recipe_quality_law_at_low_dimension() {
        // d <= 16 picks the quality law: raw B ~= 13_332 / 8 ~= 1_666 -> 2_048, and the
        // step law at B = 2048 gives exp(-1.858) ~= 0.156.
        let r = EMeansRecipe::resolve(200_000, 500, 8, 5.0, BatchLaw::Auto, None, None, None);
        assert_eq!(r.batch_law, Some(BatchLaw::Quality));
        assert_eq!(r.batch_size, 2_048);
        assert!(
            (r.alpha0 - 0.156).abs() / 0.156 < 0.01,
            "alpha0 = {}",
            r.alpha0
        );
        // An explicit law overrides the dimension rule.
        let t = EMeansRecipe::resolve(200_000, 500, 8, 5.0, BatchLaw::Throughput, None, None, None);
        assert_eq!(t.batch_size, 16_384);
    }

    #[test]
    fn recipe_overrides_and_caps() {
        // A given batch keeps the step law; a given alpha0 is used verbatim.
        let r = EMeansRecipe::resolve(10_000, 16, 32, 3.0, BatchLaw::Auto, Some(1_000), None, None);
        assert_eq!(r.batch_size, 1_000);
        assert_eq!(r.batch_law, None);
        assert_eq!(r.alpha0, step_law(10_000, 16, 3.0, 1_000));
        let r = EMeansRecipe::resolve(
            10_000,
            16,
            32,
            3.0,
            BatchLaw::Auto,
            Some(50_000),
            Some(0.3),
            None,
        );
        assert_eq!(r.batch_size, 10_000, "batch is capped at n");
        assert_eq!(r.alpha0, 0.3);
        // Tiny n: the throughput batch is capped at round_pow2(n / 2), quality at n / 2.
        let r = EMeansRecipe::resolve(100, 4, 32, 5.0, BatchLaw::Throughput, None, None, None);
        assert!(r.batch_size <= 64);
        let r = EMeansRecipe::resolve(100, 4, 8, 5.0, BatchLaw::Quality, None, None, None);
        assert!(r.batch_size <= 50);
        // Revival is off at d <= 2 unless requested.
        assert_eq!(
            EMeansRecipe::resolve(1000, 4, 2, 1.0, BatchLaw::Auto, None, None, None)
                .unused_threshold,
            0.0
        );
        assert_eq!(
            EMeansRecipe::resolve(1000, 4, 2, 1.0, BatchLaw::Auto, None, None, Some(0.2))
                .unused_threshold,
            0.2
        );
    }

    #[test]
    fn fractional_epochs_follow_faiss_layout() {
        assert_eq!(epoch_rows(1000, 2.5), vec![1000, 500, 1000]);
        assert_eq!(epoch_rows(1000, 0.5), vec![500]);
        assert_eq!(epoch_rows(300, 1.3), vec![90, 300]);
        assert_eq!(epoch_rows(1000, 3.0), vec![1000, 1000, 1000]);
        // Python's round(): ties to even.
        assert_eq!(epoch_rows(5, 1.5), vec![2, 5]);
        assert_eq!(epoch_rows(7, 1.5), vec![4, 7]);
    }

    #[test]
    fn schedule_and_helpers() {
        assert_eq!(scheduled_alpha(0.7, 0, 13), 0.7);
        assert!((scheduled_alpha(0.7, 13, 13) - 0.07).abs() < 1e-12);
        assert!(
            (scheduled_alpha(0.7, 50, 10) - 0.07).abs() < 1e-12,
            "clamped past the end"
        );
        assert!((scheduled_alpha(1.0, 5, 10) - 0.55).abs() < 1e-12);
        assert_eq!(round_power_of_two(0.3), 1);
        assert_eq!(round_power_of_two(3.0), 4);
        assert_eq!(round_power_of_two(2.8), 2);
        // Nearest by ratio, so the 1024/2048 boundary is the geometric mean ~1448.
        assert_eq!(round_power_of_two(1400.0), 1024);
        assert_eq!(round_power_of_two(1500.0), 2048);
        // seed * 1_000_003 + epoch, modulo 2^32: 1_234_003_705 fits; 5_000_015_001 wraps.
        assert_eq!(epoch_seed(1234, 3), 1_234_003_705);
        assert_eq!(epoch_seed(5000, 1), 5_000_015_001 - (1 << 32));
        assert!(epoch_seed(u64::MAX, 7) < (1 << 32));
    }

    // ---- revival ---------------------------------------------------------------

    fn rows(values: &[f32], d: usize) -> Vec<DenseVectorOwned<f32>> {
        values
            .chunks_exact(d)
            .map(|r| DenseVectorOwned::new(r.to_vec()))
            .collect()
    }

    #[test]
    fn revive_splits_heaviest_into_dead_clusters() {
        let mut centroids = rows(
            &[
                1.0, 2.0, 3.0, 4.0, // heavy donor
                -1.0, -2.0, -3.0, -4.0, // heavy donor
                9.0, 9.0, 9.0, 9.0, // dead
                0.0, 0.0, 0.0, 0.0, // dead
            ],
            4,
        );
        let mut usage = vec![10.0, 10.0, 0.01, 0.0];
        let mut rng = StdRng::seed_from_u64(0);
        let n = revive_unused(&mut centroids, &mut usage, 0.1, &mut rng);
        assert_eq!(n, 2);
        assert_eq!(usage, vec![5.0, 5.0, 5.0, 5.0]);
        for dst in [2usize, 3] {
            let row = centroids[dst].values();
            // Each target is a perturbed copy of one of the donors.
            let near = |c: &[f32]| {
                row.iter()
                    .zip(c)
                    .all(|(a, b)| (a - b).abs() <= 1e-3 * b.abs().max(1.0))
            };
            assert!(near(&[1.0, 2.0, 3.0, 4.0]) || near(&[-1.0, -2.0, -3.0, -4.0]));
        }
        // Donor and child moved apart in opposite directions.
        assert_ne!(centroids[0].values(), centroids[2].values());
        // Nothing to do when revival is off or no cluster is starved.
        assert_eq!(revive_unused(&mut centroids, &mut usage, 0.0, &mut rng), 0);
        assert_eq!(revive_unused(&mut centroids, &mut usage, 0.1, &mut rng), 0);
    }

    #[test]
    fn revive_separates_zero_donor() {
        let mut centroids = rows(&[0.0; 4], 2);
        let mut usage = vec![1.0, 0.0];
        let mut rng = StdRng::seed_from_u64(1);
        assert_eq!(revive_unused(&mut centroids, &mut usage, 0.1, &mut rng), 1);
        assert_ne!(centroids[0].values(), centroids[1].values());
    }

    // ---- the EMA update --------------------------------------------------------

    #[test]
    fn replacing_update_leaves_untouched_mass_alone() {
        // At alpha = 1 the update replaces: a touched cluster takes its batch mean, and its
        // batch count as mass; an untouched cluster keeps its mass (faiss `replace_usage`).
        let ds = Plain::from_raw(
            vec![0.0, 0.0, 2.0, 0.0, 9.0, 9.0].into_boxed_slice(),
            3,
            enc(2),
        );
        let noop = |_: &[u32], _: &mut DenseVectorOwned<f32>| {};
        let batch = [1, 0];
        let assignments = [(0.0, 0), (0.0, 0)];
        let mut centroids = rows(&[1.0, 1.0, 5.0, 5.0], 2);
        let mut usage = vec![3.0, 7.0];
        let mut pass = vec![0.0; 2];
        apply_batch(
            &ds,
            None,
            &batch,
            &assignments,
            1.0,
            &mut centroids,
            &mut usage,
            &mut pass,
            &noop,
        );
        assert_eq!(usage, vec![2.0, 7.0]);
        assert_eq!(centroids[0].values(), &[1.0, 0.0]);
        assert_eq!(centroids[1].values(), &[5.0, 5.0]);
        assert_eq!(pass, vec![2.0, 0.0]);

        // Below 1 every mass decays, and the touched centroid moves part of the way.
        let mut centroids = rows(&[1.0, 1.0, 5.0, 5.0], 2);
        let mut usage = vec![3.0, 8.0];
        apply_batch(
            &ds,
            None,
            &batch,
            &assignments,
            0.5,
            &mut centroids,
            &mut usage,
            &mut pass,
            &noop,
        );
        // Mass 0.5 * 3 + 0.5 * 2 = 2.5; weight 0.5 * 2 / 2.5 = 0.4 towards the mean (1, 0).
        assert_eq!(usage, vec![2.5, 4.0]);
        let c = centroids[0].values();
        assert!(
            (c[0] - 1.0).abs() < 1e-6 && (c[1] - 0.6).abs() < 1e-6,
            "{c:?}"
        );
    }

    // ---- end-to-end ------------------------------------------------------------

    /// `n_blobs` tight Gaussian blobs in `d` dimensions, centers far apart. Returns the
    /// dataset and each point's blob id.
    fn blobs(n_blobs: usize, per_blob: usize, d: usize, seed: u64) -> (Plain, Vec<usize>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let centers: Vec<f32> = (0..n_blobs * d)
            .map(|_| rng.gen_range(-100.0..100.0))
            .collect();
        let mut values = Vec::with_capacity(n_blobs * per_blob * d);
        let mut labels = Vec::new();
        for i in 0..n_blobs * per_blob {
            let b = i % n_blobs;
            labels.push(b);
            for x in 0..d {
                values.push(centers[b * d + x] + rng.gen_range(-1.0..1.0));
            }
        }
        let n = labels.len();
        (
            Plain::from_raw(values.into_boxed_slice(), n, enc(d)),
            labels,
        )
    }

    fn labels_of(ds: &Plain, c: &Grow) -> Vec<usize> {
        KMeans::compute_assignments(ds, c)
            .into_iter()
            .map(|(_, j)| j)
            .collect()
    }

    fn objective(ds: &Plain, c: &Grow) -> f64 {
        KMeans::compute_assignments(ds, c)
            .into_iter()
            .map(|(v, _)| v as f64)
            .sum()
    }

    /// The found partition equals the blob partition (up to relabeling).
    fn assert_recovers(found: &[usize], truth: &[usize], k: usize) {
        let mut map = vec![usize::MAX; k];
        for (&f, &t) in found.iter().zip(truth) {
            if map[t] == usize::MAX {
                map[t] = f;
            }
            assert_eq!(map[t], f, "blob {t} split across clusters");
        }
        let distinct: std::collections::HashSet<_> = map.iter().collect();
        assert_eq!(distinct.len(), k, "two blobs merged into one cluster");
    }

    #[test]
    fn train_recovers_blobs() {
        let (ds, truth) = blobs(8, 500, 16, 7);
        let em = emeans().seed(Some(3)).build();
        let c: Grow = em.train(&ds, 8, None, enc(16));
        assert_eq!(c.len(), 8);
        assert_recovers(&labels_of(&ds, &c), &truth, 8);
    }

    #[test]
    fn train_is_deterministic_for_a_seed_and_across_thread_counts() {
        let (ds, _) = blobs(5, 400, 24, 11);
        let em = emeans_with(EMeansParams::default())
            .epochs(2.5)
            .seed(Some(42))
            .build();
        let run = |threads: usize| -> Grow {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| em.train(&ds, 12, None, enc(24)))
        };
        let a = run(1);
        assert_eq!(a.values(), run(1).values());
        assert_eq!(a.values(), run(4).values());
    }

    #[test]
    fn n_equal_k_returns_the_data() {
        let (ds, _) = blobs(2, 3, 4, 1);
        let c: Grow = emeans().build().train(&ds, 6, None, enc(4));
        assert_eq!(c.values(), ds.values());
    }

    #[test]
    fn spherical_dot_product_recovers_directions() {
        // Blobs around 4 well-separated directions at varied radii: spherical e-means under
        // the dot product (the metric of the centroid encoder) must return unit centroids and
        // find the directions, through both the flat and the index path.
        let d = 16;
        let mut rng = StdRng::seed_from_u64(4);
        let dirs: Vec<Vec<f32>> = (0..4)
            .map(|b| (0..d).map(|x| if x / 4 == b { 1.0 } else { 0.0 }).collect())
            .collect();
        let mut values = Vec::new();
        let mut truth = Vec::new();
        for i in 0..1_200 {
            let b = i % 4;
            truth.push(b);
            let r = rng.gen_range(1.0f32..5.0);
            values.extend(
                dirs[b]
                    .iter()
                    .map(|&x| r * x + rng.gen_range(-0.05f32..0.05)),
            );
        }
        let ds = PlainDenseDataset::<f32, DotProduct>::from_raw(
            values.into_boxed_slice(),
            truth.len(),
            PlainDenseQuantizer::new(d),
        );
        let em = emeans().seed(Some(2)).spherical(true).build();
        let c: PlainDenseDatasetGrowable<f32, DotProduct> =
            em.train(&ds, 4, None, PlainDenseQuantizer::new(d));
        for row in c.values().chunks_exact(d) {
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "spherical centroid norm {norm}");
        }
        let found: Vec<usize> = KMeans::compute_assignments(&ds, &c)
            .into_iter()
            .map(|a| a.1)
            .collect();
        assert_recovers(&found, &truth, 4);
        let c: PlainDenseDatasetGrowable<f32, DotProduct> = em.train_with_index(
            &ds,
            4,
            None,
            PlainDenseQuantizer::new(d),
            FlatIndex::from,
            &(),
        );
        let found: Vec<usize> = KMeans::compute_assignments(&ds, &c)
            .into_iter()
            .map(|a| a.1)
            .collect();
        assert_recovers(&found, &truth, 4);
    }

    #[test]
    fn f16_training_and_centroids() {
        // Half-precision training storage, and f16 centroids, must cluster as well as f32
        // (objective within 1% of the f32 run, same seed), measured on the f32 data.
        use crate::encoders::dense_scalar::ScalarDenseQuantizer;
        use half::f16;
        let mut rng = StdRng::seed_from_u64(13);
        let (n, d, k) = (6_000, 16, 24);
        let centers: Vec<f32> = (0..30 * d).map(|_| rng.gen_range(-3.0..3.0)).collect();
        let values: Vec<f32> = (0..n * d)
            .map(|i| centers[((i / d) % 30) * d + i % d] + rng.gen_range(-1.0f32..1.0))
            .collect();
        let half_values: Vec<f16> = values.iter().map(|&x| f16::from_f32(x)).collect();
        let ds = Plain::from_raw(values.into_boxed_slice(), n, enc(d));
        let ds16 = PlainDenseDataset::<f16, SquaredEuclideanDistance>::from_raw(
            half_values.into_boxed_slice(),
            n,
            PlainDenseQuantizer::new(d),
        );
        let em = emeans().seed(Some(5)).build();
        let base = objective(&ds, &em.train(&ds, k, None, enc(d)));
        let c32: Grow = em.train(&ds16, k, None, enc(d));
        let o32 = objective(&ds, &c32);
        assert!(o32 <= 1.01 * base, "f16 training {o32} vs f32 {base}");
        let c16: crate::DenseDatasetGrowable<
            ScalarDenseQuantizer<f32, f16, SquaredEuclideanDistance>,
        > = em.train(&ds16, k, None, ScalarDenseQuantizer::new(d));
        let as_f32: Vec<f32> = c16.values().iter().map(|x| x.to_f32()).collect();
        let c16 = Grow::from_raw(as_f32, k, enc(d));
        let o16 = objective(&ds, &c16);
        assert!(o16 <= 1.01 * base, "f16 centroids {o16} vs f32 {base}");
    }

    #[test]
    fn heavy_weight_pulls_a_centroid() {
        // Three points; with k = 1 the centroid is their weighted mean.
        let ds = Plain::from_raw(
            vec![0.0, 0.0, 10.0, 0.0, 0.0, 1.0].into_boxed_slice(),
            3,
            enc(2),
        );
        let em = emeans().seed(Some(0)).build();
        let c: Grow = em.train(&ds, 1, Some(vec![1.0, 8.0, 1.0]), enc(2));
        let v = c.values();
        assert!(
            (v[0] - 8.0).abs() < 1e-4 && (v[1] - 0.1).abs() < 1e-4,
            "{v:?}"
        );
    }

    #[test]
    fn refine_sees_members_and_shapes_the_result() {
        // A refine that zeroes the last coordinate must hold in every returned centroid, and
        // its members must be valid, ascending training ids.
        let (ds, _) = blobs(4, 200, 6, 8);
        let n = ds.len() as u32;
        let em = emeans().seed(Some(1)).build();
        let c: Grow = em.train_refined(&ds, 4, None, enc(6), |members, c| {
            assert!(members.windows(2).all(|w| w[0] < w[1]) && members.iter().all(|&i| i < n));
            *c.as_mut().last_mut().unwrap() = 0.0;
        });
        assert!(c.values().chunks_exact(6).all(|r| r[5] == 0.0));
    }

    /// E-means at 5 epochs (faiss's default) should land close to (typically below) Lloyd's
    /// objective at 10 iterations on overlapping clusters.
    #[test]
    fn objective_is_competitive_with_lloyd() {
        let mut rng = StdRng::seed_from_u64(123);
        let (n, d, k) = (20_000, 32, 64);
        let centers: Vec<f32> = (0..50 * d).map(|_| rng.gen_range(-3.0..3.0)).collect();
        let values: Vec<f32> = (0..n * d)
            .map(|i| centers[((i / d) % 50) * d + i % d] + rng.gen_range(-1.0f32..1.0))
            .collect();
        let ds = Plain::from_raw(values.into_boxed_slice(), n, enc(d));
        let lloyd: Grow = KMeansBuilder::new()
            .epochs(10.0)
            .seed(Some(1))
            .build()
            .train(&ds, k, None, enc(d));
        let em: Grow = emeans().seed(Some(1)).build().train(&ds, k, None, enc(d));
        let (ol, oe) = (objective(&ds, &lloyd), objective(&ds, &em));
        assert!(oe <= 1.03 * ol, "e-means objective {oe} vs Lloyd {ol}");
    }

    #[test]
    fn sparse_smoke_with_pruning() {
        // Two disjoint vocabularies; e-means over sparse vectors must separate them.
        let mut rng = StdRng::seed_from_u64(3);
        let mut growable =
            PlainSparseDatasetGrowable::new(PlainSparseQuantizer::<u16, f32, DotProduct>::new(
                64, 64,
            ));
        let mut truth = Vec::new();
        for i in 0..400 {
            let b = i % 2;
            truth.push(b);
            let mut comps: Vec<u16> = (0..6)
                .map(|_| (b * 32 + rng.gen_range(0..32)) as u16)
                .collect();
            comps.sort_unstable();
            comps.dedup();
            let vals: Vec<f32> = comps.iter().map(|_| rng.gen_range(0.5f32..1.0)).collect();
            growable.push(SparseVectorView::new(&comps, &vals));
        }
        let ds: crate::PlainSparseDataset<u16, f32, DotProduct> = growable.into();
        let em = emeans().seed(Some(1)).spherical(true).build();
        let c: PlainSparseDatasetGrowable<u16, f32, DotProduct> =
            em.train(&ds, 2, None, PlainSparseQuantizer::new(64, 64));
        let found: Vec<usize> = KMeans::compute_assignments(&ds, &c)
            .into_iter()
            .map(|a| a.1)
            .collect();
        assert_recovers(&found, &truth, 2);
        // Pruning bounds every centroid's nnz. (Recovery is not asserted under pruning: on
        // this fixture documents that miss every kept component score 0 against both
        // centroids, and k-means fails the same way.)
        let c: PlainSparseDatasetGrowable<u16, f32, DotProduct> = em.train_refined(
            &ds,
            2,
            None,
            PlainSparseQuantizer::new(64, 64),
            sparse_centroid_pruning(&ds, 2.0),
        );
        assert!(
            c.iter().all(|v| v.len() <= 12),
            "pruned centroids stay sparse"
        );
    }

    // ---- callers that take a KMeans ---------------------------------------------------

    #[test]
    fn pq_trains_with_emeans() {
        // Product quantization through `train_with_kmeans` with e-means must reconstruct about as well
        // as the default k-means codebooks (e-means usually reconstructs slightly better).
        use crate::core::vector_encoder::DenseVectorEncoder;
        use crate::encoders::pq::ProductQuantizer;
        let mut rng = StdRng::seed_from_u64(5);
        let (n, d) = (3_000, 16);
        let centers: Vec<f32> = (0..40 * d).map(|_| rng.gen_range(-3.0..3.0)).collect();
        let values: Vec<f32> = (0..n * d)
            .map(|i| centers[((i / d) % 40) * d + i % d] + rng.gen_range(-1.0f32..1.0))
            .collect();
        let ds = Plain::from_raw(values.into_boxed_slice(), n, enc(d));
        let recon_mse = |pq: &ProductQuantizer<4, SquaredEuclideanDistance>| -> f64 {
            let mut codes = Vec::new();
            ds.iter()
                .map(|v| {
                    codes.clear();
                    pq.push_encoded(v, &mut codes);
                    let r = pq.decode_vector(DenseVectorView::new(&codes));
                    v.values()
                        .iter()
                        .zip(r.values())
                        .map(|(a, b)| ((a - b) * (a - b)) as f64)
                        .sum::<f64>()
                })
                .sum::<f64>()
                / n as f64
        };
        let em = emeans().seed(Some(2)).build();
        let pq_em = ProductQuantizer::<4, SquaredEuclideanDistance>::train_with_kmeans(&ds, &em);
        let pq_km = ProductQuantizer::<4, SquaredEuclideanDistance>::train_with_kmeans_options(
            &ds,
            25,
            Some(2),
        );
        let (e, k) = (recon_mse(&pq_em), recon_mse(&pq_km));
        assert!(
            e <= 1.05 * k,
            "e-means PQ reconstruction {e} vs k-means {k}"
        );
    }
}
