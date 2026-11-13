use super::emeans::{self, EMeansParams};
use super::index_assign::{CentroidIndexBuilder, IndexRefresh, IndexedAssigner, UpdateCtx};
use super::lloyd::{self, LloydParams};
use crate::Dataset;
use crate::core::dataset::{DatasetGrowable, MeanDataset};
use crate::core::distances::Distance;
use crate::core::index::Index;
use crate::core::vector::{SparseVectorOwned, SparseVectorView};
use crate::core::vector_encoder::{
    QueryEvaluator, SparseDataEncoder, SparseVectorEncoder, VectorEncoder,
};
use crate::{ComponentType, VectorId};
use rand::Rng;
use rand::rngs::StdRng;
use rand::seq::index;
use rayon::prelude::*;

/// K-means over any [`MeanDataset`] (dense or sparse, any encoder).
///
/// The representation and the clustering metric are independent:
/// - the **training dataset** `D` only has to be averageable ([`MeanDataset`]); its own metric is
///   ignored;
/// - the **centroids** are written into a growable dataset `G` whose encoder the caller passes, and
///   that encoder fixes their storage precision;
/// - the **clustering metric** is the distance of the centroid index `Q` used for assignment
///   (e.g. a [`FlatIndex`](crate::FlatIndex) over `G`).
///
/// Means always accumulate in `f32`, and assignment queries are the training vectors decoded to
/// `f32`.
///
/// # Solvers
///
/// `KMeans` is the k-means problem (choose `k` centroids minimising the distance of every
/// vector to its nearest one) together with everything every way of solving it shares:
/// initialisation from `k` distinct training vectors, flat or index-accelerated assignment,
/// weights, the `refine` hook, spherical mode, seeding and the conversion to `G`. How the
/// centroids are fitted is the [`Solver`], chosen with [`KMeansBuilder::solver`]:
/// - [`Solver::Lloyd`] (the default): Lloyd's algorithm, a full pass and an exact mean update
///   per iteration ([`LloydParams`]).
/// - [`Solver::EMeans`]: e-means, mini-batch updates with an exponential moving average and
///   fitted scaling laws, ported from faiss ([`EMeansParams`]).
///
/// # Precision of dense centroids
///
/// `T = f16` training storage halves the corpus footprint at equal clustering quality. For
/// centroids stored as `C`, it depends on how the binary is compiled: with hardware f16
/// conversion enabled (e.g. `-C target-cpu=native` on x86 with F16C), an f16 centroid index is
/// both smaller and faster than f32; without it the f16 -> f32 conversion falls back to slow
/// scalar code, so portable builds should keep `C = f32`.
///
/// Means are written into `C` via [`FromF32::from_f32_saturating`](crate::FromF32::from_f32_saturating). For `f16`/`bf16` that is IEEE
/// `from_f32` rounding, so magnitudes above the type's finite max become `±inf` rather than
/// clamping; callers must keep stored means inside the finite range of `C`. `FixedU8`/`FixedU16`
/// are *unsigned*: every negative component clamps to `0`, so they destroy centred data.
#[derive(Clone, Debug)]
pub struct KMeans {
    solver: Solver,
    verbose: bool,
    /// Optional fixed seed for the RNG.  `None` = drawn from entropy (non-deterministic).
    /// Setting a seed makes clustering fully reproducible across runs and thread counts.
    seed: Option<u64>,
    /// Spherical k-means: assign against, and return, unit-norm centroids.
    spherical: bool,
    /// Centroid-index maintenance for [`train_with_index`](Self::train_with_index).
    index_refresh: IndexRefresh,
}

/// How [`KMeans`] fits its centroids. Both solvers optimise the same objective; they differ in
/// the row schedule, the update rule, the handling of empty clusters and which centroids they
/// return.
#[derive(Clone, Debug, PartialEq)]
pub enum Solver {
    /// Lloyd's algorithm: every iteration assigns all vectors, then replaces each centroid by
    /// the mean of its members and splits empty clusters.
    Lloyd(LloydParams),
    /// E-means (Bessa et al., arXiv:2611.09501, ported from faiss `contrib/e_means`):
    /// mini-batches over permuted epochs, an exponential-moving-average update, mass-based
    /// revival of starved clusters and exact closing means.
    EMeans(EMeansParams),
}

impl Default for Solver {
    fn default() -> Self {
        Solver::Lloyd(LloydParams::default())
    }
}

/// Assigns training vectors to centroids: called with the current centroids, the update's
/// position in training, and the ids of the vectors to assign (all of them when `None`).
/// Returns `(distance, centroid id)` per vector, in `ids` order.
pub(super) type Assign<'a, G> = dyn FnMut(&G, UpdateCtx, Option<&[u32]>) -> Vec<(f32, usize)> + 'a;

/// One training run, as the solvers see it: the data, the target `k` and the shared settings.
pub(super) struct Run<'a, D: MeanDataset, G: DatasetGrowable> {
    pub dataset: &'a D,
    pub k: usize,
    pub weights: Option<&'a [f32]>,
    pub encoder: &'a G::Encoder,
    /// The resolved seed: the configured one, or one drawn from entropy.
    pub seed: u64,
    pub spherical: bool,
    pub verbose: bool,
}

impl<D, G> Run<'_, D, G>
where
    D: MeanDataset + Sync,
    G: DatasetGrowable,
    for<'a> &'a D::Vector: Into<<G::Encoder as VectorEncoder>::InputVector<'a>>,
{
    /// `k` distinct training vectors drawn with `rng`, normalized in spherical mode.
    pub fn initial_centroids(&self, rng: &mut StdRng) -> Vec<D::Vector> {
        let mut centroids: Vec<D::Vector> = index::sample(rng, self.dataset.len(), self.k)
            .into_iter()
            .map(|i| self.dataset.vector(i as u32))
            .collect();
        self.normalize_if_spherical(&mut centroids);
        centroids
    }

    /// In spherical mode, scale every centroid to unit L2 norm.
    pub fn normalize_if_spherical(&self, centroids: &mut [D::Vector]) {
        if self.spherical {
            centroids.iter_mut().for_each(|c| normalize_l2(c.as_mut()));
        }
    }

    /// Stores `means` in a fresh centroid dataset.
    pub fn to_dataset(&self, means: &[D::Vector]) -> G {
        to_dataset(means, self.encoder)
    }

    /// The (weighted) mean and weighted count of every cluster of `groups` that has members.
    pub fn means(&self, groups: &ClusterGroups) -> Vec<Option<(D::Vector, f32)>> {
        group_means(self.dataset, self.weights, groups)
    }
}

/// The vectors of an assignment grouped by cluster, each cluster's members in ascending vector
/// id. The fixed member order fixes every mean's summation order, so results do not depend on
/// the thread count.
pub(super) struct ClusterGroups {
    offsets: Vec<usize>,
    members: Vec<u32>,
}

impl ClusterGroups {
    /// Groups the vectors `ids` (all of them when `None`) by their `assignments`, which are in
    /// `ids` order, into `k` clusters (a stable counting sort).
    pub fn new(ids: Option<&[u32]>, assignments: &[(f32, usize)], k: usize) -> Self {
        let mut offsets = vec![0usize; k + 1];
        for &(_, c) in assignments {
            offsets[c + 1] += 1;
        }
        for c in 0..k {
            offsets[c + 1] += offsets[c];
        }
        let mut members = vec![0u32; assignments.len()];
        let mut cursor = offsets.clone();
        let mut place = |id: u32, c: usize| {
            members[cursor[c]] = id;
            cursor[c] += 1;
        };
        match ids {
            None => {
                for (i, &(_, c)) in assignments.iter().enumerate() {
                    place(i as u32, c);
                }
            }
            Some(ids) => {
                assert_eq!(ids.len(), assignments.len());
                let mut order: Vec<usize> = (0..ids.len()).collect();
                order.par_sort_unstable_by_key(|&p| ids[p]);
                for p in order {
                    place(ids[p], assignments[p].1);
                }
            }
        }
        Self { offsets, members }
    }

    /// Number of clusters.
    pub fn k(&self) -> usize {
        self.offsets.len() - 1
    }

    /// The members of cluster `c`, in ascending vector id.
    pub fn members(&self, c: usize) -> &[u32] {
        &self.members[self.offsets[c]..self.offsets[c + 1]]
    }
}

/// The (weighted) mean and weighted count of every cluster of `groups` that has members
/// (`None` for the others), computed in parallel over clusters.
pub(super) fn group_means<D: MeanDataset + Sync>(
    dataset: &D,
    weights: Option<&[f32]>,
    groups: &ClusterGroups,
) -> Vec<Option<(D::Vector, f32)>> {
    (0..groups.k())
        .into_par_iter()
        .map(|c| {
            let members = groups.members(c);
            (!members.is_empty()).then(|| {
                let count = members
                    .iter()
                    .map(|&i| weights.map_or(1.0, |w| w[i as usize]))
                    .sum::<f32>();
                (dataset.mean(members, weights), count)
            })
        })
        .collect()
}

/// The imbalance factor `k · Σ count² / (Σ count)²` of a clustering with these per-cluster
/// (weighted) sizes: 1 when perfectly balanced, larger otherwise.
pub(super) fn imbalance_factor(counts: impl Iterator<Item = f64> + Clone) -> f64 {
    let k = counts.clone().count() as f64;
    let total: f64 = counts.clone().sum();
    let squares: f64 = counts.map(|c| c * c).sum();
    k * squares / (total * total)
}

/// Scale `values` to unit L2 norm in place; all-zero values stay zero.
pub(super) fn normalize_l2(values: &mut [f32]) {
    let norm: f32 = values.iter().map(|x| x * x).sum::<f32>().sqrt();
    // Take inverse of norm, take 0 if norm is 0
    let inv_norm = f32::from(norm > 0.0) * (1.0 / norm.max(f32::MIN_POSITIVE));
    values.iter_mut().for_each(|x| *x *= inv_norm);
}

/// Store `means` in a fresh growable dataset with the given encoder.
fn to_dataset<M, G>(means: &[M], encoder: &G::Encoder) -> G
where
    G: DatasetGrowable,
    for<'a> &'a M: Into<<G::Encoder as VectorEncoder>::InputVector<'a>>,
{
    let mut centroids = G::with_capacity(encoder.clone(), means.len());
    for mean in means {
        centroids.push(mean.into());
    }
    centroids
}

/// Run `assign` over the training vectors `ids` (all of them when `None`) in parallel,
/// with one decoded-query buffer per rayon worker, overwritten per vector. Results are in
/// `ids` order.
pub(super) fn assign_each<D>(
    dataset: &D,
    ids: Option<&[u32]>,
    assign: impl Fn(&D::Vector) -> (f32, usize) + Sync,
) -> Vec<(f32, usize)>
where
    D: MeanDataset + Sync,
{
    let run = |buffer: &mut Option<D::Vector>, i: u32| {
        let query = match buffer {
            Some(query) => {
                dataset.vector_into(i, query);
                query
            }
            None => buffer.insert(dataset.vector(i)),
        };
        assign(query)
    };
    if let Some(ids) = ids {
        return ids
            .par_iter()
            .map_init(|| None::<D::Vector>, |buffer, &i| run(buffer, i))
            .collect();
    }
    (0..dataset.len() as u32)
        .into_par_iter()
        .map_init(|| None::<D::Vector>, |buffer, i| run(buffer, i))
        .collect()
}

/// Flat assignment of the vectors `ids` (all of them when `None`), in `ids` order.
fn flat_assignments_of<D, C>(dataset: &D, ids: Option<&[u32]>, centroids: &C) -> Vec<(f32, usize)>
where
    D: MeanDataset + Sync,
    C: Dataset + Sync,
    C::Encoder: VectorEncoder<QueryParams = ()>,
    <C::Encoder as VectorEncoder>::Distance: Distance,
    for<'a> &'a D::Vector: Into<<C::Encoder as VectorEncoder>::QueryVector<'a>>,
{
    assert!(!centroids.is_empty(), "no centroids to assign to");
    assign_each(dataset, ids, |query| {
        let evaluator = centroids.encoder().query_evaluator(query.into(), &());
        let best = centroids
            .iter()
            .enumerate()
            .map(|(ci, centroid)| (evaluator.compute_distance(centroid), ci))
            .reduce(|best, next| {
                if next.0.is_better_than(&best.0) {
                    next
                } else {
                    best
                }
            })
            .unwrap();
        (best.0.distance(), best.1)
    })
}

impl KMeans {
    /// The solver this configuration fits centroids with.
    pub fn solver(&self) -> &Solver {
        &self.solver
    }

    /// Flat assignment: for each vector, its nearest centroid by exhaustive scan of `centroids`
    /// under the centroids' own metric.
    ///
    /// Returns `(distance, centroid)` per vector. Of tied centroids, the first wins.
    pub fn compute_assignments<D, C>(dataset: &D, centroids: &C) -> Vec<(f32, usize)>
    where
        D: MeanDataset + Sync,
        C: Dataset + Sync,
        C::Encoder: VectorEncoder<QueryParams = ()>,
        <C::Encoder as VectorEncoder>::Distance: Distance,
        for<'a> &'a D::Vector: Into<<C::Encoder as VectorEncoder>::QueryVector<'a>>,
    {
        flat_assignments_of(dataset, None, centroids)
    }

    /// Assignment through an index: for each vector, its top-1 result in `index`, under the
    /// index's metric.
    ///
    /// Returns `(distance, centroid)` per vector.
    pub fn assign_with_index<D, Q>(
        dataset: &D,
        index: &Q,
        search_params: &Q::SearchParams,
    ) -> Vec<(f32, usize)>
    where
        D: MeanDataset + Sync,
        Q: Index + Sync,
        Q::SearchParams: Sync,
        for<'a> &'a D::Vector: Into<Q::Query<'a>>,
    {
        assign_each(dataset, None, |query| {
            let best = index
                .search(query.into(), 1, search_params)
                .into_iter()
                .next()
                .expect("centroid index returned no result during k-means assignment");
            (best.distance.distance(), best.vector as usize)
        })
    }

    /// Runs k-means with `k` clusters on any [`MeanDataset`], with flat (exhaustive) assignment.
    ///
    /// Centroids are stored in a growable dataset `G` built with `centroid_encoder`, which fixes
    /// both their storage precision and the **clustering metric**: every training vector is
    /// assigned to its nearest centroid under `G`'s metric (see [`KMeans`]). With `weights`,
    /// centroids are weighted means.
    ///
    /// Which centroids are returned is up to the [`Solver`] (see [`LloydParams`] and
    /// [`EMeansParams`]). With a fixed [`KMeansBuilder::seed`] the result is deterministic
    /// across runs and thread counts.
    pub fn train<D, G>(
        &self,
        training_dataset: &D,
        k: usize,
        weights: Option<Vec<f32>>,
        centroid_encoder: G::Encoder,
    ) -> G
    where
        D: MeanDataset + Sync,
        G: DatasetGrowable + Clone + Sync,
        G::Encoder: VectorEncoder<QueryParams = ()>,
        <G::Encoder as VectorEncoder>::Distance: Distance + From<f32>,
        for<'a> &'a D::Vector: Into<<G::Encoder as VectorEncoder>::InputVector<'a>>
            + Into<<G::Encoder as VectorEncoder>::QueryVector<'a>>,
    {
        self.train_refined(training_dataset, k, weights, centroid_encoder, |_, _| {})
    }

    /// Like [`train`](Self::train), but every centroid update is followed by
    /// `refine(members, &mut centroid)`, where `members` are the ids of the training vectors
    /// that produced the update (all members of the cluster for Lloyd, the cluster's members in
    /// the mini-batch for e-means), in ascending order. The refined centroid is the one used
    /// from then on: for the next assignment, for splitting or reviving clusters, and as the
    /// result.
    ///
    /// This is how a representation-specific policy enters the generic loop, e.g. pruning sparse
    /// centroids to their largest components while k-means runs.
    pub fn train_refined<D, G>(
        &self,
        training_dataset: &D,
        k: usize,
        weights: Option<Vec<f32>>,
        centroid_encoder: G::Encoder,
        refine: impl Fn(&[u32], &mut D::Vector) + Sync,
    ) -> G
    where
        D: MeanDataset + Sync,
        G: DatasetGrowable + Clone + Sync,
        G::Encoder: VectorEncoder<QueryParams = ()>,
        <G::Encoder as VectorEncoder>::Distance: Distance + From<f32>,
        for<'a> &'a D::Vector: Into<<G::Encoder as VectorEncoder>::InputVector<'a>>
            + Into<<G::Encoder as VectorEncoder>::QueryVector<'a>>,
    {
        let mut assign = |centroids: &G, _: UpdateCtx, ids: Option<&[u32]>| {
            flat_assignments_of(training_dataset, ids, centroids)
        };
        self.fit::<D, G>(
            training_dataset,
            k,
            weights,
            &centroid_encoder,
            &mut assign,
            &refine,
        )
    }

    /// Like [`train`](Self::train), but assigns through an [`Index`] over the centroids, built
    /// by `build_centroid_index` (any `FnMut(G) -> Q` works) and kept in step with them under this
    /// configuration's [`IndexRefresh`] policy: by default it is rebuilt for every update (a
    /// Lloyd iteration, an e-means mini-batch). Every vector takes the index's top result, and
    /// its distance is recomputed exactly against the current centroid under `G`'s metric,
    /// which must be the index's.
    ///
    /// **Reproducibility caveat:** with a fixed [`KMeansBuilder::seed`], initialization and the
    /// update step are deterministic, but end-to-end reproducibility additionally requires the
    /// index to be deterministic in build and search. A parallel HNSW build generally is not.
    pub fn train_with_index<D, G, Q>(
        &self,
        training_dataset: &D,
        k: usize,
        weights: Option<Vec<f32>>,
        centroid_encoder: G::Encoder,
        build_centroid_index: impl CentroidIndexBuilder<G, Index = Q>,
        search_params: &Q::SearchParams,
    ) -> G
    where
        D: MeanDataset + Sync,
        G: DatasetGrowable + Clone + Sync,
        G::Encoder: VectorEncoder<QueryParams = ()>,
        <G::Encoder as VectorEncoder>::Distance: Distance + From<f32>,
        for<'a> &'a D::Vector: Into<<G::Encoder as VectorEncoder>::InputVector<'a>>
            + Into<<G::Encoder as VectorEncoder>::QueryVector<'a>>,
        Q: Index + Sync,
        Q::SearchParams: Sync,
        for<'a> &'a D::Vector: Into<Q::Query<'a>>,
    {
        self.train_with_index_refined(
            training_dataset,
            k,
            weights,
            centroid_encoder,
            build_centroid_index,
            search_params,
            |_, _| {},
        )
    }

    /// Like [`train_with_index`](Self::train_with_index), but every centroid update is followed
    /// by `refine(members, &mut centroid)`, as in [`train_refined`](Self::train_refined).
    #[allow(clippy::too_many_arguments)]
    pub fn train_with_index_refined<D, G, Q>(
        &self,
        training_dataset: &D,
        k: usize,
        weights: Option<Vec<f32>>,
        centroid_encoder: G::Encoder,
        build_centroid_index: impl CentroidIndexBuilder<G, Index = Q>,
        search_params: &Q::SearchParams,
        refine: impl Fn(&[u32], &mut D::Vector) + Sync,
    ) -> G
    where
        D: MeanDataset + Sync,
        G: DatasetGrowable + Clone + Sync,
        G::Encoder: VectorEncoder<QueryParams = ()>,
        <G::Encoder as VectorEncoder>::Distance: Distance + From<f32>,
        for<'a> &'a D::Vector: Into<<G::Encoder as VectorEncoder>::InputVector<'a>>
            + Into<<G::Encoder as VectorEncoder>::QueryVector<'a>>,
        Q: Index + Sync,
        Q::SearchParams: Sync,
        for<'a> &'a D::Vector: Into<Q::Query<'a>>,
    {
        let mut assigner = IndexedAssigner::new(build_centroid_index, self.index_refresh);
        let mut assign = |centroids: &G, ctx: UpdateCtx, ids: Option<&[u32]>| {
            if ctx.update == 0 {
                // A new run (or Lloyd restart): its centroids are unrelated to the last index.
                assigner.reset();
            }
            assigner.prepare(ctx, || centroids.clone());
            assigner.assign(training_dataset, ids, centroids, search_params)
        };
        let centroids = self.fit::<D, G>(
            training_dataset,
            k,
            weights,
            &centroid_encoder,
            &mut assign,
            &refine,
        );
        if self.verbose {
            println!(
                "Centroid index: {} builds + {} refreshes, {:.2?} total",
                assigner.builds, assigner.refreshes, assigner.maintenance_time
            );
        }
        centroids
    }

    /// The driver shared by every entry point and solver: validates the input, resolves the
    /// seed, then hands the run to the solver. `assign` maps the current centroids to one
    /// `(distance, centroid)` per requested training vector, under `G`'s metric.
    fn fit<D, G>(
        &self,
        training_dataset: &D,
        k: usize,
        weights: Option<Vec<f32>>,
        centroid_encoder: &G::Encoder,
        assign: &mut Assign<'_, G>,
        refine: &(impl Fn(&[u32], &mut D::Vector) + Sync),
    ) -> G
    where
        D: MeanDataset + Sync,
        G: DatasetGrowable,
        <G::Encoder as VectorEncoder>::Distance: Distance + From<f32>,
        for<'a> &'a D::Vector: Into<<G::Encoder as VectorEncoder>::InputVector<'a>>,
    {
        let n = training_dataset.len();
        if n == k {
            if self.verbose {
                println!("WARNING: number of training data is equal to the number of clusters.");
            }
            let all: Vec<D::Vector> = (0..n as u32).map(|i| training_dataset.vector(i)).collect();
            return to_dataset(&all, centroid_encoder);
        }
        assert!(k > 0, "k must be positive");
        assert!(
            k < n,
            "k ({k}) must not exceed the number of training vectors ({n})"
        );
        if let Some(w) = &weights {
            assert_eq!(
                w.len(),
                n,
                "weights must have one entry per training vector"
            );
        }
        let run = Run::<D, G> {
            dataset: training_dataset,
            k,
            weights: weights.as_deref(),
            encoder: centroid_encoder,
            seed: self.seed.unwrap_or_else(|| rand::thread_rng().r#gen()),
            spherical: self.spherical,
            verbose: self.verbose,
        };
        if self.verbose {
            println!(
                "Clustering {n} points ({} nnz, dim {}) to {k} clusters{}",
                training_dataset.nnz(),
                training_dataset.input_dim(),
                if self.spherical { ", spherical" } else { "" },
            );
        }
        match &self.solver {
            Solver::Lloyd(params) => lloyd::fit::<D, G, <G::Encoder as VectorEncoder>::Distance>(
                &run, params, assign, refine,
            ),
            Solver::EMeans(params) => emeans::fit(&run, params, assign, refine),
        }
    }
}

#[derive(Clone, Debug)]
pub struct KMeansBuilder {
    solver: Solver,
    verbose: bool,
    max_points_per_centroid: usize,
    seed: Option<u64>,
    spherical: bool,
    index_refresh: IndexRefresh,
}

impl Default for KMeansBuilder {
    fn default() -> Self {
        KMeansBuilder {
            solver: Solver::default(),
            verbose: false,
            max_points_per_centroid: 256,
            seed: None,
            spherical: false,
            index_refresh: IndexRefresh::default(),
        }
    }
}

impl KMeansBuilder {
    pub fn new() -> Self {
        KMeansBuilder::default()
    }

    /// The solver that fits the centroids (default: [`Solver::Lloyd`] with
    /// [`LloydParams::default`]).
    pub fn solver(mut self, solver: Solver) -> KMeansBuilder {
        self.solver = solver;
        self
    }

    /// Shorthand for the Lloyd solver's [`LloydParams::n_iter`]. Selects [`Solver::Lloyd`]
    /// (with default parameters otherwise) if another solver was set.
    pub fn n_iter(mut self, n_iter: usize) -> KMeansBuilder {
        self.lloyd_params().n_iter = n_iter;
        self
    }

    /// Shorthand for the Lloyd solver's [`LloydParams::n_redo`]. Selects [`Solver::Lloyd`]
    /// (with default parameters otherwise) if another solver was set.
    pub fn n_redo(mut self, n_redo: usize) -> KMeansBuilder {
        self.lloyd_params().n_redo = n_redo;
        self
    }

    fn lloyd_params(&mut self) -> &mut LloydParams {
        if !matches!(self.solver, Solver::Lloyd(_)) {
            self.solver = Solver::Lloyd(LloydParams::default());
        }
        match &mut self.solver {
            Solver::Lloyd(params) => params,
            Solver::EMeans(_) => unreachable!(),
        }
    }

    pub fn verbose(mut self, verbose: bool) -> KMeansBuilder {
        self.verbose = verbose;
        self
    }

    // pub fn min_points_per_centroid(mut self, min_points_per_centroid: usize) -> KMeansBuilder {
    //     self.min_points_per_centroid = min_points_per_centroid;
    //     self
    // }

    pub fn max_points_per_centroid(mut self, max_points_per_centroid: usize) -> KMeansBuilder {
        self.max_points_per_centroid = max_points_per_centroid;
        self
    }

    // pub fn sample_size(mut self, sample_size: usize) -> KMeansBuilder {
    //     self.sample_size = Some(sample_size);
    //     self
    // }

    /// Fix the RNG seed for reproducible clustering.
    /// `None` (default) draws one from entropy — different results every run.
    /// `Some(s)` gives identical results for the same data, seed and solver.
    pub fn seed(mut self, seed: Option<u64>) -> KMeansBuilder {
        self.seed = seed;
        self
    }

    /// Spherical k-means: L2-normalize the initial centroids and every centroid used for
    /// assignment, and return unit-norm centroids. (E-means runs its moving average on
    /// unnormalized positions and normalizes the snapshot it assigns against, as in faiss.)
    pub fn spherical(mut self, spherical: bool) -> KMeansBuilder {
        self.spherical = spherical;
        self
    }

    /// Centroid-index maintenance for [`KMeans::train_with_index`]: how often the index is
    /// refreshed (in updates or epochs) and how many of its results are re-ranked exactly.
    /// Default: refresh every update, top result only.
    pub fn index_refresh(mut self, index_refresh: IndexRefresh) -> KMeansBuilder {
        self.index_refresh = index_refresh;
        self
    }

    pub fn build(self) -> KMeans {
        KMeans {
            solver: self.solver,
            verbose: self.verbose,
            seed: self.seed,
            spherical: self.spherical,
            index_refresh: self.index_refresh,
        }
    }
}

/// A `refine` for [`KMeans::train_refined`] / [`KMeans::train_with_index_refined`] on sparse
/// data: prune each centroid to its `⌈factor · a⌉` largest-magnitude components, where `a` is
/// the average nnz of the vectors of `dataset` assigned to it.
///
/// Exact sparse means are much denser than the vectors they average, which makes both
/// assignment and any later scan over the centroids expensive. Pruning during training keeps
/// each centroid consistent with the assignment it drives, and makes every iteration cheaper.
/// Empty clusters are left unpruned. `dataset` must be the training dataset (member IDs index
/// it).
pub fn sparse_centroid_pruning<D>(
    dataset: &D,
    factor: f32,
) -> impl Fn(&[u32], &mut SparseVectorOwned<<D::Encoder as SparseDataEncoder>::InputComponentType, f32>)
+ Sync
+ '_
where
    D: Dataset + Sync,
    D::Encoder: SparseVectorEncoder,
{
    move |members, centroid| {
        if members.is_empty() {
            return;
        }
        let total: usize = members
            .iter()
            .map(|&i| dataset.get(i as VectorId).len())
            .sum();
        let keep = (factor * total as f32 / members.len() as f32).ceil() as usize;
        *centroid = top_components(centroid.as_view(), keep);
    }
}

/// The `keep` largest-magnitude components of `vector`, in component order.
fn top_components<C: ComponentType>(
    vector: SparseVectorView<'_, C, f32>,
    keep: usize,
) -> SparseVectorOwned<C, f32> {
    let mut top: Vec<usize> = (0..vector.len()).collect();
    if keep < top.len() {
        top.select_nth_unstable_by(keep, |&a, &b| {
            vector.values()[b]
                .abs()
                .total_cmp(&vector.values()[a].abs())
        });
        top.truncate(keep);
        top.sort_unstable();
    }
    SparseVectorOwned::new(
        top.iter().map(|&j| vector.components()[j]).collect(),
        top.iter().map(|&j| vector.values()[j]).collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::flat_index::FlatIndex;
    use crate::core::vector::DenseVectorView;
    use crate::core::vector::SparseVectorView;
    use crate::datasets::dense_dataset::DenseDatasetGrowable;
    use crate::distances::DotProduct;
    use crate::distances::SquaredEuclideanDistance;
    use crate::encoders::dense_scalar::{PlainDenseQuantizer, ScalarDenseQuantizer};
    use crate::encoders::sparse_scalar::PlainSparseQuantizer;
    use crate::{Float, FromF32, ValueType};
    use crate::{
        PlainDenseDataset, PlainDenseDatasetGrowable, PlainSparseDataset,
        PlainSparseDatasetGrowable,
    };

    type Centroids<V> = PlainDenseDataset<V, SquaredEuclideanDistance>;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn cluster_groups_sort_members_by_id() {
        // Batch ids out of order: each cluster's members come back in ascending id.
        let ids = [9u32, 2, 7, 4, 0];
        let assignments = [(0.0, 1usize), (0.0, 0), (0.0, 1), (0.0, 1), (0.0, 0)];
        let g = ClusterGroups::new(Some(&ids), &assignments, 3);
        assert_eq!(g.k(), 3);
        assert_eq!(g.members(0), &[0, 2]);
        assert_eq!(g.members(1), &[4, 7, 9]);
        assert!(g.members(2).is_empty());
        // Without ids, members are positions.
        let g = ClusterGroups::new(None, &assignments, 3);
        assert_eq!(g.members(0), &[1, 4]);
        assert_eq!(g.members(1), &[0, 2, 3]);
    }

    #[test]
    fn imbalance_factor_is_one_when_balanced() {
        assert_eq!(imbalance_factor([5.0, 5.0, 5.0].into_iter()), 1.0);
        // All mass in one of two clusters: 2 * 100 / 100.
        assert_eq!(imbalance_factor([10.0, 0.0].into_iter()), 2.0);
    }

    #[test]
    fn n_iter_and_n_redo_select_lloyd() {
        use super::super::emeans::EMeansParams;
        let km = KMeansBuilder::new()
            .solver(Solver::EMeans(EMeansParams::default()))
            .n_iter(3)
            .n_redo(2)
            .build();
        assert_eq!(
            km.solver(),
            &Solver::Lloyd(LloydParams {
                n_iter: 3,
                n_redo: 2
            })
        );
        assert_eq!(KMeansBuilder::new().build().solver(), &Solver::default());
    }

    #[test]
    fn compute_assignments_picks_nearest_centroid() {
        let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(2);
        let mut dataset_builder = PlainDenseDatasetGrowable::new(encoder.clone());
        dataset_builder.push(DenseVectorView::new(&[0.0f32, 0.0]));
        dataset_builder.push(DenseVectorView::new(&[10.0f32, 10.0]));

        let dataset: PlainDenseDataset<f32, SquaredEuclideanDistance> = dataset_builder.into();
        let centroids = vec![0.0f32, 0.0, 9.0, 9.0];
        let centroids_data = centroids.into_boxed_slice();
        let centroids_len = centroids_data.len() / dataset.output_dim();
        let centroids_dataset = PlainDenseDataset::<f32, SquaredEuclideanDistance>::from_raw(
            centroids_data,
            centroids_len,
            encoder.clone(),
        );
        let assignments: Vec<(f32, usize)> =
            KMeans::compute_assignments(&dataset, &centroids_dataset)
                .into_iter()
                .collect();

        assert_eq!(assignments.len(), dataset.len());
        assert_eq!(assignments[0].1, 0);
        assert_eq!(assignments[1].1, 1);
    }

    #[test]
    fn train_builds_two_distinct_clusters() {
        let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(2);
        let mut dataset_builder = PlainDenseDatasetGrowable::new(encoder.clone());
        dataset_builder.push(DenseVectorView::new(&[0.0f32, 0.0]));
        dataset_builder.push(DenseVectorView::new(&[0.0f32, 1.0]));
        dataset_builder.push(DenseVectorView::new(&[9.0f32, 9.0]));
        dataset_builder.push(DenseVectorView::new(&[10.0f32, 10.0]));

        let dataset: PlainDenseDataset<f32, SquaredEuclideanDistance> = dataset_builder.into();
        let kmeans = KMeansBuilder::new().n_iter(5).n_redo(2).build();
        let centroids: PlainDenseDatasetGrowable<f32, SquaredEuclideanDistance> =
            kmeans.train(&dataset, 2, None, PlainDenseQuantizer::new(2));

        assert_eq!(centroids.len(), 2);
        let assignments = KMeans::compute_assignments(&dataset, &centroids);
        let assignments: Vec<(f32, usize)> = assignments.into_iter().collect();
        let cluster_ids: std::collections::HashSet<usize> =
            assignments.iter().map(|&(_, idx)| idx).collect();
        assert_eq!(cluster_ids.len(), 2);
        assert_eq!(assignments.len(), dataset.len());
    }

    // ---- train_with_index: dual storage precision (T train, C centroids) -----
    //
    // ANN-assignment k-means via in-crate `FlatIndex` (no kannolo). Two well-separated
    // blobs with integer coords (exact in f16). Covers T=C=f32, T=C=f16, and the
    // recommended hybrid T=f16 / C=f32.

    /// Two blobs: points 0..3 near the origin, points 3..6 near (10, 10).
    const BLOB_POINTS: [[f32; 2]; 6] = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [10.0, 10.0],
        [10.0, 11.0],
        [11.0, 10.0],
    ];

    /// Cluster `BLOB_POINTS` into 2 via `train_with_index` with training storage `T`
    /// and centroid / index storage `C`.
    fn blob_partition<T, C>() -> Vec<usize>
    where
        T: Float
            + ValueType
            + FromF32
            + num_traits::ToPrimitive
            + num_traits::FromPrimitive
            + Clone,
        C: Float
            + ValueType
            + FromF32
            + num_traits::ToPrimitive
            + num_traits::FromPrimitive
            + Clone,
    {
        let encoder = PlainDenseQuantizer::<T, SquaredEuclideanDistance>::new(2);
        let mut builder = PlainDenseDatasetGrowable::new(encoder);
        for pt in BLOB_POINTS.iter() {
            let v: Vec<T> = pt.iter().map(|&x| T::from_f32_saturating(x)).collect();
            builder.push(DenseVectorView::new(&v[..]));
        }
        let dataset: Centroids<T> = builder.into();

        let kmeans = KMeansBuilder::new()
            .n_iter(10)
            .n_redo(3)
            .seed(Some(42))
            .build();
        let centroids: DenseDatasetGrowable<
            ScalarDenseQuantizer<f32, C, SquaredEuclideanDistance>,
        > = kmeans.train_with_index(
            &dataset,
            2,
            None,
            ScalarDenseQuantizer::new(2),
            FlatIndex::from,
            &(),
        );
        assert_eq!(centroids.len(), 2, "expected exactly 2 centroids");

        // Final labels via the same ANN path (works for T != C; compute_assignments
        // requires matching storage types).
        let index = FlatIndex::from(centroids);
        KMeans::assign_with_index(&dataset, &index, &())
            .into_iter()
            .map(|(_, cluster)| cluster)
            .collect()
    }

    /// A correct partition puts {0,1,2} in one cluster and {3,4,5} in the other.
    fn assert_recovers_blobs(p: &[usize]) {
        assert_eq!(p.len(), 6);
        assert_eq!(p[0], p[1], "blob A points 0,1 must co-cluster");
        assert_eq!(p[1], p[2], "blob A points 1,2 must co-cluster");
        assert_eq!(p[3], p[4], "blob B points 3,4 must co-cluster");
        assert_eq!(p[4], p[5], "blob B points 4,5 must co-cluster");
        assert_ne!(p[0], p[3], "the two blobs must be in different clusters");
    }

    fn assert_same_partition(a: &[usize], b: &[usize]) {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            for j in 0..a.len() {
                assert_eq!(
                    a[i] == a[j],
                    b[i] == b[j],
                    "co-assignment of points {i},{j} differs"
                );
            }
        }
    }

    #[test]
    fn train_with_index_f32_recovers_blobs() {
        assert_recovers_blobs(&blob_partition::<f32, f32>());
    }

    #[test]
    fn train_with_index_f16_recovers_blobs() {
        assert_recovers_blobs(&blob_partition::<half::f16, half::f16>());
    }

    #[test]
    fn train_with_index_f16_train_f32_centroids_recovers_blobs() {
        // Recommended scale path: f16 training corpus, f32 centroid index.
        assert_recovers_blobs(&blob_partition::<half::f16, f32>());
    }

    #[test]
    fn train_with_index_precisions_agree_on_partition() {
        let f32p = blob_partition::<f32, f32>();
        let f16p = blob_partition::<half::f16, half::f16>();
        let hybrid = blob_partition::<half::f16, f32>();
        assert_same_partition(&f32p, &f16p);
        assert_same_partition(&f32p, &hybrid);
    }

    // ---- sparse k-means ----------------------------------------

    fn sparse_dataset(
        vectors: &[(&[u16], &[f32])],
        dim: usize,
    ) -> PlainSparseDataset<u16, f32, DotProduct> {
        let mut builder = PlainSparseDatasetGrowable::<u16, f32, DotProduct>::new(
            PlainSparseQuantizer::new(dim, dim),
        );
        for (c, v) in vectors {
            builder.push(SparseVectorView::new(c, v));
        }
        builder.into()
    }

    /// Two topics on disjoint vocabularies, {0..4} and {10..14}, clustered under clustering
    /// metric `M`. Returns the label of each point.
    fn sparse_topic_labels<M>() -> Vec<usize>
    where
        M: crate::ScalarSparseSupportedDistance + From<f32>,
    {
        let mut vectors: Vec<(Vec<u16>, Vec<f32>)> = Vec::new();
        let mut rng = StdRng::seed_from_u64(7);
        for i in 0..40 {
            let base: u16 = if i % 2 == 0 { 0 } else { 10 };
            let comps: Vec<u16> = (base..base + 5).filter(|_| rng.gen_bool(0.7)).collect();
            let comps = if comps.is_empty() { vec![base] } else { comps };
            let vals = comps.iter().map(|_| rng.gen_range(0.5f32..2.0)).collect();
            vectors.push((comps, vals));
        }
        let refs: Vec<(&[u16], &[f32])> = vectors
            .iter()
            .map(|(c, v)| (c.as_slice(), v.as_slice()))
            .collect();
        let dataset = sparse_dataset(&refs, 20);

        let kmeans = KMeansBuilder::new()
            .n_iter(10)
            .n_redo(3)
            .seed(Some(42))
            .spherical(true)
            .build();
        let centroids: PlainSparseDatasetGrowable<u16, f32, M> =
            kmeans.train(&dataset, 2, None, PlainSparseQuantizer::new(20, 20));
        assert_eq!(centroids.len(), 2);

        KMeans::compute_assignments(&dataset, &centroids)
            .into_iter()
            .map(|(_, c)| c)
            .collect()
    }

    fn assert_recovers_topics(labels: &[usize]) {
        for i in 0..labels.len() {
            assert_eq!(labels[i], labels[i % 2], "point {i} in the wrong topic");
        }
        assert_ne!(labels[0], labels[1]);
    }

    /// `train_refined` applies the refinement to every centroid it keeps: here each sparse
    /// centroid is cut to its two largest components, and the topics are still recovered.
    #[test]
    fn train_refined_applies_refinement() {
        use crate::SparseVectorOwned;

        let mut vectors: Vec<(Vec<u16>, Vec<f32>)> = Vec::new();
        for i in 0..40u16 {
            let base: u16 = if i % 2 == 0 { 0 } else { 10 };
            let comps: Vec<u16> = (base..base + 5).collect();
            let vals = comps.iter().map(|&c| 1.0 + ((c + i) % 3) as f32).collect();
            vectors.push((comps, vals));
        }
        let refs: Vec<(&[u16], &[f32])> = vectors
            .iter()
            .map(|(c, v)| (c.as_slice(), v.as_slice()))
            .collect();
        let dataset = sparse_dataset(&refs, 20);

        let top2 = |_: &[u32], centroid: &mut SparseVectorOwned<u16, f32>| {
            let mut entries: Vec<(u16, f32)> = centroid.iter().collect();
            entries.sort_by(|a, b| b.1.total_cmp(&a.1));
            entries.truncate(2);
            entries.sort_by_key(|e| e.0);
            *centroid = SparseVectorOwned::new(
                entries.iter().map(|e| e.0).collect(),
                entries.iter().map(|e| e.1).collect(),
            );
        };
        let kmeans = KMeansBuilder::new().n_iter(5).seed(Some(42)).build();
        let centroids: PlainSparseDatasetGrowable<u16, f32, DotProduct> =
            kmeans.train_refined(&dataset, 2, None, PlainSparseQuantizer::new(20, 20), top2);

        assert!(centroids.iter().all(|c| c.len() <= 2));
        let labels: Vec<usize> = KMeans::compute_assignments(&dataset, &centroids)
            .into_iter()
            .map(|(_, c)| c)
            .collect();
        assert_recovers_topics(&labels);
    }

    /// `sparse_centroid_pruning` keeps each centroid within `⌈factor · average member nnz⌉`
    /// components, keeping the largest ones, and the topics are still recovered.
    #[test]
    fn sparse_centroid_pruning_bounds_centroid_nnz() {
        let mut vectors: Vec<(Vec<u16>, Vec<f32>)> = Vec::new();
        for i in 0..40u16 {
            let base: u16 = if i % 2 == 0 { 0 } else { 10 };
            let comps: Vec<u16> = (base..base + 5).collect();
            let vals = comps.iter().map(|&c| 1.0 + ((c + i) % 3) as f32).collect();
            vectors.push((comps, vals));
        }
        let refs: Vec<(&[u16], &[f32])> = vectors
            .iter()
            .map(|(c, v)| (c.as_slice(), v.as_slice()))
            .collect();
        let dataset = sparse_dataset(&refs, 20);

        // Every vector has 5 nonzeros, so factor 0.5 allows ⌈2.5⌉ = 3 per centroid.
        let kmeans = KMeansBuilder::new().n_iter(5).seed(Some(42)).build();
        let centroids: PlainSparseDatasetGrowable<u16, f32, DotProduct> = kmeans.train_refined(
            &dataset,
            2,
            None,
            PlainSparseQuantizer::new(20, 20),
            sparse_centroid_pruning(&dataset, 0.5),
        );
        assert!(centroids.iter().all(|c| c.len() == 3));
        let labels: Vec<usize> = KMeans::compute_assignments(&dataset, &centroids)
            .into_iter()
            .map(|(_, c)| c)
            .collect();
        assert_recovers_topics(&labels);
    }

    #[test]
    fn top_components_keeps_largest_magnitudes_in_component_order() {
        let v = SparseVectorView::new(&[1_u16, 4, 6, 9], &[0.5_f32, -3.0, 2.0, 1.0]);
        let top = top_components(v, 2);
        assert_eq!(top.components(), &[4, 6]);
        assert_eq!(top.values(), &[-3.0, 2.0]);
        assert_eq!(top_components(v, 10).values(), v.values());
    }

    /// Spherical normalization works on any vector's stored values: the norm of a sparse vector
    /// is the norm of its nonzeros, and its components are untouched; zero stays zero.
    #[test]
    fn normalize_l2_dense_and_sparse_values() {
        let mut dense = crate::DenseVectorOwned::new(vec![3.0f32, 4.0]);
        normalize_l2(dense.as_mut());
        assert_eq!(dense.values(), &[0.6, 0.8]);

        let mut zero = crate::DenseVectorOwned::new(vec![0.0f32, 0.0]);
        normalize_l2(zero.as_mut());
        assert_eq!(zero.values(), &[0.0, 0.0]);

        let mut sparse = crate::SparseVectorOwned::new(vec![2u16, 7], vec![3.0f32, 4.0]);
        normalize_l2(sparse.as_mut());
        assert_eq!(sparse.components(), &[2, 7]);
        assert_eq!(sparse.values(), &[0.6, 0.8]);
    }

    /// The index variant applies the refinement too.
    #[test]
    fn train_with_index_refined_applies_refinement() {
        use crate::SparseVectorOwned;

        let vectors: Vec<(Vec<u16>, Vec<f32>)> = (0..40u16)
            .map(|i| {
                let base: u16 = if i % 2 == 0 { 0 } else { 10 };
                let comps: Vec<u16> = (base..base + 5).collect();
                let vals = comps.iter().map(|&c| 1.0 + ((c + i) % 3) as f32).collect();
                (comps, vals)
            })
            .collect();
        let refs: Vec<(&[u16], &[f32])> = vectors
            .iter()
            .map(|(c, v)| (c.as_slice(), v.as_slice()))
            .collect();
        let dataset = sparse_dataset(&refs, 20);

        let first2 = |_: &[u32], centroid: &mut SparseVectorOwned<u16, f32>| {
            *centroid = SparseVectorOwned::new(
                centroid.components().iter().take(2).copied().collect(),
                centroid.values().iter().take(2).copied().collect(),
            );
        };
        let kmeans = KMeansBuilder::new().n_iter(5).seed(Some(42)).build();
        let centroids: PlainSparseDatasetGrowable<u16, f32, DotProduct> = kmeans
            .train_with_index_refined(
                &dataset,
                2,
                None,
                PlainSparseQuantizer::new(20, 20),
                FlatIndex::from,
                &(),
                first2,
            );
        assert!(centroids.iter().all(|c| c.len() <= 2));
    }

    #[test]
    fn train_sparse_dot_product_recovers_disjoint_topics() {
        assert_recovers_topics(&sparse_topic_labels::<DotProduct>());
    }

    #[test]
    fn train_sparse_euclidean_recovers_disjoint_topics() {
        assert_recovers_topics(&sparse_topic_labels::<SquaredEuclideanDistance>());
    }
}
