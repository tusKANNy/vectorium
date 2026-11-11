//! Index-accelerated nearest-centroid assignment, shared by every [`Clustering`] algorithm.
//!
//! [`Clustering::train_with_index`] assigns training vectors through an index over the current
//! centroids. Clustering moves the centroids after every *update* (a Lloyd iteration, an
//! e-means mini-batch), so the index must be kept in step with them. Two knobs control the
//! trade-off between index maintenance and assignment accuracy, identically for every
//! algorithm:
//!
//! * **When** the index is refreshed — [`IndexRefresh::every`], counted in updates or in
//!   epochs (full passes over the training set). For Lloyd an update *is* an epoch; for
//!   e-means an epoch is many mini-batch updates.
//! * **How** it is refreshed — [`CentroidIndexBuilder::refresh`]. The default rebuilds the
//!   index from scratch (right for HNSW). Builders with reusable structure can do much less:
//!   an IVF over the centroids keeps its coarse quantizer and only re-files centroids into
//!   lists, as faiss's e-means IVF assignment does.
//!
//! Between refreshes the index is stale: it was built over older centroid positions. Each
//! vector's top [`IndexRefresh::candidates`] index results are therefore re-ranked by their
//! exact distance to the **current** centroids, under the centroids' own metric, so staleness
//! can only change *which* centroids are considered, never the distances clustering sees.
//!
//! [`Clustering`]: super::Clustering
//! [`Clustering::train_with_index`]: super::Clustering::train_with_index

use crate::core::dataset::MeanDataset;
use crate::core::distances::Distance;
use crate::core::index::Index;
use crate::core::vector_encoder::{QueryEvaluator, VectorEncoder};
use crate::{Dataset, VectorId};
use std::time::{Duration, Instant};

/// Builds (and refreshes) the index used to assign training vectors to centroids `G`.
///
/// Any `FnMut(G) -> Q` closure is a builder whose refresh is a full rebuild, so
/// `|c| HNSW::build_index(c, &config)` or `FlatIndex::from` work as-is. Implement the trait
/// directly when an index can be updated more cheaply than it is built.
pub trait CentroidIndexBuilder<G> {
    /// The index type produced.
    type Index;

    /// Builds an index over `centroids` from scratch. Called for the first update.
    fn build(&mut self, centroids: G) -> Self::Index;

    /// Brings `index` up to date with moved `centroids`. The default rebuilds from scratch.
    fn refresh(&mut self, index: Self::Index, centroids: G) -> Self::Index {
        drop(index);
        self.build(centroids)
    }
}

impl<G, Q, F> CentroidIndexBuilder<G> for F
where
    F: FnMut(G) -> Q,
{
    type Index = Q;

    fn build(&mut self, centroids: G) -> Q {
        self(centroids)
    }
}

/// How often the centroid index is refreshed during [`train_with_index`].
///
/// [`train_with_index`]: super::Clustering::train_with_index
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefreshEvery {
    /// Every `n` updates (Lloyd iterations or e-means mini-batches).
    Updates(usize),
    /// At the start of every `n`-th epoch (pass over the training set).
    Epochs(usize),
}

/// Centroid-index maintenance policy for [`train_with_index`], shared by all algorithms.
///
/// The default — refresh every update, use only the top result — is the classic behaviour:
/// a fresh index for every Lloyd iteration or e-means batch.
///
/// [`train_with_index`]: super::Clustering::train_with_index
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexRefresh {
    /// Refresh cadence.
    pub every: RefreshEvery,
    /// Index results per vector re-ranked by exact distance to the current centroids (>= 1).
    pub candidates: usize,
}

impl Default for IndexRefresh {
    fn default() -> Self {
        Self {
            every: RefreshEvery::Updates(1),
            candidates: 1,
        }
    }
}

impl IndexRefresh {
    /// Refresh every `n` updates.
    pub fn every_updates(n: usize) -> Self {
        Self {
            every: RefreshEvery::Updates(n.max(1)),
            ..Self::default()
        }
    }

    /// Refresh at the start of every `n`-th epoch.
    pub fn every_epochs(n: usize) -> Self {
        Self {
            every: RefreshEvery::Epochs(n.max(1)),
            ..Self::default()
        }
    }

    /// Re-rank this many index results per vector against the current centroids.
    pub fn with_candidates(mut self, candidates: usize) -> Self {
        self.candidates = candidates.max(1);
        self
    }
}

/// Where an update sits in training, for deciding whether the index is due a refresh.
#[derive(Clone, Copy, Debug)]
pub(crate) struct UpdateCtx {
    /// Update counter within the current run (0-based).
    pub update: usize,
    /// Epoch counter within the current run (0-based).
    pub epoch: usize,
    /// Whether this is the first update of its epoch.
    pub epoch_start: bool,
    /// Some centroids jumped since the last update (split to revive an empty or starved
    /// cluster), so a stale index cannot find them: refresh regardless of cadence.
    pub centroids_jumped: bool,
}

/// Owns a [`CentroidIndexBuilder`] and its current index for one training run.
pub(crate) struct IndexedAssigner<G, B: CentroidIndexBuilder<G>> {
    builder: B,
    index: Option<B::Index>,
    policy: IndexRefresh,
    pub builds: usize,
    pub refreshes: usize,
    pub maintenance_time: Duration,
    _centroids: std::marker::PhantomData<fn(G)>,
}

impl<G, B: CentroidIndexBuilder<G>> IndexedAssigner<G, B> {
    pub fn new(builder: B, policy: IndexRefresh) -> Self {
        Self {
            builder,
            index: None,
            policy,
            builds: 0,
            refreshes: 0,
            maintenance_time: Duration::ZERO,
            _centroids: std::marker::PhantomData,
        }
    }

    /// Drops the current index so the next [`prepare`](Self::prepare) builds from scratch.
    pub fn reset(&mut self) {
        self.index = None;
    }

    fn due(&self, ctx: UpdateCtx) -> bool {
        if ctx.centroids_jumped {
            return true;
        }
        match self.policy.every {
            RefreshEvery::Updates(n) => ctx.update.is_multiple_of(n.max(1)),
            RefreshEvery::Epochs(n) => ctx.epoch_start && ctx.epoch.is_multiple_of(n.max(1)),
        }
    }

    /// Builds the index on the first call and refreshes it whenever the policy says so.
    /// `centroids` is only invoked when the index needs them.
    pub fn prepare(&mut self, ctx: UpdateCtx, centroids: impl FnOnce() -> G) {
        if self.index.is_some() && !self.due(ctx) {
            return;
        }
        let t0 = Instant::now();
        self.index = Some(match self.index.take() {
            None => {
                self.builds += 1;
                self.builder.build(centroids())
            }
            Some(old) => {
                self.refreshes += 1;
                self.builder.refresh(old, centroids())
            }
        });
        self.maintenance_time += t0.elapsed();
    }

    /// Assigns the training vectors `ids` (all of them when `None`) to their nearest
    /// `current` centroid: the best of their top index results, re-ranked under `current`'s
    /// metric. Returns `(distance, centroid)` per vector, in `ids` order. Of candidates at
    /// equal distance, the lowest centroid id wins.
    pub fn assign<D>(
        &self,
        dataset: &D,
        ids: Option<&[u32]>,
        current: &G,
        search_params: &<B::Index as Index>::SearchParams,
    ) -> Vec<(f32, usize)>
    where
        D: MeanDataset + Sync,
        G: Dataset + Sync,
        G::Encoder: VectorEncoder<QueryParams = ()>,
        <G::Encoder as VectorEncoder>::Distance: Distance,
        for<'a> &'a D::Vector: Into<<G::Encoder as VectorEncoder>::QueryVector<'a>>,
        B::Index: Index + Sync,
        <B::Index as Index>::SearchParams: Sync,
        for<'a> &'a D::Vector: Into<<B::Index as Index>::Query<'a>>,
    {
        let index = self
            .index
            .as_ref()
            .expect("prepare() must run before assign()");
        let candidates = self.policy.candidates.clamp(1, current.len().max(1));
        super::kmeans::KMeans::assign_each(dataset, ids, |query| {
            let evaluator = current.encoder().query_evaluator(query.into(), &());
            index
                .search(query.into(), candidates, search_params)
                .into_iter()
                .map(|r| {
                    let c = r.vector as usize;
                    (evaluator.compute_distance(current.get(c as VectorId)), c)
                })
                .reduce(|best, next| {
                    if next.0.is_better_than(&best.0) || (next.0 == best.0 && next.1 < best.1) {
                        next
                    } else {
                        best
                    }
                })
                .map(|(d, c)| (d.distance(), c))
                .expect("centroid index returned no result during clustering assignment")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::{Clustering, ClusteringConfig, EMeansBuilder, KMeans, KMeansBuilder};
    use crate::core::flat_index::FlatIndex;
    use crate::{
        DatasetGrowable, PlainDenseDataset, PlainDenseDatasetGrowable, PlainDenseQuantizer,
        SquaredEuclideanDistance,
    };
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    type Plain = PlainDenseDataset<f32, SquaredEuclideanDistance>;
    type Grow = PlainDenseDatasetGrowable<f32, SquaredEuclideanDistance>;

    fn enc(d: usize) -> PlainDenseQuantizer<f32, SquaredEuclideanDistance> {
        PlainDenseQuantizer::new(d)
    }

    /// A flat-index builder that counts builds and refreshes.
    #[derive(Clone, Default)]
    struct Counting {
        builds: Arc<AtomicUsize>,
        refreshes: Arc<AtomicUsize>,
    }

    impl CentroidIndexBuilder<Grow> for Counting {
        type Index = FlatIndex<Grow>;
        fn build(&mut self, centroids: Grow) -> Self::Index {
            self.builds.fetch_add(1, Ordering::Relaxed);
            FlatIndex::from(centroids)
        }
        fn refresh(&mut self, _old: Self::Index, centroids: Grow) -> Self::Index {
            self.refreshes.fetch_add(1, Ordering::Relaxed);
            FlatIndex::from(centroids)
        }
    }

    impl Counting {
        fn counts(&self) -> (usize, usize) {
            (
                self.builds.load(Ordering::Relaxed),
                self.refreshes.load(Ordering::Relaxed),
            )
        }
    }

    fn blobs(n_blobs: usize, per_blob: usize, d: usize, seed: u64) -> (Plain, Vec<usize>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let centers: Vec<f32> = (0..n_blobs * d)
            .map(|_| rng.gen_range(-100.0..100.0))
            .collect();
        let mut values = Vec::new();
        let mut labels = Vec::new();
        for i in 0..n_blobs * per_blob {
            let b = i % n_blobs;
            labels.push(b);
            values.extend((0..d).map(|x| centers[b * d + x] + rng.gen_range(-1.0f32..1.0)));
        }
        let n = labels.len();
        (
            Plain::from_raw(values.into_boxed_slice(), n, enc(d)),
            labels,
        )
    }

    fn assert_recovers(ds: &Plain, c: &Grow, truth: &[usize], k: usize) {
        let found: Vec<usize> = KMeans::compute_assignments(ds, c)
            .into_iter()
            .map(|a| a.1)
            .collect();
        let mut map = vec![usize::MAX; k];
        for (&f, &t) in found.iter().zip(truth) {
            if map[t] == usize::MAX {
                map[t] = f;
            }
            assert_eq!(map[t], f, "blob {t} split across clusters");
        }
        assert_eq!(
            map.iter().collect::<std::collections::HashSet<_>>().len(),
            k
        );
    }

    #[test]
    fn emeans_refresh_cadence_counts() {
        let (ds, _) = blobs(4, 500, 8, 1);
        let k = 4;
        // Fix a small batch so there are many updates per epoch; revival off so only the
        // cadence triggers refreshes (revival forces one, see `centroid_jump_forces_refresh`).
        let base = EMeansBuilder::new()
            .seed(Some(1))
            .nepochs(3.0)
            .batch_size(Some(100))
            .unused_threshold(Some(0.0));
        let recipe = base.clone().build().recipe(ds.len(), k, 8);
        let total = recipe.total_steps;
        assert_eq!(recipe.epoch_steps, vec![20, 20, 20]);

        // Default: a fresh index for every mini-batch.
        let b = Counting::default();
        let _: Grow = base
            .clone()
            .build()
            .train_with_index(&ds, k, None, enc(8), b.clone(), &());
        assert_eq!(b.counts(), (1, total - 1));

        // Every 7 updates: refresh at updates 7, 14, ..., < total.
        let b = Counting::default();
        let em = base
            .clone()
            .index_refresh(IndexRefresh::every_updates(7))
            .build();
        let _: Grow = em.train_with_index(&ds, k, None, enc(8), b.clone(), &());
        assert_eq!(b.counts(), (1, (total - 1) / 7));

        // Once per epoch: refreshed at the start of epochs 1 and 2.
        let b = Counting::default();
        let em = base.index_refresh(IndexRefresh::every_epochs(1)).build();
        let _: Grow = em.train_with_index(&ds, k, None, enc(8), b.clone(), &());
        assert_eq!(b.counts(), (1, 2));
    }

    #[test]
    fn centroid_jump_forces_refresh() {
        let (ds, _) = blobs(2, 10, 4, 1);
        let b = Counting::default();
        let mut a = IndexedAssigner::<Grow, _>::new(b.clone(), IndexRefresh::every_updates(100));
        let ctx = |update, jumped| UpdateCtx {
            update,
            epoch: 0,
            epoch_start: update == 0,
            centroids_jumped: jumped,
        };
        let grow = || {
            let mut g = Grow::new(enc(4));
            for v in ds.iter() {
                g.push(v);
            }
            g
        };
        a.prepare(ctx(0, false), grow);
        a.prepare(ctx(1, false), grow);
        assert_eq!(b.counts(), (1, 0), "not due yet");
        a.prepare(ctx(2, true), grow);
        assert_eq!(b.counts(), (1, 1), "a jump refreshes immediately");
    }

    #[test]
    fn splits_and_revivals_force_refresh() {
        // With a cadence that never comes due, only moved centroids can trigger a refresh:
        // k-means splits of empty clusters and e-means revivals of starved ones must both do so.
        let never = IndexRefresh::every_updates(usize::MAX);
        // Three distinct points, each repeated 100 times, and k = 6. Duplicate centroids tie and
        // the first wins, so the others go empty (k-means splits them) or starve (e-means
        // revives them), on most updates. d = 4: revival is off by default at d <= 2.
        let mut values = Vec::new();
        for i in 0..300 {
            values.extend([(i % 3) as f32 * 10.0, 1.0, 2.0, 3.0]);
        }
        let dup = Plain::from_raw(values.into_boxed_slice(), 300, enc(4));
        let b = Counting::default();
        let km = KMeansBuilder::new()
            .n_iter(5)
            .seed(Some(1))
            .index_refresh(never)
            .build();
        let _: Grow = km.train_with_index(&dup, 6, None, enc(4), b.clone(), &());
        let (builds, refreshes) = b.counts();
        assert_eq!(builds, 1);
        assert!(refreshes > 0, "k-means splits must refresh the index");

        let b = Counting::default();
        let em = EMeansBuilder::new()
            .seed(Some(1))
            .batch_size(Some(30))
            .unused_threshold(Some(0.1))
            .index_refresh(never)
            .build();
        let _: Grow = em.train_with_index(&dup, 6, None, enc(4), b.clone(), &());
        let (builds, refreshes) = b.counts();
        assert_eq!(builds, 1);
        assert!(refreshes > 0, "e-means revivals must refresh the index");
    }

    #[test]
    fn kmeans_refresh_cadence_counts_and_resets_per_redo() {
        let (ds, _) = blobs(4, 200, 8, 2);
        let b = Counting::default();
        let km = KMeansBuilder::new()
            .n_iter(6)
            .n_redo(2)
            .seed(Some(3))
            .build();
        let _: Grow = km.train_with_index(&ds, 4, None, enc(8), b.clone(), &());
        // Each redo builds once and refreshes for iterations 1..6.
        assert_eq!(b.counts(), (2, 10));

        let b = Counting::default();
        let km = KMeansBuilder::new()
            .n_iter(6)
            .seed(Some(3))
            .index_refresh(IndexRefresh::every_updates(3))
            .build();
        let _: Grow = km.train_with_index(&ds, 4, None, enc(8), b.clone(), &());
        assert_eq!(b.counts(), (1, 1));
    }

    /// Overlapping clusters, so assignments are genuinely contested.
    fn overlapping(n: usize, d: usize, seed: u64) -> Plain {
        let mut rng = StdRng::seed_from_u64(seed);
        let centers: Vec<f32> = (0..20 * d).map(|_| rng.gen_range(-3.0..3.0)).collect();
        let values: Vec<f32> = (0..n * d)
            .map(|i| centers[((i / d) % 20) * d + i % d] + rng.gen_range(-1.0f32..1.0))
            .collect();
        Plain::from_raw(values.into_boxed_slice(), n, enc(d))
    }

    fn objective(ds: &Plain, c: &Grow) -> f64 {
        KMeans::compute_assignments(ds, c)
            .into_iter()
            .map(|a| a.0 as f64)
            .sum()
    }

    fn assert_close(a: &Grow, b: &Grow) {
        assert_eq!(Dataset::len(a), Dataset::len(b));
        for (x, y) in a.values().iter().zip(b.values()) {
            assert!((x - y).abs() <= 1e-4 * x.abs().max(1.0), "{x} vs {y}");
        }
    }

    #[test]
    fn never_refreshed_index_with_all_candidates_is_exact() {
        // Re-ranking every centroid exactly makes even a never-refreshed index exact, so
        // the index path must reproduce flat training for both algorithms.
        let ds = overlapping(3_000, 8, 1);
        let k = 16;
        let stale = IndexRefresh::every_updates(usize::MAX).with_candidates(k);
        let km = KMeansBuilder::new().n_iter(6).seed(Some(2));
        let flat: Grow = km.clone().build().train(&ds, k, None, enc(8));
        let idx: Grow = km.index_refresh(stale).build().train_with_index(
            &ds,
            k,
            None,
            enc(8),
            FlatIndex::from,
            &(),
        );
        assert_close(&flat, &idx);
        let em = EMeansBuilder::new().seed(Some(2)).nepochs(2.0);
        let flat: Grow = em.clone().build().train(&ds, k, None, enc(8));
        let idx: Grow = em.index_refresh(stale).build().train_with_index(
            &ds,
            k,
            None,
            enc(8),
            FlatIndex::from,
            &(),
        );
        assert_close(&flat, &idx);
    }

    /// A stale index changes which local optimum a single run lands in (one different early
    /// assignment changes the trajectory), so compare means over several seeds rather than
    /// one run.
    #[test]
    fn moderately_stale_index_stays_close_to_flat_on_average() {
        let ds = overlapping(6_000, 8, 3);
        let k = 16;
        let stale = IndexRefresh::every_updates(3).with_candidates(3);
        let seeds = 1..=5u64;
        let mean = |f: &dyn Fn(u64) -> Grow| -> f64 {
            seeds.clone().map(|s| objective(&ds, &f(s))).sum::<f64>() / 5.0
        };
        let km = |s| KMeansBuilder::new().n_iter(10).seed(Some(s));
        let flat = mean(&|s| km(s).build().train(&ds, k, None, enc(8)));
        let idx = mean(&|s| {
            km(s).index_refresh(stale).build().train_with_index(
                &ds,
                k,
                None,
                enc(8),
                FlatIndex::from,
                &(),
            )
        });
        assert!(idx <= 1.05 * flat, "kmeans mean {idx} vs flat {flat}");
        let em = |s| EMeansBuilder::new().seed(Some(s));
        let flat = mean(&|s| Clustering::train(&em(s).build(), &ds, k, None, enc(8)));
        let idx = mean(&|s| {
            em(s).index_refresh(stale).build().train_with_index(
                &ds,
                k,
                None,
                enc(8),
                FlatIndex::from,
                &(),
            )
        });
        assert!(idx <= 1.05 * flat, "emeans mean {idx} vs flat {flat}");
    }

    #[test]
    fn default_kmeans_index_path_matches_flat_path() {
        // With a fresh exact index every iteration, train_with_index reproduces train.
        let (ds, _) = blobs(5, 300, 10, 9);
        let km = KMeansBuilder::new().n_iter(5).seed(Some(11)).build();
        let flat: Grow = km.train(&ds, 7, None, enc(10));
        let indexed: Grow = km.train_with_index(&ds, 7, None, enc(10), FlatIndex::from, &());
        assert_close(&flat, &indexed);
    }

    #[test]
    fn clustering_config_dispatches() {
        let (ds, truth) = blobs(4, 300, 8, 6);
        for cfg in [
            ClusteringConfig::from(KMeansBuilder::new().seed(Some(1))),
            ClusteringConfig::from(EMeansBuilder::new().seed(Some(1))),
        ] {
            let c: Grow = cfg.train(&ds, 4, None, enc(8));
            assert_recovers(&ds, &c, &truth, 4);
            let c: Grow = cfg.train_with_index(&ds, 4, None, enc(8), FlatIndex::from, &());
            assert_recovers(&ds, &c, &truth, 4);
        }
        assert!(matches!(
            ClusteringConfig::default(),
            ClusteringConfig::KMeans(_)
        ));
    }
}
