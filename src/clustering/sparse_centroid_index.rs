//! Inverted index over sparse centroids, to accelerate k-means assignment under the dot
//! product ([`KMeans::train_with_index`]).
//!
//! **Scope.** This is a helper for assigning training vectors to centroids, the sparse
//! dot-product counterpart of assigning through a [`FlatIndex`]: with every query component
//! scored it computes the exact dot product with every centroid, only in a different order.
//! It is not a retrieval index, and it has no pruning beyond the optional
//! [`doc_cut`](SparseCentroidIndexSearchParams::doc_cut). Retrieval indexes over sparse
//! collections (block-max pruning, Seismic-style blocks, …) belong to kANNolo or to their own
//! crate.
//!
//! **How it works.** Flat assignment scores every training vector against every centroid
//! with a full sparse dot product. Sparse centroids touch only a small part of the
//! vocabulary, so most of those products are zero. [`SparseCentroidIndex`] stores, for every
//! component, the centroids that have it (CSR layout: one offsets array and one contiguous
//! postings array), and scores a vector by walking only the postings of its own components,
//! accumulating one `f32` score per centroid.
//! [`doc_cut`](SparseCentroidIndexSearchParams::doc_cut) walks only the vector's
//! largest-magnitude components, an approximation (k-means recomputes the exact distance of
//! the chosen centroid anyway, see [`index_assign`](super::index_assign)).
//!
//! **Types.** Centroid values can be any [`ValueType`] (`f32`, `f16`, `bf16`, fixed point), so
//! the index can be built over centroids stored at reduced precision. Queries are `f32`
//! sparse vectors, which is what k-means passes: training vectors are decoded to `f32`
//! whatever their storage type. Only the dot product is supported.
//!
//! Ported from Seismic's `FastInvertedIndex`, which clusters posting lists the same way.
//!
//! [`KMeans::train_with_index`]: super::KMeans::train_with_index
//! [`FlatIndex`]: crate::FlatIndex

use std::cell::RefCell;
use std::num::NonZeroUsize;

use crate::core::dataset::ScoredVector;
use crate::core::index::Index;
use crate::core::vector::SparseVectorView;
use crate::utils::numeric_markers::ToF32Fast;
use crate::{
    ComponentType, Dataset, DotProduct, Float, FromF32, ScalarSparseDatasetGrowable, ValueType,
    VectorId,
};

/// Search parameters of [`SparseCentroidIndex`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SparseCentroidIndexSearchParams {
    /// Score only the query's `doc_cut` largest-magnitude components. `None` scores all of
    /// them, which gives the exact dot product.
    ///
    /// If the truncated query finds no centroid with a positive score (its largest components
    /// appear in no centroid, e.g. because centroids were pruned), the query is scored again
    /// with all its components, so it is not sent to an arbitrary centroid.
    pub doc_cut: Option<NonZeroUsize>,
}

impl SparseCentroidIndexSearchParams {
    /// Exact scoring: every query component is used.
    pub fn exact() -> Self {
        Self { doc_cut: None }
    }

    /// Approximate scoring with the query's `doc_cut` largest-magnitude components.
    ///
    /// # Panics
    /// If `doc_cut` is 0.
    pub fn with_doc_cut(doc_cut: usize) -> Self {
        Self {
            doc_cut: Some(
                NonZeroUsize::new(doc_cut).expect("SparseCentroidIndex: doc_cut must be > 0"),
            ),
        }
    }
}

/// CSR inverted index over a set of sparse centroids with values of type `V`, searched by dot
/// product with `f32` sparse queries.
///
/// `postings[offsets[c]..offsets[c + 1]]` holds the `(centroid, value)` pairs of component
/// `c`, in centroid order.
///
/// Searches accumulate scores in a thread-local buffer of one `f32` per centroid, reused
/// across searches so that a search allocates nothing but its result. The buffer keeps the
/// size of the largest index searched on that thread.
#[derive(Clone, Debug)]
pub struct SparseCentroidIndex<C: ComponentType, V: ValueType = f32> {
    n_centroids: usize,
    offsets: Box<[u32]>,
    postings: Box<[(u32, V)]>,
    _component: std::marker::PhantomData<C>,
}

thread_local! {
    /// Per-thread score accumulator, see [`SparseCentroidIndex`].
    static SCORES: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    /// Per-thread buffer for the positions of the query's top components.
    static TOP: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

impl<C: ComponentType, V: ValueType> SparseCentroidIndex<C, V> {
    /// Builds the index over `centroids`, whose components must all be `< dim`.
    ///
    /// # Panics
    /// If a component is `>= dim`, or if the total number of nonzeros overflows `u32`.
    pub fn from_vectors<'a>(
        dim: usize,
        centroids: impl IntoIterator<Item = SparseVectorView<'a, C, V>>,
    ) -> Self
    where
        C: 'a,
        V: 'a,
    {
        let centroids: Vec<SparseVectorView<'a, C, V>> = centroids.into_iter().collect();
        let total_postings: usize = centroids.iter().map(|c| c.components().len()).sum();
        assert!(
            total_postings <= u32::MAX as usize,
            "SparseCentroidIndex: {total_postings} postings overflow the u32 offsets"
        );

        // Counting sort by component: count, prefix-sum, fill.
        let mut offsets = vec![0_u32; dim + 1];
        let n_centroids = centroids.len();
        for (slot, centroid) in centroids.iter().enumerate() {
            for &c in centroid.components() {
                let c = c.as_();
                assert!(
                    c < dim,
                    "SparseCentroidIndex: centroid {slot} has component {c}, out of bounds for dim {dim}"
                );
                offsets[c + 1] += 1;
            }
        }
        for i in 1..=dim {
            offsets[i] += offsets[i - 1];
        }

        let mut postings = vec![(0_u32, V::zero()); offsets[dim] as usize];
        let mut cursors = offsets[..dim].to_vec();
        for (slot, centroid) in centroids.iter().enumerate() {
            for (&c, &v) in centroid.components().iter().zip(centroid.values()) {
                let cursor = &mut cursors[c.as_()];
                postings[*cursor as usize] = (slot as u32, v);
                *cursor += 1;
            }
        }

        Self {
            n_centroids,
            offsets: offsets.into_boxed_slice(),
            postings: postings.into_boxed_slice(),
            _component: std::marker::PhantomData,
        }
    }

    /// Builds the index over every vector of a sparse centroid dataset.
    pub fn from_dataset<W>(centroids: &ScalarSparseDatasetGrowable<C, W, V, DotProduct>) -> Self
    where
        W: ValueType + Float,
        V: Float + FromF32,
    {
        Self::from_vectors(centroids.input_dim(), centroids.iter())
    }

    /// Number of indexed centroids.
    pub fn len(&self) -> usize {
        self.n_centroids
    }

    /// Whether the index holds no centroid.
    pub fn is_empty(&self) -> bool {
        self.n_centroids == 0
    }

    #[inline]
    fn postings(&self, component: usize) -> &[(u32, V)] {
        // Query components beyond the indexed dimensionality have no centroid.
        if component + 1 >= self.offsets.len() {
            return &[];
        }
        &self.postings[self.offsets[component] as usize..self.offsets[component + 1] as usize]
    }

    /// Adds `qv` times the postings of `component` into `scores`.
    #[inline]
    fn accumulate_component(&self, component: usize, qv: f32, scores: &mut [f32]) {
        for &(slot, cv) in self.postings(component) {
            scores[slot as usize] += qv * cv.to_f32_fast();
        }
    }

    /// Accumulates the dot product of `query` (restricted to its `doc_cut` largest components)
    /// with every centroid into `scores`, which must be zeroed and of length `len()`. Returns
    /// whether `doc_cut` actually dropped some components.
    fn accumulate(
        &self,
        query: SparseVectorView<'_, C, f32>,
        doc_cut: Option<NonZeroUsize>,
        scores: &mut [f32],
    ) -> bool {
        let components = query.components();
        let values = query.values();
        match doc_cut.map(NonZeroUsize::get) {
            Some(cut) if cut < components.len() => {
                TOP.with_borrow_mut(|top| {
                    top.clear();
                    top.extend(0..components.len() as u32);
                    top.select_nth_unstable_by(cut, |&a, &b| {
                        values[b as usize]
                            .abs()
                            .total_cmp(&values[a as usize].abs())
                    });
                    for &j in &top[..cut] {
                        let j = j as usize;
                        self.accumulate_component(components[j].as_(), values[j], scores);
                    }
                });
                true
            }
            _ => {
                for (&c, &qv) in components.iter().zip(values) {
                    self.accumulate_component(c.as_(), qv, scores);
                }
                false
            }
        }
    }
}

impl<C, W, V> From<&ScalarSparseDatasetGrowable<C, W, V, DotProduct>> for SparseCentroidIndex<C, V>
where
    C: ComponentType,
    W: ValueType + Float,
    V: ValueType + Float + FromF32,
{
    fn from(centroids: &ScalarSparseDatasetGrowable<C, W, V, DotProduct>) -> Self {
        Self::from_dataset(centroids)
    }
}

impl<C, W, V> From<ScalarSparseDatasetGrowable<C, W, V, DotProduct>> for SparseCentroidIndex<C, V>
where
    C: ComponentType,
    W: ValueType + Float,
    V: ValueType + Float + FromF32,
{
    fn from(centroids: ScalarSparseDatasetGrowable<C, W, V, DotProduct>) -> Self {
        Self::from_dataset(&centroids)
    }
}

impl<C: ComponentType, V: ValueType> Index for SparseCentroidIndex<C, V> {
    type Query<'q> = SparseVectorView<'q, C, f32>;
    type Distance = DotProduct;
    type SearchParams = SparseCentroidIndexSearchParams;

    /// The `k` centroids with the largest (possibly `doc_cut`-approximated) dot product with
    /// `query`, best first. Of tied centroids, the one with the smallest id wins, as in flat
    /// assignment.
    fn search<'q>(
        &self,
        query: Self::Query<'q>,
        k: usize,
        search_params: &Self::SearchParams,
    ) -> Vec<ScoredVector<DotProduct>> {
        let n = self.n_centroids;
        if n == 0 || k == 0 {
            return Vec::new();
        }
        SCORES.with_borrow_mut(|scores| {
            scores.clear();
            scores.resize(n, 0.0);
            let truncated = self.accumulate(query, search_params.doc_cut, scores);

            // The truncated query found no positive evidence for any centroid: its top
            // components miss every centroid, so every score is (at best) zero and the
            // argmax would be an arbitrary centroid. Score it exactly instead.
            if truncated && scores.iter().all(|&s| s <= 0.0) {
                scores.fill(0.0);
                self.accumulate(query, None, scores);
            }

            if k == 1 {
                let mut best = 0;
                let mut best_score = scores[0];
                for (i, &s) in scores.iter().enumerate().skip(1) {
                    if s > best_score {
                        best = i;
                        best_score = s;
                    }
                }
                return vec![ScoredVector {
                    distance: DotProduct(best_score),
                    vector: best as VectorId,
                }];
            }

            // Best first; ties by smallest id, so the order is total and deterministic.
            let by_score = |&a: &u32, &b: &u32| {
                scores[b as usize]
                    .total_cmp(&scores[a as usize])
                    .then(a.cmp(&b))
            };
            let mut order: Vec<u32> = (0..n as u32).collect();
            let take = k.min(n);
            if take < n {
                order.select_nth_unstable_by(take, by_score);
                order.truncate(take);
            }
            order.sort_unstable_by(by_score);
            order
                .into_iter()
                .map(|i| ScoredVector {
                    distance: DotProduct(scores[i as usize]),
                    vector: i as VectorId,
                })
                .collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::{KMeans, KMeansBuilder, Solver};
    use crate::{DatasetGrowable, PlainSparseDatasetGrowable, PlainSparseQuantizer};
    use half::f16;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    const DIM: usize = 200;

    fn random_sparse(
        n: usize,
        nnz: usize,
        seed: u64,
    ) -> PlainSparseDatasetGrowable<u16, f32, DotProduct> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut ds = PlainSparseDatasetGrowable::<u16, f32, DotProduct>::new(
            PlainSparseQuantizer::new(DIM, DIM),
        );
        for _ in 0..n {
            let mut comps: Vec<u16> = (0..nnz).map(|_| rng.gen_range(0..DIM as u16)).collect();
            comps.sort_unstable();
            comps.dedup();
            let vals: Vec<f32> = comps.iter().map(|_| rng.gen_range(0.01..2.0)).collect();
            ds.push(SparseVectorView::new(&comps, &vals));
        }
        ds
    }

    fn exact_dot<V: ValueType>(
        a: SparseVectorView<'_, u16, f32>,
        b: SparseVectorView<'_, u16, V>,
    ) -> f32 {
        let (mut i, mut j, mut s) = (0, 0, 0.0);
        let (ac, av, bc, bv) = (a.components(), a.values(), b.components(), b.values());
        while i < ac.len() && j < bc.len() {
            match ac[i].cmp(&bc[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    s += av[i] * bv[j].to_f32().unwrap();
                    i += 1;
                    j += 1;
                }
            }
        }
        s
    }

    #[test]
    fn exact_search_matches_brute_force() {
        let centroids = random_sparse(64, 30, 1);
        let queries = random_sparse(300, 20, 2);
        let index = SparseCentroidIndex::from(&centroids);
        assert_eq!(index.len(), 64);
        for q in queries.iter() {
            let got = index.search(q, 1, &SparseCentroidIndexSearchParams::exact())[0];
            let best = centroids
                .iter()
                .map(|c| exact_dot(q, c))
                .fold(f32::MIN, f32::max);
            let got_exact = exact_dot(q, centroids.get(got.vector));
            assert!(
                (got_exact - best).abs() <= 1e-4 * best.abs().max(1.0),
                "{got_exact} vs {best}"
            );
            assert!((got.distance.0 - got_exact).abs() <= 1e-4 * best.abs().max(1.0));
        }
    }

    #[test]
    fn f16_centroids_match_brute_force() {
        let centroids_f32 = random_sparse(64, 30, 11);
        let mut centroids = ScalarSparseDatasetGrowable::<u16, f32, f16, DotProduct>::new(
            crate::ScalarSparseQuantizer::new(DIM, DIM),
        );
        for c in centroids_f32.iter() {
            centroids.push(c);
        }
        let queries = random_sparse(300, 20, 12);
        let index: SparseCentroidIndex<u16, f16> = SparseCentroidIndex::from(&centroids);
        for q in queries.iter() {
            let got = index.search(q, 1, &SparseCentroidIndexSearchParams::exact())[0];
            let best = centroids
                .iter()
                .map(|c| exact_dot(q, c))
                .fold(f32::MIN, f32::max);
            let got_exact = exact_dot(q, centroids.get(got.vector));
            assert!(
                (got_exact - best).abs() <= 1e-4 * best.abs().max(1.0),
                "{got_exact} vs {best}"
            );
        }
    }

    #[test]
    fn top_k_is_sorted_with_ties_by_id_and_matches_full_sort() {
        let centroids = random_sparse(32, 30, 3);
        let index = SparseCentroidIndex::from(&centroids);
        let queries = random_sparse(50, 25, 4);
        for q in queries.iter() {
            let all = index.search(q, 32, &SparseCentroidIndexSearchParams::exact());
            assert_eq!(all.len(), 32);
            assert!(all.windows(2).all(|w| {
                w[0].distance.0 > w[1].distance.0
                    || (w[0].distance.0 == w[1].distance.0 && w[0].vector < w[1].vector)
            }));
            for k in [2, 5, 31] {
                let top = index.search(q, k, &SparseCentroidIndexSearchParams::exact());
                assert_eq!(top, all[..k]);
            }
        }
    }

    #[test]
    fn doc_cut_scores_only_the_largest_components() {
        let centroids = random_sparse(32, 30, 3);
        let index = SparseCentroidIndex::from(&centroids);
        let q = random_sparse(1, 25, 4);
        let q = q.get(0);
        let j = (0..q.values().len())
            .max_by(|&a, &b| q.values()[a].total_cmp(&q.values()[b]))
            .unwrap();
        let one = index.search(q, 1, &SparseCentroidIndexSearchParams::with_doc_cut(1))[0];
        let expected = centroids
            .iter()
            .map(|c| {
                c.components()
                    .iter()
                    .position(|&cc| cc == q.components()[j])
                    .map_or(0.0, |p| c.values()[p] * q.values()[j])
            })
            .fold(f32::MIN, f32::max);
        assert!(expected > 0.0, "test setup: the top component must hit some centroid");
        assert!((one.distance.0 - expected).abs() < 1e-5);
    }

    /// A query whose largest component is in no centroid used to score 0 everywhere under
    /// `doc_cut` and go to centroid 0. It must now fall back to exact scoring.
    #[test]
    fn doc_cut_missing_every_centroid_falls_back_to_exact() {
        let mut centroids = PlainSparseDatasetGrowable::<u16, f32, DotProduct>::new(
            PlainSparseQuantizer::new(DIM, DIM),
        );
        centroids.push(SparseVectorView::new(&[1_u16], &[1.0_f32]));
        centroids.push(SparseVectorView::new(&[2_u16], &[1.0_f32]));
        centroids.push(SparseVectorView::new(&[3_u16], &[1.0_f32]));
        let index = SparseCentroidIndex::from(&centroids);

        // Component 50 (largest) is in no centroid; component 3 points to centroid 2.
        let query = SparseVectorView::new(&[3_u16, 50], &[0.5_f32, 9.0]);
        let got = index.search(query, 1, &SparseCentroidIndexSearchParams::with_doc_cut(1))[0];
        assert_eq!(got.vector, 2);
        assert!((got.distance.0 - 0.5).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "doc_cut must be > 0")]
    fn zero_doc_cut_is_rejected() {
        SparseCentroidIndexSearchParams::with_doc_cut(0);
    }

    #[test]
    #[should_panic(expected = "out of bounds for dim")]
    fn out_of_range_component_is_rejected() {
        let comps = [5_u16];
        let vals = [1.0_f32];
        SparseCentroidIndex::<u16, f32>::from_vectors(4, [SparseVectorView::new(&comps, &vals)]);
    }

    #[test]
    fn train_with_exact_index_matches_flat_training() {
        let data = random_sparse(2000, 25, 5);
        let kmeans = KMeansBuilder::new()
            .solver(Solver::Lloyd)
            .epochs(4.0)
            .seed(Some(7))
            .build();
        let enc = PlainSparseQuantizer::<u16, f32, DotProduct>::new(DIM, DIM);
        let flat: PlainSparseDatasetGrowable<u16, f32, DotProduct> =
            kmeans.train(&data, 20, None, enc.clone());
        let indexed: PlainSparseDatasetGrowable<u16, f32, DotProduct> = kmeans.train_with_index(
            &data,
            20,
            None,
            enc,
            |c: PlainSparseDatasetGrowable<u16, f32, DotProduct>| SparseCentroidIndex::from(c),
            &SparseCentroidIndexSearchParams::exact(),
        );
        let obj = |c: &PlainSparseDatasetGrowable<u16, f32, DotProduct>| -> f64 {
            KMeans::compute_assignments(&data, c)
                .iter()
                .map(|&(d, _)| d as f64)
                .sum()
        };
        let (of, oi) = (obj(&flat), obj(&indexed));
        assert!(
            (of - oi).abs() <= 1e-3 * of.abs(),
            "flat {of} vs indexed {oi}"
        );
    }

    /// k-means with f16 centroids, assigned through an index over those f16 centroids.
    #[test]
    fn train_with_f16_centroid_index_matches_flat_training() {
        let data = random_sparse(2000, 25, 6);
        let kmeans = KMeansBuilder::new()
            .solver(Solver::Lloyd)
            .epochs(4.0)
            .seed(Some(7))
            .build();
        type G = ScalarSparseDatasetGrowable<u16, f32, f16, DotProduct>;
        let enc = crate::ScalarSparseQuantizer::<u16, f32, f16, DotProduct>::new(DIM, DIM);
        let flat: G = kmeans.train(&data, 20, None, enc.clone());
        let indexed: G = kmeans.train_with_index(
            &data,
            20,
            None,
            enc,
            |c: G| SparseCentroidIndex::from(c),
            &SparseCentroidIndexSearchParams::exact(),
        );
        let obj = |c: &G| -> f64 {
            KMeans::compute_assignments(&data, c)
                .iter()
                .map(|&(d, _)| d as f64)
                .sum()
        };
        let (of, oi) = (obj(&flat), obj(&indexed));
        assert!(
            (of - oi).abs() <= 1e-3 * of.abs(),
            "flat {of} vs indexed {oi}"
        );
    }
}
