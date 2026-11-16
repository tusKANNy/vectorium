//! Inverted index over sparse centroids, for index-accelerated k-means assignment under the
//! dot product ([`KMeans::train_with_index`]).
//!
//! Flat assignment scores every training vector against every centroid with a full sparse
//! dot product. Sparse centroids touch only a small part of the vocabulary, so most of those
//! products are zero. [`SimpleSparseInvertedIndex`] stores, for every component, the centroids that
//! have it (CSR layout: one offsets array and one contiguous postings array), and scores a
//! vector by walking only the postings of its own components, accumulating into one score per
//! centroid. Walking all of the vector's components yields the exact dot product with every
//! centroid; [`SimpleSparseInvertedIndexSearchParams::doc_cut`] walks only its largest-magnitude ones,
//! an approximation (the exact distance of the chosen centroid is recomputed by k-means
//! anyway, see [`index_assign`](super::index_assign)).
//!
//! Ported from Seismic's `FastInvertedIndex`, which clusters posting lists the same way.
//!
//! [`KMeans::train_with_index`]: super::KMeans::train_with_index

use std::cell::RefCell;

use crate::core::dataset::ScoredVector;
use crate::core::index::Index;
use crate::core::vector::SparseVectorView;
use crate::{ComponentType, Dataset, DotProduct, PlainSparseDatasetGrowable, VectorId};

/// Search parameters of [`SimpleSparseInvertedIndex`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SimpleSparseInvertedIndexSearchParams {
    /// Score only the query's `doc_cut` largest-magnitude components. `None` scores all of
    /// them, which gives the exact dot product.
    pub doc_cut: Option<usize>,
}

impl SimpleSparseInvertedIndexSearchParams {
    /// Exact scoring: every query component is used.
    pub fn exact() -> Self {
        Self { doc_cut: None }
    }

    /// Approximate scoring with the query's `doc_cut` largest-magnitude components.
    pub fn with_doc_cut(doc_cut: usize) -> Self {
        Self {
            doc_cut: Some(doc_cut),
        }
    }
}

/// CSR inverted index over a set of sparse `f32` centroids, searched by dot product.
///
/// `postings[offsets[c]..offsets[c + 1]]` holds the `(centroid, value)` pairs of component
/// `c`, in centroid order.
#[derive(Clone, Debug)]
pub struct SimpleSparseInvertedIndex<C: ComponentType> {
    n_centroids: usize,
    offsets: Box<[u32]>,
    postings: Box<[(u32, f32)]>,
    _component: std::marker::PhantomData<C>,
}

thread_local! {
    /// Per-thread score accumulator, reused across searches so a search allocates nothing
    /// but its result.
    static SCORES: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    /// Per-thread buffer for the positions of the query's top components.
    static TOP: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

impl<C: ComponentType> SimpleSparseInvertedIndex<C> {
    /// Builds the index over `centroids`, whose components are all `< dim`.
    pub fn from_vectors<'a>(
        dim: usize,
        centroids: impl IntoIterator<Item = SparseVectorView<'a, C, f32>>,
    ) -> Self
    where
        C: 'a,
    {
        let centroids: Vec<SparseVectorView<'a, C, f32>> = centroids.into_iter().collect();
        let total_postings: usize = centroids.iter().map(|c| c.components().len()).sum();
        assert!(
            total_postings <= u32::MAX as usize,
            "SimpleSparseInvertedIndex: {total_postings} postings overflow the u32 offsets"
        );

        // Counting sort by component: count, prefix-sum, fill.
        let mut offsets = vec![0_u32; dim + 1];
        let n_centroids = centroids.len();
        for centroid in &centroids {
            for &c in centroid.components() {
                offsets[c.as_() + 1] += 1;
            }
        }
        for i in 1..=dim {
            offsets[i] += offsets[i - 1];
        }

        let mut postings = vec![(0_u32, 0_f32); offsets[dim] as usize];
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

    /// Number of indexed centroids.
    pub fn len(&self) -> usize {
        self.n_centroids
    }

    /// Whether the index holds no centroid.
    pub fn is_empty(&self) -> bool {
        self.n_centroids == 0
    }

    #[inline]
    fn postings(&self, component: usize) -> &[(u32, f32)] {
        // Components beyond the indexed dimensionality have no centroid.
        if component + 1 >= self.offsets.len() {
            return &[];
        }
        &self.postings[self.offsets[component] as usize..self.offsets[component + 1] as usize]
    }

    /// Accumulates the dot product of `query` (restricted to `doc_cut` components) with every
    /// centroid into `scores` (which must be zeroed and of length `len()`).
    fn accumulate(
        &self,
        query: SparseVectorView<'_, C, f32>,
        doc_cut: Option<usize>,
        scores: &mut [f32],
    ) {
        let components = query.components();
        let values = query.values();
        match doc_cut {
            Some(cut) if cut < components.len() => TOP.with_borrow_mut(|top| {
                top.clear();
                top.extend(0..components.len() as u32);
                top.select_nth_unstable_by(cut, |&a, &b| {
                    values[b as usize]
                        .abs()
                        .total_cmp(&values[a as usize].abs())
                });
                for &j in &top[..cut] {
                    let qv = values[j as usize];
                    for &(slot, cv) in self.postings(components[j as usize].as_()) {
                        scores[slot as usize] += qv * cv;
                    }
                }
            }),
            _ => {
                for (&c, &qv) in components.iter().zip(values) {
                    for &(slot, cv) in self.postings(c.as_()) {
                        scores[slot as usize] += qv * cv;
                    }
                }
            }
        }
    }
}

impl<C: ComponentType> From<&PlainSparseDatasetGrowable<C, f32, DotProduct>>
    for SimpleSparseInvertedIndex<C>
{
    fn from(centroids: &PlainSparseDatasetGrowable<C, f32, DotProduct>) -> Self {
        Self::from_vectors(centroids.input_dim(), centroids.iter())
    }
}

impl<C: ComponentType> From<PlainSparseDatasetGrowable<C, f32, DotProduct>>
    for SimpleSparseInvertedIndex<C>
{
    fn from(centroids: PlainSparseDatasetGrowable<C, f32, DotProduct>) -> Self {
        Self::from(&centroids)
    }
}

impl<C: ComponentType> Index for SimpleSparseInvertedIndex<C> {
    type Query<'q> = SparseVectorView<'q, C, f32>;
    type Distance = DotProduct;
    type SearchParams = SimpleSparseInvertedIndexSearchParams;

    /// The `k` centroids with the largest (possibly `doc_cut`-approximated) dot product with
    /// `query`, best first. Of tied centroids, the first wins, as in flat assignment.
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
            self.accumulate(query, search_params.doc_cut, scores);

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

            let mut order: Vec<usize> = (0..n).collect();
            let take = k.min(n);
            // Stable sort: ties keep centroid order.
            order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
            order[..take]
                .iter()
                .map(|&i| ScoredVector {
                    distance: DotProduct(scores[i]),
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
    use crate::{DatasetGrowable, PlainSparseQuantizer};
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

    fn exact_dot(a: SparseVectorView<'_, u16, f32>, b: SparseVectorView<'_, u16, f32>) -> f32 {
        let (mut i, mut j, mut s) = (0, 0, 0.0);
        let (ac, av, bc, bv) = (a.components(), a.values(), b.components(), b.values());
        while i < ac.len() && j < bc.len() {
            match ac[i].cmp(&bc[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    s += av[i] * bv[j];
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
        let index = SimpleSparseInvertedIndex::from(&centroids);
        assert_eq!(index.len(), 64);
        for q in queries.iter() {
            let got = index.search(q, 1, &SimpleSparseInvertedIndexSearchParams::exact())[0];
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
    fn top_k_is_sorted_and_doc_cut_limits_components() {
        let centroids = random_sparse(32, 30, 3);
        let index = SimpleSparseInvertedIndex::from(&centroids);
        let q = random_sparse(1, 25, 4);
        let q = q.get(0);
        let top = index.search(q, 5, &SimpleSparseInvertedIndexSearchParams::exact());
        assert_eq!(top.len(), 5);
        assert!(top.windows(2).all(|w| w[0].distance.0 >= w[1].distance.0));

        // doc_cut = 1: only the largest query component is scored.
        let j = (0..q.values().len())
            .max_by(|&a, &b| q.values()[a].total_cmp(&q.values()[b]))
            .unwrap();
        let one = index.search(
            q,
            1,
            &SimpleSparseInvertedIndexSearchParams::with_doc_cut(1),
        )[0];
        let expected = centroids
            .iter()
            .map(|c| {
                c.components()
                    .iter()
                    .position(|&cc| cc == q.components()[j])
                    .map_or(0.0, |p| c.values()[p] * q.values()[j])
            })
            .fold(f32::MIN, f32::max);
        assert!((one.distance.0 - expected).abs() < 1e-5);
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
            |c: PlainSparseDatasetGrowable<u16, f32, DotProduct>| {
                SimpleSparseInvertedIndex::from(c)
            },
            &SimpleSparseInvertedIndexSearchParams::exact(),
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
}
