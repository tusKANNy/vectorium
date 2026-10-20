use crate::Dataset;
use crate::core::dataset::{DatasetGrowable, MeanDataset};
use crate::core::distances::{Distance, dot_product_dense};
use crate::core::index::Index;
use crate::core::vector::{DenseVectorView, SparseVectorOwned, SparseVectorView};
use crate::core::vector_encoder::{
    QueryEvaluator, SparseDataEncoder, SparseVectorEncoder, VectorEncoder,
};
use crate::{ComponentType, VectorId};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::index;
use rayon::prelude::*;
use std::time::Instant;

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
pub struct KMeans {
    n_iter: usize,
    n_redo: usize,
    verbose: bool,
    /// Optional fixed seed for the RNG.  `None` = `from_entropy()` (non-deterministic).
    /// Setting a seed makes clustering fully reproducible across runs and thread counts.
    seed: Option<u64>,
    /// Spherical k-means: L2-normalize each centroid after every update step.
    spherical: bool,
}

impl KMeans {
    /// Computes the imbalance factor of the clustering.
    /// Smaller unfairness factor means more balanced clusters.
    ///
    /// # Arguments
    ///
    /// * `histograms`: vector storing how many vectors are assigned to each cluster
    /// * `k`: the number of clusters
    ///
    /// returns: the imbalance factor as f32
    ///
    #[inline]
    fn imbalance_factor(histograms: &[f32], k: usize) -> f32 {
        let hist_view = DenseVectorView::new(histograms);
        let unfairness_factor = dot_product_dense(hist_view, hist_view).distance();
        let total: f32 = histograms.iter().sum();
        unfairness_factor * k as f32 / (total * total)
    }

    /// Scale `values` to unit L2 norm in place; all-zero values stay zero.
    fn normalize_l2(values: &mut [f32]) {
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

    /// Run `assign` over the training vectors in parallel, with one decoded-query buffer per
    /// rayon worker, overwritten per vector.
    fn assign_each<D>(
        dataset: &D,
        assign: impl Fn(&D::Vector) -> (f32, usize) + Sync,
    ) -> Vec<(f32, usize)>
    where
        D: MeanDataset + Sync,
    {
        (0..dataset.len() as u32)
            .into_par_iter()
            .map_init(
                || None::<D::Vector>,
                |buffer, i| {
                    let query = match buffer {
                        Some(query) => {
                            dataset.vector_into(i, query);
                            query
                        }
                        None => buffer.insert(dataset.vector(i)),
                    };
                    assign(query)
                },
            )
            .collect()
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
        assert!(!centroids.is_empty(), "no centroids to assign to");
        Self::assign_each(dataset, |query| {
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
        Self::assign_each(dataset, |query| {
            let best = index
                .search(query.into(), 1, search_params)
                .into_iter()
                .next()
                .expect("centroid index returned no result during k-means assignment");
            (best.distance.distance(), best.vector as usize)
        })
    }

    /// Computes the centroids as the (weighted) means of the vectors assigned to each cluster,
    /// passes each through `refine` together with its members, then splits empty clusters.
    ///
    /// Returns the number of splits, the (weighted) size of each cluster and the new centroids.
    ///
    /// Members are grouped by a stable counting sort, so each cluster averages its members in
    /// ascending point order and the results are bit-identical across thread counts.
    fn update_and_split<D>(
        dataset: &D,
        weights: Option<&[f32]>,
        k: usize,
        assignments: &[(f32, usize)],
        rng: &mut StdRng,
        spherical: bool,
        refine: &(impl Fn(&[u32], &mut D::Vector) + Sync),
    ) -> (usize, Vec<f32>, Vec<D::Vector>)
    where
        D: MeanDataset + Sync,
    {
        let n = dataset.len();

        let mut offsets = vec![0usize; k + 1];
        for &(_, ci) in assignments {
            offsets[ci + 1] += 1;
        }
        for c in 0..k {
            offsets[c + 1] += offsets[c];
        }
        let mut grouped = vec![0u32; n];
        let mut cursor = offsets.clone();
        for (i, &(_, ci)) in assignments.iter().enumerate() {
            grouped[cursor[ci]] = i as u32;
            cursor[ci] += 1;
        }

        let (mut centroids, mut histograms): (Vec<D::Vector>, Vec<f32>) = (0..k)
            .into_par_iter()
            .map(|ci| {
                let members = &grouped[offsets[ci]..offsets[ci + 1]];
                let count = members
                    .iter()
                    .map(|&i| weights.map_or(1.0, |w| w[i as usize]))
                    .sum::<f32>();
                let mut centroid = dataset.mean(members, weights);
                refine(members, &mut centroid);
                (centroid, count)
            })
            .unzip();

        // Splits clusters
        let mut n_splits = 0;
        let mut cj;
        let epsilon = 1.0 / 1024.;

        for ci in 0..k {
            if histograms[ci] != 0.0 {
                continue;
            }
            cj = 0;
            loop {
                let p = (histograms[cj] - 1.0) / (n - k) as f32;
                let r = Rng::r#gen::<f32>(rng);
                if r < p {
                    break;
                }
                cj = (cj + 1) % k;
            }

            centroids[ci] = centroids[cj].clone();
            for j in 0..centroids[ci].as_mut().len() {
                if j % 2 == 0 {
                    centroids[ci].as_mut()[j] *= 1.0 + epsilon;
                    centroids[cj].as_mut()[j] *= 1.0 - epsilon;
                } else {
                    centroids[ci].as_mut()[j] *= 1.0 - epsilon;
                    centroids[cj].as_mut()[j] *= 1.0 + epsilon;
                }
            }

            histograms[ci] = histograms[cj] / 2.0;
            histograms[cj] /= 2.0;
            n_splits += 1;
        }

        if spherical {
            for centroid in centroids.iter_mut() {
                Self::normalize_l2(centroid.as_mut());
            }
        }

        (n_splits, histograms, centroids)
    }

    /// Runs k-means with `k` clusters on any [`MeanDataset`], with flat (exhaustive) assignment.
    ///
    /// Centroids are stored in a growable dataset `G` built with `centroid_encoder`, which fixes
    /// both their storage precision and the **clustering metric**: every training vector is
    /// assigned to its nearest centroid under `G`'s metric (see [`KMeans`]). With `weights`,
    /// centroids are weighted means.
    ///
    /// The objective is the sum of the assignment distances, compared through the metric's
    /// [`Distance`] ordering (minimized for squared Euclidean, maximized for dot product). The
    /// centroids of the best iteration are returned. With a fixed [`KMeansBuilder::seed`] the
    /// result is deterministic across runs and thread counts.
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
    /// assigned to that centroid. The refined centroid is the one used from then on: for the next
    /// assignment, for splitting empty clusters, and as the result.
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
        self.fit::<D, G, <G::Encoder as VectorEncoder>::Distance>(
            training_dataset,
            k,
            weights,
            centroid_encoder,
            |centroids| Self::compute_assignments(training_dataset, centroids),
            refine,
        )
    }

    /// Like [`train`](Self::train), but assigns through an [`Index`]: at each iteration
    /// `build_centroid_index` builds an index over the current centroids and every training
    /// vector takes its top-1 result there. The index's metric is the clustering metric.
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
        build_centroid_index: impl Fn(G) -> Q,
        search_params: &Q::SearchParams,
    ) -> G
    where
        D: MeanDataset + Sync,
        G: DatasetGrowable + Clone,
        for<'a> &'a D::Vector: Into<<G::Encoder as VectorEncoder>::InputVector<'a>>,
        Q: Index + Sync,
        Q::SearchParams: Sync,
        Q::Distance: From<f32>,
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
        build_centroid_index: impl Fn(G) -> Q,
        search_params: &Q::SearchParams,
        refine: impl Fn(&[u32], &mut D::Vector) + Sync,
    ) -> G
    where
        D: MeanDataset + Sync,
        G: DatasetGrowable + Clone,
        for<'a> &'a D::Vector: Into<<G::Encoder as VectorEncoder>::InputVector<'a>>,
        Q: Index + Sync,
        Q::SearchParams: Sync,
        Q::Distance: From<f32>,
        for<'a> &'a D::Vector: Into<Q::Query<'a>>,
    {
        self.fit::<D, G, Q::Distance>(
            training_dataset,
            k,
            weights,
            centroid_encoder,
            |centroids| {
                let index = build_centroid_index(centroids.clone());
                Self::assign_with_index(training_dataset, &index, search_params)
            },
            refine,
        )
    }

    /// The k-means loop shared by [`train_refined`](Self::train_refined) and
    /// [`train_with_index_refined`](Self::train_with_index_refined). `assign` maps the current centroids to one
    /// `(distance, centroid)` per training vector, under metric `M`; `refine` is applied to every
    /// updated centroid.
    fn fit<D, G, M>(
        &self,
        training_dataset: &D,
        k: usize,
        weights: Option<Vec<f32>>,
        centroid_encoder: G::Encoder,
        assign: impl Fn(&G) -> Vec<(f32, usize)>,
        refine: impl Fn(&[u32], &mut D::Vector) + Sync,
    ) -> G
    where
        D: MeanDataset + Sync,
        G: DatasetGrowable + Clone,
        M: Distance + From<f32>,
        for<'a> &'a D::Vector: Into<<G::Encoder as VectorEncoder>::InputVector<'a>>,
    {
        let n = training_dataset.len();
        let single = |i: usize| training_dataset.vector(i as u32);

        if n == k {
            if self.verbose {
                println!("WARNING: number of training data is equal to the number of clusters.");
            }
            let means: Vec<D::Vector> = (0..n).map(single).collect();
            return Self::to_dataset(&means, &centroid_encoder);
        }
        assert!(
            k < n,
            "k ({k}) must not exceed the number of training vectors ({n})"
        );

        if self.verbose {
            println!(
                "Clustering {} points ({} nnz, dim {}) to {} clusters, redo {} times, {} iterations",
                n,
                training_dataset.nnz(),
                training_dataset.input_dim(),
                k,
                self.n_redo,
                self.n_iter
            );
        }

        let mut best_obj: Option<M> = None;
        let mut best_centroids = G::new(centroid_encoder.clone());

        let w = weights.as_deref();
        // RNG used only for empty-cluster splitting.
        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        for redo in 0..self.n_redo {
            // Offset by `redo` so each restart draws a different initial sample
            // (a fixed `s + 1` would make every redo identical under a fixed seed).
            let mut init_rng = match self.seed {
                Some(s) => StdRng::seed_from_u64(s + 1 + redo as u64),
                None => StdRng::from_entropy(),
            };
            let init: Vec<D::Vector> = index::sample(&mut init_rng, n, k)
                .into_iter()
                .map(single)
                .collect();
            let mut centroids: G = Self::to_dataset(&init, &centroid_encoder);

            let mut average_imbalance_factor = 0.0;
            let mut total_splits = 0;

            for i in 0..self.n_iter {
                let t0 = Instant::now();
                let assignments = assign(&centroids);
                let assign_time = t0.elapsed();

                let t0 = Instant::now();
                let obj: f32 = assignments.iter().map(|&(v, _)| v).sum();

                let (n_split, histograms, means) = Self::update_and_split(
                    training_dataset,
                    w,
                    k,
                    &assignments,
                    &mut rng,
                    self.spherical,
                    &refine,
                );
                let new_centroids: G = Self::to_dataset(&means, &centroid_encoder);
                let imbalance_factor = Self::imbalance_factor(&histograms, k);
                let split_time = t0.elapsed();

                average_imbalance_factor += imbalance_factor;
                total_splits += n_split;

                let obj = M::from(obj);
                if best_obj.is_none_or(|best| obj < best) {
                    if self.verbose {
                        println!("New best objective: {} (keep new clusters)", obj.distance());
                    }
                    best_obj = Some(obj);
                    best_centroids = new_centroids.clone();
                }
                centroids = new_centroids;

                if self.verbose {
                    println!(
                        "Iteration {i}, imbalance: {imbalance_factor:.4}, splits: {n_split}, \
                         centroid nnz: {}, assign: {assign_time:.2?}, split: {split_time:.2?}",
                        centroids.nnz()
                    );
                }
            }

            if self.verbose {
                println!(
                    "Outer iteration {redo} -- average imbalance: {:.4}, splits: {total_splits}",
                    average_imbalance_factor / (self.n_iter + 1) as f32,
                );
            }
        }

        best_centroids
    }
}

pub struct KMeansBuilder {
    n_iter: usize,
    n_redo: usize,
    verbose: bool,
    max_points_per_centroid: usize,
    seed: Option<u64>,
    spherical: bool,
}

impl Default for KMeansBuilder {
    fn default() -> Self {
        KMeansBuilder {
            n_iter: 10,
            n_redo: 1,
            verbose: false,
            max_points_per_centroid: 256,
            seed: None,
            spherical: false,
        }
    }
}

impl KMeansBuilder {
    pub fn new() -> Self {
        KMeansBuilder::default()
    }

    pub fn n_iter(mut self, n_iter: usize) -> KMeansBuilder {
        self.n_iter = n_iter;
        self
    }

    pub fn n_redo(mut self, n_redo: usize) -> KMeansBuilder {
        self.n_redo = n_redo;
        self
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
    /// `None` (default) uses `from_entropy()` — different results every run.
    /// `Some(s)` uses `seed_from_u64(s)` — identical results for the same data and seed.
    pub fn seed(mut self, seed: Option<u64>) -> KMeansBuilder {
        self.seed = seed;
        self
    }

    /// Spherical k-means: L2-normalize centroids after every update step.
    pub fn spherical(mut self, spherical: bool) -> KMeansBuilder {
        self.spherical = spherical;
        self
    }

    pub fn build(self) -> KMeans {
        KMeans {
            n_iter: self.n_iter,
            n_redo: self.n_redo,
            verbose: self.verbose,
            seed: self.seed,
            spherical: self.spherical,
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

    // ---- centroid update: correctness ----------------------------------------

    #[test]
    fn update_and_split_matches_reference_means() {
        // Two non-empty clusters => no splits, so the returned centroids are
        // exactly the per-cluster means. Compare against hand-computed means.
        let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(2);
        let mut builder = PlainDenseDatasetGrowable::new(encoder);
        let pts = [
            [0.0f32, 0.0],
            [2.0, 0.0],
            [1.0, 3.0], // cluster 0 -> mean [1, 1]
            [10.0, 10.0],
            [12.0, 10.0],
            [11.0, 13.0], // cluster 1 -> mean [11, 11]
        ];
        for p in pts.iter() {
            builder.push(DenseVectorView::new(&p[..]));
        }
        let dataset: PlainDenseDataset<f32, SquaredEuclideanDistance> = builder.into();

        let assignments = vec![
            (0.0, 0usize),
            (0.0, 0),
            (0.0, 0),
            (0.0, 1),
            (0.0, 1),
            (0.0, 1),
        ];
        let mut rng = StdRng::seed_from_u64(0);
        let (n_splits, hist, centroids) = KMeans::update_and_split(
            &dataset,
            None,
            2,
            &assignments,
            &mut rng,
            false,
            &|_: &[u32], _: &mut _| {},
        );

        assert_eq!(n_splits, 0, "no empty clusters => no splits");
        assert_eq!(hist, vec![3.0, 3.0]);

        let c0 = centroids[0].values().to_vec();
        let c1 = centroids[1].values().to_vec();
        for (a, b) in c0.iter().zip([1.0f32, 1.0].iter()) {
            assert!((a - b).abs() < 1e-5, "cluster 0 mean {a} != {b}");
        }
        for (a, b) in c1.iter().zip([11.0f32, 11.0].iter()) {
            assert!((a - b).abs() < 1e-5, "cluster 1 mean {a} != {b}");
        }
    }

    #[test]
    fn update_and_split_fires_split_on_empty_cluster() {
        // k=3 but assignments only ever use clusters 0 and 1 => cluster 2 is empty
        // and must be split off a populated one.
        let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(2);
        let mut builder = PlainDenseDatasetGrowable::new(encoder);
        for p in [[0.0f32, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]].iter() {
            builder.push(DenseVectorView::new(&p[..]));
        }
        let dataset: PlainDenseDataset<f32, SquaredEuclideanDistance> = builder.into();
        let assignments = vec![(0.0, 0usize), (0.0, 0), (0.0, 1), (0.0, 1)];
        let mut rng = StdRng::seed_from_u64(0);
        let (n_splits, _hist, centroids) = KMeans::update_and_split(
            &dataset,
            None,
            3,
            &assignments,
            &mut rng,
            false,
            &|_: &[u32], _: &mut _| {},
        );
        assert_eq!(n_splits, 1, "one empty cluster => one split");
        assert_eq!(centroids.len(), 3);
    }

    #[test]
    fn update_deterministic_across_thread_counts() {
        // Non-integer random data so float reassociation would surface as bit diffs
        // (integer-valued data is exactly representable in f32 and hides reordering).
        let n = 2000usize;
        let d = 16usize;
        let k = 32usize;
        let mut data_rng = StdRng::seed_from_u64(42);
        let mut raw = vec![0.0f32; n * d];
        for x in raw.iter_mut() {
            *x = data_rng.gen_range(-1.0f32..1.0);
        }
        let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(d);
        let mut builder = PlainDenseDatasetGrowable::new(encoder);
        for i in 0..n {
            builder.push(DenseVectorView::new(&raw[i * d..(i + 1) * d]));
        }
        let dataset: PlainDenseDataset<f32, SquaredEuclideanDistance> = builder.into();
        let assignments: Vec<(f32, usize)> = (0..n).map(|i| (0.0f32, i % k)).collect();

        let mut reference: Option<Vec<u32>> = None;
        for t in [1usize, 2, 4, 8] {
            let bits = rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap()
                .install(|| {
                    let mut rng = StdRng::seed_from_u64(0);
                    let (_, _, centroids) = KMeans::update_and_split(
                        &dataset,
                        None,
                        k,
                        &assignments,
                        &mut rng,
                        false,
                        &|_: &[u32], _: &mut _| {},
                    );
                    (0..k)
                        .flat_map(|ci| {
                            centroids[ci]
                                .values()
                                .iter()
                                .map(|x| x.to_bits())
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<u32>>()
                });
            match &reference {
                None => reference = Some(bits),
                Some(ref_bits) => {
                    assert_eq!(&bits, ref_bits, "centroid bits differ at thread count {t}")
                }
            }
        }
    }

    #[test]
    fn update_weighted_means_match_reference() {
        // 4 points, d=2, k=2 with non-uniform weights.
        // cluster 0: pts 0,1 with weights 1,3 -> mean (1*[0,0] + 3*[2,0]) / 4 = [1.5, 0]
        // cluster 1: pts 2,3 with weights 2,2 -> mean (2*[10,10] + 2*[12,10]) / 4 = [11, 10]
        let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(2);
        let mut builder = PlainDenseDatasetGrowable::new(encoder);
        for p in [[0.0f32, 0.0], [2.0, 0.0], [10.0, 10.0], [12.0, 10.0]].iter() {
            builder.push(DenseVectorView::new(&p[..]));
        }
        let dataset: PlainDenseDataset<f32, SquaredEuclideanDistance> = builder.into();
        let weights = [1.0f32, 3.0, 2.0, 2.0];
        let assignments = vec![(0.0, 0usize), (0.0, 0), (0.0, 1), (0.0, 1)];
        let mut rng = StdRng::seed_from_u64(0);
        let (n_splits, hist, centroids) = KMeans::update_and_split(
            &dataset,
            Some(&weights),
            2,
            &assignments,
            &mut rng,
            false,
            &|_: &[u32], _: &mut _| {},
        );

        assert_eq!(n_splits, 0);
        assert_eq!(hist, vec![4.0, 4.0]);
        let c0 = centroids[0].values().to_vec();
        let c1 = centroids[1].values().to_vec();
        for (a, b) in c0.iter().zip([1.5f32, 0.0].iter()) {
            assert!((a - b).abs() < 1e-5, "cluster 0 weighted mean {a} != {b}");
        }
        for (a, b) in c1.iter().zip([11.0f32, 10.0].iter()) {
            assert!((a - b).abs() < 1e-5, "cluster 1 weighted mean {a} != {b}");
        }
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

    #[test]
    fn update_and_split_sparse_matches_reference_means() {
        // cluster 0: {0:1, 2:3} and {2:1, 5:2} -> {0:0.5, 2:2, 5:1}
        // cluster 1: {7:4} and {7:2, 8:6}      -> {7:3, 8:3}
        let dataset = sparse_dataset(
            &[
                (&[0, 2], &[1.0, 3.0]),
                (&[7], &[4.0]),
                (&[2, 5], &[1.0, 2.0]),
                (&[7, 8], &[2.0, 6.0]),
            ],
            10,
        );
        let assignments = vec![(0.0, 0usize), (0.0, 1), (0.0, 0), (0.0, 1)];
        let mut rng = StdRng::seed_from_u64(0);
        let (n_splits, hist, centroids) = KMeans::update_and_split(
            &dataset,
            None,
            2,
            &assignments,
            &mut rng,
            false,
            &|_: &[u32], _: &mut _| {},
        );

        assert_eq!(n_splits, 0);
        assert_eq!(hist, vec![2.0, 2.0]);
        let c0 = &centroids[0];
        assert_eq!(c0.components(), &[0, 2, 5]);
        assert_eq!(c0.values(), &[0.5, 2.0, 1.0]);
        let c1 = &centroids[1];
        assert_eq!(c1.components(), &[7, 8]);
        assert_eq!(c1.values(), &[3.0, 3.0]);
    }

    #[test]
    fn update_and_split_sparse_fires_split_on_empty_cluster() {
        let dataset = sparse_dataset(
            &[
                (&[0], &[1.0]),
                (&[0, 1], &[1.0, 1.0]),
                (&[5], &[1.0]),
                (&[5, 6], &[1.0, 1.0]),
            ],
            10,
        );
        let assignments = vec![(0.0, 0usize), (0.0, 0), (0.0, 1), (0.0, 1)];
        let mut rng = StdRng::seed_from_u64(0);
        let (n_splits, _, centroids) = KMeans::update_and_split(
            &dataset,
            None,
            3,
            &assignments,
            &mut rng,
            false,
            &|_: &[u32], _: &mut _| {},
        );
        assert_eq!(n_splits, 1);
        assert_eq!(centroids.len(), 3);
        assert!(!centroids[2].is_empty(), "split centroid must not be empty");
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
        KMeans::normalize_l2(dense.as_mut());
        assert_eq!(dense.values(), &[0.6, 0.8]);

        let mut zero = crate::DenseVectorOwned::new(vec![0.0f32, 0.0]);
        KMeans::normalize_l2(zero.as_mut());
        assert_eq!(zero.values(), &[0.0, 0.0]);

        let mut sparse = crate::SparseVectorOwned::new(vec![2u16, 7], vec![3.0f32, 4.0]);
        KMeans::normalize_l2(sparse.as_mut());
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

    #[test]
    fn update_sparse_deterministic_across_thread_counts() {
        let n = 2000usize;
        let dim = 500usize;
        let k = 32usize;
        let mut data_rng = StdRng::seed_from_u64(42);
        let mut vectors: Vec<(Vec<u16>, Vec<f32>)> = Vec::new();
        for _ in 0..n {
            let mut comps: Vec<u16> = (0..20).map(|_| data_rng.gen_range(0..dim as u16)).collect();
            comps.sort_unstable();
            comps.dedup();
            let vals = comps
                .iter()
                .map(|_| data_rng.gen_range(0.0f32..1.0))
                .collect();
            vectors.push((comps, vals));
        }
        let refs: Vec<(&[u16], &[f32])> = vectors
            .iter()
            .map(|(c, v)| (c.as_slice(), v.as_slice()))
            .collect();
        let dataset = sparse_dataset(&refs, dim);
        let assignments: Vec<(f32, usize)> = (0..n).map(|i| (0.0f32, i % k)).collect();

        let mut reference: Option<Vec<u32>> = None;
        for t in [1usize, 2, 4, 8] {
            let bits = rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap()
                .install(|| {
                    let mut rng = StdRng::seed_from_u64(0);
                    let (_, _, centroids) = KMeans::update_and_split(
                        &dataset,
                        None,
                        k,
                        &assignments,
                        &mut rng,
                        true,
                        &|_: &[u32], _: &mut _| {},
                    );
                    centroids
                        .iter()
                        .flat_map(|c| c.values().iter().map(|x| x.to_bits()).collect::<Vec<_>>())
                        .collect::<Vec<u32>>()
                });
            match &reference {
                None => reference = Some(bits),
                Some(ref_bits) => {
                    assert_eq!(&bits, ref_bits, "centroid bits differ at thread count {t}")
                }
            }
        }
    }
}
