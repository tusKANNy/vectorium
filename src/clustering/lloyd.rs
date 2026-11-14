//! Lloyd's algorithm, the default [`Solver`](super::Solver) of [`KMeans`](super::KMeans).
//!
//! Every iteration (one epoch) assigns all training vectors to their nearest centroid,
//! replaces each centroid by the (weighted) mean of its members and splits empty clusters off
//! populated ones. A run returns its final centroids; restarts are handled by
//! [`KMeans`](super::KMeans).

use super::index_assign::UpdateCtx;
use super::kmeans::{Assign, ClusterGroups, Run, group_means, imbalance_factor, normalize_l2};
use crate::core::dataset::{DatasetGrowable, MeanDataset};
use crate::core::distances::Distance;
use crate::core::vector_encoder::VectorEncoder;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

/// Runs Lloyd's algorithm for `run.epochs` iterations and returns the final centroids.
pub(super) fn fit<D, G>(
    run: &Run<'_, D, G>,
    assign: &mut Assign<'_, G>,
    refine: &(impl Fn(&[u32], &mut D::Vector) + Sync),
) -> G
where
    D: MeanDataset + Sync,
    G: DatasetGrowable,
    <G::Encoder as VectorEncoder>::Distance: Distance + From<f32>,
    for<'a> &'a D::Vector: Into<<G::Encoder as VectorEncoder>::InputVector<'a>>,
{
    let k = run.k;
    // Validated by `KMeansBuilder::build` to be a whole number.
    let n_iter = run.epochs as usize;
    if run.verbose {
        println!("Lloyd: {n_iter} iterations");
    }
    // RNG used only for empty-cluster splitting.
    let mut split_rng = StdRng::seed_from_u64(run.seed);
    let mut init_rng = StdRng::seed_from_u64(run.seed.wrapping_add(1));
    let mut centroids: G = run.to_dataset(&run.initial_centroids(&mut init_rng));

    let mut total_imbalance = 0.0;
    let mut total_splits = 0;
    let mut split_last = false;

    for i in 0..n_iter {
        let t0 = Instant::now();
        let ctx = UpdateCtx {
            update: i,
            epoch: i,
            epoch_start: true,
            centroids_jumped: split_last,
        };
        let assignments = assign(&centroids, ctx, None);
        let assign_time = t0.elapsed();

        let t0 = Instant::now();
        // The objective of the centroids this iteration assigned against (not of the updated
        // ones below).
        let obj: f32 = assignments.iter().map(|&(v, _)| v).sum();
        let (n_split, histograms, means) = update_and_split(
            run.dataset,
            run.weights,
            k,
            &assignments,
            &mut split_rng,
            run.spherical,
            refine,
        );
        centroids = run.to_dataset(&means);
        let imbalance = imbalance_factor(histograms.iter().map(|&h| h as f64));
        let split_time = t0.elapsed();

        total_imbalance += imbalance;
        total_splits += n_split;
        split_last = n_split > 0;

        if run.verbose {
            println!(
                "Iteration {i}, objective: {}, imbalance: {imbalance:.4}, splits: {n_split}, \
                 centroid nnz: {}, assign: {assign_time:.2?}, split: {split_time:.2?}",
                <G::Encoder as VectorEncoder>::Distance::from(obj).distance(),
                centroids.nnz()
            );
        }
    }

    if run.verbose {
        println!(
            "Average imbalance: {:.4}, splits: {total_splits}",
            total_imbalance / n_iter.max(1) as f64,
        );
    }
    centroids
}

/// Computes the centroids as the (weighted) means of the vectors assigned to each cluster,
/// passes each through `refine` together with its members, then splits empty clusters.
///
/// Returns the number of splits, the (weighted) size of each cluster and the new centroids.
pub(super) fn update_and_split<D>(
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
    let groups = ClusterGroups::new(None, assignments, k);
    let (mut centroids, mut histograms): (Vec<D::Vector>, Vec<f32>) =
        group_means(dataset, weights, &groups)
            .into_par_iter()
            .enumerate()
            .map(|(c, mean)| match mean {
                Some((mut centroid, count)) => {
                    refine(groups.members(c), &mut centroid);
                    (centroid, count)
                }
                // Overwritten by the split below.
                None => (dataset.mean(&[], weights), 0.0),
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
            normalize_l2(centroid.as_mut());
        }
    }

    (n_splits, histograms, centroids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::{KMeans, KMeansBuilder};
    use crate::core::vector::{DenseVectorView, SparseVectorView};
    use crate::distances::{DotProduct, SquaredEuclideanDistance};
    use crate::encoders::dense_scalar::PlainDenseQuantizer;
    use crate::encoders::sparse_scalar::PlainSparseQuantizer;
    use crate::{
        PlainDenseDataset, PlainDenseDatasetGrowable, PlainSparseDataset,
        PlainSparseDatasetGrowable,
    };
    use rand::Rng;
    use std::sync::atomic::{AtomicUsize, Ordering};

    type Plain = PlainDenseDataset<f32, SquaredEuclideanDistance>;
    type Grow = PlainDenseDatasetGrowable<f32, SquaredEuclideanDistance>;

    /// Overlapping clusters in `d` dimensions, so restarts genuinely differ.
    fn overlapping(n: usize, d: usize, seed: u64) -> Plain {
        let mut rng = StdRng::seed_from_u64(seed);
        let centers: Vec<f32> = (0..20 * d).map(|_| rng.gen_range(-3.0..3.0)).collect();
        let values: Vec<f32> = (0..n * d)
            .map(|i| centers[((i / d) % 20) * d + i % d] + rng.gen_range(-1.0f32..1.0))
            .collect();
        Plain::from_raw(values.into_boxed_slice(), n, PlainDenseQuantizer::new(d))
    }

    fn objective(ds: &Plain, c: &Grow) -> f64 {
        KMeans::compute_assignments(ds, c)
            .into_iter()
            .map(|(v, _)| v as f64)
            .sum()
    }

    /// With several restarts, the result is the restart whose *returned* centroids score best,
    /// which is the best of the same restarts run one at a time: restart `r` of seed `s` is a
    /// single run of seed `s + r`. Restarts belong to `KMeans`, so this holds for both solvers.
    #[test]
    fn restarts_compete_on_their_returned_centroids() {
        use crate::clustering::{EMeansParams, Solver};
        let ds = overlapping(4_000, 8, 5);
        let k = 16;
        for solver in [Solver::Lloyd, Solver::EMeans(EMeansParams::default())] {
            let train = |seed: u64, n_redo: usize| -> Grow {
                KMeansBuilder::new()
                    .solver(solver.clone())
                    .epochs(4.0)
                    .n_redo(n_redo)
                    .seed(Some(seed))
                    .build()
                    .train(&ds, k, None, PlainDenseQuantizer::new(8))
            };
            let singles: Vec<Grow> = (0..3).map(|r| train(10 + r, 1)).collect();
            let objectives: Vec<f64> = singles.iter().map(|c| objective(&ds, c)).collect();
            assert!(
                objectives.windows(2).any(|w| w[0] != w[1]),
                "{solver:?}: restarts should differ: {objectives:?}"
            );
            let best = (0..3)
                .min_by(|&a, &b| objectives[a].total_cmp(&objectives[b]))
                .unwrap();
            assert_eq!(train(10, 3).values(), singles[best].values(), "{solver:?}");
        }
    }

    /// Each run returns its last update, even when an earlier iteration scored better. Two
    /// well-separated blobs whose third coordinate is 0; a `refine` sets every centroid's third
    /// coordinate to its update number (1-based), so the objective gets worse at every
    /// iteration. The result must still be the last update's centroids.
    #[test]
    fn a_run_returns_its_last_update() {
        let mut rng = StdRng::seed_from_u64(2);
        let mut values = Vec::new();
        for i in 0..400 {
            let base = if i % 2 == 0 { 0.0 } else { 100.0 };
            values.extend([
                base + rng.gen_range(-1.0f32..1.0),
                base + rng.gen_range(-1.0f32..1.0),
                0.0,
            ]);
        }
        let ds = Plain::from_raw(values.into_boxed_slice(), 400, PlainDenseQuantizer::new(3));
        let (k, n_iter) = (2, 5);
        let calls = AtomicUsize::new(0);
        let tag = |_: &[u32], c: &mut crate::DenseVectorOwned<f32>| {
            let update = calls.fetch_add(1, Ordering::Relaxed) / k;
            c.as_mut()[2] = (update + 1) as f32;
        };
        let c: Grow = KMeansBuilder::new()
            .epochs(n_iter as f64)
            .seed(Some(1))
            .build()
            .train_refined(&ds, k, None, PlainDenseQuantizer::new(3), tag);
        assert_eq!(
            calls.load(Ordering::Relaxed),
            k * n_iter,
            "no empty clusters"
        );
        assert!(
            c.values()
                .chunks_exact(3)
                .all(|row| row[2] == n_iter as f32),
            "{:?}",
            c.values()
        );
    }

    /// In spherical mode the initial centroids are unit-norm, so a long training vector drawn
    /// as an initial centroid does not attract every vector under the dot product.
    #[test]
    fn spherical_normalizes_initial_centroids() {
        use crate::clustering::kmeans::Run;
        let ds = PlainDenseDataset::<f32, DotProduct>::from_raw(
            vec![100.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.1].into_boxed_slice(),
            4,
            PlainDenseQuantizer::new(2),
        );
        let encoder = PlainDenseQuantizer::new(2);
        let run = Run::<_, PlainDenseDatasetGrowable<f32, DotProduct>> {
            dataset: &ds,
            k: 4,
            weights: None,
            encoder: &encoder,
            epochs: 1.0,
            seed: 0,
            spherical: true,
            verbose: false,
        };
        // k = n: every vector, the long one included, is an initial centroid.
        for c in run.initial_centroids(&mut StdRng::seed_from_u64(0)) {
            let row = c.values();
            let norm = (row[0] * row[0] + row[1] * row[1]).sqrt();
            assert!((norm - 1.0).abs() < 1e-6, "initial centroid norm {norm}");
        }
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
        let (n_splits, hist, centroids) = update_and_split(
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
        let (n_splits, _hist, centroids) = update_and_split(
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
                    let (_, _, centroids) = update_and_split(
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
        let (n_splits, hist, centroids) = update_and_split(
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
        let (n_splits, hist, centroids) = update_and_split(
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
        let (n_splits, _, centroids) = update_and_split(
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
                    let (_, _, centroids) = update_and_split(
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
