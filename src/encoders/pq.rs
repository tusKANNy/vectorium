use std::marker::PhantomData;

use serde::{Deserialize, Serialize};

use crate::PlainDenseDataset;
use crate::PlainDenseQuantizer;
use crate::SpaceUsage;
use crate::clustering::KMeansBuilder;
use crate::core::vector::{DenseVectorOwned, DenseVectorView};
use crate::core::vector_encoder::{DenseVectorEncoder, QueryEvaluator, VectorEncoder};
use crate::distances::{Distance, DotProduct, SquaredEuclideanDistance};
use crate::{Dataset, DatasetGrowable, PlainDenseDatasetGrowable};

/// Minimum dataset size below which no sampling is performed for PQ training
const PQ_TRAIN_SAMPLING_MIN_SIZE: usize = 1_000_000;
/// Fraction of dataset to sample when dataset exceeds PQ_TRAIN_SAMPLING_MIN_SIZE
const PQ_TRAIN_SAMPLE_RATE: f64 = 0.10;

pub trait ProductQuantizerDistance: Distance + From<f32> {
    fn compute_query_distance_table<const M: usize>(
        encoder: &ProductQuantizer<M, Self>,
        query: DenseVectorView<'_, f32>,
    ) -> Vec<f32>;
}

impl ProductQuantizerDistance for SquaredEuclideanDistance {
    fn compute_query_distance_table<const M: usize>(
        encoder: &ProductQuantizer<M, Self>,
        query: DenseVectorView<'_, f32>,
    ) -> Vec<f32> {
        encoder.compute_euclidean_distance_table(query)
    }
}

impl ProductQuantizerDistance for DotProduct {
    fn compute_query_distance_table<const M: usize>(
        encoder: &ProductQuantizer<M, Self>,
        query: DenseVectorView<'_, f32>,
    ) -> Vec<f32> {
        encoder.compute_dot_product_table(query)
    }
}

/// Number of centroids per subspace (always 256 = 2^8)
const KSUB: usize = 256;
/// Centroids processed per tile in the distance-table build. Must divide `KSUB`.
///
/// The `TILE` accumulators stay in registers across all `dsub` dimension passes,
/// so each table element is written exactly once rather than read-modify-written
/// on every pass. The width balances two costs: the per-(tile, dimension) setup
/// -- broadcasting `q_k`, address computation, loop bookkeeping -- is paid once
/// per step regardless of `TILE`, so a narrow tile amortizes it over too little
/// arithmetic, while a wide one raises register pressure.
const TILE: usize = 32;
const DEFAULT_KMEANS_N_ITER: usize = 25;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct ProductQuantizer<const M: usize, D>
where
    D: ProductQuantizerDistance,
{
    d: usize,
    dsub: usize,
    /// Flat centroid buffer, SoA layout: [M × dsub × KSUB].
    /// Subspace m occupies centroids[m*dsub*KSUB .. (m+1)*dsub*KSUB].
    /// Dimension k of subspace m: centroids[m*dsub*KSUB + k*KSUB .. m*dsub*KSUB + (k+1)*KSUB].
    /// Centroid i, subspace m, dimension k: centroids[m*dsub*KSUB + k*KSUB + i].
    ///
    /// SoA layout makes distance table build trivially auto-vectorizable: for each
    /// dimension k, the inner loop over KSUB centroid values is a contiguous f32
    /// slice with no stride.
    centroids: Box<[f32]>,
    #[serde(skip)]
    _distance: PhantomData<D>,
}

impl<const M: usize, D> ProductQuantizer<M, D>
where
    D: ProductQuantizerDistance,
{
    #[inline]
    pub fn train<Ds: crate::ScalarDenseSupportedDistance>(
        training_data: &PlainDenseDataset<f32, Ds>,
    ) -> Self {
        Self::train_with_kmeans_options(training_data, DEFAULT_KMEANS_N_ITER, None)
    }

    /// Train a ProductQuantizer while controlling the inner k-means seed and iteration count.
    /// This keeps the outer PQ training API unchanged for existing callers while allowing
    /// reproducible bin-specific runs to match older pipelines more closely.
    #[inline]
    pub fn train_with_kmeans_options<Ds: crate::ScalarDenseSupportedDistance>(
        training_data: &PlainDenseDataset<f32, Ds>,
        n_iter: usize,
        seed: Option<u64>,
    ) -> Self {
        let d = training_data.output_dim();
        assert_eq!(M % 4, 0, "M ({}) is not divisible by 4", M);
        assert_eq!(d % M, 0, "d ({}) is not divisible by M ({})", d, M);
        let dsub = d / M;
        let centroids = Self::train_centroids(training_data, dsub, n_iter, seed).into_boxed_slice();
        Self {
            d,
            dsub,
            centroids,
            _distance: PhantomData,
        }
    }

    /// Returns a flat Vec<f32> of length M * dsub * KSUB in SoA layout: [M × dsub × KSUB].
    /// KMeans output is AoS [KSUB × dsub]; we transpose each subspace to SoA [dsub × KSUB]
    /// so that the distance table build inner loop is over contiguous KSUB f32 values.
    fn train_centroids<Ds: crate::ScalarDenseSupportedDistance>(
        training_data: &PlainDenseDataset<f32, Ds>,
        dsub: usize,
        n_iter: usize,
        seed: Option<u64>,
    ) -> Vec<f32> {
        use rayon::prelude::*;

        println!("Running K-Means for {} subspaces", M);

        // Each rayon task trains one subspace, gets AoS centroids from KMeans,
        // and transposes them to SoA before returning.
        let parts: Vec<Vec<f32>> = (0..M)
            .into_par_iter()
            .map(|m| {
                let quantizer = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(dsub);
                let mut sub_dataset =
                    PlainDenseDatasetGrowable::<f32, SquaredEuclideanDistance>::with_capacity(
                        quantizer,
                        training_data.len(),
                    );

                for vector in training_data.iter() {
                    let start = m * dsub;
                    let end = start + dsub;
                    let sub_vector = DenseVectorView::new(&vector.values()[start..end]);
                    sub_dataset.push(sub_vector);
                }

                let kmeans = KMeansBuilder::new().n_iter(n_iter).seed(seed).build();
                let trained: PlainDenseDatasetGrowable<f32, SquaredEuclideanDistance> =
                    kmeans.train(&sub_dataset, KSUB, None, PlainDenseQuantizer::new(dsub));
                // KMeans returns AoS: [c0_d0, c0_d1, .., c0_{dsub-1}, c1_d0, ..]
                // Transpose to SoA: [d0_c0, d0_c1, .., d0_{KSUB-1}, d1_c0, ..]
                let aos = trained.values();
                let mut soa = vec![0.0_f32; KSUB * dsub];
                for c in 0..KSUB {
                    for d in 0..dsub {
                        soa[d * KSUB + c] = aos[c * dsub + d];
                    }
                }
                soa
            })
            .collect();

        println!("K-Means finished");

        let mut flat = Vec::with_capacity(M * dsub * KSUB);
        for part in parts {
            flat.extend_from_slice(&part);
        }
        flat
    }

    fn sample_random_dataset<Ds: crate::ScalarDenseSupportedDistance>(
        dataset: &PlainDenseDataset<f32, Ds>,
        sample_size: usize,
    ) -> PlainDenseDataset<f32, Ds> {
        use rand::seq::SliceRandom;
        let d = dataset.output_dim();
        let dataset_size = dataset.len();

        // Create random indices
        let mut indices: Vec<usize> = (0..dataset_size).collect();
        let mut rng = rand::thread_rng();
        indices.partial_shuffle(&mut rng, sample_size);

        // Extract sampled vectors
        let mut values = Vec::with_capacity(sample_size * d);
        for &idx in &indices[..sample_size] {
            let start = idx * d;
            values.extend_from_slice(&dataset.values()[start..start + d]);
        }

        PlainDenseDataset::<f32, Ds>::from_raw(
            values.into_boxed_slice(),
            sample_size,
            dataset.encoder().clone(),
        )
    }

    /// [`sample_random_dataset`](Self::sample_random_dataset) from a half-precision source,
    /// upcasting the sampled rows to `f32` so k-means trains exactly as it does for an `f32`
    /// collection.
    ///
    /// When `sample_size` covers the whole dataset the rows are taken in their original order
    /// rather than shuffled: k-means seeding consumes the leading rows, so preserving the order
    /// keeps this path comparable with the `f32` one.
    fn sample_random_dataset_upcast<Ds: crate::ScalarDenseSupportedDistance>(
        dataset: &PlainDenseDataset<half::f16, Ds>,
        sample_size: usize,
    ) -> PlainDenseDataset<f32, Ds> {
        use rand::seq::SliceRandom;
        let d = dataset.output_dim();
        let dataset_size = dataset.len();
        let sample_size = sample_size.min(dataset_size);

        let indices: Vec<usize> = if sample_size == dataset_size {
            (0..dataset_size).collect()
        } else {
            let mut indices: Vec<usize> = (0..dataset_size).collect();
            let mut rng = rand::thread_rng();
            indices.partial_shuffle(&mut rng, sample_size);
            indices
        };

        let mut values = Vec::with_capacity(sample_size * d);
        for &idx in &indices[..sample_size] {
            let start = idx * d;
            values.extend(
                dataset.values()[start..start + d]
                    .iter()
                    .map(|v| v.to_f32()),
            );
        }

        PlainDenseDataset::<f32, Ds>::from_raw(
            values.into_boxed_slice(),
            sample_size,
            PlainDenseQuantizer::<f32, Ds>::new(d),
        )
    }

    /// Compute the training sample size based on the automatic sampling strategy.
    ///
    /// - If dataset size ≤ PQ_SAMPLING_MIN_SIZE: no sampling (returns None)
    /// - If dataset size > PQ_SAMPLING_MIN_SIZE: sample max(PQ_SAMPLE_RATE * size, PQ_SAMPLING_MIN_SIZE)
    fn compute_training_sample_size(dataset_len: usize) -> Option<usize> {
        if dataset_len <= PQ_TRAIN_SAMPLING_MIN_SIZE {
            None
        } else {
            let sample_by_rate = (dataset_len as f64 * PQ_TRAIN_SAMPLE_RATE) as usize;
            Some(sample_by_rate.max(PQ_TRAIN_SAMPLING_MIN_SIZE))
        }
    }

    /// Build a `ProductQuantizer` from pre-trained centroids in natural AoS layout.
    ///
    /// `centroids_aos` must have length `M * KSUB * (d / M)` with layout
    /// `[M × KSUB × dsub]`: centroid `i` of subspace `m` at
    /// `centroids_aos[m*KSUB*dsub + i*dsub .. m*KSUB*dsub + (i+1)*dsub]`.
    /// Internally the buffer is transposed to SoA for efficient distance table builds.
    #[inline]
    pub fn from_pretrained(d: usize, centroids_aos: Vec<f32>) -> Self {
        assert_eq!(M % 4, 0, "M ({}) is not divisible by 4", M);
        assert_eq!(d % M, 0, "d ({}) is not divisible by M ({})", d, M);
        let dsub = d / M;
        assert_eq!(
            centroids_aos.len(),
            M * KSUB * dsub,
            "centroids length must equal M * KSUB * dsub = {} * {} * {} = {}",
            M,
            KSUB,
            dsub,
            M * KSUB * dsub
        );
        // Transpose each subspace from AoS [KSUB × dsub] to SoA [dsub × KSUB].
        let mut soa = vec![0.0_f32; M * dsub * KSUB];
        for m in 0..M {
            let aos_sub = &centroids_aos[m * KSUB * dsub..];
            let soa_sub = &mut soa[m * dsub * KSUB..];
            for c in 0..KSUB {
                for k in 0..dsub {
                    soa_sub[k * KSUB + c] = aos_sub[c * dsub + k];
                }
            }
        }
        Self {
            d,
            dsub,
            centroids: soa.into_boxed_slice(),
            _distance: PhantomData,
        }
    }

    /// Return centroids in AoS layout `[M × KSUB × dsub]`:
    /// `aos[m * KSUB * dsub + k * dsub + d]` = centroid `k`, dimension `d`, subspace `m`.
    ///
    /// Transposes the internal SoA buffer. Use this to save centroids in the format
    /// expected by `MultiVecTwoLevelProductQuantizer::from_pretrained` and the offline loaders.
    pub fn centroids_as_aos(&self) -> Vec<f32> {
        let dsub = self.dsub;
        let mut aos = vec![0.0f32; M * KSUB * dsub];
        for m in 0..M {
            let soa = self.subspace_centroids(m); // [d * KSUB .. (d+1)*KSUB] for each d
            for k in 0..KSUB {
                for d in 0..dsub {
                    aos[m * KSUB * dsub + k * dsub + d] = soa[d * KSUB + k];
                }
            }
        }
        aos
    }

    #[inline]
    pub fn input_dim(&self) -> usize {
        self.d
    }

    #[inline]
    pub fn output_dim(&self) -> usize {
        self.code_size()
    }

    #[inline]
    pub fn ksub(&self) -> usize {
        KSUB
    }

    #[inline]
    pub fn dsub(&self) -> usize {
        self.dsub
    }

    #[inline]
    pub fn d(&self) -> usize {
        self.d
    }

    /// Code size in bytes (one byte per subspace)
    #[inline]
    pub fn code_size(&self) -> usize {
        M
    }

    /// Returns the SoA centroid slice for subspace `m`: length `dsub * KSUB`.
    /// Layout within the slice: dimension `k` occupies `[k*KSUB .. (k+1)*KSUB]`.
    #[inline]
    fn subspace_centroids(&self, m: usize) -> &[f32] {
        let start = m * self.dsub * KSUB;
        &self.centroids[start..start + self.dsub * KSUB]
    }

    fn compute_euclidean_distance_table(&self, query: DenseVectorView<'_, f32>) -> Vec<f32> {
        assert_eq!(query.len(), self.d());
        let dsub = self.dsub;
        let mut table = vec![0.0_f32; KSUB * M];

        // Tiled loop: process TILE centroids at a time.
        // The TILE partial accumulators `acc` are kept in registers across all
        // dsub dimension passes, so table_sub is written exactly once per
        // element — no read-modify-write. See TILE's docs for the width choice.
        for (m, (table_sub, query_sub)) in table
            .chunks_mut(KSUB)
            .zip(query.values().chunks(dsub))
            .enumerate()
        {
            let centroids = self.subspace_centroids(m);
            for i in (0..KSUB).step_by(TILE) {
                let mut acc = [0.0_f32; TILE];
                for (k, &q_k) in query_sub.iter().enumerate() {
                    let col = &centroids[k * KSUB + i..k * KSUB + i + TILE];
                    for (a, &c) in acc.iter_mut().zip(col) {
                        let d = q_k - c;
                        *a += d * d;
                    }
                }
                table_sub[i..i + TILE].copy_from_slice(&acc);
            }
        }
        table
    }

    fn compute_dot_product_table(&self, query: DenseVectorView<'_, f32>) -> Vec<f32> {
        assert_eq!(query.len(), self.d());
        let dsub = self.dsub;
        let mut table = vec![0.0_f32; KSUB * M];

        // Same tiling strategy as the euclidean table: TILE-wide accumulators in
        // registers across all dsub passes → single write per table element.
        for (m, (table_sub, query_sub)) in table
            .chunks_mut(KSUB)
            .zip(query.values().chunks(dsub))
            .enumerate()
        {
            let centroids = self.subspace_centroids(m);
            for i in (0..KSUB).step_by(TILE) {
                let mut acc = [0.0_f32; TILE];
                for (k, &q_k) in query_sub.iter().enumerate() {
                    let col = &centroids[k * KSUB + i..k * KSUB + i + TILE];
                    for (a, &c) in acc.iter_mut().zip(col) {
                        *a += q_k * c;
                    }
                }
                table_sub[i..i + TILE].copy_from_slice(&acc);
            }
        }
        table
    }

    fn compute_distance(&self, distance_table: &[f32], code: DenseVectorView<'_, u8>) -> f32 {
        assert_eq!(code.len(), M);
        // 4 independent accumulators break the sequential dependency chain,
        // giving the CPU 4-way ILP on the table lookups.
        // M is always divisible by 4 (asserted at train/from_pretrained time).
        let code_bytes = code.values();
        let mut acc = [0.0_f32; 4];
        let mut ptr = 0_usize;
        for subcode in code_bytes.as_chunks::<4>().0 {
            unsafe {
                acc[0] += *distance_table.get_unchecked(ptr + subcode[0] as usize);
                acc[1] += *distance_table.get_unchecked(ptr + KSUB + subcode[1] as usize);
                acc[2] += *distance_table.get_unchecked(ptr + 2 * KSUB + subcode[2] as usize);
                acc[3] += *distance_table.get_unchecked(ptr + 3 * KSUB + subcode[3] as usize);
            }
            ptr += 4 * KSUB;
        }
        acc[0] + acc[1] + acc[2] + acc[3]
    }

    fn decode_vector(&self, vector: DenseVectorView<'_, u8>) -> DenseVectorOwned<f32> {
        assert_eq!(vector.len(), M);
        let mut decoded = Vec::with_capacity(self.d());
        for (sub_id, &code_byte) in vector.values().iter().enumerate() {
            // SoA: centroid i, dim k → centroids[k * KSUB + i]
            let centroids = self.subspace_centroids(sub_id);
            let i = code_byte as usize;
            for k in 0..self.dsub {
                decoded.push(centroids[k * KSUB + i]);
            }
        }
        DenseVectorOwned::new(decoded)
    }
}

impl<const M: usize, D> SpaceUsage for ProductQuantizer<M, D>
where
    D: ProductQuantizerDistance,
{
    fn space_usage_bytes(&self) -> usize {
        2 * std::mem::size_of::<usize>() + self.centroids.len() * std::mem::size_of::<f32>()
    }
}

impl<const M: usize, D> DenseVectorEncoder for ProductQuantizer<M, D>
where
    D: ProductQuantizerDistance,
{
    type InputValueType = f32;
    type OutputValueType = u8;

    fn decode_vector<'a>(
        &self,
        encoded: DenseVectorView<'a, Self::OutputValueType>,
    ) -> DenseVectorOwned<f32> {
        self.decode_vector(encoded)
    }

    #[inline]
    fn push_encoded<'a, OutputContainer>(
        &self,
        input: DenseVectorView<'a, Self::InputValueType>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<Self::OutputValueType>,
    {
        assert_eq!(input.len(), self.d());
        for (sub_id, query_sub) in input.values().chunks(self.dsub).enumerate() {
            let centroids = self.subspace_centroids(sub_id);
            // Accumulate per-centroid squared distances dimension by dimension.
            // KSUB=256 is a compile-time constant so `dists` lives on the stack.
            let mut dists = [0.0_f32; KSUB];
            let q0 = query_sub[0];
            let col0 = &centroids[0..KSUB];
            for (acc, &c_val) in dists.iter_mut().zip(col0) {
                let diff = q0 - c_val;
                *acc = diff * diff;
            }
            for (k, &q_k) in query_sub.iter().enumerate().skip(1) {
                let col = &centroids[k * KSUB..(k + 1) * KSUB];
                for (acc, &c_val) in dists.iter_mut().zip(col) {
                    let diff = q_k - c_val;
                    *acc += diff * diff;
                }
            }
            let best_idx = dists
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u8)
                .unwrap_or(0);
            output.extend(std::iter::once(best_idx));
        }
    }
}

/// The codebooks and codes do not depend on the metric, only the query distance table does.
impl<const M: usize, D, D2> crate::core::vector_encoder::WithDistance<D2> for ProductQuantizer<M, D>
where
    D: ProductQuantizerDistance,
    D2: ProductQuantizerDistance,
{
    type Output = ProductQuantizer<M, D2>;

    fn with_distance(&self) -> Self::Output {
        ProductQuantizer {
            d: self.d,
            dsub: self.dsub,
            centroids: self.centroids.clone(),
            _distance: PhantomData,
        }
    }
}

impl<const M: usize, D> VectorEncoder for ProductQuantizer<M, D>
where
    D: ProductQuantizerDistance,
{
    type Distance = D;
    type InputVector<'a> = DenseVectorView<'a, f32>;
    type QueryVector<'q> = DenseVectorView<'q, f32>;
    type EncodedVector<'a> = DenseVectorView<'a, u8>;
    type Evaluator<'e>
        = ProductQuantizerQueryEvaluator<'e, M, D>
    where
        Self: 'e;

    type QueryParams = ();

    fn query_evaluator<'e>(
        &'e self,
        query: Self::QueryVector<'_>,
        _params: &(),
    ) -> Self::Evaluator<'e> {
        assert_eq!(query.len(), self.d());
        ProductQuantizerQueryEvaluator::new(self, query)
    }

    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = self.decode_vector(vector);
        ProductQuantizerQueryEvaluator::new(self, decoded.as_view())
    }

    fn input_dim(&self) -> usize {
        self.d()
    }

    fn output_dim(&self) -> usize {
        self.code_size()
    }
}

pub struct ProductQuantizerQueryEvaluator<'a, const M: usize, D>
where
    D: ProductQuantizerDistance,
{
    encoder: &'a ProductQuantizer<M, D>,
    distance_table: Vec<f32>,
}

impl<'a, const M: usize, D> ProductQuantizerQueryEvaluator<'a, M, D>
where
    D: ProductQuantizerDistance,
{
    fn new(encoder: &'a ProductQuantizer<M, D>, query: DenseVectorView<'_, f32>) -> Self {
        let table = D::compute_query_distance_table(encoder, query);
        Self {
            encoder,
            distance_table: table,
        }
    }
}

impl<'a, const M: usize, D> QueryEvaluator<DenseVectorView<'_, u8>>
    for ProductQuantizerQueryEvaluator<'a, M, D>
where
    D: ProductQuantizerDistance,
{
    type Distance = D;

    fn compute_distance(&self, vector: DenseVectorView<'_, u8>) -> Self::Distance {
        let raw = self.encoder.compute_distance(&self.distance_table, vector);
        D::from(raw)
    }
}

use crate::dataset::ConvertFrom;

/// Train a product quantizer on `dataset` (sampling the training set on large inputs) and encode
/// every vector.
///
/// The source is taken by **reference**, so the plain vectors survive the call — a caller that
/// still needs them (to compute ground truth, or to encode a second time at another `M`) does not
/// have to clone the dataset first.
impl<const M: usize, D, Ds> ConvertFrom<&PlainDenseDataset<f32, Ds>>
    for crate::DenseDataset<ProductQuantizer<M, D>>
where
    D: ProductQuantizerDistance + 'static,
    Ds: crate::ScalarDenseSupportedDistance,
{
    type Config = ();

    fn convert_from(dataset: &PlainDenseDataset<f32, Ds>, _config: ()) -> Self {
        let sample_size = ProductQuantizer::<M, D>::compute_training_sample_size(dataset.len());

        let pq_encoder = match sample_size {
            Some(size) => {
                println!(
                    "Sampling {} vectors from {} for PQ training",
                    size,
                    dataset.len()
                );
                let sample = ProductQuantizer::<M, D>::sample_random_dataset(dataset, size);
                ProductQuantizer::<M, D>::train(&sample)
            }
            // Small enough to train on directly — no copy of any kind.
            None => ProductQuantizer::<M, D>::train(dataset),
        };

        crate::DenseDataset::<ProductQuantizer<M, D>>::from_flat_par(
            pq_encoder,
            dataset.values(),
            dataset.len(),
        )
    }
}

/// Same, from a source held at half precision.
///
/// Lets a caller that already has the collection as `f16` — which is how the graph indexes store
/// it — encode without first materializing an `f32` copy. k-means still trains in `f32`, on an
/// upcast copy of the training sample only; the encode pass upcasts one vector at a time.
impl<const M: usize, D, Ds> ConvertFrom<&PlainDenseDataset<half::f16, Ds>>
    for crate::DenseDataset<ProductQuantizer<M, D>>
where
    D: ProductQuantizerDistance + 'static,
    Ds: crate::ScalarDenseSupportedDistance,
{
    type Config = ();

    fn convert_from(dataset: &PlainDenseDataset<half::f16, Ds>, _config: ()) -> Self {
        let sample_size = ProductQuantizer::<M, D>::compute_training_sample_size(dataset.len());

        let training_data: PlainDenseDataset<f32, Ds> = match sample_size {
            Some(size) => {
                println!(
                    "Sampling {} vectors from {} for PQ training",
                    size,
                    dataset.len()
                );
                ProductQuantizer::<M, D>::sample_random_dataset_upcast(dataset, size)
            }
            // Small enough to train on directly; the upcast copy is bounded by
            // `PQ_TRAIN_SAMPLING_MIN_SIZE` vectors.
            None => ProductQuantizer::<M, D>::sample_random_dataset_upcast(dataset, dataset.len()),
        };
        let pq_encoder = ProductQuantizer::<M, D>::train(&training_data);

        crate::DenseDataset::<ProductQuantizer<M, D>>::from_flat_par_upcast(
            pq_encoder,
            dataset.values(),
            dataset.len(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const M_TEST: usize = 4;
    const D_TEST: usize = 8; // dsub = 2

    fn make_pq_and_data() -> (
        ProductQuantizer<M_TEST, SquaredEuclideanDistance>,
        Vec<Vec<f32>>,
    ) {
        // Build 300 random-ish training vectors dimension 8.
        let seed_vecs: Vec<Vec<f32>> = (0u32..300)
            .map(|i| {
                (0..D_TEST)
                    .map(|j| ((i * 17 + j as u32 * 31) % 100) as f32 / 100.0)
                    .collect()
            })
            .collect();

        let quantizer = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(D_TEST);
        let mut training_data =
            PlainDenseDatasetGrowable::<f32, SquaredEuclideanDistance>::with_capacity(
                quantizer,
                seed_vecs.len(),
            );
        for v in &seed_vecs {
            training_data.push(DenseVectorView::new(v));
        }
        let training_data: PlainDenseDataset<f32, SquaredEuclideanDistance> = training_data.into();

        let pq = ProductQuantizer::<M_TEST, SquaredEuclideanDistance>::train(&training_data);
        (pq, seed_vecs)
    }

    /// The table-build tile loop steps by TILE with no remainder handling, so
    /// TILE must divide KSUB exactly.
    #[test]
    fn tile_divides_ksub() {
        assert_eq!(KSUB % TILE, 0, "TILE ({TILE}) must divide KSUB ({KSUB})");
    }

    /// The tiled table build must agree with a straightforward untiled
    /// reference, for both metrics. This is what pins the tile width as a pure
    /// performance knob with no effect on results.
    #[test]
    fn tiled_table_matches_untiled_reference() {
        let (pq, vecs) = make_pq_and_data();
        let query = DenseVectorView::new(&vecs[2]);

        let dsub = pq.dsub();
        let l2 = pq.compute_euclidean_distance_table(query);
        for m in 0..M_TEST {
            let centroids = pq.subspace_centroids(m);
            let q_sub = &vecs[2][m * dsub..(m + 1) * dsub];
            for i in 0..KSUB {
                let mut want = 0.0_f32;
                for (k, &q_k) in q_sub.iter().enumerate() {
                    let d = q_k - centroids[k * KSUB + i];
                    want += d * d;
                }
                let got = l2[m * KSUB + i];
                assert!(
                    (got - want).abs() <= 1e-4 * want.abs().max(1e-3),
                    "euclidean table m={m} i={i}: got {got}, want {want}"
                );
            }
        }

        let pq_dot =
            ProductQuantizer::<M_TEST, DotProduct>::from_pretrained(D_TEST, pq.centroids_as_aos());
        let dot = pq_dot.compute_dot_product_table(query);
        for m in 0..M_TEST {
            let centroids = pq_dot.subspace_centroids(m);
            let q_sub = &vecs[2][m * dsub..(m + 1) * dsub];
            for i in 0..KSUB {
                let mut want = 0.0_f32;
                for (k, &q_k) in q_sub.iter().enumerate() {
                    want += q_k * centroids[k * KSUB + i];
                }
                let got = dot[m * KSUB + i];
                assert!(
                    (got - want).abs() <= 1e-4 * want.abs().max(1e-3),
                    "dot table m={m} i={i}: got {got}, want {want}"
                );
            }
        }
    }

    /// Building a PQ dataset borrows its source and accepts a source of either metric. The
    /// assertions on `plain` *after* the conversions are the point: they would not compile if the
    /// source were moved.
    #[test]
    fn convert_from_borrows_the_source_and_accepts_either_source_metric() {
        use crate::dataset::ConvertInto;

        let (_, vecs) = make_pq_and_data();
        let quantizer = PlainDenseQuantizer::<f32, DotProduct>::new(D_TEST);
        let mut growable =
            PlainDenseDatasetGrowable::<f32, DotProduct>::with_capacity(quantizer, vecs.len());
        for v in &vecs {
            growable.push(DenseVectorView::new(v));
        }
        let plain: PlainDenseDataset<f32, DotProduct> = growable.into();

        // An inner-product source.
        let pq_ip: crate::DenseDataset<ProductQuantizer<M_TEST, DotProduct>> =
            (&plain).convert_into(());
        // The same source again, for the other target metric. Only possible because `plain` lives.
        let pq_l2: crate::DenseDataset<ProductQuantizer<M_TEST, SquaredEuclideanDistance>> =
            (&plain).convert_into(());

        assert_eq!(plain.len(), vecs.len());
        assert_eq!(pq_ip.len(), vecs.len());
        assert_eq!(pq_l2.len(), vecs.len());
        assert_eq!(pq_ip.encoder().output_dim(), M_TEST);
        assert_eq!(pq_l2.encoder().output_dim(), M_TEST);
    }

    /// PQ's k-means runs in squared-Euclidean space over sub-vectors it builds itself, so the
    /// source dataset's metric never reaches training. This asserts that at the bit level: same
    /// vectors, two source metrics, one k-means seed, byte-identical centroids. If the metric ever
    /// did leak into training, the inner-product path would drift from the Euclidean one and this
    /// test fails.
    #[test]
    fn training_is_independent_of_the_source_metric() {
        let (_, vecs) = make_pq_and_data();

        let mut l2 = PlainDenseDatasetGrowable::<f32, SquaredEuclideanDistance>::with_capacity(
            PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(D_TEST),
            vecs.len(),
        );
        let mut ip = PlainDenseDatasetGrowable::<f32, DotProduct>::with_capacity(
            PlainDenseQuantizer::<f32, DotProduct>::new(D_TEST),
            vecs.len(),
        );
        for v in &vecs {
            l2.push(DenseVectorView::new(v));
            ip.push(DenseVectorView::new(v));
        }
        let l2: PlainDenseDataset<f32, SquaredEuclideanDistance> = l2.into();
        let ip: PlainDenseDataset<f32, DotProduct> = ip.into();

        // A fixed seed makes k-means deterministic; `train` itself passes `None`.
        let from_l2 =
            ProductQuantizer::<M_TEST, SquaredEuclideanDistance>::train_with_kmeans_options(
                &l2,
                DEFAULT_KMEANS_N_ITER,
                Some(42),
            );
        let from_ip =
            ProductQuantizer::<M_TEST, SquaredEuclideanDistance>::train_with_kmeans_options(
                &ip,
                DEFAULT_KMEANS_N_ITER,
                Some(42),
            );

        assert_eq!(from_l2.d, from_ip.d);
        assert_eq!(from_l2.dsub, from_ip.dsub);
        assert_eq!(from_l2.centroids, from_ip.centroids);

        // And the codes those centroids produce agree, which is what actually reaches a query.
        for v in &vecs {
            let (mut a, mut b) = (Vec::new(), Vec::new());
            from_l2.push_encoded(DenseVectorView::new(v), &mut a);
            from_ip.push_encoded(DenseVectorView::new(v), &mut b);
            assert_eq!(a, b);
        }
    }

    /// Encoding a vector then decoding it should reconstruct something close to
    /// the original (exact when query == a centroid, approximate otherwise).
    #[test]
    fn train_encode_decode_roundtrip() {
        let (pq, vecs) = make_pq_and_data();
        // Encode the first vector.
        let query = DenseVectorView::new(&vecs[0]);
        let mut codes: Vec<u8> = Vec::with_capacity(M_TEST);
        pq.push_encoded(query, &mut codes);
        assert_eq!(codes.len(), M_TEST);

        // Decode it back.
        let decoded = pq.decode_vector(DenseVectorView::new(&codes));
        assert_eq!(decoded.values().len(), D_TEST);

        // The decoded vector should be the nearest centroid per subspace, so its
        // distance from the original query should be <= max possible centroid gap.
        // Just check all values are finite.
        for v in decoded.values() {
            assert!(v.is_finite(), "decoded value is not finite");
        }
    }

    /// For each subspace the table slot chosen by push_encoded must equal
    /// argmin over that subspace's centroids of the Euclidean distance.
    #[test]
    fn table_lookup_matches_encode() {
        let (pq, vecs) = make_pq_and_data();
        let query_vals = &vecs[1];
        let query = DenseVectorView::new(query_vals);

        // Get codes from push_encoded.
        let mut codes: Vec<u8> = Vec::with_capacity(M_TEST);
        pq.push_encoded(query, &mut codes);

        // Build the distance table independently and verify each code matches
        // the argmin of the corresponding KSUB entries.
        let table = pq.compute_euclidean_distance_table(query);
        for m in 0..M_TEST {
            let sub_table = &table[m * KSUB..(m + 1) * KSUB];
            let best = sub_table
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i as u8)
                .unwrap();
            assert_eq!(
                codes[m], best,
                "subspace {m}: push_encoded gave code {} but table argmin is {}",
                codes[m], best
            );
        }
    }

    /// The upcasting encode path must assign exactly the same centroids as the `f32` path when
    /// both are handed the same trained quantizer and the same (already-rounded) values.
    ///
    /// The full `ConvertFrom` is not compared end to end because PQ's k-means is randomly seeded
    /// and its training sample is drawn at random, so two trainings do not share a codebook. What
    /// the `f16` path changes is the *encode*, and that is what this pins.
    #[test]
    fn f16_source_encodes_exactly_like_the_equivalent_f32_source() {
        use half::f16;

        let (pq, vecs) = make_pq_and_data();
        let n = vecs.len();
        let source_f16: Vec<f16> = vecs
            .iter()
            .flat_map(|v| v.iter().map(|&x| f16::from_f32(x)))
            .collect();
        let source_f32: Vec<f32> = source_f16.iter().map(|v| v.to_f32()).collect();

        let from_f32 = crate::DenseDataset::from_flat_par(pq.clone(), &source_f32, n);
        let from_f16 = crate::DenseDataset::from_flat_par_upcast(pq, &source_f16, n);

        assert_eq!(from_f16.values(), from_f32.values());
    }
}
