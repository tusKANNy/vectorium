//! Clustering algorithms behind one interface.
//!
//! [`Clustering`] is the seam every consumer of clustering goes through (product quantizers
//! here, IVF in kannolo): code that is generic over `Cl: Clustering` can swap Lloyd's
//! [`KMeans`] for mini-batch [`EMeans`] without changing anything else, the same way encoders
//! and indexes are swapped elsewhere in the ecosystem. [`ClusteringConfig`] is the
//! runtime-chosen form (from a CLI flag or a config file).
//!
//! Every algorithm shares [`KMeans`]'s contract: it clusters any [`MeanDataset`] (dense or
//! sparse), writes the centroids into a growable dataset `G` built from the encoder the caller
//! passes, and clusters under `G`'s metric. The entry points mirror [`KMeans`]'s:
//! * [`Clustering::train`] / [`Clustering::train_refined`] — exact (flat) assignment.
//! * [`Clustering::train_with_index`] / [`Clustering::train_with_index_refined`] — assignment
//!   through an index over the current centroids (e.g. an HNSW or IVF when `k` is large),
//!   maintained under the algorithm's [`IndexRefresh`] policy.

pub mod emeans;
pub mod index_assign;
pub mod kmeans;

pub use emeans::{BatchLaw, EMeans, EMeansBuilder, EMeansRecipe};
pub use index_assign::{CentroidIndexBuilder, IndexRefresh, RefreshEvery};
pub use kmeans::{KMeans, KMeansBuilder, sparse_centroid_pruning};

use crate::core::dataset::{DatasetGrowable, MeanDataset};
use crate::core::distances::Distance;
use crate::core::index::Index;
use crate::core::vector_encoder::VectorEncoder;

/// A clustering algorithm that turns a training set into `k` centroids.
///
/// Implementations carry their own hyperparameters (iterations, epochs, seed, ...) and are
/// `Sync`, so one instance can be shared across parallel jobs such as the per-subspace
/// trainings of a product quantizer.
///
/// The bounds on every method are [`KMeans`]'s: `D` is the (averageable) training dataset, `G`
/// the growable centroid dataset whose encoder fixes the centroids' precision and the
/// clustering metric, and `Q` the centroid index for the `_with_index` forms.
pub trait Clustering: Sync {
    /// Clusters `training_dataset` into `k` centroids with exact (flat) assignment, passing
    /// every updated centroid through `refine(members, &mut centroid)`, where `members` are
    /// the ids of the training vectors that produced the update. `weights`, if given, has one
    /// entry per training vector and turns every mean into a weighted mean.
    fn train_refined<D, G>(
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
            + Into<<G::Encoder as VectorEncoder>::QueryVector<'a>>;

    /// Like [`train_refined`](Clustering::train_refined), but assigns through an index over
    /// the current centroids. `build_centroid_index` builds and refreshes it (any
    /// `FnMut(G) -> Q` closure works, refreshing by full rebuild); how often is the algorithm's
    /// [`IndexRefresh`] policy. Index results are re-ranked by exact distance under `G`'s
    /// metric, which must be the index's.
    #[allow(clippy::too_many_arguments)]
    fn train_with_index_refined<D, G, Q>(
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
        for<'a> &'a D::Vector: Into<Q::Query<'a>>;

    /// [`train_refined`](Clustering::train_refined) without refinement.
    fn train<D, G>(
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

    /// [`train_with_index_refined`](Clustering::train_with_index_refined) without refinement.
    fn train_with_index<D, G, Q>(
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
}

impl Clustering for KMeans {
    fn train_refined<D, G>(
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
        KMeans::train_refined(self, training_dataset, k, weights, centroid_encoder, refine)
    }

    fn train_with_index_refined<D, G, Q>(
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
        KMeans::train_with_index_refined(
            self,
            training_dataset,
            k,
            weights,
            centroid_encoder,
            build_centroid_index,
            search_params,
            refine,
        )
    }
}

/// A clustering algorithm chosen at run time (from a config file or a CLI flag).
///
/// [`Clustering`] has generic methods, so it cannot be a trait object; this enum is the
/// runtime-dispatch form. It implements [`Clustering`] by forwarding to the chosen algorithm,
/// so it can be passed anywhere a `Cl: Clustering` is expected, and stored in build-parameter
/// structs. The default is Lloyd's k-means with its default settings.
#[derive(Clone, Debug)]
pub enum ClusteringConfig {
    KMeans(KMeans),
    EMeans(EMeans),
}

impl Default for ClusteringConfig {
    fn default() -> Self {
        ClusteringConfig::KMeans(KMeansBuilder::new().build())
    }
}

impl From<KMeans> for ClusteringConfig {
    fn from(c: KMeans) -> Self {
        ClusteringConfig::KMeans(c)
    }
}

impl From<EMeans> for ClusteringConfig {
    fn from(c: EMeans) -> Self {
        ClusteringConfig::EMeans(c)
    }
}

impl From<KMeansBuilder> for ClusteringConfig {
    fn from(b: KMeansBuilder) -> Self {
        ClusteringConfig::KMeans(b.build())
    }
}

impl From<EMeansBuilder> for ClusteringConfig {
    fn from(b: EMeansBuilder) -> Self {
        ClusteringConfig::EMeans(b.build())
    }
}

impl Clustering for ClusteringConfig {
    fn train_refined<D, G>(
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
        match self {
            ClusteringConfig::KMeans(c) => {
                Clustering::train_refined(c, training_dataset, k, weights, centroid_encoder, refine)
            }
            ClusteringConfig::EMeans(c) => {
                Clustering::train_refined(c, training_dataset, k, weights, centroid_encoder, refine)
            }
        }
    }

    fn train_with_index_refined<D, G, Q>(
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
        match self {
            ClusteringConfig::KMeans(c) => Clustering::train_with_index_refined(
                c,
                training_dataset,
                k,
                weights,
                centroid_encoder,
                build_centroid_index,
                search_params,
                refine,
            ),
            ClusteringConfig::EMeans(c) => Clustering::train_with_index_refined(
                c,
                training_dataset,
                k,
                weights,
                centroid_encoder,
                build_centroid_index,
                search_params,
                refine,
            ),
        }
    }
}
