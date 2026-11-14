//! K-means clustering with interchangeable solvers.
//!
//! [`KMeans`] owns everything shared by every way of fitting k-means centroids: initialisation,
//! flat or index-accelerated assignment ([`IndexRefresh`], [`CentroidIndexBuilder`]), weights,
//! the `refine` hook, spherical mode, seeding, restarts, the epoch budget and the conversion
//! of the centroids into a growable dataset. The [`Solver`] chosen with [`KMeansBuilder::solver`] decides how the
//! centroids are fitted:
//! * [`Solver::Lloyd`] (the default): Lloyd's algorithm.
//! * [`Solver::EMeans`] ([`EMeansParams`]): e-means, mini-batch k-means with an exponential
//!   moving average and fitted scaling laws, ported from faiss `contrib/e_means`.
//!
//! The entry points are [`KMeans::train`] / [`KMeans::train_refined`] (exact, flat assignment)
//! and [`KMeans::train_with_index`] / [`KMeans::train_with_index_refined`] (assignment
//! through an index over the current centroids, e.g. an HNSW or IVF when `k` is large).

pub mod emeans;
pub mod index_assign;
pub mod kmeans;
pub mod lloyd;

pub use emeans::{BatchLaw, EMeansParams, EMeansRecipe};
pub use index_assign::{CentroidIndexBuilder, IndexRefresh, RefreshEvery};
pub use kmeans::{KMeans, KMeansBuilder, Solver, sparse_centroid_pruning};
