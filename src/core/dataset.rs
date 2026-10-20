//! Core dataset abstractions used by the search/indexing primitives.
//! Includes type-safe traits to iterate, look up ranges, and integrate with encoders.
use crate::core::sealed;
use crate::core::vector_encoder::{DenseVectorEncoder, SparseDataEncoder, VectorEncoder};

/// Unique identifier assigned to each vector stored inside a dataset.
pub type VectorId = u64;

/// Holds a vector identifier together with the distance reported by a search evaluator.
///
/// Ordering is based solely on `distance`; the `vector` field is not considered.
/// This avoids the extra integer comparison on every heap operation.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct ScoredItemGeneric<D, T> {
    pub distance: D,
    pub vector: T,
}

impl<D: PartialOrd, T: PartialEq> PartialOrd for ScoredItemGeneric<D, T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl<D: Ord, T: Eq> Ord for ScoredItemGeneric<D, T> {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.cmp(&other.distance)
    }
}

/// Helper type that maintains the ordering guarantees expected by callers of range-based APIs.
/// The tuple order (distance, range start, range end) is enforced so callers can merge ranges deterministically.
#[derive(Debug, PartialEq, Clone)]
pub struct ScoredRange<D> {
    pub distance: D,
    pub range: std::ops::Range<usize>,
}

impl<D: Eq> Eq for ScoredRange<D> {}

impl<D: Ord> PartialOrd for ScoredRange<D> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<D: Ord> Ord for ScoredRange<D> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .cmp(&other.distance)
            .then_with(|| self.range.start.cmp(&other.range.start))
            .then_with(|| self.range.end.cmp(&other.range.end))
    }
}

/// Specialized alias where the vector handle is a `VectorId`.
pub type ScoredVector<D> = ScoredItemGeneric<D, VectorId>;

/// Conversion helper that enforces datasets must be constructible from other datasets.
pub trait ConvertFrom<T: Dataset>: Dataset {
    /// Construction parameters, `()` for encoders that learn everything from the source data.
    ///
    /// This mirrors [`Index::SearchParams`](crate::core::index::Index::SearchParams), down to
    /// there being exactly one entry point: just as every search passes its params (`&()` when
    /// there are none), every conversion passes its config. A caller therefore never has to know
    /// which kind of encoder it is holding — `()` is a configuration, not the absence of one.
    type Config;

    /// Build this dataset type from `value` under `config`.
    fn convert_from(value: T, config: Self::Config) -> Self;
}

/// Mirror helper that forwards into `ConvertFrom` implementations.
pub trait ConvertInto<T: Dataset>: Dataset {
    /// Construction parameters of the target dataset — see [`ConvertFrom::Config`].
    type Config;

    /// Consume this dataset and produce another dataset wired by `ConvertFrom`, under `config`.
    fn convert_into(self, config: Self::Config) -> T;
}

impl<T, U> ConvertInto<U> for T
where
    T: Dataset,
    U: ConvertFrom<T>,
{
    type Config = <U as ConvertFrom<T>>::Config;

    fn convert_into(self, config: Self::Config) -> U {
        U::convert_from(self, config)
    }
}

/// Core abstraction for any collection of vectors paired with a `VectorEncoder`.
/// Implementations must expose the encoder but may choose arbitrary storage strategies.
/// Each vector is identified by a `VectorId` assigned at insertion time. The id is just the index of the vector in the dataset.
/// Each vector can be retrieved either by id or by specifying the underlying storage range directly.
/// The latter provides a faster access time on sparse datasets because we don't need to translate the id into a range first, by looking up an offsets table. This saves one memory access per vector retrieval and provides the oppurtunity for prefetching the vector data.
/// No advantage is expected for dense datasets, where the range can be trivially computed from the id and the vector dimensionality.
pub trait Dataset: sealed::Sealed {
    type Encoder: VectorEncoder;

    /// Owned dataset produced by bulk operations such as [`Self::permute`].
    ///
    /// Each dataset sets this to its *frozen* variant, which for the frozen aliases
    /// (`DenseDataset`, `SparseDataset`, …) is `Self`. Naming the frozen type rather than `Self`
    /// keeps `permute` from imposing a "constructible from a `Vec`" bound on every storage
    /// backend, which would rule out read-only ones such as a memory map.
    ///
    /// The forwarding impl for `&T` defers to `T::Owned`, since it cannot return a reference to
    /// a dataset it just allocated.
    type Owned: Dataset<Encoder = Self::Encoder>;

    /// Shared encoder instance used to encode, query, and decode vectors.
    fn encoder(&self) -> &Self::Encoder;

    /// Dimensionality of the input vectors accepted by the encoder.
    fn input_dim(&self) -> usize {
        self.encoder().input_dim()
    }

    /// Dimensionality of the output vectors produced by the encoder.
    fn output_dim(&self) -> usize {
        self.encoder().output_dim()
    }

    /// Number of vectors stored in the dataset.
    fn len(&self) -> usize;

    /// Number of non-zero components in the underlying representation.
    fn nnz(&self) -> usize;

    /// Translate a persisted `VectorId` into a storage range.
    fn range_from_id(&self, id: VectorId) -> std::ops::Range<usize>;

    /// Translate a storage range back to the opaque `VectorId`.
    fn id_from_range(&self, range: std::ops::Range<usize>) -> VectorId;

    /// Check whether the dataset contains any vectors.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the representation of the vector with the given id.
    fn get(&self, index: VectorId) -> <Self::Encoder as VectorEncoder>::EncodedVector<'_>;

    /// Get a vector directly specifying the range in the underlying storage.
    fn get_with_range(
        &self,
        range: std::ops::Range<usize>,
    ) -> <Self::Encoder as VectorEncoder>::EncodedVector<'_>;

    /// Iterate through raw encoded vectors for cases where the caller is responsible for decoding.
    fn iter(&self) -> impl Iterator<Item = <Self::Encoder as VectorEncoder>::EncodedVector<'_>>;

    /// Touch the provided range to hint that it will be accessed soon, prefetching at most
    /// `max_bytes` of it (per underlying array). See [`crate::utils::prefetch_read_slice`].
    fn prefetch_with_range(&self, range: std::ops::Range<usize>, max_bytes: usize);

    /// Returns a copy of this dataset whose rows are reordered so that the row currently
    /// at `old_id` ends up at position `permutation[old_id]`.
    ///
    /// Implementations reorder the encoded storage directly instead of decoding and
    /// re-encoding, so the copy is lossless even for quantized or packed encodings.
    ///
    /// # Panics
    ///
    /// Panics if `permutation` is not a bijection of `0..self.len()`.
    fn permute(&self, permutation: &[usize]) -> Self::Owned;
}

impl<T> sealed::Sealed for &T where T: Dataset {}

/// Allows borrowing any dataset, so callers can pass references through the public APIs
/// without having to duplicate impls or rely on trait objects.
impl<T> Dataset for &T
where
    T: Dataset,
{
    type Encoder = T::Encoder;
    type Owned = T::Owned;

    #[inline]
    fn encoder(&self) -> &Self::Encoder {
        (*self).encoder()
    }

    #[inline]
    fn len(&self) -> usize {
        (*self).len()
    }

    #[inline]
    fn nnz(&self) -> usize {
        (*self).nnz()
    }

    #[inline]
    fn range_from_id(&self, id: VectorId) -> std::ops::Range<usize> {
        (*self).range_from_id(id)
    }

    #[inline]
    fn id_from_range(&self, range: std::ops::Range<usize>) -> VectorId {
        (*self).id_from_range(range)
    }

    #[inline]
    fn get(&self, index: VectorId) -> <Self::Encoder as VectorEncoder>::EncodedVector<'_> {
        (*self).get(index)
    }

    #[inline]
    fn get_with_range(
        &self,
        range: std::ops::Range<usize>,
    ) -> <Self::Encoder as VectorEncoder>::EncodedVector<'_> {
        (*self).get_with_range(range)
    }

    #[inline]
    fn prefetch_with_range(&self, range: std::ops::Range<usize>, max_bytes: usize) {
        (*self).prefetch_with_range(range, max_bytes)
    }

    #[inline]
    fn iter(&self) -> impl Iterator<Item = <Self::Encoder as VectorEncoder>::EncodedVector<'_>> {
        (*self).iter()
    }

    #[inline]
    fn permute(&self, permutation: &[usize]) -> Self::Owned {
        (*self).permute(permutation)
    }
}

/// A dataset whose vectors can be averaged, which is what k-means needs from a representation
/// to compute its centroids (see [`KMeans`](crate::KMeans)).
pub trait MeanDataset: Dataset {
    /// An `f32` vector in the dataset's input space: a decoded vector or a mean.
    ///
    /// Its values, exposed through `AsMut<[f32]>`, are the vector's stored coordinates: all of
    /// them for a dense vector, the nonzeros for a sparse one. Representation-independent code
    /// (k-means) may therefore compute a norm over them or scale them, without knowing which
    /// representation it holds.
    type Vector: AsMut<[f32]> + Clone + Send;

    /// The vector `id`, decoded to `f32`.
    fn vector(&self, id: u32) -> Self::Vector;

    /// Overwrite `out` with the vector `id` decoded to `f32`, reusing its storage where the
    /// representation allows. `out` must come from this dataset ([`vector`](Self::vector) or
    /// [`mean`](Self::mean)).
    #[inline]
    fn vector_into(&self, id: u32, out: &mut Self::Vector) {
        *out = self.vector(id);
    }

    /// The weighted mean of the vectors `ids`, accumulated in `f32` in the order given.
    ///
    /// `weights`, if any, is indexed by vector id. An empty `ids` yields an all-zero (dense) or
    /// empty (sparse) vector.
    fn mean(&self, ids: &[u32], weights: Option<&[f32]>) -> Self::Vector;

    /// The residual `v − c` of the vector `id` against `centroid`, a vector of this dataset's
    /// input space (e.g. a [`mean`](Self::mean)). For a sparse dataset its nonzeros are the
    /// union of the two vectors' components.
    fn residual(&self, id: u32, centroid: &Self::Vector) -> Self::Vector;
}

/// Marker trait representing any dataset whose encoder exposes the dense-vector contract.
///
/// Dense layout, query expectations, and decoding helpers are defined by `DenseVectorEncoder`,
/// so this marker only makes sense when the encoder implements that trait.
/// Requiring the encoder bound keeps downstream consumers honest about what they can rely on.
pub trait DenseData: Dataset<Encoder: DenseVectorEncoder> {}

/// Marker trait for datasets backed by shared sparse-data helpers.
/// Both plain `SparseVectorEncoder`s and packed encoders expose the common component/value types
/// via `SparseDataEncoder`. Consumers can rely on those helpers without needing to distinguish the layout.
pub trait SparseData: Dataset<Encoder: SparseDataEncoder> {}

pub trait DatasetGrowable: Dataset {
    /// Create a new growable dataset owning the provided encoder.
    fn new(encoder: Self::Encoder) -> Self;

    /// Create a new growable dataset with the given encoder and reserved certain capacity for vectors.
    fn with_capacity(encoder: Self::Encoder, capacity: usize) -> Self;

    /// Append another vector, encoded through the dataset encoder, into storage.
    fn push<'a>(&mut self, vec: <Self::Encoder as VectorEncoder>::InputVector<'a>);
}

/// Validates that `permutation` is a bijection of `0..n`, panicking otherwise.
pub(crate) fn assert_valid_permutation(permutation: &[usize], n: usize) {
    assert_eq!(
        permutation.len(),
        n,
        "permutation length ({}) must match dataset length ({n})",
        permutation.len()
    );

    let mut seen = vec![false; n];
    for (old_id, &new_id) in permutation.iter().enumerate() {
        assert!(
            new_id < n,
            "permutation[{old_id}] = {new_id} is out of bounds for a dataset of {n} vectors"
        );
        assert!(
            !std::mem::replace(&mut seen[new_id], true),
            "permutation maps more than one vector to position {new_id}"
        );
    }
}

/// Inverts `permutation` (`old_id -> new_id`) into `new_id -> old_id`, the order in which
/// implementations gather rows. Callers must validate it with [`assert_valid_permutation`] first.
pub(crate) fn invert_permutation(permutation: &[usize]) -> Vec<usize> {
    let mut old_by_new = vec![0usize; permutation.len()];
    for (old_id, &new_id) in permutation.iter().enumerate() {
        old_by_new[new_id] = old_id;
    }
    old_by_new
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vector::DenseVectorView;
    use crate::datasets::dense_dataset::DenseDatasetGrowable;
    use crate::distances::DotProduct;
    use crate::distances::SquaredEuclideanDistance;
    use crate::{DenseDataset, PlainDenseQuantizer};

    #[test]
    fn scored_range_order_obeys_distance_then_start() {
        let r1 = ScoredRange {
            distance: DotProduct::from(2.0),
            range: 1..5,
        };
        let r2 = ScoredRange {
            distance: DotProduct::from(1.0),
            range: 0..2,
        };
        assert!(r1 < r2);

        let r1 = ScoredRange {
            distance: DotProduct::from(1.0),
            range: 0..2,
        };
        let r2 = ScoredRange {
            distance: DotProduct::from(1.0),
            range: 1..3,
        };
        assert!(r1 < r2);
        let r1 = ScoredRange {
            distance: SquaredEuclideanDistance::from(1.0),
            range: 1..5,
        };
        let r2 = ScoredRange {
            distance: SquaredEuclideanDistance::from(2.0),
            range: 0..2,
        };
        assert!(r1 < r2);

        let r1 = ScoredRange {
            distance: SquaredEuclideanDistance::from(1.0),
            range: 0..2,
        };
        let r2 = ScoredRange {
            distance: SquaredEuclideanDistance::from(1.0),
            range: 1..3,
        };
        assert!(r1 < r2);
    }

    #[test]
    fn dataset_trait_helpers_report_dimensions_and_empty() {
        type Encoder = PlainDenseQuantizer<f32, DotProduct>;

        let encoder = Encoder::new(2);
        let mut growable = DenseDatasetGrowable::new(encoder);
        growable.push(DenseVectorView::new(&[1.0f32, 0.0]));

        assert_eq!(growable.input_dim(), 2);
        assert_eq!(growable.output_dim(), 2);
        assert!(!growable.is_empty());
        assert!(growable.nnz() > 0);
    }

    #[test]
    fn dataset_reference_forwarding_implements_all_methods() {
        type Encoder = PlainDenseQuantizer<f32, DotProduct>;

        let encoder = Encoder::new(2);
        let mut growable = DenseDatasetGrowable::new(encoder);
        growable.push(DenseVectorView::new(&[1.0f32, 0.5]));
        growable.push(DenseVectorView::new(&[0.0f32, 1.0]));
        growable.push(DenseVectorView::new(&[2.0f32, 1.0]));

        let dataset: DenseDataset<Encoder> = growable.into();

        fn exercise(dataset: &DenseDataset<Encoder>) {
            assert_eq!(<&DenseDataset<Encoder> as Dataset>::len(&dataset), 3);
            assert_eq!(<&DenseDataset<Encoder> as Dataset>::nnz(&dataset), 6);
            assert!(!<&DenseDataset<Encoder> as Dataset>::is_empty(&dataset));
            assert_eq!(<&DenseDataset<Encoder> as Dataset>::input_dim(&dataset), 2);
            assert_eq!(<&DenseDataset<Encoder> as Dataset>::output_dim(&dataset), 2);

            let first_range = <&DenseDataset<Encoder> as Dataset>::range_from_id(&dataset, 0);
            let second_range = <&DenseDataset<Encoder> as Dataset>::range_from_id(&dataset, 1);
            let third_range = <&DenseDataset<Encoder> as Dataset>::range_from_id(&dataset, 2);
            assert_eq!(first_range, 0..2);
            assert_eq!(second_range, 2..4);
            assert_eq!(third_range, 4..6);

            assert_eq!(
                <&DenseDataset<Encoder> as Dataset>::id_from_range(&dataset, first_range.clone()),
                0
            );
            assert_eq!(
                <&DenseDataset<Encoder> as Dataset>::id_from_range(&dataset, second_range.clone()),
                1
            );
            assert_eq!(
                <&DenseDataset<Encoder> as Dataset>::id_from_range(&dataset, third_range.clone()),
                2
            );

            <&DenseDataset<Encoder> as Dataset>::prefetch_with_range(
                &dataset,
                first_range.clone(),
                256,
            );
            <&DenseDataset<Encoder> as Dataset>::prefetch_with_range(
                &dataset,
                second_range.clone(),
                256,
            );
            <&DenseDataset<Encoder> as Dataset>::prefetch_with_range(
                &dataset,
                third_range.clone(),
                256,
            );

            let first_vector = <&DenseDataset<Encoder> as Dataset>::get(&dataset, 0);
            assert_eq!(first_vector, DenseVectorView::new(&[1.0f32, 0.5]));

            let second_vector =
                <&DenseDataset<Encoder> as Dataset>::get_with_range(&dataset, second_range.clone());
            assert_eq!(second_vector, DenseVectorView::new(&[0.0f32, 1.0]));

            let iterated = <&DenseDataset<Encoder> as Dataset>::iter(&dataset)
                .map(|vec| vec.values().to_vec())
                .collect::<Vec<_>>();

            assert_eq!(
                iterated,
                vec![vec![1.0f32, 0.5], vec![0.0f32, 1.0], vec![2.0f32, 1.0]]
            );
        }

        exercise(&dataset);
    }

    /// Permuting through a reference must produce the same owned dataset as permuting the value.
    #[test]
    fn permute_through_a_reference_matches_permuting_the_value() {
        type Encoder = PlainDenseQuantizer<f32, DotProduct>;

        let encoder = Encoder::new(2);
        let mut growable = DenseDatasetGrowable::new(encoder);
        growable.push(DenseVectorView::new(&[10.0f32, 11.0]));
        growable.push(DenseVectorView::new(&[20.0f32, 21.0]));
        growable.push(DenseVectorView::new(&[30.0f32, 31.0]));

        let dataset: DenseDataset<Encoder> = growable.into();
        let permutation = [2usize, 0, 1];

        let borrowed: &DenseDataset<Encoder> = &dataset;
        let via_reference = <&DenseDataset<Encoder> as Dataset>::permute(&borrowed, &permutation);

        assert_eq!(via_reference, dataset.permute(&permutation));
        for (old_id, &new_id) in permutation.iter().enumerate() {
            assert_eq!(
                via_reference.get(new_id as VectorId),
                dataset.get(old_id as VectorId)
            );
        }
    }
}
