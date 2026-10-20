//! Packed sparse dataset helpers keep the encodings in a single concatenated buffer and expose growable/immutable APIs.
//! The module reuses `PackedSparseVectorEncoder` implementations so conversions and queries stay efficient.
use serde::{Deserialize, Serialize};

use crate::PackedSparseVectorEncoder;
use crate::SpaceUsage;
use crate::core::dataset::{assert_valid_permutation, invert_permutation};
use crate::core::sealed;
use crate::dataset::ConvertFrom;
use crate::utils::prefetch_read_slice;
use crate::{Dataset, DatasetGrowable, PackedVectorView, SparseData, VectorId};

use rayon::prelude::*;

/// A growable packed dataset.
///
/// # Examples
///
/// ```
/// use vectorium::{
///     DotPacking8U16F16Encoder, PackedSparseDatasetGrowable, DatasetGrowable, Dataset,
///     SparseVectorView,
/// };
///
/// let encoder = DotPacking8U16F16Encoder::new(4);
/// let mut dataset = PackedSparseDatasetGrowable::new(encoder);
/// dataset.push(SparseVectorView::new(&[0_u16, 2], &[1.0_f32, 2.0]));
/// assert_eq!(dataset.len(), 1);
/// ```
pub type PackedSparseDatasetGrowable<E> = PackedSparseDatasetGeneric<
    E,
    Vec<usize>,
    Vec<<E as PackedSparseVectorEncoder>::PackedDataType>,
>;

impl<E> PackedSparseDatasetGrowable<E>
where
    E: PackedSparseVectorEncoder,
{
    #[inline]
    pub fn new(encoder: E) -> Self {
        crate::DatasetGrowable::new(encoder)
    }

    #[inline]
    pub fn with_capacity(encoder: E, capacity: usize) -> Self {
        crate::DatasetGrowable::with_capacity(encoder, capacity)
    }
}

/// An immutable packed dataset.
///
/// # Examples
///
/// ```
/// use vectorium::{
///     Dataset, DotPacking8U16ScalarU8Encoder, DotProduct, DatasetGrowable, PackedSparseDataset,
///     PlainSparseDataset, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVectorView,
/// };
///
/// let quantizer = PlainSparseQuantizer::<u16, f32, DotProduct>::new(3, 3);
/// let mut sparse = PlainSparseDatasetGrowable::new(quantizer);
/// sparse.push(SparseVectorView::new(&[0_u16], &[1.0_f32]));
/// let frozen: PlainSparseDataset<u16, f32, DotProduct> = sparse.into();
/// // Trains the encoder on `frozen`, then encodes every vector.
/// let packed: PackedSparseDataset<DotPacking8U16ScalarU8Encoder> = frozen.into();
/// assert_eq!(packed.len(), 1);
/// ```
pub type PackedSparseDataset<E> = PackedSparseDatasetGeneric<
    E,
    Box<[usize]>,
    Box<[<E as PackedSparseVectorEncoder>::PackedDataType]>,
>;

/// Dataset storing variable-length packed encodings in a single concatenated `data` array.
///
/// Vector boundaries are stored in `offsets`, exactly like in `SparseDataset`:
/// - `offsets.len() == len() + 1`
/// - `offsets[0] == 0`
/// - vector `i` lives in `data[offsets[i]..offsets[i+1]]`.
///
/// Packed dataset storing variable-length encodings with offsets.
///
/// # Example
/// ```
/// use vectorium::{
///     Dataset, DotPacking8U16ScalarU8Encoder, DotProduct, DatasetGrowable, PackedSparseDataset,
///     PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVectorView, VectorEncoder,
/// };
///
/// let quantizer = PlainSparseQuantizer::<u16, f32, DotProduct>::new(5, 5);
/// let mut sparse = PlainSparseDatasetGrowable::new(quantizer);
/// sparse.push(SparseVectorView::new(&[1_u16, 3], &[1.0, 2.0]));
///
/// let frozen: vectorium::PlainSparseDataset<u16, f32, DotProduct> = sparse.into();
/// let packed: PackedSparseDataset<DotPacking8U16ScalarU8Encoder> = frozen.into();
/// let range = packed.range_from_id(0);
/// let v = packed.get_with_range(range);
/// assert!(!v.data().is_empty());
/// ```
#[derive(Default, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct PackedSparseDatasetGeneric<E, Offsets, Data>
where
    E: PackedSparseVectorEncoder,
    Offsets: AsRef<[usize]>,
    Data: AsRef<[E::PackedDataType]>,
{
    offsets: Offsets,
    data: Data,
    encoder: E,
    nnz: usize,
}

impl<E, Offsets, Data> sealed::Sealed for PackedSparseDatasetGeneric<E, Offsets, Data>
where
    E: PackedSparseVectorEncoder,
    Offsets: AsRef<[usize]>,
    Data: AsRef<[E::PackedDataType]>,
{
}

impl<E, Offsets, Data> PackedSparseDatasetGeneric<E, Offsets, Data>
where
    E: PackedSparseVectorEncoder,
    Offsets: AsRef<[usize]>,
    Data: AsRef<[E::PackedDataType]>,
{
    #[inline]
    /// Return the raw offsets array that demarcates each vector slice.
    pub fn offsets(&self) -> &[usize] {
        self.offsets.as_ref()
    }

    #[inline]
    /// Return the pooled packed data buffer shared by all vectors.
    pub fn data(&self) -> &[E::PackedDataType] {
        self.data.as_ref()
    }

    #[inline]
    pub fn nnz(&self) -> usize {
        self.nnz
    }

    #[inline]
    /// Translate vector id into a data range.
    pub fn range_from_id(&self, id: VectorId) -> std::ops::Range<usize> {
        let index = id as usize;
        let offsets = self.offsets.as_ref();
        assert!(index + 1 < offsets.len(), "Index out of bounds.");
        offsets[index]..offsets[index + 1]
    }

    #[inline]
    /// Translate a packed data range back into the owning id.
    pub fn id_from_range(&self, range: std::ops::Range<usize>) -> VectorId {
        let offsets = self.offsets.as_ref();
        let idx = offsets.binary_search(&range.start).unwrap();
        assert_eq!(
            offsets[idx + 1],
            range.end,
            "Range does not match vector boundaries."
        );
        idx as VectorId
    }

    /// Parallel iterator over dataset encoded vectors.
    ///
    /// Each item is an `E::EncodedVector<'_>` borrowing its slice from the dataset `data`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rayon::iter::ParallelIterator;
    /// use vectorium::{
    ///     DotPacking8U16F16Encoder, PackedSparseDataset, PackedSparseDatasetGrowable,
    ///     DatasetGrowable, Dataset, SparseVectorView,
    /// };
    ///
    /// let encoder = DotPacking8U16F16Encoder::new(4);
    /// let mut growable = PackedSparseDatasetGrowable::new(encoder);
    /// growable.push(SparseVectorView::new(&[0_u16, 1], &[1.0_f32, 2.0]));
    /// let packed: PackedSparseDataset<_> = growable.into();
    /// let collected: Vec<_> = packed.par_iter().collect();
    /// assert_eq!(collected.len(), packed.len());
    /// ```
    #[inline]
    pub fn par_iter(&self) -> impl IndexedParallelIterator<Item = E::EncodedVector<'_>> + '_
    where
        for<'a> E::EncodedVector<'a>: Send,
        Offsets: Sync,
        Data: Sync,
    {
        let offsets = self.offsets.as_ref();
        let data = self.data.as_ref();

        // https://github.com/rayon-rs/rayon/pull/789
        offsets.par_windows(2).map(move |window| {
            let start = window[0];
            let end = window[1];
            PackedVectorView::new(&data[start..end])
        })
    }
}

impl<E> DatasetGrowable
    for PackedSparseDatasetGeneric<
        E,
        Vec<usize>,
        Vec<<E as PackedSparseVectorEncoder>::PackedDataType>,
    >
where
    E: PackedSparseVectorEncoder,
{
    #[inline]
    fn new(encoder: E) -> Self {
        Self {
            offsets: vec![0],
            data: Vec::new(),
            encoder,
            nnz: 0,
        }
    }

    fn with_capacity(encoder: E, capacity: usize) -> Self {
        let mut offsets = Vec::with_capacity(capacity + 1);
        offsets.push(0);
        Self {
            offsets,
            data: Vec::with_capacity(capacity),
            encoder,
            nnz: 0,
        }
    }

    #[inline]
    /// Append a packed encoding directly into the dataset buffer.
    ///
    /// The implementation relies on `push_encoded` to extend the pooled `data` vector,
    /// keeping allocations at a minimum.
    fn push<'a>(&mut self, vec: E::InputVector<'a>) {
        self.nnz += vec.components().len(); // Capture length before move if needed? Copy view is cheap.

        self.encoder.push_encoded(vec, &mut self.data);
        self.offsets.push(self.data.len());
    }
}

impl<E, Offsets, Data> SpaceUsage for PackedSparseDatasetGeneric<E, Offsets, Data>
where
    E: PackedSparseVectorEncoder,
    E: SpaceUsage,
    Offsets: AsRef<[usize]> + SpaceUsage,
    Data: AsRef<[E::PackedDataType]> + SpaceUsage,
{
    fn space_usage_bytes(&self) -> usize {
        self.encoder.space_usage_bytes()
            + self.offsets.space_usage_bytes()
            + self.data.space_usage_bytes()
            + self.nnz.space_usage_bytes()
    }
}

impl<E, Offsets, Data> Dataset for PackedSparseDatasetGeneric<E, Offsets, Data>
where
    E: PackedSparseVectorEncoder,
    Offsets: AsRef<[usize]>,
    Data: AsRef<[E::PackedDataType]>,
{
    type Encoder = E;
    /// Frozen, `Box<[_]>`-backed variant; identical to `Self` for [`PackedSparseDataset`].
    type Owned = PackedSparseDatasetGeneric<E, Box<[usize]>, Box<[E::PackedDataType]>>;

    #[inline]
    fn nnz(&self) -> usize {
        self.nnz
    }

    #[inline]
    fn range_from_id(&self, id: VectorId) -> std::ops::Range<usize> {
        let index = id as usize;
        let offsets = self.offsets.as_ref();
        assert!(index + 1 < offsets.len(), "Index out of bounds.");
        offsets[index]..offsets[index + 1]
    }

    #[inline]
    fn id_from_range(&self, range: std::ops::Range<usize>) -> VectorId {
        let offsets = self.offsets.as_ref();
        let idx = offsets.binary_search(&range.start).unwrap();
        assert_eq!(
            offsets[idx + 1],
            range.end,
            "Range does not match vector boundaries."
        );
        idx as VectorId
    }

    #[inline]
    fn encoder(&self) -> &E {
        &self.encoder
    }

    #[inline]
    fn len(&self) -> usize {
        self.offsets.as_ref().len().saturating_sub(1)
    }

    #[inline]
    fn get(&self, index: VectorId) -> E::EncodedVector<'_> {
        let range = self.range_from_id(index);
        self.get_with_range(range)
    }

    #[inline]
    fn get_with_range<'a>(&'a self, range: std::ops::Range<usize>) -> E::EncodedVector<'a> {
        let slice = &self.data.as_ref()[range];
        PackedVectorView::new(slice)
    }

    #[inline]
    fn prefetch_with_range(&self, range: std::ops::Range<usize>, max_bytes: usize) {
        prefetch_read_slice(&self.data.as_ref()[range], max_bytes);
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = E::EncodedVector<'a>> {
        let offsets = self.offsets.as_ref();
        let data = self.data.as_ref();
        offsets
            .windows(2)
            .map(move |w| PackedVectorView::new(&data[w[0]..w[1]]))
    }

    fn permute(&self, permutation: &[usize]) -> Self::Owned {
        let n_vecs = self.offsets.as_ref().len() - 1;
        assert_valid_permutation(permutation, n_vecs);

        // Each row is padded to a whole number of `PackedDataType` words, so its encoding does
        // not depend on its position: copying the range verbatim is lossless.
        let old_by_new = invert_permutation(permutation);
        let offsets = self.offsets.as_ref();
        let source = self.data.as_ref();

        let mut new_offsets = Vec::with_capacity(n_vecs + 1);
        new_offsets.push(0);
        let mut permuted = Vec::with_capacity(source.len());

        for old_id in old_by_new {
            permuted.extend_from_slice(&source[offsets[old_id]..offsets[old_id + 1]]);
            new_offsets.push(permuted.len());
        }

        PackedSparseDatasetGeneric {
            offsets: new_offsets.into_boxed_slice(),
            data: permuted.into_boxed_slice(),
            encoder: self.encoder.clone(),
            // Reordering rows moves non-zeros around but never adds or drops any.
            nnz: self.nnz,
        }
    }
}

impl<E, Offsets, Data> SparseData for PackedSparseDatasetGeneric<E, Offsets, Data>
where
    E: PackedSparseVectorEncoder,
    Offsets: AsRef<[usize]>,
    Data: AsRef<[E::PackedDataType]>,
{
}

impl<E> From<PackedSparseDatasetGrowable<E>> for PackedSparseDataset<E>
where
    E: PackedSparseVectorEncoder,
{
    fn from(dataset: PackedSparseDatasetGrowable<E>) -> Self {
        PackedSparseDatasetGeneric {
            offsets: dataset.offsets.into_boxed_slice(),
            data: dataset.data.into_boxed_slice(),
            encoder: dataset.encoder,
            nnz: dataset.nnz,
        }
    }
}

impl<E> ConvertFrom<PackedSparseDatasetGrowable<E>> for PackedSparseDataset<E>
where
    E: PackedSparseVectorEncoder,
{
    type Config = ();

    fn convert_from(dataset: PackedSparseDatasetGrowable<E>, _config: ()) -> Self {
        dataset.into()
    }
}

impl<E> From<PackedSparseDataset<E>> for PackedSparseDatasetGrowable<E>
where
    E: PackedSparseVectorEncoder,
{
    fn from(dataset: PackedSparseDataset<E>) -> Self {
        PackedSparseDatasetGeneric {
            offsets: dataset.offsets.to_vec(),
            data: dataset.data.to_vec(),
            encoder: dataset.encoder,
            nnz: dataset.nnz,
        }
    }
}

impl<E> ConvertFrom<PackedSparseDataset<E>> for PackedSparseDatasetGrowable<E>
where
    E: PackedSparseVectorEncoder,
{
    type Config = ();

    fn convert_from(dataset: PackedSparseDataset<E>, _config: ()) -> Self {
        dataset.into()
    }
}

impl<EIn, S, V> From<crate::datasets::sparse_dataset::SparseDatasetGeneric<EIn, S>>
    for PackedSparseDataset<crate::encoders::blocked_sparse::BlockedSparseEncoder<V>>
where
    EIn: crate::SparseVectorEncoder<OutputComponentType = u16>,
    EIn::OutputValueType: crate::ValueType + crate::Float,
    S: crate::core::storage::SparseStorage<EIn>,
    V: crate::encoders::blocked_sparse::BlockValues,
{
    /// Trains the value format on `dataset` (no-op for `f16`) and packs every vector into
    /// blocks. Values of any float type are widened to `f32` per entry; no copy of the dataset
    /// is made.
    fn from(dataset: crate::datasets::sparse_dataset::SparseDatasetGeneric<EIn, S>) -> Self {
        use crate::encoders::blocked_sparse::BlockedSparseEncoder;
        use num_traits::ToPrimitive as _;

        let dim = dataset.output_dim();
        let mut encoder = BlockedSparseEncoder::<V>::new(dim);
        encoder.train(&dataset);

        let mut offsets = Vec::with_capacity(dataset.len() + 1);
        offsets.push(0);
        let mut data = Vec::new();
        for v in dataset.iter() {
            encoder.push_blocks(
                v.components(),
                v.values().iter().map(|x| x.to_f32().unwrap()),
                &mut data,
            );
            offsets.push(data.len());
        }

        PackedSparseDatasetGeneric {
            offsets: offsets.into_boxed_slice(),
            data: data.into_boxed_slice(),
            encoder,
            nnz: dataset.nnz(),
        }
    }
}

impl<EIn, S, V>
    crate::dataset::ConvertFrom<crate::datasets::sparse_dataset::SparseDatasetGeneric<EIn, S>>
    for PackedSparseDataset<crate::encoders::blocked_sparse::BlockedSparseEncoder<V>>
where
    EIn: crate::SparseVectorEncoder<OutputComponentType = u16>,
    EIn::OutputValueType: crate::ValueType + crate::Float,
    S: crate::core::storage::SparseStorage<EIn>,
    V: crate::encoders::blocked_sparse::BlockValues,
{
    type Config = ();

    fn convert_from(
        dataset: crate::datasets::sparse_dataset::SparseDatasetGeneric<EIn, S>,
        _config: (),
    ) -> Self {
        dataset.into()
    }
}

impl<EIn, S, C, V> From<crate::datasets::sparse_dataset::SparseDatasetGeneric<EIn, S>>
    for PackedSparseDataset<crate::encoders::dotpacking8::encoder::DotPacking8Encoder<C, V>>
where
    EIn: crate::SparseVectorEncoder<OutputComponentType = C>,
    EIn::OutputValueType: crate::ValueType + crate::Float,
    for<'a> EIn::EncodedVector<'a>: crate::VectorView,
    S: crate::core::storage::SparseStorage<EIn>,
    C: crate::encoders::dotpacking8::encoder::PackedComponent,
    V: crate::encoders::dotpacking8::values::ValueCodec,
{
    /// Trains the encoder on `dataset` (component order, then value parameters) and encodes
    /// every vector. Values are widened to `f32`; no copy of the dataset is made.
    fn from(dataset: crate::datasets::sparse_dataset::SparseDatasetGeneric<EIn, S>) -> Self {
        use crate::encoders::dotpacking8::encoder::DotPacking8Encoder;
        use num_traits::ToPrimitive as _;

        let dim = dataset.output_dim();
        let mut encoder = DotPacking8Encoder::<C, V>::new(dim);
        encoder.train(&dataset);

        let mut offsets = Vec::with_capacity(dataset.len() + 1);
        offsets.push(0);
        let mut data = Vec::new();
        let mut values = Vec::new();
        for v in dataset.iter() {
            values.clear();
            values.extend(v.values().iter().map(|x| x.to_f32().unwrap()));
            encoder.push_encoded(
                crate::SparseVectorView::new(v.components(), &values),
                &mut data,
            );
            offsets.push(data.len());
        }

        PackedSparseDatasetGeneric {
            offsets: offsets.into_boxed_slice(),
            data: data.into_boxed_slice(),
            encoder,
            nnz: dataset.nnz(),
        }
    }
}

impl<EIn, S, C, V> ConvertFrom<crate::datasets::sparse_dataset::SparseDatasetGeneric<EIn, S>>
    for PackedSparseDataset<crate::encoders::dotpacking8::encoder::DotPacking8Encoder<C, V>>
where
    EIn: crate::SparseVectorEncoder<OutputComponentType = C>,
    EIn::OutputValueType: crate::ValueType + crate::Float,
    for<'a> EIn::EncodedVector<'a>: crate::VectorView,
    S: crate::core::storage::SparseStorage<EIn>,
    C: crate::encoders::dotpacking8::encoder::PackedComponent,
    V: crate::encoders::dotpacking8::values::ValueCodec,
{
    type Config = ();

    fn convert_from(
        dataset: crate::datasets::sparse_dataset::SparseDatasetGeneric<EIn, S>,
        _config: (),
    ) -> Self {
        dataset.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BlockedSparseEncoder;
    use crate::DotPacking8U16F16Encoder;
    use crate::core::vector::SparseVectorView;
    use crate::core::vector_encoder::{SparseDataEncoder, VectorEncoder};

    /// Four DotPacking8 FixedU8 rows of differing encoded lengths.
    fn permutable_packed() -> PackedSparseDataset<crate::DotPacking8U16FixedU8Encoder> {
        use crate::DatasetGrowable;

        let rows: [(&[u16], &[f32]); 4] = [
            (&[1, 10, 100], &[1.5, 2.0, 2.5]),
            (&[2, 11], &[0.5, 1.0]),
            (&[0, 5, 50, 200, 400], &[0.25, 0.5, 0.75, 1.0, 1.25]),
            (&[300], &[2.0]),
        ];

        let mut growable =
            PackedSparseDatasetGrowable::new(crate::DotPacking8U16FixedU8Encoder::new(505));
        for (components, values) in rows {
            growable.push(SparseVectorView::new(components, values));
        }
        growable.into()
    }

    /// A verbatim copy is exact, while decoding and re-quantizing through `FixedU8Q` would
    /// lose precision on every permutation.
    #[test]
    fn packed_permute_is_lossless() {
        let dataset = permutable_packed();
        let permutation = [2usize, 0, 3, 1];

        let permuted = dataset.permute(&permutation);
        let encoder = dataset.encoder();

        assert_eq!(permuted.len(), dataset.len());
        assert_eq!(permuted.nnz(), dataset.nnz());
        for (old_id, &new_id) in permutation.iter().enumerate() {
            let expected = encoder.decode_vector(dataset.get(old_id as VectorId));
            let actual = encoder.decode_vector(permuted.get(new_id as VectorId));
            assert_eq!(actual.components(), expected.components());
            // Exact equality, no tolerance: the bytes were copied, not recomputed.
            assert_eq!(actual.values(), expected.values());
        }
    }

    #[test]
    fn packed_permute_copies_row_bytes_verbatim() {
        let dataset = permutable_packed();
        let permutation = [2usize, 0, 3, 1];

        let permuted = dataset.permute(&permutation);

        for (old_id, &new_id) in permutation.iter().enumerate() {
            assert_eq!(
                permuted.get(new_id as VectorId).data(),
                dataset.get(old_id as VectorId).data()
            );
        }
    }

    #[test]
    fn packed_permute_then_inverse_round_trips() {
        let dataset = permutable_packed();
        let permutation = [2usize, 0, 3, 1];
        let inverse = crate::core::dataset::invert_permutation(&permutation);

        let round_tripped = dataset.permute(&permutation).permute(&inverse);

        assert_eq!(round_tripped.offsets(), dataset.offsets());
        assert_eq!(round_tripped.data(), dataset.data());
    }

    #[test]
    #[should_panic(expected = "permutation length")]
    fn packed_permute_rejects_wrong_length() {
        permutable_packed().permute(&[0, 1, 2]);
    }

    #[test]
    #[should_panic(expected = "more than one vector to position")]
    fn packed_permute_rejects_non_bijective_permutation() {
        permutable_packed().permute(&[0, 1, 1, 3]);
    }

    #[test]
    fn conversion_and_dot_product() {
        use crate::DatasetGrowable;
        use crate::QueryEvaluator as _;
        use crate::VectorEncoder as _;
        use crate::core::vector::SparseVectorView;
        use crate::distances::Distance as _;
        use crate::{DotProduct, PlainSparseDataset, PlainSparseDatasetGrowable};
        use half::f16;

        let dim = 505;

        let mut growable: PlainSparseDatasetGrowable<u16, f32, DotProduct> =
            PlainSparseDatasetGrowable::new(
                crate::PlainSparseQuantizer::<u16, f32, DotProduct>::new(dim, dim),
            );

        let v0_components = vec![1_u16, 10, 100];
        let v0_values = vec![1.5_f32, 2.0, 2.5];

        growable.push(SparseVectorView::new(&v0_components, &v0_values));

        let v1_components = vec![2_u16, 11];
        let v1_values = vec![0.5_f32, 1.0];

        growable.push(SparseVectorView::new(&v1_components, &v1_values));

        let frozen: PlainSparseDataset<u16, f32, DotProduct> = growable.into();

        let dataset: PackedSparseDataset<BlockedSparseEncoder> = frozen.into();

        // 3 query components, below SPARSE_QUERY_THRESHOLD: the jump_dot path.
        let query = SparseVectorView::new(&[1_u16, 10, 11][..], &[2.0_f32, 3.0, 4.0][..]);
        let evaluator = dataset.encoder().query_evaluator(query, &());

        let d0 = evaluator.compute_distance(dataset.get(0)).distance();
        let d1 = evaluator.compute_distance(dataset.get(1)).distance();

        // Values are stored as f16; the kernel's summation order differs from this one.
        let r = |x: f32| f16::from_f32(x).to_f32();
        let expected0 = r(1.5) * 2.0 + r(2.0) * 3.0;
        let expected1 = r(1.0) * 4.0;

        assert!((d0 - expected0).abs() < 1e-6, "{d0} vs {expected0}");
        assert!((d1 - expected1).abs() < 1e-6, "{d1} vs {expected1}");
    }

    #[test]
    fn packed_growable_immutable_roundtrip() {
        use crate::PackedSparseDatasetGrowable;
        use crate::core::vector::SparseVectorView;
        use crate::{DatasetGrowable, PackedSparseDataset};

        let dim = 8;
        let encoder = DotPacking8U16F16Encoder::new(dim);
        let mut growable = PackedSparseDatasetGrowable::new(encoder);

        growable.push(SparseVectorView::new(&[1_u16, 4], &[1.0_f32, 2.0]));
        growable.push(SparseVectorView::new(&[2_u16], &[3.0_f32]));

        let frozen: PackedSparseDataset<DotPacking8U16F16Encoder> = growable.into();
        assert_eq!(frozen.len(), 2);
        assert_eq!(frozen.nnz(), 3);

        let mut growable_again: PackedSparseDatasetGrowable<DotPacking8U16F16Encoder> =
            frozen.into();
        growable_again.push(SparseVectorView::new(&[7_u16], &[4.0_f32]));

        assert_eq!(growable_again.len(), 3);
        assert_eq!(growable_again.nnz(), 4);
    }

    #[test]
    fn packed_dataset_iter_and_convert_from_traits_work() {
        use crate::DataBlock;
        use crate::PackedSparseDataset;
        use crate::PackedSparseDatasetGrowable;
        use crate::PlainSparseDataset;
        use crate::PlainSparseDatasetGrowable;
        use crate::core::vector::PackedVectorView;
        use crate::encoders::sparse_scalar::PlainSparseQuantizer;
        use crate::{DotProduct, SparseVectorView};

        let quantizer = PlainSparseQuantizer::<u16, f32, DotProduct>::new(6, 6);
        let mut growable = PlainSparseDatasetGrowable::new(quantizer);
        growable.push(SparseVectorView::new(&[0_u16, 2], &[1.0_f32, 2.0]));
        growable.push(SparseVectorView::new(&[1_u16, 3], &[3.0_f32, 4.0]));

        let frozen: PlainSparseDataset<u16, f32, DotProduct> = growable.into();
        let packed: PackedSparseDataset<BlockedSparseEncoder> =
            PackedSparseDataset::convert_from(frozen.clone(), ());

        assert_eq!(packed.nnz(), 4);
        assert_eq!(packed.len(), 2);

        let range = packed.range_from_id(1);
        let view: PackedVectorView<'_, DataBlock> = packed.get_with_range(range);
        assert!(!view.is_empty());

        let iter_values: Vec<_> = packed.iter().collect();
        assert_eq!(iter_values.len(), 2);

        let par_iter_values: Vec<_> = packed.par_iter().collect();
        assert_eq!(par_iter_values.len(), 2);

        let growable_again: PackedSparseDatasetGrowable<BlockedSparseEncoder> =
            PackedSparseDatasetGrowable::convert_from(packed.clone(), ());
        let packed_again = PackedSparseDataset::convert_from(growable_again, ());
        assert_eq!(packed_again.nnz(), packed.nnz());
    }

    #[test]
    fn packed_dataset_offsets_data_and_prefetch() {
        use crate::PackedSparseDatasetGrowable;
        use crate::core::vector::SparseVectorView;

        let dim = 5;
        let encoder = DotPacking8U16F16Encoder::new(dim);
        let mut growable = PackedSparseDatasetGrowable::new(encoder);
        growable.push(SparseVectorView::new(&[1_u16], &[1.0_f32]));

        let dataset: PackedSparseDataset<DotPacking8U16F16Encoder> = growable.into();
        assert_eq!(dataset.nnz(), 1);
        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset.offsets(), &[0, dataset.data().len()]);
        assert_eq!(dataset.encoder().output_dim(), dim);
        dataset.prefetch_with_range(0..dataset.data().len(), 256);
        let mut iter = dataset.iter();
        assert!(iter.next().is_some());
        assert!(dataset.par_iter().count() > 0);
        assert!(dataset.space_usage_bytes() > 0);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds.")]
    fn packed_dataset_range_from_id_panics_when_invalid() {
        let encoder = DotPacking8U16F16Encoder::new(3);
        let mut growable = PackedSparseDatasetGrowable::new(encoder);
        growable.push(SparseVectorView::new(&[0_u16], &[0.5_f32]));
        let dataset: PackedSparseDataset<DotPacking8U16F16Encoder> = growable.into();
        let _ = dataset.range_from_id(1);
    }

    #[test]
    #[should_panic]
    fn packed_dataset_id_from_range_panics_when_mismatch() {
        let encoder = DotPacking8U16F16Encoder::new(3);
        let mut growable = PackedSparseDatasetGrowable::new(encoder);
        growable.push(SparseVectorView::new(&[0_u16], &[0.5_f32]));
        let dataset: PackedSparseDataset<DotPacking8U16F16Encoder> = growable.into();
        let _ = dataset.id_from_range(1..2);
    }

    #[test]
    fn packed_dataset_with_capacity_pushes_vectors() {
        let encoder = DotPacking8U16F16Encoder::new(3);
        let mut growable = PackedSparseDatasetGrowable::with_capacity(encoder, 2);
        assert_eq!(growable.offsets().len(), 1);
        growable.push(SparseVectorView::new(&[0_u16], &[1.0_f32]));
        growable.push(SparseVectorView::new(&[1_u16], &[2.0_f32]));
        assert_eq!(growable.nnz(), 2);
        let frozen: PackedSparseDataset<DotPacking8U16F16Encoder> = growable.into();
        assert_eq!(frozen.len(), 2);
    }
}
