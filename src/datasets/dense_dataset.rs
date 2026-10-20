use serde::{Deserialize, Serialize};

use crate::SpaceUsage;
use crate::core::dataset::MeanDataset;
use crate::core::dataset::{assert_valid_permutation, invert_permutation};
use crate::core::sealed;
use crate::core::vector::DenseVectorOwned;
use crate::core::vector_encoder::{DenseVectorEncoder, VectorEncoder};
use crate::{Dataset, DatasetGrowable, VectorId};

use rayon::prelude::*;

/// Growable dense dataset backed by a `Vec` buffer.
///
/// This alias is the preferred entry point for building datasets incrementally.
/// The example below shows how to push vectors and freeze the dataset.
///
/// # Examples
///
/// ```
/// use vectorium::datasets::dense_dataset::DenseDatasetGrowable;
/// use vectorium::encoders::dense_scalar::ScalarDenseQuantizer;
/// use vectorium::distances::DotProduct;
/// use vectorium::core::vector::DenseVectorView;
/// use vectorium::DatasetGrowable;
/// use vectorium::Dataset;
///
/// let encoder = ScalarDenseQuantizer::<f32, f32, DotProduct>::new(2);
/// let mut growable = DenseDatasetGrowable::new(encoder);
/// growable.push(DenseVectorView::new(&[1.0, 0.0]));
/// growable.push(DenseVectorView::new(&[0.0, 1.0]));
/// assert_eq!(growable.len(), 2);
/// ```
pub type DenseDatasetGrowable<E> =
    DenseDatasetGeneric<E, Vec<<E as DenseVectorEncoder>::OutputValueType>>;

/// Immutable dataset backed by a boxed slice.
///
/// Use this alias for workloads that need a compact, read-only dataset.
///
/// # Examples
///
/// ```
/// use vectorium::datasets::dense_dataset::DenseDatasetGrowable;
/// use vectorium::{DenseDataset, Dataset, DatasetGrowable};
/// use vectorium::encoders::dense_scalar::ScalarDenseQuantizer;
/// use vectorium::distances::DotProduct;
/// use vectorium::core::vector::DenseVectorView;
///
/// let encoder = ScalarDenseQuantizer::<f32, f32, DotProduct>::new(2);
/// let mut growable = DenseDatasetGrowable::new(encoder);
/// growable.push(DenseVectorView::new(&[1.0, 0.0]));
/// growable.push(DenseVectorView::new(&[0.0, 1.0]));
/// let dataset: DenseDataset<_> = growable.into();
/// assert_eq!(dataset.len(), 2);
/// ```
pub type DenseDataset<E> =
    DenseDatasetGeneric<E, Box<[<E as DenseVectorEncoder>::OutputValueType]>>;

#[derive(Default, PartialEq, Debug, Clone, Serialize, Deserialize)]
/// Shared implementation for growable and frozen dense datasets.
/// Wraps a contiguous buffer and the encoder so callers can treat both variants with the same API.
pub struct DenseDatasetGeneric<E, Data>
where
    E: DenseVectorEncoder,
    Data: AsRef<[E::OutputValueType]>,
{
    n_vecs: usize,
    data: Data,
    encoder: E,
}

impl<E, Data> sealed::Sealed for DenseDatasetGeneric<E, Data>
where
    E: DenseVectorEncoder,
    Data: AsRef<[E::OutputValueType]>,
{
}

impl<E, Data> SpaceUsage for DenseDatasetGeneric<E, Data>
where
    E: DenseVectorEncoder,
    Data: AsRef<[E::OutputValueType]> + SpaceUsage,
{
    fn space_usage_bytes(&self) -> usize {
        self.n_vecs.space_usage_bytes()
            + self.encoder.space_usage_bytes()
            + self.data.space_usage_bytes()
    }
}

impl<E, Data> DenseDatasetGeneric<E, Data>
where
    E: DenseVectorEncoder,
    Data: AsRef<[E::OutputValueType]>,
{
    /// Build a dataset from its raw encoded buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectorium::DenseDataset;
    /// use vectorium::encoders::dense_scalar::ScalarDenseQuantizer;
    /// use vectorium::distances::DotProduct;
    /// use vectorium::Dataset;
    ///
    /// let encoder = ScalarDenseQuantizer::<f32, f32, DotProduct>::new(2);
    /// let dataset: DenseDataset<_> =
    ///     DenseDataset::from_raw(vec![1.0f32, 2.0, 3.0, 4.0].into_boxed_slice(), 2, encoder);
    /// assert_eq!(dataset.len(), 2);
    /// ```
    #[inline]
    pub fn from_raw(data: Data, n_vecs: usize, encoder: E) -> Self {
        assert_eq!(
            data.as_ref().len(),
            n_vecs * encoder.output_dim(),
            "Data length must equal n_vecs * encoder.output_dim()"
        );
        Self {
            n_vecs,
            data,
            encoder,
        }
    }

    /// Access the contiguous storage backing the dataset.
    ///
    /// This is the same buffer that gets populated by `DenseDatasetGrowable::push`.
    #[inline]
    pub fn values(&self) -> &[E::OutputValueType] {
        self.data.as_ref()
    }

    /// Number of stored elements (all components are considered non-zero here).
    #[inline]
    pub fn nnz(&self) -> usize {
        self.data.as_ref().len()
    }

    // Note: par_iter returns EncodedVector (view)
    /// Iterate the dataset in parallel without reallocating temporary buffers.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectorium::DenseDataset;
    /// use vectorium::encoders::dense_scalar::ScalarDenseQuantizer;
    /// use vectorium::distances::DotProduct;
    /// use vectorium::Dataset;
    /// use rayon::iter::ParallelIterator;
    ///
    /// let encoder = ScalarDenseQuantizer::<f32, f32, DotProduct>::new(2);
    /// let dataset: DenseDataset<_> =
    ///     DenseDataset::from_raw(vec![1.0f32, 2.0, 3.0, 4.0].into_boxed_slice(), 2, encoder);
    /// let collected: Vec<_> = dataset.par_iter().map(|view| view.values().to_vec()).collect();
    /// assert_eq!(collected.len(), dataset.len());
    /// ```
    #[inline]
    pub fn par_iter(&self) -> impl ParallelIterator<Item = E::EncodedVector<'_>> {
        let m = self.encoder.output_dim();
        let data = self.data.as_ref();
        let n = self.n_vecs;

        (0..n).into_par_iter().map(move |i| {
            let start = i * m;
            let end = start + m;
            DenseVectorView::new(&data[start..end])
        })
    }
}

impl<E, Data> MeanDataset for DenseDatasetGeneric<E, Data>
where
    E: DenseVectorEncoder,
    Data: AsRef<[E::OutputValueType]>,
{
    type Vector = DenseVectorOwned<f32>;

    fn vector(&self, id: u32) -> Self::Vector {
        self.encoder().decode_vector(self.get(id as VectorId))
    }

    #[inline]
    fn vector_into(&self, id: u32, out: &mut Self::Vector) {
        self.encoder()
            .decode_into(self.get(id as VectorId), out.as_mut());
    }

    fn residual(&self, id: u32, centroid: &Self::Vector) -> Self::Vector {
        let mut v = self.vector(id);
        for (x, &c) in v.as_mut().iter_mut().zip(centroid.values()) {
            *x -= c;
        }
        v
    }

    fn mean(&self, ids: &[u32], weights: Option<&[f32]>) -> Self::Vector {
        let mut sum = vec![0.0f32; self.input_dim()];
        let mut decoded = vec![0.0f32; self.input_dim()];
        let mut count = 0.0f32;
        for &i in ids {
            let w = weights.map_or(1.0, |w| w[i as usize]);
            count += w;
            self.encoder()
                .decode_into(self.get(i as VectorId), &mut decoded);
            for (s, &x) in sum.iter_mut().zip(&decoded) {
                *s += x * w;
            }
        }
        if count > 0.0 {
            let inv = 1.0 / count;
            sum.iter_mut().for_each(|s| *s *= inv);
        }
        DenseVectorOwned::new(sum)
    }
}

impl<E, Data> Dataset for DenseDatasetGeneric<E, Data>
where
    E: DenseVectorEncoder,
    Data: AsRef<[E::OutputValueType]>,
{
    type Encoder = E;
    /// Frozen, `Box<[_]>`-backed variant. For `DenseDataset<E>` this *is* `Self`; for the
    /// growable variant it is the frozen counterpart, which is what a bulk copy should yield.
    type Owned = DenseDatasetGeneric<E, Box<[E::OutputValueType]>>;

    #[inline]
    fn encoder(&self) -> &E {
        &self.encoder
    }

    #[inline]
    fn len(&self) -> usize {
        self.n_vecs
    }

    #[inline]
    fn nnz(&self) -> usize {
        self.data.as_ref().len()
    }

    #[inline]
    fn range_from_id(&self, id: VectorId) -> std::ops::Range<usize> {
        let m = self.encoder.output_dim();
        let index = id as usize;
        assert!(index < self.n_vecs, "Index out of bounds.");
        let start = index * m;
        start..start + m
    }

    #[inline]
    fn id_from_range(&self, range: std::ops::Range<usize>) -> VectorId {
        let m = self.encoder.output_dim();

        if m == 0 {
            assert_eq!(
                range.start, range.end,
                "Range does not match vector boundaries."
            );
            return 0;
        }

        assert!(
            range.start.is_multiple_of(m),
            "Range does not match vector boundaries."
        );
        assert_eq!(
            range.end,
            range.start + m,
            "Range does not match vector boundaries."
        );
        let idx = range.start / m;
        assert!(idx < self.n_vecs, "Index out of bounds.");
        idx as VectorId
    }

    #[inline]
    fn get(&self, index: VectorId) -> E::EncodedVector<'_> {
        assert!(index < self.n_vecs as VectorId, "Index out of bounds.");
        let m = self.encoder.output_dim();
        let start = index as usize * m;
        let end = start + m;
        DenseVectorView::new(&self.data.as_ref()[start..end])
    }

    #[inline]
    fn get_with_range(&self, range: std::ops::Range<usize>) -> E::EncodedVector<'_> {
        DenseVectorView::new(&self.data.as_ref()[range])
    }

    #[inline]
    fn iter(&self) -> impl Iterator<Item = E::EncodedVector<'_>> {
        let m = self.encoder.output_dim();
        let data = self.data.as_ref();
        let n = self.n_vecs;

        (0..n).map(move |i| {
            let start = i * m;
            let end = start + m;
            DenseVectorView::new(&data[start..end])
        })
    }

    #[inline]
    fn prefetch_with_range(&self, range: std::ops::Range<usize>, max_bytes: usize) {
        crate::utils::prefetch_read_slice(&self.data.as_ref()[range], max_bytes);
    }

    fn permute(&self, permutation: &[usize]) -> Self::Owned {
        let n_vecs = self.n_vecs;
        assert_valid_permutation(permutation, n_vecs);

        // Gather in the new order rather than scatter into a pre-filled buffer: appending rows
        // only needs `Copy` (guaranteed by `ValueType`), while scattering would need `Default`.
        let dim = self.encoder.output_dim();
        let source = self.data.as_ref();
        let mut permuted = Vec::with_capacity(n_vecs * dim);
        for old_id in invert_permutation(permutation) {
            let old_start = old_id * dim;
            permuted.extend_from_slice(&source[old_start..old_start + dim]);
        }

        DenseDatasetGeneric {
            n_vecs,
            data: permuted.into_boxed_slice(),
            encoder: self.encoder.clone(),
        }
    }
}

// DatasetGrowable implementation
// We need to implement push which takes EncodedVector?
// No, DatasetGrowable takes InputVectorType to push.
// But InputVectorType was removed from VectorEncoder trait.
// It is now defined by `encode_vector` method signature which takes `DenseVectorView`.
// So we should adapt `DatasetGrowable` trait or implementation.
// `DenseDatasetGeneric` assumes inputs are compatible with `E`.

use crate::core::vector::DenseVectorView;
use crate::{Float, FromF32, ValueType};

impl<E> DatasetGrowable for DenseDatasetGeneric<E, Vec<E::OutputValueType>>
where
    E: DenseVectorEncoder,
{
    fn new(encoder: E) -> Self {
        Self {
            n_vecs: 0,
            data: Vec::new(),
            encoder,
        }
    }

    fn with_capacity(encoder: E, capacity: usize) -> Self {
        Self {
            n_vecs: 0,
            data: Vec::with_capacity(capacity * encoder.output_dim()),
            encoder,
        }
    }

    fn push<'a>(&mut self, vec: E::InputVector<'a>) {
        self.encoder.push_encoded(vec, &mut self.data);
        self.n_vecs += 1;
    }
}

impl<E> DenseDatasetGrowable<E>
where
    E: DenseVectorEncoder,
{
    /// Build a new growable dataset using the provided encoder.
    #[inline]
    pub fn new(encoder: E) -> Self {
        crate::DatasetGrowable::new(encoder)
    }

    /// Build a growable dataset with the provided encoder and reserved capacity.
    #[inline]
    pub fn with_capacity(encoder: E, capacity: usize) -> Self {
        crate::DatasetGrowable::with_capacity(encoder, capacity)
    }

    /// Return how many vectors can be stored without growing the underlying buffer.
    ///
    /// This mirrors the behavior of `Vec::capacity`, but expressed in vector units instead of scalar components.
    pub fn capacity(&self) -> usize {
        if self.encoder.output_dim() == 0 {
            0
        } else {
            self.data.capacity() / self.encoder.output_dim()
        }
    }

    /// Make room for `additional` vectors without extra reallocations.
    ///
    /// The argument counts vectors, so the method multiplies it by the encoder output dimension.
    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional * self.encoder.output_dim());
    }
}

impl<VIn, VOut, D>
    DenseDatasetGrowable<crate::encoders::dense_scalar::ScalarDenseQuantizer<VIn, VOut, D>>
where
    VIn: ValueType + crate::Float,
    VOut: ValueType + crate::Float + crate::FromF32,
    D: crate::distances::Distance + crate::encoders::dense_scalar::ScalarDenseSupportedDistance,
{
    /// Convenience constructor that creates a quantizer and an empty dataset for any scalar quantizer.
    pub fn with_dim(dim: usize) -> Self {
        let encoder = crate::encoders::dense_scalar::ScalarDenseQuantizer::new(dim);
        crate::DatasetGrowable::new(encoder)
    }

    /// Convenience constructor that also preallocates enough space for `capacity` vectors.
    pub fn with_dim_and_capacity(dim: usize, capacity: usize) -> Self {
        let encoder = crate::encoders::dense_scalar::ScalarDenseQuantizer::new(dim);
        Self {
            n_vecs: 0,
            data: Vec::with_capacity(capacity * dim),
            encoder,
        }
    }
}

impl<E> DenseDataset<E>
where
    E: DenseVectorEncoder,
{
    /// Build an immutable dataset by encoding all vectors in parallel.
    ///
    /// `flat_input`: all input vector values concatenated in row-major order;
    /// layout `[vec0_v0, ..., vec0_vD, vec1_v0, ...]`.
    ///
    /// Each vector is encoded independently on a rayon thread pool, straight into its own
    /// row of the final buffer: the slab is allocated once up front and each worker is handed
    /// the `&mut [OutputValueType]` slice it owns, so there is no per-vector allocation and no
    /// reassembly copy. Significantly faster than sequential `push` for any non-trivial
    /// encoder (e.g. PQ).
    ///
    /// Requires a **fixed-width** encoder: every record must be exactly
    /// [`output_dim`](crate::core::vector_encoder::VectorEncoder::output_dim) values long, since each row is a
    /// pre-sized slice of the slab. A record of any other length panics (see [`SliceSink`](crate::utils::SliceSink))
    /// rather than shifting every subsequent vector. Variable-length encodings belong in a
    /// packed dataset ([`PackedSparseDataset`](crate::datasets::packed_dataset::PackedSparseDataset)),
    /// which carries per-vector offsets.
    pub fn from_flat_par(encoder: E, flat_input: &[E::InputValueType], n_vecs: usize) -> Self
    where
        E: Sync,
        E::InputValueType: Sync,
        E::OutputValueType: Send,
    {
        let input_dim = encoder.input_dim();
        let output_dim = encoder.output_dim();

        assert_eq!(
            flat_input.len(),
            n_vecs * input_dim,
            "flat_input length must equal n_vecs * input_dim"
        );

        // Zero-initialized so encoders that OR their output into place (bit-plane packers) see
        // a clean row; every encoder then overwrites exactly `output_dim` values, which
        // `SliceSink::finish` enforces — a short record would otherwise leave those zeros
        // indistinguishable from encoder output.
        let mut data = vec![num_traits::Zero::zero(); n_vecs * output_dim];
        data.par_chunks_mut(output_dim)
            .zip(flat_input.par_chunks_exact(input_dim))
            .for_each(|(out, chunk)| {
                let mut sink = crate::utils::SliceSink::new(out);
                encoder.push_encoded(DenseVectorView::new(chunk), &mut sink);
                sink.finish();
            });

        Self::from_raw(data.into_boxed_slice(), n_vecs, encoder)
    }
}

impl<E> DenseDataset<E>
where
    E: DenseVectorEncoder<InputValueType = f32>,
{
    /// [`from_flat_par`](Self::from_flat_par) for a source held at half precision.
    ///
    /// The compressing encoders (PQ, RaBitQ) take `f32` input, so a caller holding an `f16`
    /// collection would otherwise have to materialize a full `f32` copy
    /// Each worker upcasts one vector at a time into a reusable scratch buffer, so
    /// the extra memory is `output_dim` floats per thread.
    ///
    /// The resulting codes are those of the `f16` values, which differ from codes derived from
    /// the original `f32` collection by at most the `f16` rounding of the input.
    pub fn from_flat_par_upcast<S>(encoder: E, flat_input: &[S], n_vecs: usize) -> Self
    where
        E: Sync,
        E::OutputValueType: Send,
        S: crate::ValueType + Sync,
    {
        let input_dim = encoder.input_dim();
        let output_dim = encoder.output_dim();

        assert_eq!(
            flat_input.len(),
            n_vecs * input_dim,
            "flat_input length must equal n_vecs * input_dim"
        );

        let mut data = vec![num_traits::Zero::zero(); n_vecs * output_dim];
        data.par_chunks_mut(output_dim)
            .zip(flat_input.par_chunks_exact(input_dim))
            .for_each_init(
                || vec![0.0f32; input_dim],
                |scratch, (out, chunk)| {
                    for (dst, src) in scratch.iter_mut().zip(chunk) {
                        *dst = src
                            .to_f32()
                            .expect("source value is not representable as f32");
                    }
                    let mut sink = crate::utils::SliceSink::new(out);
                    encoder.push_encoded(DenseVectorView::new(scratch), &mut sink);
                    sink.finish();
                },
            );

        Self::from_raw(data.into_boxed_slice(), n_vecs, encoder)
    }
}

impl<E> From<DenseDatasetGrowable<E>> for DenseDataset<E>
where
    E: DenseVectorEncoder,
{
    fn from(dataset: DenseDatasetGrowable<E>) -> Self {
        Self {
            n_vecs: dataset.n_vecs,
            data: dataset.data.into_boxed_slice(),
            encoder: dataset.encoder,
        }
    }
}

use crate::dataset::ConvertFrom;
use crate::distances::{DotProduct, SquaredEuclideanDistance};
use crate::encoders::dense_scalar::ScalarDenseQuantizer;
use crate::encoders::dense_scalar::ScalarDenseSupportedDistance;

/// Convert ScalarDenseDataset from SquaredEuclideanDistance to DotProduct without copying data
impl<VIn, VOut> From<DenseDataset<ScalarDenseQuantizer<VIn, VOut, SquaredEuclideanDistance>>>
    for DenseDataset<ScalarDenseQuantizer<VIn, VOut, DotProduct>>
where
    VIn: crate::ValueType + crate::Float,
    VOut: crate::ValueType + crate::Float + crate::FromF32,
{
    fn from(
        dataset: DenseDataset<ScalarDenseQuantizer<VIn, VOut, SquaredEuclideanDistance>>,
    ) -> Self {
        Self {
            n_vecs: dataset.n_vecs,
            data: dataset.data,
            encoder: ScalarDenseQuantizer::new(dataset.encoder.output_dim()),
        }
    }
}

/// Convert ScalarDenseDataset from DotProduct to SquaredEuclideanDistance without copying data
impl<VIn, VOut> From<DenseDataset<ScalarDenseQuantizer<VIn, VOut, DotProduct>>>
    for DenseDataset<ScalarDenseQuantizer<VIn, VOut, SquaredEuclideanDistance>>
where
    VIn: crate::ValueType + crate::Float,
    VOut: crate::ValueType + crate::Float + crate::FromF32,
{
    fn from(dataset: DenseDataset<ScalarDenseQuantizer<VIn, VOut, DotProduct>>) -> Self {
        Self {
            n_vecs: dataset.n_vecs,
            data: dataset.data,
            encoder: ScalarDenseQuantizer::new(dataset.encoder.output_dim()),
        }
    }
}

impl<SrcIn, Mid, DstOut, D, SrcStorage, DstStorage>
    ConvertFrom<&DenseDatasetGeneric<ScalarDenseQuantizer<SrcIn, Mid, D>, SrcStorage>>
    for DenseDatasetGeneric<ScalarDenseQuantizer<Mid, DstOut, D>, DstStorage>
where
    SrcIn: ValueType + Float,
    Mid: ValueType + Float + FromF32,
    DstOut: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
    ScalarDenseQuantizer<SrcIn, Mid, D>:
        DenseVectorEncoder<InputValueType = SrcIn, OutputValueType = Mid>,
    ScalarDenseQuantizer<Mid, DstOut, D>:
        DenseVectorEncoder<InputValueType = Mid, OutputValueType = DstOut>,
    SrcStorage: AsRef<[Mid]>,
    DstStorage: From<Box<[DstOut]>> + AsRef<[DstOut]>,
{
    type Config = ();

    fn convert_from(
        source: &DenseDatasetGeneric<ScalarDenseQuantizer<SrcIn, Mid, D>, SrcStorage>,
        _config: (),
    ) -> Self {
        let m = source.encoder.output_dim();
        let encoder = ScalarDenseQuantizer::<Mid, DstOut, D>::new(m);

        // Treat source data as a contiguous array of dense vectors of type Mid.
        let mut new_data = Vec::with_capacity(source.data.as_ref().len());
        let src_data = source.data.as_ref();

        for chunk in src_data.chunks_exact(m) {
            let vec_view = DenseVectorView::new(chunk);
            encoder.push_encoded(vec_view, &mut new_data);
        }

        Self {
            n_vecs: source.n_vecs,
            data: new_data.into_boxed_slice().into(),
            encoder,
        }
    }
}

impl<E, Data> crate::core::dataset::DenseData for DenseDatasetGeneric<E, Data>
where
    E: DenseVectorEncoder,
    Data: AsRef<[E::OutputValueType]>,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vector::DenseVectorView;
    use crate::dataset::ConvertFrom;
    use crate::distances::DotProduct;
    use crate::encoders::dense_scalar::ScalarDenseQuantizer;

    type PermuteEncoder = ScalarDenseQuantizer<f32, f32, DotProduct>;

    /// Rows `[10, 11] [20, 21] [30, 31] [40, 41]`, so each row is easy to identify.
    fn permutable_dense() -> DenseDataset<PermuteEncoder> {
        let mut growable = DenseDatasetGrowable::new(PermuteEncoder::new(2));
        for i in 1..=4 {
            let base = (i * 10) as f32;
            growable.push(DenseVectorView::new(&[base, base + 1.0]));
        }
        growable.into()
    }

    #[test]
    fn dense_residual_subtracts_per_coordinate() {
        use crate::core::dataset::MeanDataset;
        let c = crate::DenseVectorOwned::new(vec![1.0_f32, 2.5]);
        let r = permutable_dense().residual(1, &c);
        assert_eq!(r.values(), &[19.0, 18.5]);
    }

    #[test]
    fn dense_permute_moves_each_row_to_its_target_slot() {
        let dataset = permutable_dense();
        let permutation = [2usize, 0, 3, 1];

        let permuted = dataset.permute(&permutation);

        assert_eq!(permuted.len(), dataset.len());
        for (old_id, &new_id) in permutation.iter().enumerate() {
            assert_eq!(
                permuted.get(new_id as VectorId).values(),
                dataset.get(old_id as VectorId).values()
            );
        }
    }

    #[test]
    fn dense_permute_with_identity_is_a_no_op() {
        let dataset = permutable_dense();
        assert_eq!(dataset.permute(&[0, 1, 2, 3]), dataset);
    }

    #[test]
    fn dense_permute_then_inverse_round_trips() {
        let dataset = permutable_dense();
        let permutation = [2usize, 0, 3, 1];
        let inverse = crate::core::dataset::invert_permutation(&permutation);

        assert_eq!(dataset.permute(&permutation).permute(&inverse), dataset);
    }

    #[test]
    #[should_panic(expected = "permutation length")]
    fn dense_permute_rejects_wrong_length() {
        permutable_dense().permute(&[0, 1, 2]);
    }

    #[test]
    #[should_panic(expected = "more than one vector to position")]
    fn dense_permute_rejects_non_bijective_permutation() {
        permutable_dense().permute(&[0, 1, 1, 3]);
    }

    #[test]
    fn dense_dataset_range_and_id_are_consistent() {
        type Encoder = ScalarDenseQuantizer<f32, f32, DotProduct>;
        let encoder = Encoder::new(2);

        let mut growable = DenseDatasetGrowable::new(encoder);
        growable.push(DenseVectorView::new(&[1.0f32, 0.0]));
        growable.push(DenseVectorView::new(&[0.0f32, 1.0]));

        let frozen: DenseDataset<Encoder> = growable.into();
        let range = frozen.range_from_id(1);
        assert_eq!(range, 2..4);
        assert_eq!(frozen.id_from_range(range), 1);
    }

    #[test]
    fn dense_dataset_from_raw_rebuilds_vectors() {
        type Encoder = ScalarDenseQuantizer<f32, f32, DotProduct>;
        let encoder = Encoder::new(2);
        let raw = vec![1.0f32, 2.0, 3.0, 4.0];

        let dataset = DenseDataset::from_raw(raw.into_boxed_slice(), 2, encoder);
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.output_dim(), 2);
        let first = dataset.get(0);
        assert_eq!(first.values(), &[1.0f32, 2.0]);
    }

    #[test]
    fn dense_dataset_values_par_iter_and_space_usage() {
        type Encoder = ScalarDenseQuantizer<f32, f32, DotProduct>;

        let encoder = Encoder::new(2);
        let mut growable = DenseDatasetGrowable::new(encoder);
        growable.push(DenseVectorView::new(&[1.0f32, 2.0]));
        growable.push(DenseVectorView::new(&[3.0f32, 4.0]));

        let dataset: DenseDataset<Encoder> = growable.into();
        assert_eq!(dataset.values(), &[1.0f32, 2.0, 3.0, 4.0]);
        assert_eq!(dataset.nnz(), 4);
        assert!(dataset.space_usage_bytes() > 0);

        let iter_values: Vec<Vec<_>> = dataset.iter().map(|v| v.values().to_vec()).collect();
        assert_eq!(iter_values, vec![vec![1.0f32, 2.0], vec![3.0f32, 4.0]]);

        let par_values: Vec<Vec<_>> = dataset.par_iter().map(|v| v.values().to_vec()).collect();
        assert_eq!(par_values.len(), dataset.len());
        assert_eq!(par_values, iter_values);

        dataset.prefetch_with_range(0..2, 256);
    }

    #[test]
    #[should_panic(expected = "Data length must equal n_vecs * encoder.output_dim()")]
    fn dense_dataset_from_raw_length_mismatch_panics() {
        type Encoder = ScalarDenseQuantizer<f32, f32, DotProduct>;
        let encoder = Encoder::new(2);
        let _ = DenseDataset::from_raw(vec![1.0f32, 2.0].into_boxed_slice(), 2, encoder);
    }

    #[test]
    #[should_panic(expected = "Range does not match vector boundaries.")]
    fn id_from_range_bad_range_panics() {
        type Encoder = ScalarDenseQuantizer<f32, f32, DotProduct>;
        let encoder = Encoder::new(2);
        let dataset =
            DenseDataset::from_raw(vec![1.0f32, 2.0, 3.0, 4.0].into_boxed_slice(), 2, encoder);
        let _ = dataset.id_from_range(1..3);
    }

    #[test]
    fn id_from_range_zero_dim_returns_zero() {
        type Encoder = ScalarDenseQuantizer<f32, f32, DotProduct>;
        let encoder = Encoder::new(0);
        let dataset = DenseDataset::from_raw(vec![].into_boxed_slice(), 5, encoder);
        assert_eq!(dataset.id_from_range(0..0), 0);
    }

    #[test]
    fn get_with_range_returns_expected_view() {
        type Encoder = ScalarDenseQuantizer<f32, f32, DotProduct>;
        let encoder = Encoder::new(2);
        let dataset = DenseDataset::from_raw(
            vec![10.0f32, 20.0, 30.0, 40.0].into_boxed_slice(),
            2,
            encoder,
        );
        let view = dataset.get_with_range(2..4);
        assert_eq!(view.values(), &[30.0, 40.0]);
    }

    #[test]
    fn growable_capacity_and_reserve_affect_data() {
        type Encoder = ScalarDenseQuantizer<f32, f32, DotProduct>;
        let _encoder = Encoder::new(2);
        let mut growable =
            DenseDatasetGrowable::<ScalarDenseQuantizer<f32, f32, DotProduct>>::with_dim(2);
        assert_eq!(growable.capacity(), 0);
        growable.reserve(3);
        assert!(growable.capacity() >= 3);
        growable.push(DenseVectorView::new(&[0.0f32, 1.0]));
        assert!(growable.capacity() >= 1);
        growable.push(DenseVectorView::new(&[2.0f32, 3.0]));
        assert!(growable.len() >= 2);
        assert_eq!(growable.encoder.output_dim(), 2);
    }

    #[test]
    fn with_dim_constructors_return_dataset() {
        let growable =
            DenseDatasetGrowable::<ScalarDenseQuantizer<f32, f32, DotProduct>>::with_dim(3);
        assert_eq!(growable.encoder.input_dim(), 3);
        let with_capacity = DenseDatasetGrowable::<ScalarDenseQuantizer<f32, f32, DotProduct>>::with_dim_and_capacity(
            3, 4,
        );
        assert_eq!(with_capacity.encoder.output_dim(), 3);
        // Capacity is always >= 0 for unsigned types, so no need to assert
    }

    #[test]
    fn from_flat_par_produces_identical_output_to_sequential_push() {
        // Verify that from_flat_par is byte-for-byte identical to the sequential
        // push path. Uses ScalarDenseQuantizer<f32, f32> so the encoding is a
        // trivial identity cast — any divergence in assembly order would surface here.
        type Encoder = ScalarDenseQuantizer<f32, f32, DotProduct>;
        let n = 8;
        let dim = 3;
        let flat: Vec<f32> = (0..n * dim).map(|i| i as f32).collect();

        // Sequential path.
        let encoder = Encoder::new(dim);
        let mut sequential = DenseDatasetGrowable::new(encoder.clone());
        for chunk in flat.chunks_exact(dim) {
            sequential.push(DenseVectorView::new(chunk));
        }
        let sequential: DenseDataset<Encoder> = sequential.into();

        // Parallel path.
        let parallel = DenseDataset::from_flat_par(encoder, &flat, n);

        assert_eq!(sequential.values(), parallel.values());
        assert_eq!(sequential.len(), parallel.len());
    }

    #[test]
    fn with_dim_and_capacity_supported_for_generic_quantizers() {
        let growable =
            DenseDatasetGrowable::<ScalarDenseQuantizer<f32, f64, DotProduct>>::with_dim(4);
        assert_eq!(growable.encoder.output_dim(), 4);
        let with_capacity =
            DenseDatasetGrowable::<ScalarDenseQuantizer<f32, f64, DotProduct>>::with_dim_and_capacity(
                4, 8,
            );
        assert_eq!(with_capacity.encoder.output_dim(), 4);
        assert_eq!(with_capacity.capacity(), 8);
    }

    #[test]
    fn convert_from_dense_dataset_preserves_values() {
        type SrcEncoder = ScalarDenseQuantizer<f32, f32, DotProduct>;
        type MidEncoder = ScalarDenseQuantizer<f32, f32, DotProduct>;
        let encoder = SrcEncoder::new(2);
        let dataset =
            DenseDataset::from_raw(vec![5.0f32, 6.0, 7.0, 8.0].into_boxed_slice(), 2, encoder);
        let converted: DenseDatasetGeneric<MidEncoder, Vec<f32>> =
            ConvertFrom::convert_from(&dataset, ());
        assert_eq!(converted.len(), dataset.len());
        assert_eq!(converted.values(), dataset.values());
    }

    /// The upcasting encode path must produce exactly what the `f32` path produces when handed
    /// the same values. This is the contract that lets a caller hold the collection as `f16`
    /// without a full `f32` copy: the only difference from the `f32` path is the source's
    /// precision, never the encoding.
    #[test]
    fn from_flat_par_upcast_matches_from_flat_par_on_the_same_values() {
        use half::f16;

        let d = 8;
        let n = 40;
        let source_f16: Vec<f16> = (0..n * d)
            .map(|i| f16::from_f32((i as f32 * 0.37).sin() * 3.0))
            .collect();
        let source_f32: Vec<f32> = source_f16.iter().map(|v| v.to_f32()).collect();

        let encoder = ScalarDenseQuantizer::<f32, f16, DotProduct>::new(d);
        let from_f32 = DenseDataset::from_flat_par(encoder.clone(), &source_f32, n);
        let from_f16 = DenseDataset::from_flat_par_upcast(encoder, &source_f16, n);

        assert_eq!(from_f16.len(), from_f32.len());
        assert_eq!(from_f16.values(), from_f32.values());
    }
}
