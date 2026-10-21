use serde::{Deserialize, Serialize};
use std::hint::assert_unchecked;

use crate::SpaceUsage;
use crate::core::dataset::MeanDataset;
use crate::core::dataset::{assert_valid_permutation, invert_permutation};
use crate::core::sealed;
use crate::core::storage::{GrowableSparseStorage, ImmutableSparseStorage, SparseStorage};
use crate::core::vector::{SparseVectorOwned, SparseVectorView};
use crate::core::vector_encoder::{SparseVectorEncoder, VectorEncoder};
use crate::utils::{is_strictly_sorted, prefetch_read_slice};
use crate::{ComponentType, Float, FromF32, ValueType, VectorId};
use crate::{Dataset, DatasetGrowable, SparseData};
use num_traits::AsPrimitive;

use crate::dataset::ConvertFrom;
use crate::distances::{DotProduct, SquaredEuclideanDistance};

use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSlice};

/// A growable representation of a sparse dataset.
///
/// # Examples
///
/// ```rust
/// use vectorium::{Dataset, DatasetGrowable};
/// use vectorium::{DotProduct, PlainSparseDatasetGrowable, PlainSparseQuantizer};
///
/// // Create a new empty dataset
/// let quantizer = PlainSparseQuantizer::<u32, f32, DotProduct>::new(0, 0);
/// let dataset = PlainSparseDatasetGrowable::new(quantizer);
/// assert_eq!(dataset.len(), 0);
/// assert_eq!(dataset.nnz(), 0);
/// ```
///
pub type SparseDatasetGrowable<E> = SparseDatasetGeneric<E, GrowableSparseStorage<E>>;

// Implementation of a (immutable) sparse dataset.
pub type SparseDataset<E> = SparseDatasetGeneric<E, ImmutableSparseStorage<E>>;

/// Sparse dataset storing variable-length vectors with offsets.
///
/// # Type Parameters
///
/// - `E`: The encoder type (quantizer types are encoders)
/// - `S`: The storage backend (defaults to `GrowableSparseStorage<E>`)
///
/// # Example
/// ```
/// use vectorium::{
///     Dataset, DotProduct, DatasetGrowable, PlainSparseDataset, PlainSparseQuantizer,
///     SparseDatasetGrowable, SparseVectorView, VectorView, VectorEncoder,
/// };
///
/// let quantizer = PlainSparseQuantizer::<u16, f32, DotProduct>::new(5, 5);
/// let mut dataset = SparseDatasetGrowable::new(quantizer);
/// dataset.push(SparseVectorView::new(&[1_u16, 3], &[1.0, 2.0]));
///
/// let frozen: PlainSparseDataset<u16, f32, DotProduct> = dataset.into();
/// let v = frozen.get(0);
/// assert_eq!(v.components(), &[1_u16, 3]);
/// assert_eq!(v.values(), &[1.0, 2.0]);
///
/// let range = frozen.range_from_id(0);
/// let v = frozen.get_with_range(range);
/// assert_eq!(v.components(), &[1_u16, 3]);
/// assert_eq!(v.values(), &[1.0, 2.0]);
/// ```
#[derive(Default, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct SparseDatasetGeneric<E, S = GrowableSparseStorage<E>>
where
    E: SparseVectorEncoder,
    S: SparseStorage<E>,
{
    storage: S,
    encoder: E,
}

impl<E, S> sealed::Sealed for SparseDatasetGeneric<E, S>
where
    E: SparseVectorEncoder,
    S: SparseStorage<E>,
{
}

impl<E, S> SparseDatasetGeneric<E, S>
where
    E: SparseVectorEncoder,
    S: SparseStorage<E>,
{
    /// Number of non-zero components stored across the entire dataset.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectorium::{Dataset, DatasetGrowable};
    /// use vectorium::{DotProduct, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVectorView};
    ///
    /// let quantizer = PlainSparseQuantizer::<u32, f32, DotProduct>::new(5, 5);
    /// let mut dataset = PlainSparseDatasetGrowable::<u32, f32, DotProduct>::new(quantizer);
    ///
    /// dataset.push(SparseVectorView::new(&[0, 2], &[1.0, 2.0]));
    /// assert_eq!(dataset.nnz(), 2);
    /// ```
    #[inline]
    pub fn nnz(&self) -> usize {
        self.storage.components().as_ref().len()
    }

    /// Translate a stored `VectorId` into the component/value range in the storage buffers.
    ///
    /// The returned range indexes into `components()`/`values()` rather than into logical vector indices.
    /// # Examples
    ///
    /// ```
    /// use vectorium::{Dataset, DatasetGrowable};
    /// use vectorium::{DotProduct, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVectorView};
    ///
    /// let quantizer = PlainSparseQuantizer::<u32, f32, DotProduct>::new(5, 5);
    /// let mut dataset = PlainSparseDatasetGrowable::<u32, f32, DotProduct>::new(quantizer);
    ///
    /// dataset.push(SparseVectorView::new(&[0, 2, 4], &[1.0, 2.0, 3.0]));
    /// let range = dataset.range_from_id(0);
    /// assert_eq!(range, 0..3);
    /// ```
    #[inline]
    pub fn range_from_id(&self, id: VectorId) -> std::ops::Range<usize> {
        let offsets = self.storage.offsets().as_ref();
        let index = id as usize;
        assert!(index + 1 < offsets.len(), "Index out of bounds.");
        offsets[index]..offsets[index + 1]
    }

    /// Translate a storage range back to the `VectorId` that owns it.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectorium::{Dataset, DatasetGrowable};
    /// use vectorium::{DotProduct, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVectorView};
    ///
    /// let quantizer = PlainSparseQuantizer::<u32, f32, DotProduct>::new(5, 5);
    /// let mut dataset = PlainSparseDatasetGrowable::<u32, f32, DotProduct>::new(quantizer);
    ///
    /// dataset.push(SparseVectorView::new(&[0, 2, 4], &[1.0, 2.0, 3.0]));
    /// dataset.push(SparseVectorView::new(&[1, 3], &[4.0, 5.0]));
    /// let range = dataset.range_from_id(1);
    /// assert_eq!(dataset.id_from_range(range), 1);
    /// ```
    #[inline]
    pub fn id_from_range(&self, range: std::ops::Range<usize>) -> VectorId {
        let offsets = self.storage.offsets().as_ref();
        let idx = offsets.binary_search(&range.start).unwrap();
        assert_eq!(
            offsets[idx + 1],
            range.end,
            "Range does not match vector boundaries."
        );
        idx as VectorId
    }

    /// Parallel iterator over all vectors as slice-backed `SparseVectorView`.
    ///
    /// The returned views borrow from the shared storage buffers, so this iterator avoids
    /// heap allocations while still providing per-vector access.
    #[inline]
    pub fn par_iter(&self) -> impl IndexedParallelIterator<Item = E::EncodedVector<'_>> + '_ {
        let offsets = self.storage.offsets().as_ref();
        let components = self.storage.components().as_ref();
        let values = self.storage.values().as_ref();

        // https://github.com/rayon-rs/rayon/pull/789
        offsets.par_windows(2).map(move |window| {
            let &[start, end] = window else {
                unsafe { std::hint::unreachable_unchecked() }
            };
            SparseVectorView::new(&components[start..end], &values[start..end])
        })
    }
}

impl<E, S> MeanDataset for SparseDatasetGeneric<E, S>
where
    E: SparseVectorEncoder,
    S: SparseStorage<E>,
{
    type Vector = SparseVectorOwned<E::InputComponentType, f32>;

    fn vector(&self, id: u32) -> Self::Vector {
        self.encoder().decode_vector(self.get(id as VectorId))
    }

    fn residual(&self, id: u32, centroid: &Self::Vector) -> Self::Vector {
        let v = self.encoder().decode_vector(self.get(id as VectorId));
        let (vc, vv) = (v.components(), v.values());
        let (cc, cv) = (centroid.components(), centroid.values());
        let mut components = Vec::with_capacity(vc.len() + cc.len());
        let mut values = Vec::with_capacity(vc.len() + cc.len());
        let (mut i, mut j) = (0, 0);
        while i < vc.len() || j < cc.len() {
            if j == cc.len() || (i < vc.len() && vc[i] < cc[j]) {
                components.push(vc[i]);
                values.push(vv[i]);
                i += 1;
            } else if i == vc.len() || cc[j] < vc[i] {
                components.push(cc[j]);
                values.push(-cv[j]);
                j += 1;
            } else {
                components.push(vc[i]);
                values.push(vv[i] - cv[j]);
                i += 1;
                j += 1;
            }
        }
        SparseVectorOwned::new(components, values)
    }

    /// The exact mean: its nonzeros are the union of the members' components.
    fn mean(&self, ids: &[u32], weights: Option<&[f32]>) -> Self::Vector {
        let mut entries: Vec<(E::InputComponentType, f32)> = Vec::new();
        let mut count = 0.0f32;
        for &i in ids {
            let w = weights.map_or(1.0, |w| w[i as usize]);
            count += w;
            let vector = self.encoder().decode_vector(self.get(i as VectorId));
            entries.extend(vector.iter().map(|(c, x)| (c, x * w)));
        }
        // Stable sort: equal components keep member order, so the sums are deterministic.
        entries.sort_by_key(|&(c, _)| c);
        let inv = if count > 0.0 { 1.0 / count } else { 0.0 };
        let mut components = Vec::new();
        let mut values: Vec<f32> = Vec::new();
        for (c, x) in entries {
            if components.last() == Some(&c) {
                *values.last_mut().unwrap() += x;
            } else {
                components.push(c);
                values.push(x);
            }
        }
        values.iter_mut().for_each(|v| *v *= inv);
        SparseVectorOwned::new(components, values)
    }
}

impl<E, S> Dataset for SparseDatasetGeneric<E, S>
where
    E: SparseVectorEncoder,
    S: SparseStorage<E>,
{
    type Encoder = E;
    /// Frozen, [`ImmutableSparseStorage`]-backed variant; identical to `Self` for
    /// [`SparseDataset`].
    type Owned = SparseDatasetGeneric<E, ImmutableSparseStorage<E>>;

    #[inline]
    fn nnz(&self) -> usize {
        self.storage.components().as_ref().len()
    }

    #[inline]
    fn range_from_id(&self, id: VectorId) -> std::ops::Range<usize> {
        let offsets = self.storage.offsets().as_ref();
        let index = id as usize;
        assert!(index + 1 < offsets.len(), "Index out of bounds.");
        offsets[index]..offsets[index + 1]
    }

    #[inline]
    fn id_from_range(&self, range: std::ops::Range<usize>) -> VectorId {
        let offsets = self.storage.offsets().as_ref();
        let idx = offsets.binary_search(&range.start).unwrap();
        assert_eq!(
            offsets[idx + 1],
            range.end,
            "Range does not match vector boundaries."
        );
        idx as VectorId
    }

    /// Retrieves the components and values of the sparse vector at the specified index.
    ///
    /// This method returns a tuple containing slices of components and values
    /// of the sparse vector located at the given index.
    ///
    /// # Panics
    ///
    /// Panics if the specified index is out of range.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectorium::{Dataset, DatasetGrowable};
    /// use vectorium::VectorView;
    /// use vectorium::{DotProduct, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVectorView};
    ///
    /// let quantizer = PlainSparseQuantizer::<u32, f32, DotProduct>::new(5, 5);
    /// let mut dataset = PlainSparseDatasetGrowable::<u32, f32, DotProduct>::new(quantizer);
    ///
    /// dataset.push(SparseVectorView::new(&[0, 2, 4], &[1.0, 2.0, 3.0]));
    /// dataset.push(SparseVectorView::new(&[1, 3], &[4.0, 5.0]));
    ///
    /// let v = dataset.get(1);
    /// assert_eq!(v.components(), &[1, 3]);
    /// assert_eq!(v.values(), &[4.0, 5.0]);
    /// ```
    #[inline]
    fn get(&self, index: VectorId) -> E::EncodedVector<'_> {
        let range = self.range_from_id(index);
        self.get_with_range(range)
    }

    /// Returns a view into the specified component/value range so callers can decode subsets.
    ///
    /// # Safety
    ///
    /// The range must come from `range_from_id` or otherwise align with vector boundaries.
    #[inline]
    fn get_with_range<'a>(&'a self, range: std::ops::Range<usize>) -> E::EncodedVector<'a> {
        let components = self.storage.components().as_ref();
        let values = self.storage.values().as_ref();
        unsafe { assert_unchecked(components.len() == values.len()) };

        let v_components = &components[range.clone()];
        let v_values = &values[range];

        SparseVectorView::new(v_components, v_values)
    }

    #[inline]
    fn prefetch_with_range(&self, range: std::ops::Range<usize>, max_bytes: usize) {
        let components = self.storage.components().as_ref();
        let values = self.storage.values().as_ref();
        unsafe { assert_unchecked(components.len() == values.len()) };

        prefetch_read_slice(&components[range.clone()], max_bytes);
        prefetch_read_slice(&values[range], max_bytes);
    }

    fn encoder(&self) -> &E {
        &self.encoder
    }

    /// Returns an iterator over the vectors of the dataset.
    ///
    /// This method returns an iterator that yields references to each vector in the dataset.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectorium::{Dataset, DatasetGrowable};
    /// use vectorium::VectorView;
    /// use vectorium::{DotProduct, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVectorView};
    ///
    /// let data = vec![
    ///     (vec![0_u32, 2, 4], vec![1.0_f32, 2.0, 3.0]),
    ///     (vec![1_u32, 3], vec![4.0_f32, 5.0]),
    ///     (vec![0_u32, 1, 2, 3], vec![1.0_f32, 2.0, 3.0, 4.0]),
    /// ];
    ///
    /// let quantizer = PlainSparseQuantizer::<u32, f32, DotProduct>::new(5, 5);
    /// let mut dataset = PlainSparseDatasetGrowable::<u32, f32, DotProduct>::new(quantizer);
    /// for (c, v) in data.iter() {
    ///     dataset.push(SparseVectorView::new(c.as_slice(), v.as_slice()));
    /// }
    ///
    /// for (vec, (c, v)) in dataset.iter().zip(data.iter()) {
    ///     assert_eq!(vec.components(), c.as_slice());
    ///     assert_eq!(vec.values(), v.as_slice());
    /// }
    /// ```
    fn iter<'a>(&'a self) -> impl Iterator<Item = E::EncodedVector<'a>> {
        let offsets = self.storage.offsets().as_ref();
        let components = self.storage.components().as_ref();
        let values = self.storage.values().as_ref();
        offsets.windows(2).map(move |window| {
            let start = window[0];
            let end = window[1];
            SparseVectorView::new(&components[start..end], &values[start..end])
        })
    }

    // /// Returns an iterator over the sparse vector with id `vec_id`.
    // ///
    // /// # Panics
    // /// Panics if the `vec_id` is out of bounds.
    // fn iter(
    //     &self,
    //     vec_id: usize,
    // ) -> Zip<std::slice::Iter<C>, std::slice::Iter<V>> {
    //     let (v_components, v_values) = self.get(vec_id);

    //     v_components.iter().zip(v_values.iter())
    // }

    /// Returns the number of vectors in the dataset.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectorium::{Dataset, DatasetGrowable};
    /// use vectorium::{DotProduct, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVectorView};
    ///
    /// let quantizer = PlainSparseQuantizer::<u32, f32, DotProduct>::new(5, 5);
    /// let mut dataset = PlainSparseDatasetGrowable::<u32, f32, DotProduct>::new(quantizer);
    ///
    /// dataset.push(SparseVectorView::new(&[0, 2, 4], &[1.0, 2.0, 3.0]));
    /// dataset.push(SparseVectorView::new(&[1, 3], &[4.0, 5.0]));
    /// dataset.push(SparseVectorView::new(&[0, 1, 2, 3], &[1.0, 2.0, 3.0, 4.0]));
    ///
    /// assert_eq!(dataset.len(), 3);
    /// ```
    fn len(&self) -> usize {
        self.storage.offsets().as_ref().len() - 1
    }

    fn permute(&self, permutation: &[usize]) -> Self::Owned {
        let n_vecs = self.len();
        assert_valid_permutation(permutation, n_vecs);

        // Rows are variable-width: gather them in the new order and rebuild the offsets by
        // prefix sum. Copying the CSR ranges verbatim keeps the encoded values bit-identical.
        let old_by_new = invert_permutation(permutation);
        let offsets = self.storage.offsets().as_ref();
        let components = self.storage.components().as_ref();
        let values = self.storage.values().as_ref();

        let mut new_offsets = Vec::with_capacity(n_vecs + 1);
        new_offsets.push(0);
        let mut new_components = Vec::with_capacity(components.len());
        let mut new_values = Vec::with_capacity(values.len());

        for old_id in old_by_new {
            let range = offsets[old_id]..offsets[old_id + 1];
            new_components.extend_from_slice(&components[range.clone()]);
            new_values.extend_from_slice(&values[range]);
            new_offsets.push(new_components.len());
        }

        SparseDatasetGeneric {
            storage: ImmutableSparseStorage::from(GrowableSparseStorage::from_parts(
                new_offsets,
                new_components,
                new_values,
            )),
            encoder: self.encoder.clone(),
        }
    }
}

impl<E> SparseDataset<E>
where
    E: SparseVectorEncoder,
{
    /// Builds a dataset from CSR arrays already in `encoder`'s output types, for
    /// readers that produce them in bulk rather than pushing vector by vector.
    /// The caller guarantees that `offsets` starts at `0`, is non-decreasing and
    /// ends at the length of both `components` and `values`, and that every
    /// vector's components are strictly ascending and below `encoder.input_dim()`.
    pub(crate) fn from_parts(
        offsets: Vec<usize>,
        components: Vec<E::OutputComponentType>,
        values: Vec<E::OutputValueType>,
        encoder: E,
    ) -> Self {
        SparseDatasetGeneric {
            storage: ImmutableSparseStorage::from(GrowableSparseStorage::from_parts(
                offsets, components, values,
            )),
            encoder,
        }
    }
}

// Unfortunately, Rust doesn't yet support specialization, meaning that we can't use From too generically (otherwise it fails due to reimplementing `From<T> for T`)

impl<E> From<SparseDatasetGrowable<E>> for SparseDataset<E>
where
    E: SparseVectorEncoder,
{
    /// Converts a mutable sparse dataset into an immutable one.
    ///
    /// This function consumes the provided `SparseDatasetGrowable<C, V>` and produces
    /// a corresponding immutable `SparseDataset<C, V>` instance. The conversion is performed
    /// by transferring ownership of the internal data structures from the mutable dataset
    /// to the immutable one.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectorium::{Dataset, DatasetGrowable};
    /// use vectorium::{DotProduct, PlainSparseDataset, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVectorView};
    ///
    /// let quantizer = PlainSparseQuantizer::<u32, f32, DotProduct>::new(5, 5);
    /// let mut growable_dataset = PlainSparseDatasetGrowable::<u32, f32, DotProduct>::new(quantizer);
    ///
    /// growable_dataset.push(SparseVectorView::new(&[0, 2, 4], &[1.0, 2.0, 3.0]));
    /// growable_dataset.push(SparseVectorView::new(&[1, 3], &[4.0, 5.0]));
    /// growable_dataset.push(SparseVectorView::new(&[0, 1, 2, 3], &[1.0, 2.0, 3.0, 4.0]));
    ///
    /// let immutable_dataset: PlainSparseDataset<u32, f32, DotProduct> = growable_dataset.into();
    /// assert_eq!(immutable_dataset.nnz(), 9);
    /// ```
    fn from(dataset: SparseDatasetGrowable<E>) -> Self {
        Self {
            storage: dataset.storage.into(),
            encoder: dataset.encoder,
        }
    }
}

impl<E> From<SparseDataset<E>> for SparseDatasetGrowable<E>
where
    E: SparseVectorEncoder,
{
    /// Converts an immutable sparse dataset into a mutable one.
    ///
    /// This function consumes the provided `SparseDataset<C, V>` and produces
    /// a corresponding mutable `SparseDatasetGrowable<C, V>` instance. The conversion is performed
    /// by transferring ownership of the internal data structures from the immutable dataset
    /// to the mutable one.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectorium::{Dataset, DatasetGrowable};
    /// use vectorium::{DotProduct, PlainSparseDataset, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVectorView};
    ///
    /// let quantizer = PlainSparseQuantizer::<u32, f32, DotProduct>::new(5, 5);
    /// let mut growable_dataset = PlainSparseDatasetGrowable::<u32, f32, DotProduct>::new(quantizer);
    ///
    /// growable_dataset.push(SparseVectorView::new(&[0, 2, 4], &[1.0, 2.0, 3.0]));
    /// growable_dataset.push(SparseVectorView::new(&[1, 3], &[4.0, 5.0]));
    ///
    /// let immutable_dataset: PlainSparseDataset<u32, f32, DotProduct> = growable_dataset.into();
    ///
    /// // Convert immutable dataset back to mutable
    /// let mut growable_dataset_again: PlainSparseDatasetGrowable<u32, f32, DotProduct> = immutable_dataset.into();
    /// growable_dataset_again.push(SparseVectorView::new(&[1, 4], &[1.0, 3.0]));
    ///
    /// assert_eq!(growable_dataset_again.nnz(), 7);
    /// ```
    fn from(dataset: SparseDataset<E>) -> Self {
        Self {
            storage: dataset.storage.into(),
            encoder: dataset.encoder,
        }
    }
}

impl<C, InValue, OutValue, D>
    SparseDatasetGeneric<
        crate::ScalarSparseQuantizer<C, InValue, OutValue, D>,
        GrowableSparseStorage<crate::ScalarSparseQuantizer<C, InValue, OutValue, D>>,
    >
where
    C: ComponentType,
    InValue: ValueType + Float,
    OutValue: ValueType + Float + FromF32,
    D: crate::ScalarSparseSupportedDistance,
{
    /// Relabels a scalar-quantized growable dataset as plain without re-encoding.
    pub fn relabel_as_plain(
        self,
    ) -> SparseDatasetGrowable<crate::PlainSparseQuantizer<C, OutValue, D>> {
        let dim = self.encoder.output_dim();
        let storage = self
            .storage
            .relabel::<crate::PlainSparseQuantizer<C, OutValue, D>>();

        SparseDatasetGeneric {
            storage,
            encoder: crate::PlainSparseQuantizer::<C, OutValue, D>::new(dim, dim),
        }
    }
}

impl<C, OutValue, D>
    SparseDatasetGeneric<
        crate::PlainSparseQuantizer<C, OutValue, D>,
        GrowableSparseStorage<crate::PlainSparseQuantizer<C, OutValue, D>>,
    >
where
    C: ComponentType,
    OutValue: ValueType + Float + FromF32,
    D: crate::ScalarSparseSupportedDistance,
{
    /// Relabels a plain-quantized growable dataset as scalar without re-encoding.
    pub fn relabel_as_scalar<InValue>(
        self,
    ) -> SparseDatasetGrowable<crate::ScalarSparseQuantizer<C, InValue, OutValue, D>>
    where
        InValue: ValueType + Float,
    {
        let dim = self.encoder.output_dim();
        let storage = self
            .storage
            .relabel::<crate::ScalarSparseQuantizer<C, InValue, OutValue, D>>();

        SparseDatasetGeneric {
            storage,
            encoder: crate::ScalarSparseQuantizer::<C, InValue, OutValue, D>::new(dim, dim),
        }
    }
}

impl<C, InValue, OutValue, D>
    SparseDatasetGeneric<
        crate::ScalarSparseQuantizer<C, InValue, OutValue, D>,
        ImmutableSparseStorage<crate::ScalarSparseQuantizer<C, InValue, OutValue, D>>,
    >
where
    C: ComponentType,
    InValue: ValueType + Float,
    OutValue: ValueType + Float + FromF32,
    D: crate::ScalarSparseSupportedDistance,
{
    /// Relabels a scalar-quantized immutable dataset as plain without re-encoding.
    pub fn relabel_as_plain(self) -> SparseDataset<crate::PlainSparseQuantizer<C, OutValue, D>> {
        let dim = self.encoder.output_dim();
        let storage = self
            .storage
            .relabel::<crate::PlainSparseQuantizer<C, OutValue, D>>();

        SparseDatasetGeneric {
            storage,
            encoder: crate::PlainSparseQuantizer::<C, OutValue, D>::new(dim, dim),
        }
    }
}

impl<C, OutValue, D>
    SparseDatasetGeneric<
        crate::PlainSparseQuantizer<C, OutValue, D>,
        ImmutableSparseStorage<crate::PlainSparseQuantizer<C, OutValue, D>>,
    >
where
    C: ComponentType,
    OutValue: ValueType + Float + FromF32,
    D: crate::ScalarSparseSupportedDistance,
{
    /// Relabels a plain-quantized immutable dataset as scalar without re-encoding.
    pub fn relabel_as_scalar<InValue>(
        self,
    ) -> SparseDataset<crate::ScalarSparseQuantizer<C, InValue, OutValue, D>>
    where
        InValue: ValueType + Float,
    {
        let dim = self.encoder.output_dim();
        let storage = self
            .storage
            .relabel::<crate::ScalarSparseQuantizer<C, InValue, OutValue, D>>();

        SparseDatasetGeneric {
            storage,
            encoder: crate::ScalarSparseQuantizer::<C, InValue, OutValue, D>::new(dim, dim),
        }
    }
}

/// Convert ScalarSparseDataset from SquaredEuclideanDistance to DotProduct without copying data
impl<C, InValue, OutValue>
    From<
        SparseDataset<crate::ScalarSparseQuantizer<C, InValue, OutValue, SquaredEuclideanDistance>>,
    > for SparseDataset<crate::ScalarSparseQuantizer<C, InValue, OutValue, DotProduct>>
where
    C: ComponentType,
    InValue: ValueType + Float,
    OutValue: ValueType + Float + FromF32,
{
    fn from(
        dataset: SparseDataset<
            crate::ScalarSparseQuantizer<C, InValue, OutValue, SquaredEuclideanDistance>,
        >,
    ) -> Self {
        let dim = dataset.encoder.output_dim();
        SparseDatasetGeneric {
            storage: dataset.storage.relabel(),
            encoder: crate::ScalarSparseQuantizer::new(dim, dim),
        }
    }
}

/// Convert ScalarSparseDataset from DotProduct to SquaredEuclideanDistance without copying data
impl<C, InValue, OutValue>
    From<SparseDataset<crate::ScalarSparseQuantizer<C, InValue, OutValue, DotProduct>>>
    for SparseDataset<crate::ScalarSparseQuantizer<C, InValue, OutValue, SquaredEuclideanDistance>>
where
    C: ComponentType,
    InValue: ValueType + Float,
    OutValue: ValueType + Float + FromF32,
{
    fn from(
        dataset: SparseDataset<crate::ScalarSparseQuantizer<C, InValue, OutValue, DotProduct>>,
    ) -> Self {
        let dim = dataset.encoder.output_dim();
        SparseDatasetGeneric {
            storage: dataset.storage.relabel(),
            encoder: crate::ScalarSparseQuantizer::new(dim, dim),
        }
    }
}

impl<C, SrcIn, Mid, DstOut, D, SrcStorage, DstStorage>
    ConvertFrom<&SparseDatasetGeneric<crate::ScalarSparseQuantizer<C, SrcIn, Mid, D>, SrcStorage>>
    for SparseDatasetGeneric<crate::ScalarSparseQuantizer<C, Mid, DstOut, D>, DstStorage>
where
    C: ComponentType,
    SrcIn: ValueType + Float,
    Mid: ValueType + Float + FromF32,
    DstOut: ValueType + Float + FromF32,
    D: crate::ScalarSparseSupportedDistance,
    crate::ScalarSparseQuantizer<C, SrcIn, Mid, D>: SparseVectorEncoder<
            InputComponentType = C,
            InputValueType = SrcIn,
            OutputComponentType = C,
            OutputValueType = Mid,
        >,
    crate::ScalarSparseQuantizer<C, Mid, DstOut, D>: SparseVectorEncoder<
            InputComponentType = C,
            InputValueType = Mid,
            OutputComponentType = C,
            OutputValueType = DstOut,
        >,
    SrcStorage: SparseStorage<crate::ScalarSparseQuantizer<C, SrcIn, Mid, D>>,
    DstStorage: SparseStorage<crate::ScalarSparseQuantizer<C, Mid, DstOut, D>>
        + From<GrowableSparseStorage<crate::ScalarSparseQuantizer<C, Mid, DstOut, D>>>,
{
    type Config = ();

    fn convert_from(
        source: &SparseDatasetGeneric<crate::ScalarSparseQuantizer<C, SrcIn, Mid, D>, SrcStorage>,
        _config: (),
    ) -> Self {
        let n_vecs = source.len();
        let nnz = source.nnz();
        let dim = source.encoder.output_dim();
        let encoder = crate::ScalarSparseQuantizer::<C, Mid, DstOut, D>::new(dim, dim);

        let mut storage =
            GrowableSparseStorage::<crate::ScalarSparseQuantizer<C, Mid, DstOut, D>>::with_capacity(
                n_vecs, nnz,
            );

        for src_vec in crate::datasets::sparse_dataset::SparseDatasetIter::new(source) {
            encoder.push_encoded(
                SparseVectorView::<C, Mid>::new(src_vec.components(), src_vec.values()),
                &mut storage.components,
                &mut storage.values,
            );
            storage.offsets.push(storage.components.len());
        }

        SparseDatasetGeneric {
            storage: storage.into(),
            encoder,
        }
    }
}

impl<C, SrcIn, Mid, DstOut, D>
    ConvertFrom<SparseDataset<crate::ScalarSparseQuantizer<C, SrcIn, Mid, D>>>
    for SparseDataset<crate::ScalarSparseQuantizer<C, Mid, DstOut, D>>
where
    C: ComponentType,
    SrcIn: ValueType + Float,
    Mid: ValueType + Float + FromF32,
    DstOut: ValueType + Float + FromF32,
    D: crate::ScalarSparseSupportedDistance,
    crate::ScalarSparseQuantizer<C, SrcIn, Mid, D>: SparseVectorEncoder<
            InputComponentType = C,
            InputValueType = SrcIn,
            OutputComponentType = C,
            OutputValueType = Mid,
        >,
    crate::ScalarSparseQuantizer<C, Mid, DstOut, D>: SparseVectorEncoder<
            InputComponentType = C,
            InputValueType = Mid,
            OutputComponentType = C,
            OutputValueType = DstOut,
        >,
{
    type Config = ();

    fn convert_from(
        source: SparseDataset<crate::ScalarSparseQuantizer<C, SrcIn, Mid, D>>,
        _config: (),
    ) -> Self {
        Self::convert_from(&source, ())
    }
}

impl<C, SrcIn, Mid, DstOut, D>
    ConvertFrom<SparseDatasetGrowable<crate::ScalarSparseQuantizer<C, SrcIn, Mid, D>>>
    for SparseDatasetGrowable<crate::ScalarSparseQuantizer<C, Mid, DstOut, D>>
where
    C: ComponentType,
    SrcIn: ValueType + Float,
    Mid: ValueType + Float + FromF32,
    DstOut: ValueType + Float + FromF32,
    D: crate::ScalarSparseSupportedDistance,
    crate::ScalarSparseQuantizer<C, SrcIn, Mid, D>: SparseVectorEncoder<
            InputComponentType = C,
            InputValueType = SrcIn,
            OutputComponentType = C,
            OutputValueType = Mid,
        >,
    crate::ScalarSparseQuantizer<C, Mid, DstOut, D>: SparseVectorEncoder<
            InputComponentType = C,
            InputValueType = Mid,
            OutputComponentType = C,
            OutputValueType = DstOut,
        >,
{
    type Config = ();

    fn convert_from(
        source: SparseDatasetGrowable<crate::ScalarSparseQuantizer<C, SrcIn, Mid, D>>,
        _config: (),
    ) -> Self {
        Self::convert_from(&source, ())
    }
}

/// A struct to iterate over the *raw* sparse dataset storage as slice-backed `SparseVectorView`.
///
/// This iterator is independent from the encoder’s `EncodedVector` choice and is
/// used by internal utilities (including the parallel iterator plumbing).
#[derive(Clone)]
pub struct SparseDatasetIter<'a, C, V>
where
    C: ComponentType,
    V: ValueType,
{
    // Last offset consumed from the front; used to rebase remaining offsets to the current slices.
    front_base_offset: usize,
    offsets: &'a [usize],
    components: &'a [C],
    values: &'a [V],
}

impl<'a, C, V> SparseDatasetIter<'a, C, V>
where
    C: ComponentType,
    V: ValueType,
{
    #[inline]
    pub fn new<E, S>(dataset: &'a SparseDatasetGeneric<E, S>) -> Self
    where
        E: SparseVectorEncoder<OutputComponentType = C, OutputValueType = V>,
        S: SparseStorage<E>,
    {
        Self {
            front_base_offset: 0,
            offsets: &dataset.storage.offsets().as_ref()[1..],
            components: dataset.storage.components().as_ref(),
            values: dataset.storage.values().as_ref(),
        }
    }
}

impl<E> DatasetGrowable for SparseDatasetGrowable<E>
where
    E: SparseVectorEncoder,
{
    /// For SparseDataset, the dimensionality `d` may be 0 if unknown when creating a new dataset.
    fn new(encoder: E) -> Self {
        Self {
            storage: GrowableSparseStorage::new(),
            encoder,
        }
    }

    fn with_capacity(encoder: E, capacity: usize) -> Self {
        Self {
            storage: GrowableSparseStorage::with_capacity(capacity, 0),
            encoder,
        }
    }

    /// Adds a new sparse vector to the dataset.
    ///
    /// The `components` parameter is assumed to be a strictly increasing sequence
    /// representing the indices of non-zero values in the vector, and `values`
    /// holds the corresponding values. Both `components` and `values` must have
    /// the same size and cannot be empty. Additionally, `components` must be sorted in strictly ascending order.
    ///
    /// # Parameters
    ///
    /// * `components`: A slice containing the indices of non-zero values in the vector.
    /// * `values`: A slice containing the corresponding values for each index in `components`.
    ///
    /// # Panics
    ///
    /// This function will panic if:
    ///
    /// * The sizes of `components` and `values` are different.
    /// * The size of either `components` or `values` is 0.
    /// * `components` is not sorted in strictly ascending order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectorium::{Dataset, DatasetGrowable};
    /// use vectorium::{DotProduct, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVectorView};
    ///
    /// let quantizer = PlainSparseQuantizer::<u32, f32, DotProduct>::new(5, 5);
    /// let mut dataset = PlainSparseDatasetGrowable::<u32, f32, DotProduct>::new(quantizer);
    /// dataset.push(SparseVectorView::new(&[0, 2, 4], &[1.0, 2.0, 3.0]));
    ///
    /// assert_eq!(dataset.len(), 1);
    /// assert_eq!(dataset.input_dim(), 5);
    /// assert_eq!(dataset.nnz(), 3);
    /// ```
    fn push<'a>(&mut self, vec: E::InputVector<'a>) {
        let components = vec.components();
        let values = vec.values();

        assert_eq!(
            components.len(),
            values.len(),
            "Vectors have different sizes"
        );

        assert!(
            is_strictly_sorted(components),
            "Components must be given in strictly ascending order"
        );

        let cur_dim = self.encoder.input_dim();
        if let Some(last_component) = components.last().map(|l| l.as_()) {
            assert!(
                last_component < cur_dim,
                "Component index {} is out of bounds for input_dim {}",
                last_component,
                cur_dim
            );
        }

        self.encoder
            .push_encoded(vec, &mut self.storage.components, &mut self.storage.values);

        self.storage.offsets.push(self.storage.components.len());
    }
}

impl<'a, C, V> Iterator for SparseDatasetIter<'a, C, V>
where
    C: ComponentType,
    V: ValueType,
{
    type Item = SparseVectorView<'a, C, V>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (&next_offset, rest) = self.offsets.split_first()?;
        self.offsets = rest;

        let (cur_components, rest) = self
            .components
            .split_at(next_offset - self.front_base_offset);
        self.components = rest;

        let (cur_values, rest) = self.values.split_at(next_offset - self.front_base_offset);
        self.values = rest;

        self.front_base_offset = next_offset;

        Some(SparseVectorView::new(cur_components, cur_values))
    }
}

impl<C, V> ExactSizeIterator for SparseDatasetIter<'_, C, V>
where
    C: ComponentType,
    V: ValueType,
{
    fn len(&self) -> usize {
        self.offsets.len()
    }
}

impl<C, V> DoubleEndedIterator for SparseDatasetIter<'_, C, V>
where
    C: ComponentType,
    V: ValueType,
{
    /// Retrieves the next vector from the end of the iterator.
    ///
    /// # Returns
    ///
    /// An option containing a tuple of slices representing the components and values of the next vector,
    /// or `None` if the end of the iterator is reached.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectorium::{Dataset, DatasetGrowable};
    /// use vectorium::datasets::sparse_dataset::SparseDatasetIter;
    /// use vectorium::{DotProduct, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVectorView, VectorView};
    ///
    /// let data = vec![
    ///     (vec![0_u32, 2, 4], vec![1.0_f32, 2.0, 3.0]),
    ///     (vec![1_u32, 3], vec![4.0_f32, 5.0]),
    ///     (vec![0_u32, 1, 2, 3], vec![1.0_f32, 2.0, 3.0, 4.0]),
    /// ];
    ///
    /// let quantizer = PlainSparseQuantizer::<u32, f32, DotProduct>::new(5, 5);
    /// let mut dataset = PlainSparseDatasetGrowable::<u32, f32, DotProduct>::new(quantizer);
    /// for (c, v) in data.iter() {
    ///     dataset.push(SparseVectorView::new(c.as_slice(), v.as_slice()));
    /// }
    ///
    /// // Use the concrete iterator type so `next_back()` is available.
    /// let mut iter = SparseDatasetIter::new(&dataset);
    /// let last = iter.next_back().unwrap();
    /// assert_eq!(last.components(), &[0, 1, 2, 3]);
    /// assert_eq!(last.values(), &[1.0, 2.0, 3.0, 4.0]);
    /// ```
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.offsets.is_empty() {
            return None;
        }
        let (&tail_offset, rest) = self.offsets.split_last()?;
        self.offsets = rest;

        let tail_offset = tail_offset - self.front_base_offset;
        let next_offset =
            *self.offsets.last().unwrap_or(&self.front_base_offset) - self.front_base_offset;

        let len = tail_offset - next_offset;

        let (rest, cur_components) = self.components.split_at(tail_offset - len);
        self.components = rest;

        let (rest, cur_values) = self.values.split_at(tail_offset - len);
        self.values = rest;

        Some(SparseVectorView::new(cur_components, cur_values))
    }
}

impl<E, S> SpaceUsage for SparseDatasetGeneric<E, S>
where
    E: SparseVectorEncoder,
    E: SpaceUsage,
    S: SparseStorage<E> + SpaceUsage,
{
    /// Returns the size of the dataset in bytes.
    fn space_usage_bytes(&self) -> usize {
        self.storage.space_usage_bytes() + self.encoder.space_usage_bytes()
    }
}

impl<C, VIn, VOut, D>
    SparseDatasetGrowable<crate::encoders::sparse_scalar::ScalarSparseQuantizer<C, VIn, VOut, D>>
where
    C: ComponentType,
    VIn: ValueType + Float,
    VOut: ValueType + Float + FromF32,
    D: crate::encoders::sparse_scalar::ScalarSparseSupportedDistance,
{
    /// Convenience constructor that builds a quantizer and an empty growable dataset.
    pub fn with_dim(dim: usize) -> Self {
        let encoder = crate::encoders::sparse_scalar::ScalarSparseQuantizer::new(dim, dim);
        crate::DatasetGrowable::new(encoder)
    }

    /// Convenience constructor that reserves space for `n_vecs` vectors and `nnz` total components.
    pub fn with_dim_and_capacity(dim: usize, n_vecs: usize, nnz: usize) -> Self {
        let encoder = crate::encoders::sparse_scalar::ScalarSparseQuantizer::new(dim, dim);
        Self {
            storage: GrowableSparseStorage::with_capacity(n_vecs, nnz),
            encoder,
        }
    }
}

impl<E, S> SparseData for SparseDatasetGeneric<E, S>
where
    E: SparseVectorEncoder,
    S: SparseStorage<E>,
{
}

impl<E> SparseDatasetGrowable<E>
where
    E: SparseVectorEncoder,
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

#[cfg(test)]
mod tests {
    use super::SparseDatasetIter;

    use crate::core::dataset::ConvertInto;
    use crate::core::dataset::DatasetGrowable;
    use crate::core::vector::SparseVectorView;
    use crate::core::vector_encoder::VectorEncoder;
    use crate::core::vector_encoder::{SparseDataEncoder, SparseVectorEncoder};
    use crate::dataset::ConvertFrom;
    use crate::encoders::sparse_scalar::ScalarSparseQuantizer;
    use crate::{
        Dataset, DotProduct, PlainSparseDataset, PlainSparseDatasetGrowable, PlainSparseQuantizer,
        SparseDataset, SparseDatasetGrowable,
    };
    use half::f16;
    use rayon::prelude::*;

    /// Four rows of different lengths (3, 1, 2, 4 non-zeros), so wrong offsets are caught.
    fn permutable_sparse() -> PlainSparseDataset<u16, f32, DotProduct> {
        let mut growable =
            PlainSparseDatasetGrowable::new(PlainSparseQuantizer::<u16, f32, DotProduct>::new(
                8, 8,
            ));
        growable.push(SparseVectorView::new(&[0_u16, 1, 2], &[1.0_f32, 1.5, 2.0]));
        growable.push(SparseVectorView::new(&[3_u16], &[3.0_f32]));
        growable.push(SparseVectorView::new(&[4_u16, 5], &[4.0_f32, 4.5]));
        growable.push(SparseVectorView::new(
            &[0_u16, 2, 6, 7],
            &[5.0_f32, 5.5, 6.0, 6.5],
        ));
        growable.into()
    }

    #[test]
    fn sparse_permute_moves_each_row_to_its_target_slot() {
        let dataset = permutable_sparse();
        let permutation = [2usize, 0, 3, 1];

        let permuted = dataset.permute(&permutation);

        assert_eq!(permuted.len(), dataset.len());
        assert_eq!(permuted.nnz(), dataset.nnz());
        for (old_id, &new_id) in permutation.iter().enumerate() {
            let expected = dataset.get(old_id as u64);
            let actual = permuted.get(new_id as u64);
            assert_eq!(actual.components(), expected.components());
            assert_eq!(actual.values(), expected.values());
        }
    }

    #[test]
    fn sparse_permute_rebuilds_consistent_offsets() {
        let permuted = permutable_sparse().permute(&[2, 0, 3, 1]);

        // Rows land in the original order 1, 3, 0, 2, whose lengths are 1, 4, 3, 2.
        let ranges: Vec<_> = (0..permuted.len())
            .map(|id| permuted.range_from_id(id as u64))
            .collect();
        assert_eq!(ranges[0].start, 0);
        assert!(ranges.windows(2).all(|w| w[0].end == w[1].start));
        let lengths: Vec<usize> = ranges.iter().map(|r| r.end - r.start).collect();
        assert_eq!(lengths, vec![1, 4, 3, 2]);
    }

    #[test]
    fn sparse_residual_merges_components() {
        use crate::core::dataset::MeanDataset;
        use crate::core::vector::SparseVectorOwned;
        // Row 0 is {0: 1.0, 1: 1.5, 2: 2.0}; the centroid shares 1, adds 5, misses 0 and 2.
        let c = SparseVectorOwned::new(vec![1_u16, 5], vec![0.5_f32, 2.0]);
        let r = permutable_sparse().residual(0, &c);
        assert_eq!(r.components(), &[0, 1, 2, 5]);
        assert_eq!(r.values(), &[1.0, 1.0, 2.0, -2.0]);
    }

    #[test]
    fn sparse_metric_relabel_keeps_the_data() {
        let dataset = permutable_sparse();
        let l2: PlainSparseDataset<u16, f32, crate::SquaredEuclideanDistance> =
            dataset.clone().into();
        assert_eq!(l2.len(), dataset.len());
        for (a, b) in l2.iter().zip(dataset.iter()) {
            assert_eq!(a.components(), b.components());
            assert_eq!(a.values(), b.values());
        }
        let back: PlainSparseDataset<u16, f32, DotProduct> = l2.into();
        assert_eq!(back, dataset);
    }

    #[test]
    fn sparse_permute_with_identity_is_a_no_op() {
        let dataset = permutable_sparse();
        assert_eq!(dataset.permute(&[0, 1, 2, 3]), dataset);
    }

    #[test]
    fn sparse_permute_then_inverse_round_trips() {
        let dataset = permutable_sparse();
        let permutation = [2usize, 0, 3, 1];
        let inverse = crate::core::dataset::invert_permutation(&permutation);

        assert_eq!(dataset.permute(&permutation).permute(&inverse), dataset);
    }

    #[test]
    #[should_panic(expected = "permutation length")]
    fn sparse_permute_rejects_wrong_length() {
        permutable_sparse().permute(&[0, 1, 2]);
    }

    #[test]
    #[should_panic(expected = "more than one vector to position")]
    fn sparse_permute_rejects_non_bijective_permutation() {
        permutable_sparse().permute(&[0, 1, 1, 3]);
    }

    #[test]
    fn sparse_dataset_iter_next_then_next_back_returns_last_vector() {
        let quantizer = PlainSparseQuantizer::<u16, f32, DotProduct>::new(6, 6);
        let mut dataset = PlainSparseDatasetGrowable::new(quantizer);

        dataset.push(SparseVectorView::new(&[0_u16, 1], &[1.0_f32, 1.0]));
        dataset.push(SparseVectorView::new(&[2_u16, 3], &[2.0_f32, 2.0]));
        dataset.push(SparseVectorView::new(&[4_u16, 5], &[3.0_f32, 3.0]));

        let mut iter = SparseDatasetIter::new(&dataset);
        let _ = iter.next().unwrap();
        let back = iter.next_back().unwrap();

        assert_eq!(back.components(), &[4_u16, 5]);
        assert_eq!(back.values(), &[3.0_f32, 3.0]);
        let back = iter.next_back().unwrap();
        assert_eq!(back.components(), &[2_u16, 3]);
        assert_eq!(back.values(), &[2.0_f32, 2.0]);

        assert!(iter.next_back().is_none());
    }

    #[test]
    fn with_dim_and_capacity_supports_generic_sparse_quantizers() {
        let growable =
            SparseDatasetGrowable::<ScalarSparseQuantizer<u16, f32, f64, DotProduct>>::with_dim(5);
        assert_eq!(growable.encoder.output_dim(), 5);
        let with_capacity =
            SparseDatasetGrowable::<ScalarSparseQuantizer<u16, f32, f64, DotProduct>>::with_dim_and_capacity(
                5, 7, 13,
            );
        assert_eq!(with_capacity.encoder.output_dim(), 5);
        assert_eq!(with_capacity.len(), 0);
    }

    #[test]
    fn sparse_growable_immutable_roundtrip() {
        let quantizer = PlainSparseQuantizer::<u16, f32, DotProduct>::new(6, 6);
        let mut growable = PlainSparseDatasetGrowable::new(quantizer);

        growable.push(SparseVectorView::new(&[0_u16, 2], &[1.0_f32, 2.0]));
        growable.push(SparseVectorView::new(&[1_u16, 3], &[3.0_f32, 4.0]));

        let frozen: PlainSparseDataset<u16, f32, DotProduct> = growable.into();
        assert_eq!(frozen.len(), 2);
        assert_eq!(frozen.nnz(), 4);

        let first = frozen.get(0);
        assert_eq!(first.components(), &[0_u16, 2]);
        assert_eq!(first.values(), &[1.0_f32, 2.0]);

        let mut growable_again: PlainSparseDatasetGrowable<u16, f32, DotProduct> = frozen.into();
        growable_again.push(SparseVectorView::new(&[4_u16], &[5.0_f32]));

        assert_eq!(growable_again.len(), 3);
        assert_eq!(growable_again.nnz(), 5);
    }

    #[test]
    fn sparse_scalar_plain_roundtrip_without_reencode() {
        let quantizer = crate::ScalarSparseQuantizer::<u16, f32, f16, DotProduct>::new(6, 6);
        let mut growable = SparseDatasetGrowable::new(quantizer);

        growable.push(SparseVectorView::new(&[0_u16, 2], &[1.0_f32, 2.0]));
        growable.push(SparseVectorView::new(&[1_u16, 3], &[3.0_f32, 4.0]));

        let growable_plain: SparseDatasetGrowable<PlainSparseQuantizer<u16, f16, DotProduct>> =
            growable.relabel_as_plain();
        let first = growable_plain.get(0);
        assert_eq!(first.values(), &[f16::from_f32(1.0), f16::from_f32(2.0)]);

        let frozen_plain: PlainSparseDataset<u16, f16, DotProduct> = growable_plain.into();
        let frozen_scalar: SparseDataset<crate::ScalarSparseQuantizer<u16, f32, f16, DotProduct>> =
            frozen_plain.relabel_as_scalar();
        let second = frozen_scalar.get(1);
        assert_eq!(second.values(), &[f16::from_f32(3.0), f16::from_f32(4.0)]);
    }

    #[test]
    fn sparse_scalar_reencode_convert_into_owned() {
        type SrcQuant = crate::ScalarSparseQuantizer<u16, f32, f16, DotProduct>;
        type DstQuant = crate::ScalarSparseQuantizer<u16, f16, f32, DotProduct>;

        let quantizer = SrcQuant::new(6, 6);
        let mut growable = SparseDatasetGrowable::new(quantizer);

        growable.push(SparseVectorView::new(&[0_u16, 2], &[1.25_f32, 2.5]));
        growable.push(SparseVectorView::new(&[1_u16, 3], &[3.5_f32, 4.25]));

        let frozen: SparseDataset<SrcQuant> = growable.into();

        let converted: SparseDataset<DstQuant> = frozen.convert_into(());

        let first = converted.get(0);
        assert_eq!(first.components(), &[0_u16, 2]);
        assert_eq!(
            first.values(),
            &[
                f32::from(f16::from_f32(1.25)),
                f32::from(f16::from_f32(2.5)),
            ]
        );

        let second = converted.get(1);
        assert_eq!(second.components(), &[1_u16, 3]);
        assert_eq!(
            second.values(),
            &[
                f32::from(f16::from_f32(3.5)),
                f32::from(f16::from_f32(4.25)),
            ]
        );
    }

    #[test]
    fn sparse_dataset_iter_range_and_prefetch_work() {
        type Encoder = PlainSparseQuantizer<u16, f32, DotProduct>;

        let quantizer = Encoder::new(6, 6);
        let mut growable = PlainSparseDatasetGrowable::new(quantizer);
        growable.push(SparseVectorView::new(&[0_u16, 2], &[1.0_f32, 2.0]));
        growable.push(SparseVectorView::new(&[1_u16, 3], &[3.0_f32, 4.0]));

        let frozen: PlainSparseDataset<u16, f32, DotProduct> = growable.into();
        assert_eq!(frozen.len(), 2);
        assert_eq!(frozen.nnz(), 4);

        let first_range = frozen.range_from_id(0);
        let second_range = frozen.range_from_id(1);
        assert_eq!(first_range, 0..2);
        assert_eq!(second_range, 2..4);
        assert_eq!(frozen.id_from_range(second_range.clone()), 1);

        let first_vec = frozen.get_with_range(first_range.clone());
        assert_eq!(first_vec.values(), &[1.0_f32, 2.0]);

        let iter_values: Vec<Vec<_>> = frozen.iter().map(|v| v.values().to_vec()).collect();
        let par_values: Vec<Vec<_>> = frozen.par_iter().map(|v| v.values().to_vec()).collect();
        assert_eq!(iter_values, par_values);

        frozen.prefetch_with_range(second_range, 256);
    }

    #[test]
    fn sparse_dataset_growable_scalar_relabel_roundtrip() {
        type ScalarQuant = ScalarSparseQuantizer<u16, f32, f16, DotProduct>;

        let mut growable = SparseDatasetGrowable::new(ScalarQuant::new(4, 4));
        growable.push(SparseVectorView::new(&[0_u16, 2], &[1.0_f32, 2.0]));
        growable.push(SparseVectorView::new(&[1_u16, 3], &[3.0_f32, 4.0]));

        let plain = growable.relabel_as_plain();
        assert_eq!(plain.nnz(), 4);

        let scalar_again = plain.relabel_as_scalar::<f32>();
        assert_eq!(scalar_again.nnz(), 4);
        let values: Vec<Vec<f32>> = scalar_again
            .iter()
            .map(|v| v.values().iter().map(|&val| val.to_f32()).collect())
            .collect();
        assert_eq!(values, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }

    #[test]
    fn sparse_dataset_immutable_relabel_plain_scalar_roundtrip() {
        type ScalarQuant = ScalarSparseQuantizer<u16, f32, f16, DotProduct>;

        let mut growable = SparseDatasetGrowable::new(ScalarQuant::new(4, 4));
        growable.push(SparseVectorView::new(&[0_u16, 2], &[1.0_f32, 2.0]));
        growable.push(SparseVectorView::new(&[1_u16, 3], &[3.0_f32, 4.0]));

        let frozen: SparseDataset<ScalarQuant> = growable.into();
        let plain_immutable = frozen.clone().relabel_as_plain();
        assert_eq!(plain_immutable.nnz(), 4);

        let scalar_immutable = plain_immutable.relabel_as_scalar::<f32>();
        assert_eq!(scalar_immutable.nnz(), 4);

        let values: Vec<Vec<f32>> = scalar_immutable
            .iter()
            .map(|v| v.values().iter().map(|&val| val.to_f32()).collect())
            .collect();
        assert_eq!(values, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }

    #[test]
    fn scalar_sparse_encoder_encode_decode_roundtrip() {
        type Encoder = ScalarSparseQuantizer<u16, f32, half::f16, DotProduct>;
        let encoder = Encoder::new(4, 4);
        let view = SparseVectorView::new(&[0_u16, 2], &[1.25_f32, 2.5_f32]);

        let mut components = Vec::new();
        let mut values = Vec::new();
        encoder.push_encoded(view, &mut components, &mut values);
        assert_eq!(components, vec![0_u16, 2]);

        let encoded_view = SparseVectorView::new(&components, &values);
        let decoded = encoder.decode_vector(encoded_view);
        assert_eq!(decoded.components(), &[0_u16, 2]);
        assert_eq!(decoded.values(), &[1.25_f32, 2.5_f32]);

        let owned = encoder.encode_vector(view);
        let owned_vals: Vec<_> = owned.values().iter().map(|v| v.to_f32()).collect();
        assert_eq!(owned.components(), &[0_u16, 2]);
        assert_eq!(owned_vals, vec![1.25_f32, 2.5_f32]);
    }

    #[test]
    fn sparse_dataset_iter_matches_parallel_and_raw_iter() {
        type Encoder = PlainSparseQuantizer<u16, f32, DotProduct>;

        let quantizer = Encoder::new(6, 6);
        let mut growable = PlainSparseDatasetGrowable::new(quantizer);
        growable.push(SparseVectorView::new(&[0_u16, 2], &[1.0_f32, 2.0]));
        growable.push(SparseVectorView::new(&[1_u16, 3], &[3.0_f32, 4.0]));

        let frozen: PlainSparseDataset<u16, f32, DotProduct> = growable.into();
        let iter_values: Vec<_> = frozen
            .iter()
            .map(|vec| (vec.components().to_vec(), vec.values().to_vec()))
            .collect();
        let par_values: Vec<_> = frozen
            .par_iter()
            .map(|vec| (vec.components().to_vec(), vec.values().to_vec()))
            .collect();
        assert_eq!(iter_values, par_values);

        let raw_values: Vec<_> = SparseDatasetIter::new(&frozen)
            .map(|vec| (vec.components().to_vec(), vec.values().to_vec()))
            .collect();
        assert_eq!(iter_values, raw_values);
    }

    #[test]
    fn sparse_dataset_convert_from_reference_reencodes() {
        type SrcQuant = ScalarSparseQuantizer<u16, f32, half::f16, DotProduct>;
        type DstQuant = ScalarSparseQuantizer<u16, half::f16, f32, DotProduct>;

        let mut growable = SparseDatasetGrowable::new(SrcQuant::new(4, 4));
        growable.push(SparseVectorView::new(&[0_u16, 2], &[1.0_f32, 2.0_f32]));
        growable.push(SparseVectorView::new(&[1_u16, 3], &[3.0_f32, 4.0_f32]));

        let converted: SparseDatasetGrowable<DstQuant> = ConvertFrom::convert_from(&growable, ());

        assert_eq!(converted.len(), growable.len());
        assert_eq!(converted.nnz(), growable.nnz());

        let values: Vec<Vec<f32>> = converted.iter().map(|vec| vec.values().to_vec()).collect();
        assert_eq!(values, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }
}
