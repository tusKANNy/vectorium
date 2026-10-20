//! Utilities for borrowing and owning vectors regardless of the encoder representation.
//! The module centralizes view traits plus owned types so the rest of the core library can
//! reason about dense, sparse, and packed data uniformly.

use crate::{ComponentType, SpaceUsage, ValueType};

/// Marker trait for vector views that end up as encoder inputs, queries, or results.
/// Implementers are cheap wrappers around slices or owned buffers, so this trait acts as
/// a lightweight way to constrain encoders without prescribing any API surface.
pub trait VectorView {}

impl<T: VectorView> VectorView for &T {}
impl<T: VectorView> VectorView for &mut T {}

/// A view over a dense vector (slice of values).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenseVectorView<'a, V: ValueType> {
    values: &'a [V],
}

impl<'a, V: ValueType> DenseVectorView<'a, V> {
    /// Create a view from a borrowed slice.
    #[inline]
    pub fn new(values: &'a [V]) -> Self {
        Self { values }
    }
    /// Number of elements accessible through the view.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }
    /// True if the view contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
    /// Access the raw slice backing the view.
    #[inline]
    pub fn values(&self) -> &'a [V] {
        self.values
    }
    /// Iterate over copies of each value.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = V> + '_ {
        self.values.iter().copied()
    }
    /// Clone the view contents into an owned vector.
    #[inline]
    pub fn to_owned(&self) -> DenseVectorOwned<V> {
        DenseVectorOwned::new(self.values.to_vec())
    }
}

impl<'a, V: ValueType> VectorView for DenseVectorView<'a, V> {}

/// A view over a sparse vector (parallel slices of component indices and values).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseVectorView<'a, C: ComponentType, V: ValueType> {
    components: &'a [C],
    values: &'a [V],
}

impl<'a, C: ComponentType, V: ValueType> SparseVectorView<'a, C, V> {
    /// Construct a sparse view from parallel component/value slices.
    /// Panics in debug builds if lengths mismatch.
    #[inline]
    pub fn new(components: &'a [C], values: &'a [V]) -> Self {
        debug_assert_eq!(components.len(), values.len());
        Self { components, values }
    }
    /// Number of nonzero entries described by this view.
    #[inline]
    pub fn len(&self) -> usize {
        self.components.len()
    }
    /// True if the view contains no entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }
    /// Component indices for the sparse vector.
    #[inline]
    pub fn components(&self) -> &'a [C] {
        self.components
    }
    /// Values corresponding to each component.
    #[inline]
    pub fn values(&self) -> &'a [V] {
        self.values
    }
    /// Iterate through component/value pairs.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (C, V)> + '_ {
        self.components
            .iter()
            .copied()
            .zip(self.values.iter().copied())
    }
    /// Clone this sparse view into owned buffers.
    #[inline]
    pub fn to_owned(&self) -> SparseVectorOwned<C, V> {
        SparseVectorOwned::new(self.components.to_vec(), self.values.to_vec())
    }
}

impl<'a, C: ComponentType, V: ValueType> VectorView for SparseVectorView<'a, C, V> {}

/// A view over a packed vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PackedVectorView<'a, T: SpaceUsage + Copy> {
    data: &'a [T],
}

impl<'a, T: SpaceUsage + Copy> PackedVectorView<'a, T> {
    /// Build a view of packed data.
    #[inline]
    pub fn new(data: &'a [T]) -> Self {
        Self { data }
    }
    /// Number of packed entries available.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }
    /// True when no packed entries exist.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    /// Access the raw packed slice.
    #[inline]
    pub fn data(&self) -> &'a [T] {
        self.data
    }
    /// Iterate over copied packed values.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        self.data.iter().copied()
    }
    /// Copy the packed data into an owned buffer.
    #[inline]
    pub fn to_owned(&self) -> PackedVectorOwned<T> {
        PackedVectorOwned::new(self.data.to_vec())
    }
}
impl<'a, T: SpaceUsage + Copy> VectorView for PackedVectorView<'a, T> {}

// --- Owned Structs ---

#[derive(Debug, Clone, PartialEq)]
/// Owned dense vector storage.
pub struct DenseVectorOwned<V: ValueType> {
    values: Vec<V>,
}

impl<V: ValueType> DenseVectorOwned<V> {
    /// Construct from owned values.
    #[inline]
    pub fn new(values: Vec<V>) -> Self {
        Self { values }
    }

    /// Return the number of values.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// True when the underlying vector is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Borrow shared slice of values.
    #[inline]
    pub fn values(&self) -> &[V] {
        &self.values
    }

    /// Iterate over copies of each value.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = V> + '_ {
        self.values.iter().copied()
    }

    /// Create a view that borrows the inner values.
    #[inline]
    pub fn as_view(&self) -> DenseVectorView<'_, V> {
        DenseVectorView::new(&self.values)
    }
}
impl<V: ValueType> VectorView for DenseVectorOwned<V> {}

impl<V: ValueType> AsMut<[V]> for DenseVectorOwned<V> {
    fn as_mut(&mut self) -> &mut [V] {
        &mut self.values
    }
}

impl<'a, V: ValueType> From<&'a DenseVectorOwned<V>> for DenseVectorView<'a, V> {
    fn from(owned: &'a DenseVectorOwned<V>) -> Self {
        owned.as_view()
    }
}

#[cfg(feature = "multivec")]
/// A view over a multivector: a contiguous flat buffer of `dim * num_vecs` values
/// representing `num_vecs` dense token vectors each of dimensionality `dim`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenseMultiVectorView<'a, V: ValueType> {
    values: &'a [V],
    dim: usize,
    num_vecs: usize,
}

#[cfg(feature = "multivec")]
impl<'a, V: ValueType> DenseMultiVectorView<'a, V> {
    /// Build a view from a flat slice.
    ///
    /// Panics if `dim == 0` or `values.len() % dim != 0`.
    #[inline]
    pub fn new(values: &'a [V], dim: usize) -> Self {
        assert!(dim > 0, "dim must be > 0");
        assert_eq!(
            values.len() % dim,
            0,
            "values.len() ({}) must be a multiple of dim ({})",
            values.len(),
            dim
        );
        let num_vecs = values.len() / dim;
        Self {
            values,
            dim,
            num_vecs,
        }
    }

    /// Dimensionality of each individual token vector.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of token vectors in this multivector.
    #[inline]
    pub fn num_vecs(&self) -> usize {
        self.num_vecs
    }

    /// Total number of scalar values (`dim * num_vecs`).
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// True when the multivector contains no token vectors.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_vecs == 0
    }

    /// Flat slice of all values (`dim * num_vecs` elements).
    #[inline]
    pub fn values(&self) -> &'a [V] {
        self.values
    }

    /// Iterate over individual token vectors as [`DenseVectorView`]s.
    #[inline]
    pub fn iter_vectors(&self) -> impl Iterator<Item = DenseVectorView<'a, V>> {
        self.values.chunks_exact(self.dim).map(DenseVectorView::new)
    }

    /// Clone the view contents into an owned multivector.
    #[inline]
    pub fn to_owned(&self) -> DenseMultiVectorOwned<V> {
        DenseMultiVectorOwned::new(self.values.to_vec(), self.dim)
    }
}

#[cfg(feature = "multivec")]
impl<'a, V: ValueType> VectorView for DenseMultiVectorView<'a, V> {}

#[cfg(feature = "multivec")]
/// Owned multivector storage: a flat buffer of `dim * num_vecs` values representing
/// `num_vecs` dense token vectors each of dimensionality `dim`.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseMultiVectorOwned<V: ValueType> {
    values: Vec<V>,
    dim: usize,
    num_vecs: usize,
}

#[cfg(feature = "multivec")]
impl<V: ValueType> DenseMultiVectorOwned<V> {
    /// Build from owned values.
    ///
    /// Panics if `dim == 0` or `values.len() % dim != 0`.
    #[inline]
    pub fn new(values: Vec<V>, dim: usize) -> Self {
        assert!(dim > 0, "dim must be > 0");
        assert_eq!(
            values.len() % dim,
            0,
            "values.len() ({}) must be a multiple of dim ({})",
            values.len(),
            dim
        );
        let num_vecs = values.len() / dim;
        Self {
            values,
            dim,
            num_vecs,
        }
    }

    /// Dimensionality of each individual token vector.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of token vectors in this multivector.
    #[inline]
    pub fn num_vecs(&self) -> usize {
        self.num_vecs
    }

    /// Total number of scalar values (`dim * num_vecs`).
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// True when the multivector contains no token vectors.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_vecs == 0
    }

    /// Flat slice of all values (`dim * num_vecs` elements).
    #[inline]
    pub fn values(&self) -> &[V] {
        &self.values
    }

    /// Iterate over individual token vectors as [`DenseVectorView`]s.
    #[inline]
    pub fn iter_vectors(&self) -> impl Iterator<Item = DenseVectorView<'_, V>> {
        self.values.chunks_exact(self.dim).map(DenseVectorView::new)
    }

    /// Borrow as a [`DenseMultiVectorView`].
    #[inline]
    pub fn as_view(&self) -> DenseMultiVectorView<'_, V> {
        DenseMultiVectorView::new(&self.values, self.dim)
    }
}

#[cfg(feature = "multivec")]
impl<V: ValueType> VectorView for DenseMultiVectorOwned<V> {}

#[cfg(feature = "multivec")]
impl<'a, V: ValueType> From<&'a DenseMultiVectorOwned<V>> for DenseMultiVectorView<'a, V> {
    fn from(owned: &'a DenseMultiVectorOwned<V>) -> Self {
        owned.as_view()
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Owned storage for sparse component/value vectors.
pub struct SparseVectorOwned<C: ComponentType, V: ValueType> {
    components: Vec<C>,
    values: Vec<V>,
}

impl<C: ComponentType, V: ValueType> SparseVectorOwned<C, V> {
    /// Build from owned component/value buffers.
    #[inline]
    pub fn new(components: Vec<C>, values: Vec<V>) -> Self {
        assert!(
            components.len() == values.len(),
            "Components and values must have the same length"
        );
        Self { components, values }
    }

    /// Number of stored entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// True if there are no entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// Access the component indices.
    #[inline]
    pub fn components(&self) -> &[C] {
        &self.components
    }

    /// Access the stored values.
    #[inline]
    pub fn values(&self) -> &[V] {
        &self.values
    }

    /// Iterate over component/value pairs.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (C, V)> + '_ {
        self.components
            .iter()
            .copied()
            .zip(self.values.iter().copied())
    }

    /// Create a borrowed view on the owned data.
    #[inline]
    pub fn as_view(&self) -> SparseVectorView<'_, C, V> {
        SparseVectorView::new(&self.components, &self.values)
    }
}
impl<C: ComponentType, V: ValueType> VectorView for SparseVectorOwned<C, V> {}

/// The values only: the components of a sparse vector are not mutable through this.
impl<C: ComponentType, V: ValueType> AsMut<[V]> for SparseVectorOwned<C, V> {
    fn as_mut(&mut self) -> &mut [V] {
        &mut self.values
    }
}

impl<'a, C: ComponentType, V: ValueType> From<&'a SparseVectorOwned<C, V>>
    for SparseVectorView<'a, C, V>
{
    fn from(owned: &'a SparseVectorOwned<C, V>) -> Self {
        owned.as_view()
    }
}

// PackedVectorOwned
#[derive(Debug, Clone, PartialEq, Eq)]
/// Owned packed vector storage.
pub struct PackedVectorOwned<T>
where
    T: SpaceUsage + Copy,
{
    data: Vec<T>,
}

impl<T> PackedVectorOwned<T>
where
    T: SpaceUsage + Copy,
{
    /// Build from packed data.
    #[inline]
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }

    /// Number of packed entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True when no packed entries exist.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Borrow the underlying packed slice.
    #[inline]
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Create a borrowable view of the packed data.
    #[inline]
    pub fn as_view(&self) -> PackedVectorView<'_, T> {
        PackedVectorView::new(&self.data)
    }

    /// Iterate over copies of the packed data.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        self.data.iter().copied()
    }
}
impl<T> VectorView for PackedVectorOwned<T> where T: SpaceUsage + Copy {}

impl<'a, T> From<&'a PackedVectorOwned<T>> for PackedVectorView<'a, T>
where
    T: SpaceUsage + Copy,
{
    fn from(owned: &'a PackedVectorOwned<T>) -> Self {
        owned.as_view()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_vector_owned_basic() {
        let values = vec![1.0f32, 2.0, 3.0];
        let v = DenseVectorOwned::new(values.clone());
        assert_eq!(v.len(), 3);
        assert_eq!(v.values(), values.as_slice());

        let iter_vals: Vec<_> = v.iter().collect();
        assert_eq!(iter_vals, values);

        let view = v.as_view();
        assert_eq!(view.values(), values.as_slice());
    }

    #[test]
    fn dense_vector_view_basic() {
        let values = vec![1.0f32, 2.0, 3.0];
        let v = DenseVectorView::new(&values);
        assert_eq!(v.len(), 3);
        assert_eq!(v.values(), values.as_slice());

        let owned = v.to_owned();
        assert_eq!(owned.values(), values.as_slice());
    }

    #[test]
    fn sparse_vector_owned_basic() {
        let comps = vec![0usize, 2usize];
        let vals = vec![1.0f32, 3.0f32];
        let v = SparseVectorOwned::new(comps.clone(), vals.clone());
        assert_eq!(v.len(), 2);
        assert_eq!(v.components(), comps.as_slice());
        assert_eq!(v.values(), vals.as_slice());

        let iter_vals: Vec<_> = v.iter().collect();
        assert_eq!(iter_vals, vec![(0, 1.0), (2, 3.0)]);

        let view = v.as_view();
        assert_eq!(view.components(), comps.as_slice());
        assert_eq!(view.values(), vals.as_slice());
    }

    #[test]
    fn sparse_vector_view_basic() {
        let comps = vec![0usize, 2usize];
        let vals = vec![1.0f32, 3.0f32];
        let v = SparseVectorView::new(&comps, &vals);
        assert_eq!(v.len(), 2);
        assert_eq!(v.components(), comps.as_slice());
        assert_eq!(v.values(), vals.as_slice());

        let owned = v.to_owned();
        assert_eq!(owned.components(), comps.as_slice());
        assert_eq!(owned.values(), vals.as_slice());
    }

    #[test]
    fn packed_vector_owned_roundtrip() {
        let packed = PackedVectorOwned::new(vec![5_u32, 7, 9]);
        assert_eq!(packed.len(), 3);
        let view = packed.as_view();
        assert_eq!(view.len(), 3);
        assert_eq!(view.data(), &[5, 7, 9]);
        let cloned = view.to_owned();
        assert_eq!(cloned, packed);
        let collected: Vec<_> = view.iter().collect();
        assert_eq!(collected, vec![5, 7, 9]);
    }

    #[test]
    fn packed_vector_view_iterates() {
        let data = [42_u16, 100];
        let view = PackedVectorView::new(&data);
        assert_eq!(view.iter().collect::<Vec<_>>(), vec![42, 100]);
    }

    #[test]
    fn dense_vector_view_reports_empty_and_iterates() {
        let empty = DenseVectorView::<f32>::new(&[]);
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.iter().collect::<Vec<_>>(), Vec::<f32>::new());
        assert!(empty.to_owned().values().is_empty());
    }

    #[test]
    fn sparse_vector_view_reports_empty_and_iterates() {
        let empty_view = SparseVectorView::<u16, f32>::new(&[], &[]);
        assert!(empty_view.is_empty());
        assert_eq!(empty_view.len(), 0);
        assert!(empty_view.iter().next().is_none());
        let owned = empty_view.to_owned();
        assert!(owned.is_empty());
    }
}
