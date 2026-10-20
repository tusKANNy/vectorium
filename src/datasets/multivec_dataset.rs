//! Multivector dataset with offset-based variable-length storage.
//!
//! Documents are stored flat in a single buffer; an offsets array records where
//! each document starts and ends, analogous to a CSR sparse-matrix row pointer array.
//! Each slot in the dataset is a [`DenseMultiVectorView`] whose `dim` equals the
//! encoder's `token_dim` and whose `num_vecs` is derived from the stored flat length.

use serde::{Deserialize, Serialize};

use crate::SpaceUsage;
use crate::core::dataset::{assert_valid_permutation, invert_permutation};
use crate::core::sealed;
use crate::core::vector::DenseMultiVectorView;
use crate::core::vector_encoder::MultiVecEncoder;
use crate::utils::prefetch_read_slice;
use crate::{Dataset, DatasetGrowable, VectorId};

use rayon::prelude::*;

/// Growable multivector dataset backed by `Vec` flat buffers.
pub type MultiVectorDatasetGrowable<E> =
    MultiVectorDatasetGeneric<E, Vec<usize>, Vec<<E as MultiVecEncoder>::OutputValueType>>;

/// Immutable multivector dataset backed by boxed-slice flat buffers.
pub type MultiVectorDataset<E> =
    MultiVectorDatasetGeneric<E, Box<[usize]>, Box<[<E as MultiVecEncoder>::OutputValueType]>>;

/// Shared implementation for growable and frozen multivector datasets.
///
/// Documents are stored as flat buffers of `token_dim * n_tokens` values concatenated
/// in insertion order. Variable per-document token counts are tracked via `offsets`:
/// document `i` occupies `data[offsets[i]..offsets[i+1]]`.
/// The sentinel `offsets[0] == 0` is always present; `offsets.len() == n_docs + 1`.
#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct MultiVectorDatasetGeneric<E, Offsets, Data>
where
    E: MultiVecEncoder,
    Offsets: AsRef<[usize]>,
    Data: AsRef<[E::OutputValueType]>,
{
    data: Data,
    offsets: Offsets,
    encoder: E,
}

impl<E, Offsets, Data> sealed::Sealed for MultiVectorDatasetGeneric<E, Offsets, Data>
where
    E: MultiVecEncoder,
    Offsets: AsRef<[usize]>,
    Data: AsRef<[E::OutputValueType]>,
{
}

impl<E, Offsets, Data> MultiVectorDatasetGeneric<E, Offsets, Data>
where
    E: MultiVecEncoder,
    Offsets: AsRef<[usize]>,
    Data: AsRef<[E::OutputValueType]>,
{
    /// Build a dataset from pre-encoded raw buffers.
    ///
    /// `offsets` must satisfy `offsets[0] == 0` and `*offsets.last() == data.len()`.
    pub fn from_raw(data: Data, offsets: Offsets, encoder: E) -> Self {
        let offsets_slice = offsets.as_ref();
        assert!(
            !offsets_slice.is_empty(),
            "offsets must contain at least the sentinel 0"
        );
        assert_eq!(offsets_slice[0], 0, "offsets[0] must be 0");
        assert_eq!(
            *offsets_slice.last().unwrap(),
            data.as_ref().len(),
            "offsets.last() must equal data.len()"
        );
        Self {
            data,
            offsets,
            encoder,
        }
    }

    /// Access the flat encoded buffer.
    pub fn values(&self) -> &[E::OutputValueType] {
        self.data.as_ref()
    }

    /// Access the raw offsets array.
    pub fn offsets(&self) -> &[usize] {
        self.offsets.as_ref()
    }

    /// Parallel iterator over documents as [`DenseMultiVectorView`]s.
    pub fn par_iter(&self) -> impl ParallelIterator<Item = E::EncodedVector<'_>> + '_
    where
        for<'a> E::EncodedVector<'a>: Send,
    {
        let data = self.data.as_ref();
        let dim = self.encoder.output_dim();
        self.offsets.as_ref().par_windows(2).map(move |window| {
            let &[start, end] = window else {
                unsafe { std::hint::unreachable_unchecked() }
            };
            DenseMultiVectorView::new(&data[start..end], dim)
        })
    }
}

impl<E, Offsets, Data> Dataset for MultiVectorDatasetGeneric<E, Offsets, Data>
where
    E: MultiVecEncoder,
    Offsets: AsRef<[usize]>,
    Data: AsRef<[E::OutputValueType]>,
{
    type Encoder = E;
    /// Frozen, `Box<[_]>`-backed variant; identical to `Self` for [`MultiVectorDataset`].
    type Owned = MultiVectorDatasetGeneric<E, Box<[usize]>, Box<[E::OutputValueType]>>;

    fn encoder(&self) -> &E {
        &self.encoder
    }

    fn len(&self) -> usize {
        self.offsets.as_ref().len().saturating_sub(1)
    }

    fn nnz(&self) -> usize {
        self.data.as_ref().len()
    }

    fn range_from_id(&self, id: VectorId) -> std::ops::Range<usize> {
        let offsets = self.offsets.as_ref();
        let idx = id as usize;
        assert!(idx + 1 < offsets.len(), "Index out of bounds.");
        offsets[idx]..offsets[idx + 1]
    }

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

    fn get(&self, index: VectorId) -> E::EncodedVector<'_> {
        let range = self.range_from_id(index);
        DenseMultiVectorView::new(&self.data.as_ref()[range], self.encoder.output_dim())
    }

    fn get_with_range(&self, range: std::ops::Range<usize>) -> E::EncodedVector<'_> {
        DenseMultiVectorView::new(&self.data.as_ref()[range], self.encoder.output_dim())
    }

    fn prefetch_with_range(&self, range: std::ops::Range<usize>, max_bytes: usize) {
        prefetch_read_slice(&self.data.as_ref()[range], max_bytes);
    }

    fn iter(&self) -> impl Iterator<Item = E::EncodedVector<'_>> {
        let data = self.data.as_ref();
        let dim = self.encoder.output_dim();
        self.offsets
            .as_ref()
            .windows(2)
            .map(move |w| DenseMultiVectorView::new(&data[w[0]..w[1]], dim))
    }

    fn permute(&self, permutation: &[usize]) -> Self::Owned {
        let n_docs = self.len();
        assert_valid_permutation(permutation, n_docs);

        // Documents hold a variable number of tokens: gather them in the new order and rebuild
        // the offsets by prefix sum. Each document's encoding is self-contained, so copying its
        // range verbatim is equivalent to re-encoding, and lossless.
        let old_by_new = invert_permutation(permutation);
        let offsets = self.offsets.as_ref();
        let source = self.data.as_ref();

        let mut new_offsets = Vec::with_capacity(n_docs + 1);
        new_offsets.push(0);
        let mut permuted = Vec::with_capacity(source.len());

        for old_id in old_by_new {
            permuted.extend_from_slice(&source[offsets[old_id]..offsets[old_id + 1]]);
            new_offsets.push(permuted.len());
        }

        MultiVectorDatasetGeneric {
            data: permuted.into_boxed_slice(),
            offsets: new_offsets.into_boxed_slice(),
            encoder: self.encoder.clone(),
        }
    }
}

impl<E> DatasetGrowable for MultiVectorDatasetGrowable<E>
where
    E: MultiVecEncoder,
{
    fn new(encoder: E) -> Self {
        Self {
            data: Vec::new(),
            offsets: vec![0],
            encoder,
        }
    }

    fn with_capacity(encoder: E, capacity: usize) -> Self {
        let mut offsets = Vec::with_capacity(capacity + 1);
        offsets.push(0);
        Self {
            data: Vec::new(),
            offsets,
            encoder,
        }
    }

    fn push<'a>(&mut self, vec: E::InputVector<'a>) {
        assert_eq!(
            vec.dim(),
            self.encoder.input_dim(),
            "Input dim ({}) must match encoder token_dim ({})",
            vec.dim(),
            self.encoder.input_dim(),
        );
        self.encoder.push_encoded(vec, &mut self.data);
        self.offsets.push(self.data.len());
    }
}

impl<E> MultiVectorDatasetGrowable<E>
where
    E: MultiVecEncoder,
{
    /// Build a new empty growable multivector dataset.
    pub fn new(encoder: E) -> Self {
        crate::DatasetGrowable::new(encoder)
    }

    /// Build a growable dataset with preallocated capacity for `doc_capacity` documents
    /// and `data_capacity` encoded scalar values.
    pub fn with_data_capacity(encoder: E, doc_capacity: usize, data_capacity: usize) -> Self {
        let mut offsets = Vec::with_capacity(doc_capacity + 1);
        offsets.push(0);
        Self {
            data: Vec::with_capacity(data_capacity),
            offsets,
            encoder,
        }
    }
}

impl<E> MultiVectorDataset<E>
where
    E: MultiVecEncoder,
{
    /// Build an immutable dataset by encoding all documents in parallel.
    ///
    /// `flat_input`: all input token values concatenated in document order;
    /// layout `[doc0_tok0_v0, ..., doc0_tokN_vD, doc1_tok0_v0, ...]`.
    /// `doc_token_counts`: number of tokens per document.
    ///
    /// Each document is encoded independently on a rayon thread pool, then the
    /// results are assembled into a single flat buffer. This is the preferred
    /// constructor when all input data is available upfront, as it is
    /// significantly faster than sequential `push` for any non-trivial encoder.
    pub fn from_flat_par(
        encoder: E,
        flat_input: &[E::InputValueType],
        doc_token_counts: &[usize],
    ) -> Self
    where
        E: Sync,
        E::InputValueType: Sync,
        E::OutputValueType: Send,
    {
        let token_dim = encoder.input_dim();
        let n_docs = doc_token_counts.len();

        // Build per-doc ranges into flat_input (sequential, O(n_docs)).
        let mut input_offsets = Vec::with_capacity(n_docs + 1);
        input_offsets.push(0usize);
        for &count in doc_token_counts {
            input_offsets.push(input_offsets.last().unwrap() + count * token_dim);
        }

        // Encode each document on a rayon thread (encoder is Sync, flat_input is Sync).
        let encoded_docs: Vec<Vec<E::OutputValueType>> = input_offsets
            .par_windows(2)
            .map(|w| {
                let view = DenseMultiVectorView::new(&flat_input[w[0]..w[1]], token_dim);
                let mut buf = Vec::new();
                encoder.push_encoded(view, &mut buf);
                buf
            })
            .collect();

        // Assemble flat data buffer and offsets (sequential, O(total_encoded_len)).
        let total_len: usize = encoded_docs.iter().map(|d| d.len()).sum();
        let mut data = Vec::with_capacity(total_len);
        let mut offsets = Vec::with_capacity(n_docs + 1);
        offsets.push(0usize);
        for doc in &encoded_docs {
            data.extend_from_slice(doc);
            offsets.push(data.len());
        }

        Self::from_raw(data.into_boxed_slice(), offsets.into_boxed_slice(), encoder)
    }
}

impl<E> From<MultiVectorDatasetGrowable<E>> for MultiVectorDataset<E>
where
    E: MultiVecEncoder,
{
    fn from(dataset: MultiVectorDatasetGrowable<E>) -> Self {
        Self {
            data: dataset.data.into_boxed_slice(),
            offsets: dataset.offsets.into_boxed_slice(),
            encoder: dataset.encoder,
        }
    }
}

impl<E, Offsets, Data> SpaceUsage for MultiVectorDatasetGeneric<E, Offsets, Data>
where
    E: MultiVecEncoder + SpaceUsage,
    Offsets: AsRef<[usize]> + SpaceUsage,
    Data: AsRef<[E::OutputValueType]> + SpaceUsage,
{
    fn space_usage_bytes(&self) -> usize {
        self.encoder.space_usage_bytes()
            + self.data.space_usage_bytes()
            + self.offsets.space_usage_bytes()
    }
}

/// Marker trait for datasets whose encoder implements the multivector contract.
pub trait MultiVecData: Dataset<Encoder: MultiVecEncoder> {}

impl<E, Offsets, Data> MultiVecData for MultiVectorDatasetGeneric<E, Offsets, Data>
where
    E: MultiVecEncoder,
    Offsets: AsRef<[usize]>,
    Data: AsRef<[E::OutputValueType]>,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FlatIndex;
    use crate::core::dataset::invert_permutation;
    use crate::core::index::Index;
    use crate::core::vector::{DenseMultiVectorView, DenseVectorView};
    use crate::encoders::multivec_scalar::PlainMultiVecQuantizer;

    #[test]
    fn multivec_dataset_push_and_get() {
        let encoder = PlainMultiVecQuantizer::<f32>::new(2);
        let mut dataset = MultiVectorDatasetGrowable::new(encoder);

        dataset.push(DenseMultiVectorView::new(&[1.0f32, 2.0, 3.0, 4.0], 2)); // 2 tokens
        dataset.push(DenseMultiVectorView::new(
            &[5.0f32, 6.0, 7.0, 8.0, 9.0, 10.0],
            2,
        )); // 3 tokens

        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.nnz(), 10);
        assert_eq!(dataset.get(0).values(), &[1.0f32, 2.0, 3.0, 4.0]);
        assert_eq!(dataset.get(0).num_vecs(), 2);
        assert_eq!(dataset.get(1).values(), &[5.0f32, 6.0, 7.0, 8.0, 9.0, 10.0]);
        assert_eq!(dataset.get(1).num_vecs(), 3);
    }

    #[test]
    fn multivec_dataset_range_roundtrip() {
        let encoder = PlainMultiVecQuantizer::<f32>::new(2);
        let mut dataset = MultiVectorDatasetGrowable::new(encoder);

        dataset.push(DenseMultiVectorView::new(&[1.0f32, 2.0, 3.0, 4.0], 2));
        dataset.push(DenseMultiVectorView::new(&[5.0f32, 6.0, 7.0, 8.0], 2));

        assert_eq!(dataset.range_from_id(0), 0..4);
        assert_eq!(dataset.range_from_id(1), 4..8);
        assert_eq!(dataset.id_from_range(0..4), 0);
        assert_eq!(dataset.id_from_range(4..8), 1);
    }

    #[test]
    fn multivec_dataset_iter_matches_get() {
        let encoder = PlainMultiVecQuantizer::<f32>::new(2);
        let mut dataset = MultiVectorDatasetGrowable::new(encoder);

        dataset.push(DenseMultiVectorView::new(&[1.0f32, 2.0, 3.0, 4.0], 2));
        dataset.push(DenseMultiVectorView::new(&[5.0f32, 6.0], 2));

        let collected: Vec<Vec<f32>> = dataset.iter().map(|v| v.values().to_vec()).collect();
        assert_eq!(
            collected,
            vec![vec![1.0f32, 2.0, 3.0, 4.0], vec![5.0f32, 6.0]]
        );
    }

    #[test]
    fn multivec_dataset_iter_vectors() {
        let encoder = PlainMultiVecQuantizer::<f32>::new(2);
        let mut dataset = MultiVectorDatasetGrowable::new(encoder);

        dataset.push(DenseMultiVectorView::new(&[1.0f32, 2.0, 3.0, 4.0], 2));

        let tokens: Vec<Vec<f32>> = dataset
            .get(0)
            .iter_vectors()
            .map(|v| v.values().to_vec())
            .collect();
        assert_eq!(tokens, vec![vec![1.0f32, 2.0], vec![3.0f32, 4.0]]);
    }

    #[test]
    fn multivec_dataset_search_works() {
        let encoder = PlainMultiVecQuantizer::<f32>::new(2);
        let mut dataset = MultiVectorDatasetGrowable::new(encoder);

        // doc0: [[1,0],[0,1]], doc1: [[2,0],[0,2]]
        dataset.push(DenseMultiVectorView::new(&[1.0f32, 0.0, 0.0, 1.0], 2));
        dataset.push(DenseMultiVectorView::new(&[2.0f32, 0.0, 0.0, 2.0], 2));

        // query: [[1,0]] — MaxSim with doc0=1, doc1=2, so doc1 wins
        let query = DenseMultiVectorView::new(&[1.0f32, 0.0], 2);
        let results = FlatIndex::from(&dataset).search(query, 2, &());

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].vector, 1);
        assert_eq!(results[1].vector, 0);
    }

    #[test]
    fn multivec_dataset_frozen_roundtrip() {
        let encoder = PlainMultiVecQuantizer::<f32>::new(2);
        let mut growable = MultiVectorDatasetGrowable::new(encoder);

        growable.push(DenseMultiVectorView::new(&[1.0f32, 2.0], 2));
        growable.push(DenseMultiVectorView::new(&[3.0f32, 4.0, 5.0, 6.0], 2));

        let frozen: MultiVectorDataset<_> = growable.into();
        assert_eq!(frozen.len(), 2);
        assert_eq!(frozen.nnz(), 6);
        assert_eq!(frozen.get(1).values(), &[3.0f32, 4.0, 5.0, 6.0]);
        assert_eq!(frozen.get(1).dim(), 2);
        assert_eq!(frozen.get(1).num_vecs(), 2);
    }

    #[test]
    #[should_panic(expected = "Input dim")]
    fn multivec_dataset_panics_on_mismatched_dim() {
        let encoder = PlainMultiVecQuantizer::<f32>::new(3);
        let mut dataset = MultiVectorDatasetGrowable::new(encoder);
        // dim=2 doesn't match encoder token_dim=3
        dataset.push(DenseMultiVectorView::new(&[1.0f32, 2.0, 3.0, 4.0], 2));
    }

    #[test]
    fn multivec_dataset_frozen_boxes_offsets() {
        let encoder = PlainMultiVecQuantizer::<f32>::new(2);
        let mut growable = MultiVectorDatasetGrowable::new(encoder);
        growable.push(DenseMultiVectorView::new(&[1.0f32, 0.0], 2));
        growable.push(DenseMultiVectorView::new(&[0.0f32, 1.0], 2));

        let frozen: MultiVectorDataset<_> = growable.into();
        // offsets is now Box<[usize]>: [0, 2, 4]
        assert_eq!(frozen.offsets(), &[0usize, 2, 4]);
    }

    #[test]
    fn multivec_dataset_space_usage() {
        use crate::SpaceUsage;

        let encoder = PlainMultiVecQuantizer::<f32>::new(2);
        let mut dataset = MultiVectorDatasetGrowable::new(encoder);

        dataset.push(DenseMultiVectorView::new(&[1.0f32, 2.0, 3.0, 4.0], 2));
        dataset.push(DenseMultiVectorView::new(
            &[5.0f32, 6.0, 7.0, 8.0, 9.0, 10.0],
            2,
        ));

        let frozen: MultiVectorDataset<_> = dataset.into();

        // Verify space_usage_bytes() returns a reasonable non-zero value
        // Should include: encoder bytes + data buffer (10 f32s) + offsets (3 usizes)
        let space = frozen.space_usage_bytes();
        assert!(space > 0, "space_usage_bytes() should return non-zero");

        // Rough check: should be at least 10 * 4 bytes (data) + 3 * 8 bytes (offsets on 64-bit)
        assert!(
            space >= 10 * 4 + 3 * 8,
            "space_usage_bytes() should account for data and offsets"
        );
    }

    #[test]
    fn multivec_dataset_par_iter() {
        let encoder = PlainMultiVecQuantizer::<f32>::new(2);
        let mut dataset = MultiVectorDatasetGrowable::new(encoder);

        dataset.push(DenseMultiVectorView::new(&[1.0f32, 2.0, 3.0, 4.0], 2));
        dataset.push(DenseMultiVectorView::new(&[5.0f32, 6.0, 7.0, 8.0], 2));
        dataset.push(DenseMultiVectorView::new(&[9.0f32, 10.0, 11.0, 12.0], 2));

        let frozen: MultiVectorDataset<_> = dataset.into();

        // Count items in parallel
        let count = frozen.par_iter().count();
        assert_eq!(count, 3);

        // Parallel iterator should produce same values as sequential
        let par_vals: Vec<_> = frozen
            .par_iter()
            .flat_map(|v| v.values().to_vec())
            .collect();
        let seq_vals: Vec<_> = frozen.iter().flat_map(|v| v.values().to_vec()).collect();
        assert_eq!(par_vals.len(), seq_vals.len());
    }

    #[test]
    fn multivec_dataset_empty_document() {
        let encoder = PlainMultiVecQuantizer::<f32>::new(2);
        let mut dataset = MultiVectorDatasetGrowable::new(encoder);

        // Note: pushing empty document might not be supported, but test what's allowed
        dataset.push(DenseMultiVectorView::new(&[1.0f32, 2.0], 2)); // 1 token

        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset.nnz(), 2); // 1 token × 2 dims
    }

    #[test]
    fn multivec_dataset_many_small_documents() {
        let encoder = PlainMultiVecQuantizer::<f32>::new(2);
        let mut dataset = MultiVectorDatasetGrowable::new(encoder);

        // Many documents, each with single token
        for i in 0..100 {
            let val = i as f32;
            dataset.push(DenseMultiVectorView::new(&[val, val + 1.0], 2));
        }

        assert_eq!(dataset.len(), 100);
        assert_eq!(dataset.nnz(), 200); // 100 tokens × 2 dims

        // Verify we can access each one
        for i in 0..100 {
            let doc = dataset.get(i);
            assert_eq!(doc.num_vecs(), 1);
        }
    }

    #[test]
    fn multivec_dataset_large_document() {
        let encoder = PlainMultiVecQuantizer::<f32>::new(8);
        let mut dataset = MultiVectorDatasetGrowable::new(encoder);

        // Single document with many tokens
        let vals: Vec<f32> = (0..1000 * 8).map(|i| i as f32).collect();
        dataset.push(DenseMultiVectorView::new(&vals, 8));

        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset.nnz(), 1000 * 8);

        let doc = dataset.get(0);
        assert_eq!(doc.num_vecs(), 1000);
        assert_eq!(doc.dim(), 8);
    }

    #[test]
    fn multivec_dataset_search_with_pq_encoder() {
        use crate::encoders::multivec_pq::MultiVecProductQuantizer;

        let encoder = MultiVecProductQuantizer::<4, f32>::train(&{
            use crate::{PlainDenseDatasetGrowable, PlainDenseQuantizer, SquaredEuclideanDistance};
            let quantizer = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(8);
            let mut ds = PlainDenseDatasetGrowable::<f32, SquaredEuclideanDistance>::with_capacity(
                quantizer,
                256 * 2,
            );
            for _ in 0..256 * 2 {
                ds.push(DenseVectorView::new(&[0.0f32; 8]));
            }
            ds.into()
        });

        let mut dataset = MultiVectorDatasetGrowable::new(encoder);

        // Add documents with 8-dim vectors (token_dim)
        dataset.push(DenseMultiVectorView::new(
            &[
                1.0f32, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
            ],
            8,
        ));
        dataset.push(DenseMultiVectorView::new(
            &[0.0f32, 1.0, 1.0, 0.0, 0.5, 0.5, 0.5, 0.5],
            8,
        ));

        // Search with a query
        let query = DenseMultiVectorView::new(&[1.0f32, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5], 8);
        let results = FlatIndex::from(&dataset).search(query, 2, &());

        assert_eq!(results.len(), 2);
    }

    /// Four documents holding 1, 3, 2 and 4 tokens, so wrong offsets are caught.
    fn permutable_multivec() -> MultiVectorDataset<PlainMultiVecQuantizer<f32>> {
        let encoder = PlainMultiVecQuantizer::<f32>::new(2);
        let mut growable = MultiVectorDatasetGrowable::new(encoder);
        growable.push(DenseMultiVectorView::new(&[10.0f32, 11.0], 2));
        growable.push(DenseMultiVectorView::new(
            &[20.0f32, 21.0, 22.0, 23.0, 24.0, 25.0],
            2,
        ));
        growable.push(DenseMultiVectorView::new(&[30.0f32, 31.0, 32.0, 33.0], 2));
        growable.push(DenseMultiVectorView::new(
            &[40.0f32, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0],
            2,
        ));
        growable.into()
    }

    #[test]
    fn multivec_permute_moves_each_doc_to_its_target_slot() {
        let dataset = permutable_multivec();
        let permutation = [2usize, 0, 3, 1];

        let permuted = dataset.permute(&permutation);

        assert_eq!(permuted.len(), dataset.len());
        assert_eq!(permuted.nnz(), dataset.nnz());
        for (old_id, &new_id) in permutation.iter().enumerate() {
            assert_eq!(
                permuted.get(new_id as VectorId).values(),
                dataset.get(old_id as VectorId).values(),
                "document {old_id} should have landed at {new_id}"
            );
        }
    }

    #[test]
    fn multivec_permute_rebuilds_consistent_offsets() {
        let dataset = permutable_multivec();
        let permuted = dataset.permute(&[2usize, 0, 3, 1]);

        let offsets = permuted.offsets();
        assert_eq!(offsets[0], 0);
        assert_eq!(*offsets.last().unwrap(), permuted.values().len());

        // Documents of 1, 3, 2, 4 tokens land in the original order 1, 3, 0, 2, so the permuted
        // lengths (in scalar values, 2 per token) are 6, 8, 2, 4.
        let lengths: Vec<usize> = offsets.windows(2).map(|w| w[1] - w[0]).collect();
        assert_eq!(lengths, vec![6, 8, 2, 4]);
    }

    #[test]
    fn multivec_permute_then_inverse_round_trips() {
        let dataset = permutable_multivec();
        let permutation = [2usize, 0, 3, 1];

        let round_tripped = dataset
            .permute(&permutation)
            .permute(&invert_permutation(&permutation));

        assert_eq!(round_tripped.values(), dataset.values());
        assert_eq!(round_tripped.offsets(), dataset.offsets());
    }

    #[test]
    #[should_panic(expected = "permutation length")]
    fn multivec_permute_rejects_wrong_length() {
        permutable_multivec().permute(&[0, 1, 2]);
    }

    #[test]
    #[should_panic(expected = "more than one vector to position")]
    fn multivec_permute_rejects_non_bijective_permutation() {
        permutable_multivec().permute(&[0, 1, 1, 3]);
    }
}
