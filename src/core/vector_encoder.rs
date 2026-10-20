#[cfg(feature = "multivec")]
pub use crate::core::vector::{DenseMultiVectorOwned, DenseMultiVectorView};
pub use crate::core::vector::{
    DenseVectorOwned, DenseVectorView, PackedVectorOwned, PackedVectorView, SparseVectorOwned,
    SparseVectorView, VectorView,
};
use crate::{ComponentType, SpaceUsage, ValueType};

/// A query evaluator computes distances between a query and vectors.
///
/// We usually create an evaluator from a query vector once and then use it to compute
/// distances to many dataset vectors.
pub trait QueryEvaluator<V: VectorView> {
    type Distance: Copy + Ord;

    /// Calculates the distance between the evaluator’s query and `vector`.
    ///
    /// Implementations must be fast and should not mutate `vector`.
    fn compute_distance(&self, vector: V) -> Self::Distance;

    /// Calculates distances between the evaluator's query and six vectors in one call.
    ///
    /// The default dispatches to six independent `compute_distance` calls. Encoders that can
    /// compute all six distances in a single-pass kernel should override this to improve ILP
    /// and cache reuse.
    #[inline]
    fn compute_distances_batch6(&self, vectors: [V; 6]) -> [Self::Distance; 6] {
        vectors.map(|v| self.compute_distance(v))
    }
}

/// The encoder abstraction used by datasets.
///
/// Each encoder maps user-supplied inputs into an encoded representation and exposes
/// the types necessary to build evaluators and query decoders.
///
/// `Clone` lets bulk operations such as [`crate::Dataset::permute`] carry the encoder over to
/// the dataset they build.
pub trait VectorEncoder: Send + Sync + SpaceUsage + Clone {
    type Distance: Copy + Ord;

    type InputVector<'a>: VectorView;

    type QueryVector<'q>: VectorView;

    type EncodedVector<'a>: VectorView;

    /// Evaluator created from a query.
    ///
    /// This evaluator is intended to be created once and then used to compute many distances.
    /// Implementations are allowed to do expensive preprocessing here (e.g., densification,
    /// lookup tables) as long as the resulting evaluator is cheap to use.
    ///
    /// IMPORTANT: the evaluator must not borrow from the query vector.
    type Evaluator<'e>: for<'b> QueryEvaluator<Self::EncodedVector<'b>, Distance = Self::Distance>
    where
        Self: 'e;

    /// Query-side parameters, `()` for encoders whose scoring takes no configuration.
    ///
    /// These describe the *query*, not the stored data: varying them must never change how a
    /// dataset decodes. Keeping them out of the encoder is what lets one stored index serve every
    /// setting concurrently, without mutating shared state. [`FlatIndex`](crate::FlatIndex)
    /// forwards its `SearchParams` straight to this type.
    ///
    /// `Send + Sync` because a parallel search shares one set of parameters across threads.
    type QueryParams: Send + Sync;

    /// Create an evaluator from a *plain* query vector, under `params`.
    ///
    /// This is designed for search-time queries provided by the user.
    fn query_evaluator<'e>(
        &'e self,
        query: Self::QueryVector<'_>,
        params: &Self::QueryParams,
    ) -> Self::Evaluator<'e>;

    /// Create an evaluator from a dataset-internal vector.
    ///
    /// This is designed for index-build-time scenarios where we need to compute distances
    /// between vectors stored in the dataset. Implementations typically:
    ///
    /// 1) materialize/decompress the encoded vector into a plain `...Owned<..., f32>`;
    /// 2) build and return an evaluator from that plain representation.
    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e>;

    fn input_dim(&self) -> usize;
    fn output_dim(&self) -> usize;

    /// Compute the distance between two encoded dataset vectors without building an evaluator.
    ///
    /// The default implementation builds a temporary evaluator (one allocation). Encoders
    /// that can compute distances directly on their stored representation (e.g. plain dense
    /// f32/f16 quantizers) should override this to avoid any allocation in hot code paths.
    #[inline]
    fn compute_distance_between(
        &self,
        v1: Self::EncodedVector<'_>,
        v2: Self::EncodedVector<'_>,
    ) -> Self::Distance {
        self.vector_evaluator(v1).compute_distance(v2)
    }
}

/// An encoder whose stored codes do not depend on its metric, so the same codes can also be
/// scored under the metric `D2`: [`with_distance`](Self::with_distance) returns an encoder whose
/// evaluators read the encoded vectors of a dataset built by `self`, without re-encoding them.
pub trait WithDistance<D2>: VectorEncoder {
    type Output: for<'a> VectorEncoder<
            Distance = D2,
            EncodedVector<'a> = Self::EncodedVector<'a>,
            QueryVector<'a> = Self::QueryVector<'a>,
            QueryParams = Self::QueryParams,
        >;

    fn with_distance(&self) -> Self::Output;
}

/// Encoder contract for dense vectors.
///
/// The encoder consumes plain dense inputs (`DenseVectorView`) and produces dense
/// outputs with the encoder-specific `OutputValueType`. Queries are always
/// dense `f32` vectors and the encoded representation is also dense.
///
/// Implementations must support decoding encoded vectors back to `f32` and
/// provide `push_encoded`/`encode_vector` functions to perform the encoding.
pub trait DenseVectorEncoder:
    for<'a> VectorEncoder<
        InputVector<'a> = DenseVectorView<'a, Self::InputValueType>,
        QueryVector<'a> = DenseVectorView<'a, f32>,
        EncodedVector<'a> = DenseVectorView<'a, Self::OutputValueType>,
    >
{
    type InputValueType: ValueType;
    type OutputValueType: ValueType;

    /// Decode an encoded vector to a plain dense `f32` vector.
    ///
    /// Intended for index build and other internal workflows.
    fn decode_vector<'a>(&self, encoded: Self::EncodedVector<'a>) -> DenseVectorOwned<f32>;

    /// Decode an encoded vector into `out`, which must have length `input_dim`, without
    /// allocating when the encoder supports it. The default goes through
    /// [`decode_vector`](Self::decode_vector).
    #[inline]
    fn decode_into<'a>(&self, encoded: Self::EncodedVector<'a>, out: &mut [f32]) {
        out.copy_from_slice(self.decode_vector(encoded).values());
    }

    /// Encode `input` into the provided `output` container.
    ///
    /// This lets callers store the encoded values directly in dataset-owned buffers or reuse
    /// temporary containers, avoiding the extra allocation that would happen inside `encode_vector`.
    fn push_encoded<'a, OutputContainer>(
        &self,
        input: Self::InputVector<'a>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<Self::OutputValueType>;

    /// Convenience helper that builds a `DenseVectorOwned` from `input`.
    ///
    /// The default implementation reuses `push_encoded` to append into a temporary `Vec`.
    /// Avoid this helper when you can store results directly via `push_encoded`, since encoding
    /// through the temporary `Vec` performs an extra allocation.
    fn encode_vector<'a>(
        &self,
        input: Self::InputVector<'a>,
    ) -> DenseVectorOwned<Self::OutputValueType> {
        let mut values = Vec::new();
        self.push_encoded(input, &mut values);
        DenseVectorOwned::new(values)
    }
}

#[cfg(feature = "multivec")]
/// Encoder contract for multivector (late-interaction) encoders.
///
/// Each encoded "vector" is a [`DenseMultiVectorView`] carrying `token_dim` and `num_vecs`
/// alongside the flat scalar buffer. Because document lengths vary, datasets built on this
/// encoder maintain an explicit offsets array rather than a fixed stride.
///
/// Queries are also multivectors of `f32` tokens; encoders must validate
/// `query.dim() == token_dim` before building an evaluator.
pub trait MultiVecEncoder:
    for<'a> VectorEncoder<
        InputVector<'a> = DenseMultiVectorView<'a, Self::InputValueType>,
        QueryVector<'a> = DenseMultiVectorView<'a, f32>,
        EncodedVector<'a> = DenseMultiVectorView<'a, Self::OutputValueType>,
    >
{
    type InputValueType: ValueType;
    type OutputValueType: ValueType;

    /// Encode `input` and append the resulting elements into `output`.
    ///
    /// For scalar quantizers this is an element-wise type cast from `InputValueType` to
    /// `OutputValueType`. Used by datasets to populate their flat storage buffers without
    /// intermediate allocations.
    fn push_encoded<'a, OutputContainer>(
        &self,
        input: DenseMultiVectorView<'a, Self::InputValueType>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<Self::OutputValueType>;

    /// Convenience helper that encodes `input` into an owned multivector.
    ///
    /// Callers who already manage buffers should prefer `push_encoded` to avoid the
    /// extra allocation and keep working buffers alive for reuse.
    fn encode_vector<'a>(
        &self,
        input: DenseMultiVectorView<'a, Self::InputValueType>,
    ) -> DenseMultiVectorOwned<Self::OutputValueType> {
        let mut values = Vec::new();
        self.push_encoded(input, &mut values);
        DenseMultiVectorOwned::new(values, input.dim())
    }

    /// Decode an encoded multivector back to a plain `f32` multivector.
    ///
    /// Intended for index build and other internal workflows.
    fn decode_vector<'a>(
        &self,
        encoded: DenseMultiVectorView<'a, Self::OutputValueType>,
    ) -> DenseMultiVectorOwned<f32>;
}

/// A minimal shared trait capturing the sparse input/query contract, component/value types, and decode capability.
/// This lets code reason about sparse datasets without caring whether the encoder produces component/value pairs
/// or a packed blob; it also keeps trait bounds lean by avoiding the heavier `SparseVectorEncoder`/`PackedSparseVectorEncoder`
/// when only the shared behavior is needed. Implementing this trait signals that the encoder can decode the stored
/// representation back into component/value pairs, which is necessary for indexing workflows.
pub trait SparseDataEncoder
where
    for<'a> Self: VectorEncoder<
            InputVector<'a> = SparseVectorView<'a, Self::InputComponentType, Self::InputValueType>,
            QueryVector<'a> = SparseVectorView<'a, Self::InputComponentType, f32>,
        >,
{
    type InputComponentType: ComponentType;
    type InputValueType: ValueType;
    type OutputComponentType: ComponentType;
    type OutputValueType: ValueType;

    /// Decode an encoded vector to a plain sparse `f32` vector in the *input component* space.
    ///
    /// Intended for index build and other internal workflows.
    fn decode_vector<'a>(
        &self,
        encoded: Self::EncodedVector<'a>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32>;
}

/// Encoder contract for producing sparse component/value pairs.
///
/// Extends `SparseDataEncoder` by requiring the encoded representation be exposed
/// as sparse component/value slices so callers can build datasets that store explicit
/// offsets/components/values. Implementations are requested to expose `push_encoded`
/// helpers for component/value containers, while the blanket `encode_vector`
/// builds an owned sparse vector.
pub trait SparseVectorEncoder: SparseDataEncoder
where
    for<'a> Self: VectorEncoder<
        EncodedVector<'a> = SparseVectorView<'a, Self::OutputComponentType, Self::OutputValueType>,
    >,
{
    /// Encode `input` directly into dataset-owned component/value containers or reuse buffers,
    /// so datasets can populate their storage without introducing temporary allocations.
    fn push_encoded<'a, ComponentContainer, ValueContainer>(
        &self,
        input: Self::InputVector<'a>,
        components: &mut ComponentContainer,
        values: &mut ValueContainer,
    ) where
        ComponentContainer: Extend<Self::OutputComponentType>,
        ValueContainer: Extend<Self::OutputValueType>;

    /// Convenience helper that builds an owned sparse vector by replaying `push_encoded`.
    ///
    /// Callers who already manage buffers can skip this method to avoid the additional allocation
    /// and keep working buffers alive for reuse.
    fn encode_vector<'a>(
        &self,
        input: Self::InputVector<'a>,
    ) -> SparseVectorOwned<Self::OutputComponentType, Self::OutputValueType> {
        let mut components = Vec::new();
        let mut values = Vec::new();
        self.push_encoded(input, &mut components, &mut values);
        SparseVectorOwned::new(components, values)
    }
}

/// Encoder contract for encoding sparse component/value pairs packed into a single container and potentially encoded in some way.
/// For example, compression schemes such as DotVByte use this approach.
pub trait PackedSparseVectorEncoder: SparseDataEncoder
where
    for<'a> Self: VectorEncoder<EncodedVector<'a> = PackedVectorView<'a, Self::PackedDataType>>,
{
    type PackedDataType: ValueType + SpaceUsage;

    /// Encode `input` directly into an existing packed data container so datasets can
    /// avoid intermediate allocations by reusing buffers.
    fn push_encoded<'a, OutputContainer>(
        &self,
        input: Self::InputVector<'a>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<Self::PackedDataType>;

    /// Convenience helper that builds an owned packed vector by replaying `push_encoded`.
    ///
    /// Implementations expecting to reuse buffers can rely solely on `push_encoded` to keep
    /// buffers alive across encodings.
    fn encode_vector<'a>(
        &self,
        input: Self::InputVector<'a>,
    ) -> PackedVectorOwned<Self::PackedDataType> {
        let mut data = Vec::new();
        self.push_encoded(input, &mut data);
        PackedVectorOwned::new(data)
    }
}
