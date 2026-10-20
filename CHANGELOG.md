# Changelog

Changes that matter to crates depending on vectorium (kANNolo, Seismic).
Newest entries go at the top of each section.

## [0.10.0] - unreleased (branch `compression`)

This branch (`compression`) started from `main` at `3fedb2f` (0.4.0) and has since merged
`origin/main` up to its 0.10.0 (`5850880`). The tables below list what `compression` adds or changes;
the 0.5.0–0.10.0 changes of `main` themselves (RaBitQ, `l1_prune_split`, the `Config`/`QueryParams`
encoder API, generic dense/sparse k-means, `WithDistance`, `RerankIndex` moved to kANNolo, …) are in
the git history of `origin/main`. Commit hashes
below are on `compression`; "relative to `main`" means relative to 0.4.0 unless stated otherwise.

Dependents: change the vectorium dependency to `0.10.0` (or the new git `rev`).

### Coming from 0.9.4

A crate already on 0.9.4 only needs the breaking changes below that come from `compression`
(DotVByte removal; DotPacking8 `train` and `BlockedSparseEncoder` type annotations). Everything
added on `compression` follows the 0.9.4 API: conversions take a config
(`ScalarU8SparseDataset::<u16, DotProduct>::convert_from(plain, ())`, `plain.convert_into(())`),
evaluators take query parameters (`encoder.query_evaluator(query, &())`), and every new encoder has
`QueryParams = ()`. `compute_groundtruth --l1-prune` works with every sparse encoder (pruning happens
before encoding). The `rgbisect` dependency of 0.9.4 is dropped together with DotVByte.

### What kANNolo and Seismic must support

Every dataset type below must be buildable from the dependent crate. Each one is built from a
plain sparse dataset with `ConvertFrom::convert_from(plain, ())` / `.into()` (in Seismic: `convert_dataset_into::<T>()`).

| Dataset type | Components | Values | Conversion status |
|---|---|---|---|
| `BlockedSparseDataset` (jump_dot for queries with < 33 components) | `u16` | `f16`, uncompressed | available (`1569f03`) |
| `DotPacking8U16ScalarU8Dataset` | `u16` | 8-bit per-component scalar | available (`0bcdd01`) |
| `DotPacking8U16F16Dataset` | `u16` | `f16` | available (`0bcdd01`) |
| `DotPacking8U16Scalar4BitDataset` | `u16` | 4-bit per-component scalar | available (`0bcdd01`) |
| `BridgedDotPacking8U32ScalarU8Dataset` | `u32` | 8-bit per-component scalar | available (`0bcdd01`) |
| `DotPacking8U16FixedU8Dataset` | `u16` | `FixedU8Q` | available (`0bcdd01`) |
| `DotPacking8U16ScalarU16Dataset` | `u16` | 16-bit per-component scalar | available (`f3afe08`) |
| `BridgedDotPacking8U32ScalarU16Dataset` | `u32` | 16-bit per-component scalar | available (`f3afe08`) |
| `ScalarU8SparseDataset<C, D>` (plain sparse; dot product and Euclidean) | `u16`, `u32` | 8-bit per-component scalar | available (`deab0cd`), `ConvertFrom<plain sparse dataset>` |
| `ScalarU16SparseDataset<C, D>` (plain sparse; dot product and Euclidean) | `u16`, `u32` | 16-bit per-component scalar | available (`deab0cd`), `ConvertFrom<plain sparse dataset>` |
| `BlockedSparseScalarU8Dataset` (48-byte blocks, jump_dot) | `u16` | 8-bit per-component scalar | `2d91d56` |
| `BlockedSparseScalarU16Dataset` (jump_dot) | `u16` | 16-bit per-component scalar | `2d91d56` |

The README section "Choosing a sparse representation" (also the crate rustdoc) describes all of them and when to use each; measurements are in `local_docs/Benchmarks.md`.
`FixedU8Q`/`FixedU16Q` stay: the plan to remove them was cancelled (user, 2026-10-13).

How to use it when updating a dependent crate: read **Breaking changes** first (things that
stop compiling or change results), then **Renames** if the crate used `feature/quantization`,
then **Additions** for what can replace removed pieces.

---

### Breaking changes (relative to `main`)

#### Done

| Commit | Change | Who is hit | Migration |
|---|---|---|---|
| `2d91d56` | `BlockedSparseEncoder` and `BlockedSparseQueryEvaluator` take a value-format parameter (`BlockedSparseEncoder<V = BlockF16Values>`); `BlockedSparseEncoder` without parameters is still the `f16` encoder with the same serialized form and bytes. | `let e = BlockedSparseEncoder::new(dim);` with nothing else fixing the type no longer compiles ("type annotations needed"). | Write `let e: BlockedSparseEncoder = BlockedSparseEncoder::new(dim);`. |
| `2d91d56` | Blocked decoding and `DataBlock::len` find padding by its repeated component instead of a zero value: a real value stored as zero (an `f16` underflow) no longer truncates the decoded vector. | Only results of `decode_vector` / `len` on such vectors (scores were already correct). | None. |
| `a6a5ddd` | **DotVByte removed**: `DotVByteFixedU8Encoder`, `DotVByteFixedU8QueryEvaluator`, the `encoders::dotvbyte_fixedu8` module, the `SparseDataset -> PackedSparseDataset<DotVByteFixedU8Encoder>` conversion, `--encoder dotvbyte` in `compute_groundtruth`. | **Seismic**: `src/bin/build_inverted_index.rs:20,303` and `src/pylib/dotvbyte.rs` (`SeismicIndexDotVByte`). | Replace `PackedSparseDataset<DotVByteFixedU8Encoder>` with a DotPacking8 dataset, e.g. `DotPacking8U16ScalarU8Dataset`: `convert_dataset_into::<DotPacking8U16ScalarU8Dataset>()` works the same way (it trains inside). Rebuild serialized DotVByte indexes. |
| `0bcdd01` | `DotPacking8Encoder::train` takes any sparse dataset (`&SparseDatasetGeneric<E, S>` with `E::OutputComponentType = C`, any `ValueType` values, growable or immutable) instead of `&PlainSparseDataset<C, f32, D>`. Same result for the same data. `ComponentMapping::by_decreasing_frequency(dim, components_iter)` takes an iterator of component slices instead of a dataset. | Code that let `train` infer the dataset type, e.g. `let d = growable.into(); enc.train(&d);`, no longer compiles ("type annotations needed"). | Train on the growable dataset directly (`enc.train(&growable)`), or annotate the type. |
| `0db4183` | Component reordering uses **decreasing document frequency** instead of graph bisection; the `rgb` (enhanced-graph-bisection) git dependency is gone. The internal helper `permute_components_with_bisection` became `permute_components_by_frequency` (`pub(crate)`, not visible outside). | Anyone building a DotVByte (or DotPacking8) dataset: the component permutation, hence the encoded bytes and the index size, differ from `main`. Scores do not change (the permutation is applied to queries too). Serialized indexes from `main` still load (the mapping is stored). | None needed. Drop any `rgb`-related patch/feature in the dependent crate. |
| `0db4183` | The plain-sparse → `PackedSparseDataset<DotVByteFixedU8Encoder>` conversion trains the permutation on **all** vectors instead of the first 5% (min 50,000). | Build time of DotVByte conversions (one counting pass over all components; no longer a sample). | None. |
| `7c72295` | `KMeans` assignment and PQ tests use `total_cmp` instead of `partial_cmp().unwrap()`: a NaN distance no longer panics, it sorts last/first per IEEE total order. | Only behaviour on NaN distances. | None. |

---

### Renames for code written against `feature/quantization`

The DotPacking8 encoders on `compression` are type aliases of one generic
`DotPacking8Encoder<C, V>` (`C` = `u16`/`u32` component type, `V` = value codec).
Public names follow `DotPacking8<ComponentType><Quantization>`:

| `feature/quantization` | `compression` |
|---|---|
| `DotPacking8FixedU8Encoder` / `DotPacking8Fixedu8QueryEvaluator` | `DotPacking8U16FixedU8Encoder` / `DotPacking8U16FixedU8QueryEvaluator` |
| `DotPacking8ScalarU8Encoder` / `DotPacking8Scalaru8QueryEvaluator` | `DotPacking8U16ScalarU8Encoder` / `DotPacking8U16ScalarU8QueryEvaluator` |
| `DotPacking8f16Encoder` / `DotPacking8f16QueryEvaluator` | `DotPacking8U16F16Encoder` / `DotPacking8U16F16QueryEvaluator` |
| `DotPacking8Scalar4BitEncoder` / `DotPacking8Scalar4BitQueryEvaluator` | `DotPacking8U16Scalar4BitEncoder` / `DotPacking8U16Scalar4BitQueryEvaluator` |
| `OptimisticDotPacking8U32ScalarU8Encoder` / `…QueryEvaluator` | `BridgedDotPacking8U32ScalarU8Encoder` / `BridgedDotPacking8U32ScalarU8QueryEvaluator` |
| `BlockedSparseEncoder` path "v1" (`compute_distance_v1`) | `compute_distance_jump_dot` (private; public API unchanged) |

API differences besides the names:

- `new(input_dim)` is unchanged.
- `train(&PlainSparseDataset<C, f32, D>)` is generic over the distance `D` (was fixed to
  `SquaredEuclideanDistance`) and trains on **all** vectors (was a 5% sample, min 50,000).
- `train_components(iter)` is gone; `train` does both the permutation and the value
  quantizer. To set a permutation by hand: `set_component_mapping(ComponentMapping::from_permutation(..))`.
- `component_mapping()` / `inverse_component_mapping()` return `Option<&[C]>` as before.
- The quantizer types (`FixedU8Quantizer`, `ScalarU8Quantizer`, `Scalar4BitQuantizer`,
  `Centroid4BitQuantizer`, …) and the view/iterator types (`DotPacking8View`,
  `DotPacking8Iter`, `DotPacking8f16Iter`, …) are no longer public. Value codecs are
  `FixedU8Values`, `ScalarU8Values`, `F16Values`, `Scalar4BitValues` (sealed: no external
  implementations).
- Not ported (exist only on `feature/quantization`): `DotPacking8Centroid4BitEncoder`,
  `CDotPacking8*`, `DotPackingDp*`, `CDotPackingDp*`, `DotPackingSparse*`,
  `DotVByteScalarU8Encoder`, `DotVByteU32*`, `OptimisticDotVByte*`.
- **Serialized DotPacking8 encoders/datasets from `feature/quantization` do not load** on
  `compression` (different struct layout). Rebuild them. Encoded vector bytes are frozen by
  golden tests from commit `699f8b7` on.
- The u32 Bridged encoder: a stored value code of 0 marks a bridge entry; a real component is
  never quantized to code 0 (see `dotpacking8_u32_scalaru8.rs`).

---

### Additions (relative to `main`)

| Commit | Addition |
|---|---|
| `2d91d56` | Blocked ScalarU8/ScalarU16: `BlockScalarU8Values` (48-byte `DataBlockU8`) and `BlockScalarU16Values` (64-byte `DataBlockU16`), aliases `BlockedSparseScalarU8Encoder`/`…U16Encoder`, `BlockedSparseScalarU8Dataset`/`…U16Dataset` (+ `Growable`), the `Block` and `BlockValues` traits; `From`/`ConvertFrom<SparseDataset>` trains the steps; `compute_groundtruth --encoder blocked --blocked-values f16|scalaru8|scalaru16`. |
| `f3afe08` | DotPacking8 ScalarU16: `ScalarU16Values`, `DotPacking8U16ScalarU16Encoder`, `BridgedDotPacking8U32ScalarU16Encoder` with evaluators and datasets; `--dotpacking8-values scalaru16`. |
| `c57bf4e` | `compute_groundtruth`: sparse `--value-type scalaru8|scalaru16`, and `--distance euclidean` for every plain sparse value type; reports ms/query. |
| `deab0cd` | `ScalarStepSparseQuantizer<C, Code, D>` (plain sparse, `u8`/`u16` codes, one trained step per component, rounded to nearest; dot product and squared Euclidean), aliases `ScalarU8SparseQuantizer`, `ScalarU16SparseQuantizer`, datasets `ScalarU8SparseDataset<C, D>`, `ScalarU16SparseDataset<C, D>` (+ `Growable`), `StepCode`; `ConvertFrom<plain or cast-quantized sparse dataset>` trains the steps without copying the input. Replaces `ScalarSparseDataset<C, f32, FixedU8Q|FixedU16Q, D>` with better recall at the same size (see README). |
| `0bcdd01` | `From` / `ConvertFrom<SparseDataset<…>>` for every DotPacking8 dataset (one generic impl over `DotPacking8Encoder<C, V>`). Trains component order and value steps on the input, widens values of any float type to `f32` per vector, and makes no copy of the input. Produces exactly the bytes of `train` + `push`. |
| `1569f03` | `BlockedSparseEncoder`, `BlockedSparseQueryEvaluator`, `BlockedSparseDataset[Growable]`, `DataBlock`, `BLOCK_SIZE`, `SPARSE_QUERY_THRESHOLD` (= 33). Uncompressed `u16`/`f16` sparse vectors in 64-byte blocks of 16 entries. Queries with fewer than 33 components use **jump_dot** (skip every block whose last component is below the current query component); longer queries are densified. `From`/`ConvertFrom<SparseDataset<…, u16, …>>` for `PackedSparseDataset<BlockedSparseEncoder>`. Requires x86_64 with AVX2/F16C/FMA (not checked at run time). |
| `23ded88` … `d3c38f0` | DotPacking8 family: `DotPacking8U16{FixedU8,ScalarU8,F16,Scalar4Bit}Encoder`, `BridgedDotPacking8U32ScalarU8Encoder`, matching `…QueryEvaluator` and `…Dataset[Growable]` aliases. `DotProduct` only. Encode with `PackedSparseDatasetGrowable::new(encoder)` + `push`, after `encoder.train(&plain_dataset)`. No `From<SparseDataset>` conversion yet. |
| `67a3540`, `3f994c5` | `utils::train_sparse_scalar_quantizer`, `utils::train_sparse_scalar_quantizer_with_levels` (per-component scalar quantizer steps), `utils::load128_and_broadcast_to_256` (`unsafe`, SIMD helper). |
| `12211af` | `compute_groundtruth --encoder blocked` (u16 components, dot product): builds a `BlockedSparseDataset` and reports size, time and recall. |
| `4ee61ff`, `3f994c5` | `compute_groundtruth`: `--encoder dotpacking8`, `--component-type u16\|u32`, `--dotpacking8-values fixedu8\|scalaru8\|f16\|scalar4bit`, `--no-train` (fixedu8/f16 only), `--groundtruth <tsv>` (reports recall@k). Invalid combinations exit with code 2. |

### Performance-relevant changes

| Commit | Change |
|---|---|
| `7f76345` | `FlatIndex::search` (used by `Dataset::search`) marks its per-vector closure `#[inline(always)]`. Measured: no slowdown on plain sparse/dense, fixes a +4.3% regression on DotPacking8 F16. |
| `eb4ca45` | `prefetch_read_slice` (used by every `prefetch_with_range`) prefetches up to 512 bytes instead of 256. Measured on Seismic (MS MARCO cocondenser), against the 256-byte cap: blocked-sparse −19% / −15% query time, DotPacking8 F16 −8% / −4%, DotPacking8 scalar u8 −5% / −3%. Dense workloads not measured. |
| — | No nightly feature was added or removed relative to `main`. |
