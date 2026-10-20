# Vectorium

<p align="center">
    <img width="300px" src="/imgs/vectorium_logo.png" />
</p>

Vectorium is a Rust library for storing, accessing, and compressing dense and sparse embedding datasets.
Multivector support (ColBERT-style late-interaction) is available as an optional feature.
The main goal is to provide a unified dataset/encoder interface that can be shared by indexing/search crates such as
[Seismic](https://github.com/TusKANNy/seismic) and [kANNolo](https://github.com/TusKANNy/kannolo).

If you are new to KNN: *exhaustive* KNN searches score every vector in the dataset and return the top‑k closest results.
That is accurate but slow at scale. ANN indexes (HNSW, IVF, Seismic, etc.) trade a bit of accuracy for speed by building extra data structures (e.g., proximity graphs, inverted indexes) on top of the same dataset/encoder primitives.

Vectorium includes an exhaustive search API (`FlatIndex`, which implements the `Index` trait over any dataset) and a binary executable for ground-truth computation on CPU. For state‑of‑the‑art dense, sparse and multivector indexing, use these tools: [Seismic](https://github.com/TusKANNy/seismic), [kANNolo](https://github.com/TusKANNy/kannolo), [TACHIOM](https://github.com/TusKANNy/tachiom).

## Cargo features

| Feature | What it enables | Default |
|---------|-----------------|:-------:|
| `multivec` | `DenseMultiVectorView/Owned`, `MultiVectorDataset`, and all multivec encoders (`PlainMultiVecQuantizer`, `MultiVecProductQuantizer`, `MultiVecTwoLevelProductQuantizer`) | No |
| `cli` | The `compute_groundtruth` binary | No |

### Building

This repository targets nightly Rust (`rust-toolchain.toml` pins it).

If you want to compile the **library only** (dense and sparse datasets, no multivec, no binary):
```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

If you want the **`compute_groundtruth` CLI binary** (dense and sparse):
```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --features cli
```

If you want **multivector support** in the library (adds `MultiVectorDataset`, all multivec encoders):
```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --features multivec
```

If you want **both** the CLI binary and multivector support:
```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --features "cli,multivec"
```

### Using as a dependency

If you only need dense and sparse datasets:
```toml
[dependencies]
vectorium = { git = "https://github.com/TusKANNy/vectorium.git" }
```

If you also need multivector support:
```toml
[dependencies]
vectorium = { git = "https://github.com/TusKANNy/vectorium.git", features = ["multivec"] }
```

## Command-line tool
Vectorium ships a binary `compute_groundtruth` (feature `cli`) under `src/bin/` to exhaustive top‑k for a set of queries, writes a TSV file.

After building with `--features cli`, you will find `compute_groundtruth` in `target/release/`. You can also run it directly via Cargo:

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release --features cli --bin compute_groundtruth -- --help
```


This tool computes exhaustive top‑k neighbors for each query and writes the results as a TSV file. It is designed for
research/benchmarking workflows where you need exact results (ground truth) to compare against an ANN
index. It parallelizes across queries using Rayon, which is typically the right granularity for CPU ground‑truth runs.

#### Inputs and formats

- `--dataset-type dense` (default): expects `.npy` inputs (dataset + queries). The reader loads `.npy` as `f32` and can
  optionally convert the *dataset* storage via `--value-type`.
- `--dataset-type sparse`: expects Seismic binary format inputs (dataset + queries) via `read_seismic_format`.

#### Output format

Each output line is:

```text
query_id<TAB>doc_id<TAB>rank<TAB>score
```

- `query_id`: 0‑based query index.
- `doc_id`: a `VectorId` (0‑based vector index in the dataset).
- `rank`: `1..=k`.
- `score`:
  - for `--distance euclidean`: squared Euclidean distance (smaller is better);
  - for `--distance dotproduct`: dot product score (larger is better).

#### Examples

Dense vectors stored in `.npy`, Euclidean, PQ:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --features cli
./target/release/compute_groundtruth \
  -i <path>/dataset.npy \
  -q <path>/queries.npy \
  -o sift_gt_l2.tsv \
  --dataset-type dense \
  --value-type f32 \
  --distance euclidean \
  --encoder pq \
  --pq-subspaces 8
```

Dense vectors stored in `.npy`, dot product, plain (no PQ):

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --features cli
./target/release/compute_groundtruth \
  -i <path>/dataset.npy \
  -q <path>/queries.npy \
  -o sift_true_gt_ip.tsv \
  --dataset-type dense \
  --value-type f32 \
  --distance dotproduct \
  --encoder plain
```

Sparse dataset (Seismic binary), dot product, DotVByte compression:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --features cli
./target/release/compute_groundtruth \
  -i <path>/dataset.seismic \
  -q <path>/queries.seismic \
  -o sparse_gt_ip.tsv \
  --dataset-type sparse \
  --component-type u16 \
  --value-type f32 \
  --distance dotproduct \
  --encoder dotvbyte
```

#### Parameters and common combinations

- `-i, --input-file <PATH>`: dataset path (`.npy` when `--dataset-type dense`, Seismic binary when `sparse`).
- `-q, --query-file <PATH>`: query path (same format as the dataset type).
- `-o, --output-path <PATH>`: output TSV path.
- `-k, --k <N>`: number of neighbors to output per query (default: 10).
- `-d, --distance <euclidean|dotproduct>`: scoring metric (default: `euclidean`).
- `-v, --value-type <f32|f16|bf16|fixedu8|fixedu16>`: dataset storage value type (default: `f32`).
  - Dense queries are always `f32`.
  - Sparse queries are always `f32` (dataset values may be quantized).
- `--dataset-type <dense|sparse>`: dataset format (default: `dense`).
- `--component-type <u16|u32>`: sparse component type (default: `u32`).
- `--encoder <plain|pq|rabitq|rabitq-ext|dotvbyte>`: encoder (default: `plain`).
  - `pq`: dense‑only, requires `--value-type f32`, and uses `--pq-subspaces`.
  - `rabitq` / `rabitq-ext`: dense‑only, require `--value-type f32` and a dataset dimension that is a multiple of 64. Both metrics are supported.
  - `dotvbyte`: sparse‑only, requires `--component-type u16` and `--distance dotproduct`.
- `--pq-subspaces <M>`: number of PQ subspaces (`0` auto‑selects a supported value that divides the dataset dimension).
  - Product Quantization splits vectors into `M` equal‑sized chunks, so `dim % M == 0` is required.
  - Supported values are currently `{128, 96, 64, 32, 16, 8, 4}`.
- `--rabitq-query-bits <N>`: query bits for `--encoder rabitq` (`1..=8`, default `1`). Documents are always 1 bit; `rabitq-ext` scores an unquantized query and ignores this.
- `--rabitq-total-bits <N>`: document bits for `--encoder rabitq-ext` (`2`, `4` or `8`, default `4`).
- `--rabitq-seed <N>`: seed for the random orthogonal rotation (default: `42`).
- `--rabitq-rotate <true|false>`: apply the rotation (default: `true`).
- `--rabitq-faster-quant <true|false>`: for `rabitq-ext`, estimate the rescale factor once at train time rather than searching per vector (default: `true`).

Since these encoders are lossy, the output is what that encoder retrieves rather than exact
neighbours — use `--encoder plain --value-type f32` for true ground truth, and the lossy encoders to
measure how far a compression scheme falls short of it.


## Rust library

The library is organized around two core concepts:

1. A `Dataset` owns encoded vectors and an encoder.
2. A `VectorEncoder` defines the encoded representation and how to evaluate distances via a `QueryEvaluator`.

Indexes typically store a dataset plus extra search structures. During search they repeatedly need distances between a
query and *candidate vectors* inside the dataset. Vectorium is designed to make that path explicit and efficient: you can
build a query evaluator once, then score many candidate vectors without decoding them first.

### 1) Dense datasets (growable and immutable)

```rust
use vectorium::{Dataset, DenseDataset, DenseVectorView, DotProduct, DatasetGrowable, PlainDenseDatasetGrowable, PlainDenseQuantizer};

let encoder = PlainDenseQuantizer::<f32, DotProduct>::new(3);
let mut dataset = PlainDenseDatasetGrowable::new(encoder);

dataset.push(DenseVectorView::new(&[1.0, 0.0, 2.0]));
dataset.push(DenseVectorView::new(&[0.5, 1.5, 0.0]));

let frozen: DenseDataset<_> = dataset.into();
assert_eq!(frozen.len(), 2);
assert_eq!(frozen.get(0).values(), &[1.0, 0.0, 2.0]);
```

### 2) Sparse datasets (component/value pairs)

```rust
use vectorium::{Dataset, DotProduct, DatasetGrowable, PlainSparseDataset, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVectorView};

let encoder = PlainSparseQuantizer::<u16, f32, DotProduct>::new(5, 5);
let mut dataset = PlainSparseDatasetGrowable::new(encoder);

dataset.push(SparseVectorView::new(&[1_u16, 3], &[1.0, 2.0]));
dataset.push(SparseVectorView::new(&[0_u16, 4], &[0.5, 3.5]));

let frozen: PlainSparseDataset<u16, f32, DotProduct> = dataset.into();
assert_eq!(frozen.len(), 2);
assert_eq!(frozen.get(0).components(), &[1_u16, 3]);
```

### 3) Distance computation with a query evaluator (index-style)

The typical pattern for an index is: build the evaluator once, then score many candidates.

```rust
use vectorium::{Dataset, DatasetGrowable, DenseDataset, DenseVectorView, DotProduct, PlainDenseDatasetGrowable, PlainDenseQuantizer, QueryEvaluator, VectorEncoder, VectorId};

let encoder = PlainDenseQuantizer::<f32, DotProduct>::new(3);
let mut growable = PlainDenseDatasetGrowable::new(encoder);
growable.push(DenseVectorView::new(&[1.0, 0.0, 2.0]));
growable.push(DenseVectorView::new(&[0.5, 1.5, 0.0]));
let dataset: DenseDataset<_> = growable.into();

let query = DenseVectorView::new(&[0.2, 0.1, 0.7]);
// The second argument is the encoder's query parameters; this encoder has none, so `()`.
let evaluator = dataset.encoder().query_evaluator(query, &());

let candidate: VectorId = 0;
let score = evaluator.compute_distance(dataset.get(candidate));
let _ = score;
```

### 4) Exhaustive search (top‑k baseline)

`FlatIndex` wraps any dataset and searches it exhaustively. It is the brute-force implementation
of the `Index` trait, provides an exhaustive top‑k baseline:

```rust
use vectorium::{DatasetGrowable, DenseDataset, DenseVectorView, DotProduct, FlatIndex, Index, PlainDenseDatasetGrowable, PlainDenseQuantizer};

let encoder = PlainDenseQuantizer::<f32, DotProduct>::new(3);
let mut growable = PlainDenseDatasetGrowable::new(encoder);
growable.push(DenseVectorView::new(&[1.0, 0.0, 2.0]));
growable.push(DenseVectorView::new(&[0.5, 1.5, 0.0]));
let dataset: DenseDataset<_> = growable.into();

let query = DenseVectorView::new(&[0.2, 0.1, 0.7]);
let top2 = FlatIndex::from(&dataset).search(query, 2, &());
assert_eq!(top2.len(), 2);
```

### 5) Parallel ground-truth computation in Rust (query-level parallelism)

The easiest way to use multiple CPU cores is to parallelize across queries. This is the strategy used by
`compute_groundtruth`.

```rust
use rayon::prelude::*;
use vectorium::{DatasetGrowable, DenseVectorView, DotProduct, FlatIndex, Index, PlainDenseDatasetGrowable, PlainDenseQuantizer};

let encoder = PlainDenseQuantizer::<f32, DotProduct>::new(3);
let mut growable = PlainDenseDatasetGrowable::new(encoder);
growable.push(DenseVectorView::new(&[1.0, 0.0, 2.0]));
growable.push(DenseVectorView::new(&[0.5, 1.5, 0.0]));

let queries = vec![vec![0.2_f32, 0.1, 0.7], vec![1.0_f32, 0.0, 0.0]];
let results: Vec<_> = queries
    .par_iter()
    .map(|q| FlatIndex::from(&growable).search(DenseVectorView::new(q.as_slice()), 1, &()))
    .collect();

assert_eq!(results.len(), 2);
```

### 6) Multivector datasets (requires `multivec` feature)

Multivector datasets store variable-length sequences of token vectors — the representation used by late-interaction models such as ColBERT.
Enable the feature in your `Cargo.toml` first (see [Cargo features](#cargo-features)).

```rust
# #[cfg(feature = "multivec")]
# {
use vectorium::{
    Dataset, DatasetGrowable, DenseMultiVectorView, MultiVectorDataset,
    MultiVectorDatasetGrowable, PlainMultiVecQuantizer,
};

// Each document is a sequence of token vectors.  Here dim=2 and each doc has 2 tokens.
let encoder = PlainMultiVecQuantizer::<f32>::new(2);
let mut dataset = MultiVectorDatasetGrowable::new(encoder);

// Doc 0: two tokens [1.0, 0.0] and [0.0, 1.0]
dataset.push(DenseMultiVectorView::new(&[1.0_f32, 0.0, 0.0, 1.0], 2));
// Doc 1: two tokens [0.5, 0.5] and [1.0, 1.0]
dataset.push(DenseMultiVectorView::new(&[0.5_f32, 0.5, 1.0, 1.0], 2));

let frozen: MultiVectorDataset<_> = dataset.into();
assert_eq!(frozen.len(), 2);
# }
```

### 7) Range-based access and prefetch

Datasets expose `range_from_id`/`id_from_range` so callers can keep lightweight handles to the underlying storage ranges.
This is mainly useful for sparse/packed layouts, where range lookups can be a cache miss and prefetching can help.

```rust
use vectorium::{Dataset, DatasetGrowable, DenseDataset, DenseVectorView, DotProduct, PlainDenseDatasetGrowable, PlainDenseQuantizer};

let encoder = PlainDenseQuantizer::<f32, DotProduct>::new(3);
let mut growable = PlainDenseDatasetGrowable::new(encoder);
growable.push(DenseVectorView::new(&[1.0, 0.0, 2.0]));
let dataset: DenseDataset<_> = growable.into();

let id = 0;
let range = dataset.range_from_id(id);
dataset.prefetch_with_range(range.clone());
let view = dataset.get_with_range(range);
assert_eq!(view.values(), &[1.0, 0.0, 2.0]);
```

### 8) Encoders: choosing a compression scheme

Every encoder implements the same `VectorEncoder`/`QueryEvaluator` contract, so datasets, `FlatIndex`
and any index built on top behave identically whichever one you pick — only the evaluator changes.
What differs is footprint and accuracy.

| Encoder | Layout | Stored as | Metric | Bytes / vector | Built with |
|---|---|---|---|---|---|
| `PlainDenseQuantizer` | dense | `f32` | ℓ₂ / IP | `4d` | `push` |
| `ScalarDenseQuantizer` | dense | `f16` / `bf16` / `FixedU8Q` / `FixedU16Q` | ℓ₂ / IP | `2d` / `2d` / `d` / `2d` | `convert_into(())` |
| `ProductQuantizer<M, D>` | dense | `M` × `u8` codes | ℓ₂ / IP | `M` | `convert_into(())` |
| `RabitqQuantizer<D>` | dense | 1 bit/comp + metadata | ℓ₂ / IP | `d/8 + 8` | `convert_into(config)` |
| `RabitqExtQuantizer<D>` | dense | `total_bits`/comp + metadata | ℓ₂ / IP | `total_bits·d/8 + 8` | `convert_into(config)` |
| `PlainSparseQuantizer` | sparse | `f32` / `f16` | IP | varies with nnz | `push` |
| `DotVByteFixedU8Encoder` | packed sparse | group-varint `u64` blob | IP | varies with nnz | `convert_into(())` |

(The multi-vector encoders behind the `multivec` feature are covered in section 6.)

**One construction pattern, for every encoder.** Every encoded dataset is built by converting an
existing one — `plain.convert_into(config)` — which learns whatever else it needs (PQ's codebooks,
a scalar range, RaBitQ's means and rotation) from the source data. You never have to look up how a
particular encoder is constructed: it is always this call, and what changes is only the associated
`Config` it asks for. Some encoders take a real config (`RabitqConfig`, `RabitqExtConfig`); the
rest take `()`, written out as `convert_into(())`. This is how `Index::search(query, k, &params)`
already treats search parameters — one entry point, and `()` is a configuration rather than the
absence of one.

Passing the config explicitly is the point for the configured encoders: `query_bits` and
`total_bits` *are* the footprint/accuracy dial, so a build site that silently defaulted them would
hide the decision that matters most. `RabitqConfig::default()` is still one expression away.

#### Empty config: convert an existing dataset

```rust
use vectorium::dataset::ConvertInto;
use vectorium::{
    Dataset, DatasetGrowable, DenseDataset, DenseVectorView, DotProduct, FlatIndex, Index,
    PlainDenseDataset, PlainDenseDatasetGrowable, PlainDenseQuantizer, ProductQuantizer,
    VectorEncoder,
};

let d = 64;
let mut growable = PlainDenseDatasetGrowable::new(PlainDenseQuantizer::<f32, DotProduct>::new(d));
for i in 0..512usize {
    let mut v = vec![-1.0f32; d];
    v[..(i % d)].fill(1.0);
    growable.push(DenseVectorView::new(&v));
}
let plain: PlainDenseDataset<f32, DotProduct> = growable.into();

// 8 subspaces, one u8 code each: 8 bytes/vector instead of 256. Requires d % M == 0.
// `plain` is borrowed, not consumed, so it is still usable afterwards.
let pq: DenseDataset<ProductQuantizer<8, DotProduct>> = (&plain).convert_into(());
assert_eq!(pq.encoder().output_dim(), 8);

let mut query = vec![-1.0f32; d];
query[..56].fill(1.0);
let top1 = FlatIndex::from(&pq).search(DenseVectorView::new(&query), 1, &());
assert_eq!(top1.len(), 1);
```

The same `convert_into(())` call builds an `f16` dataset (`ScalarDenseDataset<f32, f16, D>`) or, for
sparse data, a DotVByte-compressed one.

#### Configured: RaBitQ and Extended RaBitQ

`RabitqQuantizer` stores 1 bit per component: it subtracts the per-component means, applies a seeded
random orthogonal rotation, and keeps the sign of each rotated residual, packed 64 bits per `u64`.
Two per-document floats (the residual norm and the code/residual cosine) turn the raw sign agreement
into an *unbiased distance estimate*, which is what separates it from a plain sign code.
`RabitqExtQuantizer` generalizes this to `total_bits ∈ {2, 4, 8}` bits per component (1 sign bit plus
`total_bits - 1` magnitude bits), making `total_bits` the footprint/accuracy dial; `total_bits = 1`
is rejected rather than duplicating the 1-bit encoder. **Both require `dim % 64 == 0`** (asserted in
`train`; there is no bit-padding).

The metric is a type parameter — `RabitqDenseDataset` for inner product,
`RabitqDenseDatasetSquaredEuclidean` for Euclidean (and likewise for `RabitqExt…`). Document codes
are **identical** either way, so the metric is a free choice at build time.

```rust
use vectorium::dataset::ConvertInto;
use vectorium::{
    Dataset, DatasetGrowable, DenseVectorView, DotProduct, FlatIndex, Index, PlainDenseDataset,
    PlainDenseDatasetGrowable, PlainDenseQuantizer, RabitqConfig, RabitqDenseDataset,
    RabitqExtConfig, RabitqExtDenseDataset, RabitqQueryParams, VectorEncoder,
};

let d = 64;
let mut growable = PlainDenseDatasetGrowable::new(PlainDenseQuantizer::<f32, DotProduct>::new(d));
for ones in [8usize, 24, 56] {
    let mut v = vec![-1.0f32; d];
    v[..ones].fill(1.0);
    growable.push(DenseVectorView::new(&v));
}
let plain: PlainDenseDataset<f32, DotProduct> = growable.into();

let mut query = vec![-1.0f32; d];
query[..56].fill(1.0);

// 1 bit per component. Documents are always 1 bit; `query_bits` scalar-quantizes the *query*
// only, so it is a search parameter rather than part of the stored index.
let rabitq: RabitqDenseDataset = (&plain).convert_into(RabitqConfig::default());
assert_eq!(rabitq.encoder().output_dim(), d / 64 + 1); // code words + 1 metadata word
let params = RabitqQueryParams::new(4);
let top1 = FlatIndex::from(&rabitq).search(DenseVectorView::new(&query), 1, &params);
assert_eq!(top1[0].vector, 2);

// 4 bits per component. `faster_quant` (the default) estimates the rescale factor once at train
// time instead of searching per vector — far faster to encode, for a marginal accuracy cost.
let config = RabitqExtConfig { total_bits: 4, ..RabitqExtConfig::default() };
let ext: RabitqExtDenseDataset = (&plain).convert_into(config);
assert_eq!(ext.encoder().output_dim(), 4 * (d / 64) + 1);
let top1 = FlatIndex::from(&ext).search(DenseVectorView::new(&query), 1, &());
assert_eq!(top1[0].vector, 2);
```

Two things worth knowing. `query_bits` is query-side state: pass a
different `RabitqQueryParams` to `search` to change it, on a loaded dataset, with no re-encoding and
no mutation of the encoder, one index serves every setting. And the extended encoder stores
codes **component-major** at those three byte-aligned widths, scoring against an **unquantized**
query: one widen plus one fused multiply-add per 16 components, with no query-side error term and
no `query_bits` dial. The intermediate widths
(3, 5, 6, 7, 9) are not supported — a space-exact code would have to be split into aligned parts,
which costs throughput without buying a useful accuracy/footprint point.


## Design notes

- **Type safety by construction:** datasets are sealed and are tied to a specific encoder type, so mixing dense/sparse/packed
  representations is prevented at compile time.
- **Evaluator-driven search:** encoders build a `QueryEvaluator` from the query once; the evaluator can then score many dataset
  vectors. This matches how most ANN indexes are structured internally.
- **Distance ordering:** distances implement `Ord`. `DotProduct` uses reversed ordering (larger is better), while
  `SquaredEuclideanDistance` uses the natural ordering (smaller is better). Distance values must not be NaN.
- **Range-based access:** `range_from_id`/`get_with_range` and `prefetch_with_range` are meant for index implementations that keep
  storage ranges around (especially for sparse/packed datasets) and want predictable, low-overhead access.

## License

MIT. See `LICENSE-MIT`.
