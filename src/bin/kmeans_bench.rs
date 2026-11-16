use std::time::Instant;

use clap::{Parser, ValueEnum};
use num_traits::FromPrimitive;
use std::path::{Path, PathBuf};
use vectorium::PlainDenseDatasetGrowable;
use vectorium::clustering::{
    EMeansParams, KMeans, KMeansBuilder, SimpleSparseInvertedIndex,
    SimpleSparseInvertedIndexSearchParams, Solver, sparse_centroid_pruning,
};
use vectorium::core::vector::DenseVectorView;
use vectorium::datasets::readers::{read_npy_f32, read_seismic_format};
use vectorium::distances::{DotProduct, SquaredEuclideanDistance};
use vectorium::{
    ComponentType, Dataset, DatasetGrowable, PlainDenseDataset, PlainDenseQuantizer,
    PlainSparseDatasetGrowable, PlainSparseQuantizer,
};

#[derive(Parser)]
#[command(about = "Benchmark k-means / e-means training over dense or sparse datasets")]
struct Args {
    /// Path to the dataset file (.npy for dense, Seismic binary for sparse)
    #[arg(long)]
    input: PathBuf,

    /// Dataset format: 'dense', 'sparse' (densified, squared Euclidean) or 'sparse-dot'
    /// (kept sparse, sparse centroids, dot-product assignment; for large vocabularies where
    /// densifying does not fit in memory). Default: dense.
    #[arg(long, default_value = "dense")]
    dataset_type: String,

    /// 'sparse-dot' only: prune each centroid after every update to its
    /// `ceil(factor * average member nnz)` largest components (`sparse_centroid_pruning`).
    /// 0 keeps the exact means.
    #[arg(long, default_value_t = 0.0)]
    centroid_pruning: f32,

    /// 'sparse-dot' only: how training assigns vectors to centroids. 'flat' scans every
    /// centroid; 'simple-sparse' walks a `SimpleSparseInvertedIndex` (inverted index) over them,
    /// rebuilt after every update.
    #[arg(long, value_enum, default_value_t = CentroidIndexArg::Flat)]
    centroid_index: CentroidIndexArg,

    /// 'simple-sparse' only: score only each vector's `doc_cut` largest components
    /// (approximate). Unset: all components (exact).
    #[arg(long)]
    doc_cut: Option<usize>,

    /// Component type for sparse datasets: 'u16' or 'u32' (default: u32)
    #[arg(long, default_value = "u32")]
    component_type: String,

    /// Number of clusters to request
    #[arg(long)]
    n_clusters: usize,

    /// K-means solver
    #[arg(long, value_enum, default_value_t = SolverArg::Lloyd)]
    solver: SolverArg,

    /// Epoch budget (default: the solver's, 10 for Lloyd and 2 for e-means). Lloyd needs a
    /// whole number; e-means accepts fractions.
    #[arg(long)]
    epochs: Option<f64>,

    /// RNG seed (default: from entropy)
    #[arg(long)]
    seed: Option<u64>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CentroidIndexArg {
    Flat,
    SimpleSparse,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum SolverArg {
    Lloyd,
    Emeans,
}

fn main() {
    let args = Args::parse();

    if args.dataset_type.eq_ignore_ascii_case("sparse-dot") {
        match args.component_type.to_lowercase().as_str() {
            "u16" => run_sparse_dot::<u16>(&args),
            "u32" => run_sparse_dot::<u32>(&args),
            other => panic!("Unsupported component type '{}'", other),
        }
        return;
    }

    let dataset = match args.dataset_type.to_lowercase().as_str() {
        "dense" => load_dense(&args.input),
        "sparse" => match args.component_type.to_lowercase().as_str() {
            "u16" => load_sparse::<u16>(&args.input),
            "u32" => load_sparse::<u32>(&args.input),
            other => panic!("Unsupported component type '{}'", other),
        },
        other => panic!("Unsupported dataset type '{}'", other),
    };

    let kmeans = build_kmeans(&args);

    println!(
        "Running k-means ({:?}) on {} vectors, {}D, {} clusters",
        args.solver,
        dataset.len(),
        dataset.output_dim(),
        args.n_clusters,
    );

    let t0 = Instant::now();
    let centroids: PlainDenseDatasetGrowable<f32, SquaredEuclideanDistance> = kmeans.train(
        &dataset,
        args.n_clusters,
        None,
        PlainDenseQuantizer::new(dataset.output_dim()),
    );
    let elapsed = t0.elapsed();

    println!("Training finished in {:.3?}", elapsed);
    println!("Resulting centroids: {}", centroids.len());

    // Mean squared distance of every vector to its nearest centroid (the k-means objective).
    let mse = KMeans::compute_assignments(&dataset, &centroids)
        .iter()
        .map(|&(d, _)| d as f64)
        .sum::<f64>()
        / dataset.len() as f64;
    println!("Mean squared distance to nearest centroid: {mse:.6e}");
}

fn load_dense(path: &Path) -> PlainDenseDataset<f32, SquaredEuclideanDistance> {
    read_npy_f32(path).expect("dataset load failed")
}

fn load_sparse<C>(path: &Path) -> PlainDenseDataset<f32, SquaredEuclideanDistance>
where
    C: ComponentType + FromPrimitive,
{
    let sparse = read_seismic_format::<C, f32, DotProduct>(path).expect("failed to read dataset");

    let dim = sparse.input_dim();
    let quantizer = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(dim);
    let mut growable = PlainDenseDatasetGrowable::new(quantizer.clone());
    let mut dense = vec![0.0f32; dim];

    for vector in sparse.iter() {
        dense.iter_mut().for_each(|v| *v = 0.0);
        for (&comp, &value) in vector.components().iter().zip(vector.values().iter()) {
            dense[comp.as_()] = value;
        }
        growable.push(DenseVectorView::new(&dense));
    }

    growable.into()
}

fn build_kmeans(args: &Args) -> KMeans {
    let solver = match args.solver {
        SolverArg::Lloyd => Solver::Lloyd,
        SolverArg::Emeans => Solver::EMeans(EMeansParams::default()),
    };
    let mut builder = KMeansBuilder::new()
        .solver(solver)
        .seed(args.seed)
        .verbose(true);
    if let Some(epochs) = args.epochs {
        builder = builder.epochs(epochs);
    }
    builder.build()
}

/// Clusters a sparse dataset without densifying it: sparse centroids, dot-product metric
/// (the vector is assigned to the centroid with the largest inner product), then reports
/// statistics of the final clustering.
fn run_sparse_dot<C>(args: &Args)
where
    C: ComponentType + FromPrimitive,
{
    let t_load = Instant::now();
    let dataset =
        read_seismic_format::<C, f32, DotProduct>(&args.input).expect("failed to read dataset");
    let n = dataset.len();
    let dim = dataset.input_dim();
    let data_nnz: usize = dataset.iter().map(|v| v.components().len()).sum();
    println!(
        "Loaded {n} sparse vectors, dim {dim}, avg nnz {:.2} in {:.3?}",
        data_nnz as f64 / n as f64,
        t_load.elapsed()
    );

    let kmeans = build_kmeans(args);
    println!(
        "Running k-means ({:?}, {} epochs, centroid pruning {}, assignment {:?}, doc_cut {:?}) on {n} vectors, {} clusters, dot product",
        args.solver,
        kmeans.epochs(),
        args.centroid_pruning,
        args.centroid_index,
        args.doc_cut,
        args.n_clusters,
    );

    let encoder = PlainSparseQuantizer::<C, f32, DotProduct>::new(dim, dim);
    let t0 = Instant::now();
    type Centroids<C> = PlainSparseDatasetGrowable<C, f32, DotProduct>;
    let pruning = args.centroid_pruning;
    let refine = |members: &[u32], centroid: &mut _| {
        if pruning > 0.0 {
            sparse_centroid_pruning(&dataset, pruning)(members, centroid);
        }
    };
    let centroids: Centroids<C> = match args.centroid_index {
        CentroidIndexArg::Flat => {
            kmeans.train_refined(&dataset, args.n_clusters, None, encoder, refine)
        }
        CentroidIndexArg::SimpleSparse => {
            let params = SimpleSparseInvertedIndexSearchParams {
                doc_cut: args.doc_cut,
            };
            kmeans.train_with_index_refined(
                &dataset,
                args.n_clusters,
                None,
                encoder,
                |c: Centroids<C>| SimpleSparseInvertedIndex::from(c),
                &params,
                refine,
            )
        }
    };
    let train_time = t0.elapsed();
    println!("Training finished in {train_time:.3?}");

    // Final, exact assignment for the metrics, through an exact index (same result as a flat
    // scan, up to floating-point summation order, and much faster at large k).
    let t1 = Instant::now();
    let final_index = SimpleSparseInvertedIndex::from(&centroids);
    let assignments = KMeans::assign_with_index(
        &dataset,
        &final_index,
        &SimpleSparseInvertedIndexSearchParams::exact(),
    );
    let assign_time = t1.elapsed();
    println!("Final exact assignment in {assign_time:.3?}");

    report_sparse_clustering(&centroids, &assignments);
    println!(
        "RESULT\tsolver={:?}\tepochs={}\tk={}\tcentroid_pruning={}\tcentroid_index={:?}\tdoc_cut={:?}\ttrain_secs={:.3}\tassign_secs={:.3}",
        args.solver,
        kmeans.epochs(),
        args.n_clusters,
        args.centroid_pruning,
        args.centroid_index,
        args.doc_cut,
        train_time.as_secs_f64(),
        assign_time.as_secs_f64()
    );
}

/// Prints the clustering quality and shape: the dot-product objective (mean inner product of
/// every vector with its centroid, higher is better), the cluster-size distribution, the imbalance factor
/// (`k_nonempty * sum(size^2) / n^2`, 1 is perfectly balanced), and centroid density.
fn report_sparse_clustering<C>(
    centroids: &PlainSparseDatasetGrowable<C, f32, DotProduct>,
    assignments: &[(f32, usize)],
) where
    C: ComponentType,
{
    let n = assignments.len();
    let k = centroids.len();

    let mut sizes = vec![0usize; k];
    let mut dot_sum = 0.0f64;
    for &(dot, c) in assignments {
        sizes[c] += 1;
        dot_sum += dot as f64;
    }

    let mut nonempty: Vec<usize> = sizes.iter().copied().filter(|&s| s > 0).collect();
    nonempty.sort_unstable();
    let k_nonempty = nonempty.len();
    let pct =
        |p: f64| nonempty[((p * (k_nonempty - 1) as f64).round() as usize).min(k_nonempty - 1)];
    let sq: f64 = nonempty.iter().map(|&s| (s as f64) * (s as f64)).sum();
    let imbalance = k_nonempty as f64 * sq / (n as f64 * n as f64);
    let singletons = nonempty.iter().filter(|&&s| s == 1).count();

    let centroid_nnz: Vec<usize> = (0..k)
        .map(|c| centroids.get(c as u64).components().len())
        .collect();
    let nnz_total: usize = centroid_nnz.iter().sum();
    let nnz_max = centroid_nnz.iter().copied().max().unwrap_or(0);

    println!(
        "Objective (mean dot with assigned centroid): {:.6}",
        dot_sum / n as f64
    );
    println!(
        "Clusters: {k} requested, {k_nonempty} non-empty, {singletons} singletons, imbalance factor {imbalance:.4}"
    );
    println!(
        "Cluster sizes: min {} p10 {} p50 {} p90 {} p99 {} max {} (mean {:.1})",
        nonempty[0],
        pct(0.10),
        pct(0.50),
        pct(0.90),
        pct(0.99),
        nonempty[k_nonempty - 1],
        n as f64 / k_nonempty as f64
    );
    println!(
        "Centroid nnz: mean {:.1} max {nnz_max}",
        nnz_total as f64 / k as f64
    );
    println!(
        "METRICS\tobjective={:.6}\tk_nonempty={k_nonempty}\tsingletons={singletons}\timbalance={imbalance:.4}\tsize_min={}\tsize_p50={}\tsize_p99={}\tsize_max={}\tcentroid_nnz_mean={:.1}\tcentroid_nnz_max={nnz_max}",
        dot_sum / n as f64,
        nonempty[0],
        pct(0.50),
        pct(0.99),
        nonempty[k_nonempty - 1],
        nnz_total as f64 / k as f64
    );
}
