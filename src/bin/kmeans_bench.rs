use std::time::Instant;

use clap::{Parser, ValueEnum};
use num_traits::FromPrimitive;
use std::path::{Path, PathBuf};
use vectorium::PlainDenseDatasetGrowable;
use vectorium::clustering::{EMeansParams, KMeans, KMeansBuilder, LloydParams, Solver};
use vectorium::core::vector::DenseVectorView;
use vectorium::datasets::readers::{read_npy_f32, read_seismic_format};
use vectorium::distances::{DotProduct, SquaredEuclideanDistance};
use vectorium::{ComponentType, Dataset, DatasetGrowable, PlainDenseDataset, PlainDenseQuantizer};

#[derive(Parser)]
#[command(about = "Benchmark k-means / e-means training over dense or sparse datasets")]
struct Args {
    /// Path to the dataset file (.npy for dense, Seismic binary for sparse)
    #[arg(long)]
    input: PathBuf,

    /// Dataset format: 'dense' or 'sparse' (default: dense)
    #[arg(long, default_value = "dense")]
    dataset_type: String,

    /// Component type for sparse datasets: 'u16' or 'u32' (default: u32)
    #[arg(long, default_value = "u32")]
    component_type: String,

    /// Number of clusters to request
    #[arg(long)]
    n_clusters: usize,

    /// K-means solver
    #[arg(long, value_enum, default_value_t = SolverArg::Lloyd)]
    solver: SolverArg,

    /// Lloyd iterations (each is one epoch)
    #[arg(long, default_value_t = 10)]
    lloyd_n_iter: usize,

    /// e-means epochs over the data (fractional allowed)
    #[arg(long, default_value_t = 5.0)]
    emeans_epochs: f64,

    /// RNG seed (default: from entropy)
    #[arg(long)]
    seed: Option<u64>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum SolverArg {
    Lloyd,
    Emeans,
}

fn main() {
    let args = Args::parse();

    let dataset = match args.dataset_type.to_lowercase().as_str() {
        "dense" => load_dense(&args.input),
        "sparse" => match args.component_type.to_lowercase().as_str() {
            "u16" => load_sparse::<u16>(&args.input),
            "u32" => load_sparse::<u32>(&args.input),
            other => panic!("Unsupported component type '{}'", other),
        },
        other => panic!("Unsupported dataset type '{}'", other),
    };

    let solver = match args.solver {
        SolverArg::Lloyd => Solver::Lloyd(LloydParams {
            n_iter: args.lloyd_n_iter,
            ..Default::default()
        }),
        SolverArg::Emeans => Solver::EMeans(EMeansParams {
            nepochs: args.emeans_epochs,
            ..Default::default()
        }),
    };
    let kmeans = KMeansBuilder::new()
        .solver(solver)
        .seed(args.seed)
        .verbose(true)
        .build();

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
