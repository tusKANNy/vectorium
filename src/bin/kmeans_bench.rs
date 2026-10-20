use std::time::Instant;

use clap::Parser;
use num_traits::FromPrimitive;
use std::path::{Path, PathBuf};
use vectorium::PlainDenseDatasetGrowable;
use vectorium::clustering::KMeansBuilder;
use vectorium::core::vector::DenseVectorView;
use vectorium::datasets::readers::{read_npy_f32, read_seismic_format};
use vectorium::distances::{DotProduct, SquaredEuclideanDistance};
use vectorium::{ComponentType, Dataset, DatasetGrowable, PlainDenseDataset, PlainDenseQuantizer};

#[derive(Parser)]
#[command(about = "Benchmark KMeans training over dense or sparse datasets")]
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

    let builder = KMeansBuilder::new().verbose(true).build();

    println!(
        "Running KMeans on {} vectors, {}D, {} clusters)",
        dataset.len(),
        dataset.output_dim(),
        args.n_clusters,
    );

    let t0 = Instant::now();
    let centroids: PlainDenseDatasetGrowable<f32, SquaredEuclideanDistance> = builder.train(
        &dataset,
        args.n_clusters,
        None,
        PlainDenseQuantizer::new(dataset.output_dim()),
    );
    let elapsed = t0.elapsed();

    println!("Training finished in {:.3?}", elapsed);
    println!("Resulting centroids: {}", centroids.len());
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
