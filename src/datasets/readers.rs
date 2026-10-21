//! File readers for loading datasets from various formats.

use std::io;
use std::path::Path;

use std::fs::File;
use std::io::{BufReader, Read};

use ndarray::Array2;
use ndarray_npy::ReadNpyExt;

use crate::encoders::dense_scalar::{PlainDenseQuantizer, ScalarDenseSupportedDistance};
use crate::encoders::sparse_scalar::{PlainSparseQuantizer, ScalarSparseSupportedDistance};
use crate::{ComponentType, Float, FromF32, SpaceUsage, ValueType};
use crate::{PlainDenseDataset, PlainSparseDataset};

/// Error type for dataset reading operations.
#[derive(Debug)]
pub enum ReaderError {
    Io(io::Error),
    Npy(ndarray_npy::ReadNpyError),
}

impl From<io::Error> for ReaderError {
    fn from(e: io::Error) -> Self {
        ReaderError::Io(e)
    }
}

impl From<ndarray_npy::ReadNpyError> for ReaderError {
    fn from(e: ndarray_npy::ReadNpyError) -> Self {
        ReaderError::Npy(e)
    }
}

impl std::fmt::Display for ReaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReaderError::Io(e) => write!(f, "IO error: {}", e),
            ReaderError::Npy(e) => write!(f, "NPY error: {}", e),
        }
    }
}

impl std::error::Error for ReaderError {}

/// TODO: make it generic over the AsRef(f32) to return a static dataset or a growable one dataset
/// Reads a dense dataset from a `.npy` file containing f32 values.
///
/// The file should contain a 2D array where:
/// - Each row is a vector
/// - Each column is a dimension
///
/// The distance type `D` must implement `ScalarDenseSupportedDistance`.
///
/// # Type Parameters
/// * `D` - The distance type (e.g., `SquaredEuclideanDistance` or `DotProduct`)
///
/// # Arguments
/// * `filename` - Path to the `.npy` file
///
/// # Example
/// ```ignore
/// use vectorium::readers::read_npy_f32;
/// use vectorium::distances::SquaredEuclideanDistance;
///
/// let dataset = read_npy_f32::<SquaredEuclideanDistance>("vectors.npy")?;
/// let evaluator = dataset.encoder().query_evaluator(query);
/// ```
pub fn read_npy_f32<D>(filename: impl AsRef<Path>) -> Result<PlainDenseDataset<f32, D>, ReaderError>
where
    D: ScalarDenseSupportedDistance,
{
    let file = std::fs::File::open(filename)?;
    let reader = io::BufReader::new(file);

    let array: Array2<f32> = Array2::read_npy(reader)?;

    let (n_vecs, d) = array.dim();
    let data: Vec<f32> = array.into_raw_vec_and_offset().0;

    let quantizer = PlainDenseQuantizer::new(d);

    Ok(PlainDenseDataset::from_raw(data.into(), n_vecs, quantizer))
}

/// Read a sparse dataset from the Seismic binary file format.
///
/// The format of this binary file is the following:
/// - Number of vectors `n_vecs` in 4 bytes, followed by `n_vecs` sparse vectors.
/// - For each vector:
///   - Its length (number of non-zero components) in 4 bytes
///   - A sorted sequence of n components in 4 bytes each (u32)
///   - Corresponding n values in 4 bytes each (f32)
///
/// # Type Parameters
/// * `C` - Component type (must be `ComponentType`)
/// * `V` - Value type (must be `ValueType + Float`)
/// * `D` - Distance type (must implement `ScalarSparseSupportedDistance`)
///
/// # Errors
/// Returns `ReaderError` if:
/// - File cannot be read
/// - Component values overflow the type C (e.g., reading a value > 2^16 when C is u16)
///
/// # Example
/// ```ignore
/// use vectorium::readers::read_seismic_format;
/// use vectorium::distances::DotProduct;
///
/// let dataset = read_seismic_format::<u32, f32, DotProduct>("vectors.bin")?;
/// ```
pub fn read_seismic_format<C, V, D>(
    filename: impl AsRef<Path>,
) -> Result<PlainSparseDataset<C, V, D>, ReaderError>
where
    C: ComponentType + num_traits::FromPrimitive,
    V: ValueType + Float + FromF32 + SpaceUsage,
    D: ScalarSparseSupportedDistance,
{
    read_seismic_format_limit(filename, None)
}

/// Read a sparse dataset from the Seismic binary file format with an optional limit on the number of vectors.
///
/// # Type Parameters
/// * `C` - Component type (must be `ComponentType`)
/// * `V` - Value type (must be `ValueType + Float`)
/// * `D` - Distance type (must implement `ScalarSparseSupportedDistance`)
///
/// # Arguments
/// * `filename` - Path to the binary file
/// * `limit` - Maximum number of vectors to read (if `None`, reads all vectors)
///
/// # Errors
/// Returns `ReaderError` if:
/// - File cannot be read
/// - Component values overflow the type C (e.g., reading a value > 2^16 when C is u16)
/// - The components of a vector are not strictly ascending
pub fn read_seismic_format_limit<C, V, D>(
    filename: impl AsRef<Path>,
    limit: Option<usize>,
) -> Result<PlainSparseDataset<C, V, D>, ReaderError>
where
    C: ComponentType + num_traits::FromPrimitive,
    V: ValueType + Float + FromF32 + SpaceUsage,
    D: ScalarSparseSupportedDistance,
{
    let path = Path::new(filename.as_ref());
    let file = File::open(path)?;
    let file_len = file.metadata()?.len() as usize;
    let mut br = BufReader::with_capacity(1 << 22, file);
    let invalid = |msg: String| ReaderError::Io(io::Error::new(io::ErrorKind::InvalidData, msg));

    let mut word = [0_u8; 4];
    br.read_exact(&mut word)?;
    let file_n_vecs = u32::from_le_bytes(word) as usize;
    let n_vecs = limit.map_or(file_n_vecs, |n| n.min(file_n_vecs));

    // One pass straight into the CSR arrays, reserved from the file size (each
    // non-zero takes 8 bytes after the header and the vector lengths) so that
    // the multi-gigabyte arrays are never reallocated.
    let file_nnz = file_len.saturating_sub(4 * (1 + file_n_vecs)) / 8;
    let nnz_hint = if file_n_vecs == 0 {
        0
    } else {
        (file_nnz as u128 * n_vecs as u128 / file_n_vecs as u128) as usize
    };
    let mut offsets = Vec::with_capacity(n_vecs + 1);
    offsets.push(0);
    let mut components: Vec<C> = Vec::with_capacity(nnz_hint);
    let mut values: Vec<V> = Vec::with_capacity(nnz_hint);
    let mut max_component: Option<u32> = None;
    let mut bytes = Vec::new();
    for _ in 0..n_vecs {
        br.read_exact(&mut word)?;
        let n = u32::from_le_bytes(word) as usize;
        bytes.resize(8 * n, 0);
        br.read_exact(&mut bytes)?;
        let (comp_bytes, value_bytes) = bytes.split_at(4 * n);

        let mut prev: Option<u32> = None;
        for &c in comp_bytes.as_chunks::<4>().0 {
            let c = u32::from_le_bytes(c);
            if prev.is_some_and(|p| p >= c) {
                return Err(invalid(format!(
                    "components of vector {} are not strictly ascending",
                    offsets.len() - 1
                )));
            }
            prev = Some(c);
            components.push(C::from_u32(c).ok_or_else(|| {
                invalid(format!(
                    "Component value {c} exceeds maximum for type (possible overflow)"
                ))
            })?);
        }
        max_component = max_component.max(prev);
        values.extend(
            value_bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&v| V::from_f32_saturating(f32::from_le_bytes(v))),
        );
        offsets.push(components.len());
    }

    let input_dim = max_component.map_or(0, |m| m as usize + 1);
    let quantizer = PlainSparseQuantizer::<C, V, D>::new(input_dim, input_dim);
    Ok(PlainSparseDataset::from_parts(
        offsets, components, values, quantizer,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dataset::Dataset;
    use crate::distances::{DotProduct, SquaredEuclideanDistance};
    use std::path::PathBuf;

    fn toy_dense_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("examples")
            .join("toy_datasets")
            .join("toy_dense.npy")
    }

    fn toy_seismic_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("examples")
            .join("toy_datasets")
            .join("toy_seismic.bin")
    }

    #[test]
    fn read_npy_dense_dataset() -> Result<(), ReaderError> {
        let dataset = read_npy_f32::<SquaredEuclideanDistance>(toy_dense_path())?;
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.output_dim(), 3);
        let first = dataset.get(0);
        assert_eq!(first.values(), &[1.0_f32, 2.0, 3.0]);
        let second = dataset.get(1);
        assert_eq!(second.values(), &[4.0_f32, 5.0, 6.0]);
        Ok(())
    }

    #[test]
    fn read_seismic_format_dataset_and_limit() -> Result<(), ReaderError> {
        let dataset = read_seismic_format::<u16, f32, DotProduct>(toy_seismic_path())?;
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.nnz(), 3);

        let first = dataset.get(0);
        assert_eq!(first.components(), &[0_u16, 2]);
        assert_eq!(first.values(), &[1.0_f32, 3.0]);

        let second = dataset.get(1);
        assert_eq!(second.components(), &[1_u16]);
        assert_eq!(second.values(), &[2.0_f32]);

        let limited =
            read_seismic_format_limit::<u16, f32, DotProduct>(toy_seismic_path(), Some(1))?;
        assert_eq!(limited.len(), 1);
        Ok(())
    }

    /// Write `vectors` in the Seismic format to a fresh temporary file.
    fn seismic_file(name: &str, vectors: &[(&[u32], &[f32])]) -> PathBuf {
        let mut bytes = (vectors.len() as u32).to_le_bytes().to_vec();
        for (components, values) in vectors {
            bytes.extend((components.len() as u32).to_le_bytes());
            components
                .iter()
                .for_each(|c| bytes.extend(c.to_le_bytes()));
            values.iter().for_each(|v| bytes.extend(v.to_le_bytes()));
        }
        let path =
            std::env::temp_dir().join(format!("vectorium_{}_{name}.bin", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn read_seismic_format_vectors_dim_and_limits() -> Result<(), ReaderError> {
        let vectors: [(&[u32], &[f32]); 4] = [
            (&[3, 7, 70_000], &[0.5, 1.5, 2.5]),
            (&[], &[]),
            (&[0], &[-1.0]),
            (&[1, 2], &[4.0, 8.0]),
        ];
        let path = seismic_file("vectors", &vectors);
        let dataset = read_seismic_format::<u32, f32, DotProduct>(&path)?;
        assert_eq!(dataset.len(), 4);
        assert_eq!(dataset.nnz(), 6);
        assert_eq!(dataset.input_dim(), 70_001);
        for (i, (components, values)) in vectors.iter().enumerate() {
            let v = dataset.get(i as u64);
            assert_eq!(v.components(), *components);
            assert_eq!(v.values(), *values);
        }
        // The dimension comes from the vectors actually read.
        let first_three = read_seismic_format_limit::<u32, f32, DotProduct>(&path, Some(3))?;
        assert_eq!((first_three.len(), first_three.nnz()), (3, 4));
        let more = read_seismic_format_limit::<u32, f32, DotProduct>(&path, Some(10))?;
        assert_eq!(more.len(), 4);
        let none = read_seismic_format_limit::<u32, f32, DotProduct>(&path, Some(0))?;
        assert_eq!((none.len(), none.input_dim()), (0, 0));
        std::fs::remove_file(path).unwrap();
        Ok(())
    }

    #[test]
    fn read_seismic_format_empty_file() -> Result<(), ReaderError> {
        let path = seismic_file("empty", &[]);
        let dataset = read_seismic_format::<u32, f32, DotProduct>(&path)?;
        assert_eq!(
            (dataset.len(), dataset.nnz(), dataset.input_dim()),
            (0, 0, 0)
        );
        std::fs::remove_file(path).unwrap();
        Ok(())
    }

    #[test]
    fn read_seismic_format_rejects_bad_files() {
        let overflow = seismic_file("overflow", &[(&[1, 70_000], &[1.0, 2.0])]);
        let err = read_seismic_format::<u16, f32, DotProduct>(&overflow).unwrap_err();
        assert!(err.to_string().contains("exceeds maximum"), "{err}");

        let unsorted = seismic_file("unsorted", &[(&[0], &[1.0]), (&[4, 2], &[1.0, 2.0])]);
        let err = read_seismic_format::<u32, f32, DotProduct>(&unsorted).unwrap_err();
        assert!(
            err.to_string()
                .contains("vector 1 are not strictly ascending"),
            "{err}"
        );
        let duplicate = seismic_file("duplicate", &[(&[2, 2], &[1.0, 2.0])]);
        assert!(read_seismic_format::<u32, f32, DotProduct>(&duplicate).is_err());

        let truncated = seismic_file("truncated", &[(&[1, 2, 3], &[1.0, 2.0, 3.0])]);
        let bytes = std::fs::read(&truncated).unwrap();
        std::fs::write(&truncated, &bytes[..bytes.len() - 2]).unwrap();
        assert!(read_seismic_format::<u32, f32, DotProduct>(&truncated).is_err());

        for p in [overflow, unsorted, duplicate, truncated] {
            std::fs::remove_file(p).unwrap();
        }
    }
}
