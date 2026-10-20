#![feature(portable_simd)]
#![feature(f16)]
#![allow(internal_features)]
#![feature(core_intrinsics)]
#![feature(gen_blocks)]
#![feature(yield_expr)]
#![doc = include_str!("../README.md")]
#![allow(non_snake_case)]

use num_traits::{AsPrimitive, ToPrimitive, Zero};
use serde::{Serialize, de::DeserializeOwned};
use std::fs::File;
use std::io::{BufReader, BufWriter};

/// Type aliases for quantized fixed-point types. You can change FRAC in the `fixed` crate to adjust the precision.
/// The `FixedU8Q` type uses 6 fractional bits, while `FixedU16Q` uses 13 fractional bits.
use fixed::FixedU8;
use fixed::FixedU16;
use fixed::types::extra::U6;
use fixed::types::extra::U13;
pub type FixedU8Q = FixedU8<U6>;
pub type FixedU16Q = FixedU16<U13>;

pub mod clustering;
pub mod core;
pub mod datasets;
pub mod encoders;
pub mod transformations;
pub mod utils;

pub use core::dataset;
pub use core::distances;
pub use core::storage;
pub use core::vector;
pub use core::vector_encoder;
pub use utils::numeric_markers;
pub use utils::space_usage;

pub use utils::numeric_markers::{Float, FromF32};
pub use utils::space_usage::SpaceUsage;

#[cfg(feature = "multivec")]
pub use core::vector::{DenseMultiVectorOwned, DenseMultiVectorView};
pub use core::vector::{
    DenseVectorOwned, DenseVectorView, PackedVectorOwned, PackedVectorView, SparseVectorOwned,
    SparseVectorView, VectorView,
};

pub use core::distances::{Distance, DotProduct, SquaredEuclideanDistance};

#[cfg(feature = "multivec")]
pub use core::vector_encoder::MultiVecEncoder;
pub use core::vector_encoder::{
    DenseVectorEncoder, PackedSparseVectorEncoder, QueryEvaluator, SparseVectorEncoder,
    VectorEncoder, WithDistance,
};

pub use clustering::{KMeans, KMeansBuilder, sparse_centroid_pruning};

pub use encoders::rabitq::{
    RabitqConfig, RabitqQuantizer, RabitqQueryEvaluator, RabitqQueryParams, RabitqSupportedDistance,
};
pub use encoders::rabitq_ext::{RabitqExtConfig, RabitqExtQuantizer, RabitqExtQueryEvaluator};
pub use transformations::fht_kac::FhtKacRotator;
pub use transformations::l1_prune::{l1_prune_split, l1_prune_split_vector};

pub use encoders::dense_scalar::{
    PlainDenseQuantizer, PlainDenseQuantizerDotProduct, PlainDenseQuantizerSquaredEuclidean,
    ScalarDenseQuantizer, ScalarDenseQuantizerDotProduct, ScalarDenseQuantizerSame,
    ScalarDenseQuantizerSquaredEuclidean, ScalarDenseQueryEvaluator, ScalarDenseSupportedDistance,
};
pub use encoders::dotvbyte_fixedu8::{DotVByteFixedU8Encoder, DotVByteFixedU8QueryEvaluator};
#[cfg(feature = "multivec")]
pub use encoders::multivec_pq::{MultiVecPQQueryEvaluator, MultiVecProductQuantizer};
#[cfg(feature = "multivec")]
pub use encoders::multivec_scalar::{
    PlainMultiVecQuantizer, ScalarMultiVecQuantizer, ScalarMultiVecQueryEvaluator,
};
#[cfg(feature = "multivec")]
pub use encoders::multivec_two_level_pq::{
    MultiVecTwoLevelPQQueryEvaluator, MultiVecTwoLevelProductQuantizer,
};
pub use encoders::pq::ProductQuantizer;
pub use encoders::sparse_scalar::{
    PlainSparseQuantizer, PlainSparseQuantizerDotProduct, ScalarSparseQuantizer,
    ScalarSparseQuantizerDotProduct, ScalarSparseQueryEvaluator, ScalarSparseSupportedDistance,
};

pub use core::dataset::{Dataset, DatasetGrowable, DenseData, MeanDataset, SparseData, VectorId};
pub use core::flat_index::FlatIndex;
pub use core::index::{Index, IndexStats};
pub use core::storage::{
    GrowableSparseStorage, ImmutableSparseStorage, SparseStorage, SparseStorageMut,
};
pub use datasets::dense_dataset::{DenseDataset, DenseDatasetGeneric, DenseDatasetGrowable};
#[cfg(feature = "multivec")]
pub use datasets::multivec_dataset::{
    MultiVecData, MultiVectorDataset, MultiVectorDatasetGeneric, MultiVectorDatasetGrowable,
};
pub use datasets::packed_dataset::{
    PackedSparseDataset, PackedSparseDatasetGeneric, PackedSparseDatasetGrowable,
};
pub use datasets::sparse_dataset::{SparseDataset, SparseDatasetGrowable};

// Useful type aliases for dense dataset types
pub type ScalarDenseDataset<VIn, VOut, D> = DenseDataset<ScalarDenseQuantizer<VIn, VOut, D>>;
pub type ScalarDenseDatasetGrowable<VIn, VOut, D> =
    DenseDatasetGrowable<ScalarDenseQuantizer<VIn, VOut, D>>;

pub type PlainDenseDataset<V, D> = ScalarDenseDataset<V, V, D>;
pub type PlainDenseDatasetGrowable<V, D> = ScalarDenseDatasetGrowable<V, V, D>;

/// RaBitQ-style binary dense dataset: rotated 1-bit sign codes with per-document factor/norm
/// metadata, scored with the RaBitQ estimator.
///
/// Defaults to inner-product scoring. Use [`RabitqDenseDatasetSquaredEuclidean`] for Euclidean
/// search; document codes are identical either way, so the metric is purely a scoring choice.
///
/// Built by converting a [`PlainDenseDataset`] — `(&plain).convert_into(config)` with a
/// [`RabitqConfig`]; stores `d/64` code words plus one metadata word per vector.
pub type RabitqDenseDataset = DenseDataset<RabitqQuantizer<DotProduct>>;

/// [`RabitqDenseDataset`] scored with the RaBitQ squared-Euclidean estimator.
pub type RabitqDenseDatasetSquaredEuclidean =
    DenseDataset<RabitqQuantizer<SquaredEuclideanDistance>>;

/// Extended RaBitQ dense dataset: rotated multi-bit document codes (a configurable
/// `total_bits ∈ {2, 4, 8}` per component, component-major) with per-document scan metadata.
/// For the 1-bit sign code use [`RabitqDenseDataset`].
///
/// Defaults to inner-product scoring. Use [`RabitqExtDenseDatasetSquaredEuclidean`] for Euclidean
/// search; document codes are identical either way, so the metric is purely a scoring choice.
///
/// Built by converting a [`PlainDenseDataset`] — `(&plain).convert_into(config)` with a
/// [`RabitqExtConfig`]; stores `total_bits · d/64` code words plus one metadata word per vector.
pub type RabitqExtDenseDataset = DenseDataset<RabitqExtQuantizer<DotProduct>>;

/// [`RabitqExtDenseDataset`] scored with the RaBitQ squared-Euclidean estimator.
pub type RabitqExtDenseDatasetSquaredEuclidean =
    DenseDataset<RabitqExtQuantizer<SquaredEuclideanDistance>>;

// Useful type aliases for sparse dataset types
pub type ScalarSparseDataset<C, W, V, D> = SparseDataset<ScalarSparseQuantizer<C, W, V, D>>;
pub type ScalarSparseDatasetGrowable<C, W, V, D> =
    SparseDatasetGrowable<ScalarSparseQuantizer<C, W, V, D>>;

pub type PlainSparseDataset<C, V, D> = SparseDataset<PlainSparseQuantizer<C, V, D>>;
pub type PlainSparseDatasetGrowable<C, V, D> = SparseDatasetGrowable<PlainSparseQuantizer<C, V, D>>;

pub use core::dataset::{ScoredRange, ScoredVector};

pub type ScoredVectorDotProduct = ScoredVector<DotProduct>;
pub type ScoredRangeDotProduct = ScoredRange<DotProduct>;
pub type ScoredVectorEuclidean = ScoredVector<SquaredEuclideanDistance>;
pub type ScoredRangeEuclidean = ScoredRange<SquaredEuclideanDistance>;

pub use datasets::readers;
pub use datasets::readers::{read_npy_f32, read_seismic_format};

/// Marker for types used as values in a dataset.
pub trait ValueType: Copy + Send + Sync + 'static + ToPrimitive + PartialOrd + Zero {}
impl<T> ValueType for T where T: Copy + Send + Sync + 'static + ToPrimitive + PartialOrd + Zero {}

/// Marker for types used as components in a dataset.
pub trait ComponentType: AsPrimitive<usize> + Copy + Send + Sync + 'static + Ord {}
impl<T> ComponentType for T where T: AsPrimitive<usize> + Copy + Send + Sync + 'static + Ord {}

/// Support trait for index types so they can save and load with a common interface.
#[derive(Debug)]
pub enum IndexIoError {
    Io(std::io::Error),
    Encode(bincode::error::EncodeError),
    Decode(bincode::error::DecodeError),
}

impl From<std::io::Error> for IndexIoError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<bincode::error::EncodeError> for IndexIoError {
    fn from(value: bincode::error::EncodeError) -> Self {
        Self::Encode(value)
    }
}

impl From<bincode::error::DecodeError> for IndexIoError {
    fn from(value: bincode::error::DecodeError) -> Self {
        Self::Decode(value)
    }
}

pub trait IndexSerializer: Sized {
    fn save_index(&self, filename: &str) -> Result<(), IndexIoError>
    where
        Self: Serialize,
    {
        let file = File::create(filename)?;
        let mut writer = BufWriter::new(file);
        let config = bincode::config::standard()
            .with_fixed_int_encoding()
            .with_little_endian();
        bincode::serde::encode_into_std_write(self, &mut writer, config)?;
        Ok(())
    }

    fn load_index(filename: &str) -> Result<Self, IndexIoError>
    where
        Self: DeserializeOwned,
    {
        let file = File::open(filename)?;
        let mut reader = BufReader::new(file);
        let config = bincode::config::standard()
            .with_fixed_int_encoding()
            .with_little_endian();
        let decoded = bincode::serde::decode_from_std_read::<Self, _, _>(&mut reader, config)?;
        Ok(decoded)
    }
}

impl<T> IndexSerializer for T where T: Dataset {}

#[cfg(test)]
mod with_distance_tests {
    use crate::core::dataset::ConvertFrom;
    use crate::{
        Dataset, DenseDataset, DenseVectorView, Distance, DotProduct, PlainDenseDataset,
        QueryEvaluator, RabitqConfig, RabitqExtConfig, RabitqExtQuantizer, RabitqQuantizer,
        RabitqQueryParams, SquaredEuclideanDistance, VectorEncoder, WithDistance,
    };

    fn data(n: usize, dim: usize) -> PlainDenseDataset<f32, DotProduct> {
        let values: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 7919 % 1000) as f32 / 100.0) - 5.0)
            .collect();
        PlainDenseDataset::from_raw(
            values.into_boxed_slice(),
            n,
            crate::ScalarDenseQuantizer::new(dim),
        )
    }

    /// Scoring dot-product-built codes under L2 through `with_distance` gives the distances of
    /// the same encoder built for L2 (RaBitQ codes are metric-independent and deterministic).
    #[test]
    fn rabitq_codes_score_under_the_other_metric() {
        let ds = data(50, 128);
        let q: Vec<f32> = (0..128).map(|i| (i % 13) as f32 / 3.0).collect();
        let q = DenseVectorView::new(&q);
        let qp = RabitqQueryParams { query_bits: 4 };

        let dp: DenseDataset<RabitqQuantizer<DotProduct>> =
            ConvertFrom::convert_from(&ds, RabitqConfig::default());
        let l2: DenseDataset<RabitqQuantizer<SquaredEuclideanDistance>> =
            ConvertFrom::convert_from(&ds, RabitqConfig::default());
        let relabelled = WithDistance::<SquaredEuclideanDistance>::with_distance(dp.encoder());
        let (a, b) = (
            relabelled.query_evaluator(q, &qp),
            l2.encoder().query_evaluator(q, &qp),
        );
        for i in 0..ds.len() as u64 {
            assert_eq!(a.compute_distance(dp.get(i)), b.compute_distance(l2.get(i)));
        }

        let config = RabitqExtConfig::default();
        let dp: DenseDataset<RabitqExtQuantizer<DotProduct>> =
            ConvertFrom::convert_from(&ds, config);
        let l2: DenseDataset<RabitqExtQuantizer<SquaredEuclideanDistance>> =
            ConvertFrom::convert_from(&ds, config);
        let relabelled = WithDistance::<SquaredEuclideanDistance>::with_distance(dp.encoder());
        let (a, b) = (
            relabelled.query_evaluator(q, &()),
            l2.encoder().query_evaluator(q, &()),
        );
        for i in 0..ds.len() as u64 {
            assert_eq!(a.compute_distance(dp.get(i)), b.compute_distance(l2.get(i)));
        }
    }

    /// Sparse scalar values scored under L2 through `with_distance` give the exact `‖q − v‖²`,
    /// for f32 and f16 storage (each against its decoded values).
    #[test]
    fn sparse_scalar_values_score_under_the_other_metric() {
        use crate::core::vector::SparseVectorView;
        use crate::{
            DatasetGrowable, PlainSparseDataset, PlainSparseDatasetGrowable, PlainSparseQuantizer,
            ScalarSparseDataset, core::vector_encoder::SparseDataEncoder,
        };
        let mut g = PlainSparseDatasetGrowable::<u16, f32, DotProduct>::new(
            PlainSparseQuantizer::new(16, 16),
        );
        g.push(SparseVectorView::new(
            &[0_u16, 3, 9],
            &[1.5_f32, -2.0, 0.25],
        ));
        g.push(SparseVectorView::new(
            &[1_u16, 3, 15],
            &[4.0_f32, 0.5, -1.0],
        ));
        g.push(SparseVectorView::new(&[7_u16], &[3.0_f32]));
        let ds: PlainSparseDataset<u16, f32, DotProduct> = g.into();
        let (qc, qv) = ([0_u16, 3, 7, 15], [2.0_f32, 1.0, -1.5, 0.5]);
        let q = SparseVectorView::new(&qc, &qv);
        let dense = |c: &[u16], v: &[f32]| {
            let mut d = vec![0.0f32; 16];
            for (&ci, &vi) in c.iter().zip(v) {
                d[ci as usize] = vi;
            }
            d
        };
        let qd = dense(&qc, &qv);

        let l2 = WithDistance::<SquaredEuclideanDistance>::with_distance(ds.encoder());
        let e = l2.query_evaluator(q, &());
        for i in 0..ds.len() as u64 {
            let v = ds.get(i);
            let vd = dense(v.components(), v.values());
            let exact: f32 = qd.iter().zip(&vd).map(|(a, b)| (a - b) * (a - b)).sum();
            assert_eq!(e.compute_distance(v).distance(), exact);
        }

        let ds16: ScalarSparseDataset<u16, f32, half::f16, DotProduct> =
            ConvertFrom::convert_from(&ds, ());
        let l2 = WithDistance::<SquaredEuclideanDistance>::with_distance(ds16.encoder());
        let e = l2.query_evaluator(q, &());
        for i in 0..ds16.len() as u64 {
            let decoded = ds16.encoder().decode_vector(ds16.get(i));
            let vd = dense(decoded.components(), decoded.values());
            let exact: f32 = qd.iter().zip(&vd).map(|(a, b)| (a - b) * (a - b)).sum();
            let got = e.compute_distance(ds16.get(i)).distance();
            assert!(
                (got - exact).abs() <= 1e-5 * exact.max(1.0),
                "{got} vs {exact}"
            );
        }
    }

    /// PQ codes scored under L2 through `with_distance` give `‖q − decode(code)‖²`.
    #[test]
    fn pq_codes_score_under_the_other_metric() {
        use crate::DenseVectorEncoder;
        use crate::encoders::pq::ProductQuantizer;
        let ds = data(600, 32);
        let q: Vec<f32> = (0..32).map(|i| (i % 5) as f32).collect();
        let dp: DenseDataset<ProductQuantizer<8, DotProduct>> = ConvertFrom::convert_from(&ds, ());
        let l2 = WithDistance::<SquaredEuclideanDistance>::with_distance(dp.encoder());
        let e = l2.query_evaluator(DenseVectorView::new(&q), &());
        for i in 0..ds.len() as u64 {
            let decoded = dp.encoder().decode_vector(dp.get(i));
            let exact: f32 = q
                .iter()
                .zip(decoded.values())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            let got = e.compute_distance(dp.get(i)).distance();
            assert!(
                (got - exact).abs() <= 1e-3 * exact.max(1.0),
                "{got} vs {exact}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vector::DenseVectorView;
    use crate::datasets::dense_dataset::DenseDatasetGrowable;
    use crate::distances::DotProduct;
    use crate::encoders::dense_scalar::ScalarDenseQuantizer;
    use crate::{DatasetGrowable, DenseDataset};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards")
            .as_nanos();
        let mut path = std::env::temp_dir();
        path.push(format!(
            "vectorium_{}_{}_{}.bin",
            name,
            std::process::id(),
            nanos
        ));
        path
    }

    #[test]
    fn index_serializer_round_trip_dense_dataset() {
        let encoder = ScalarDenseQuantizer::<f32, f32, DotProduct>::new(2);
        let mut growable = DenseDatasetGrowable::new(encoder);
        growable.push(DenseVectorView::new(&[1.0, 0.0]));
        growable.push(DenseVectorView::new(&[0.0, 1.0]));
        let dataset: DenseDataset<_> = growable.into();

        let path = temp_path("dense_round_trip");
        dataset.save_index(path.to_str().unwrap()).unwrap();
        let loaded = DenseDataset::<_>::load_index(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(dataset, loaded);
    }

    #[test]
    fn index_serializer_fails_on_corrupt_file() {
        let path = temp_path("corrupt");
        std::fs::write(&path, b"not a valid index").unwrap();

        let result = DenseDataset::<ScalarDenseQuantizer<f32, f32, DotProduct>>::load_index(
            path.to_str().unwrap(),
        );
        std::fs::remove_file(&path).unwrap();

        assert!(result.is_err());
    }
}
