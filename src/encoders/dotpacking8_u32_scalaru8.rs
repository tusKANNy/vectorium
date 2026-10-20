use crate::encoders::dotpacking8::encoder::{DotPacking8Encoder, DotPacking8QueryEvaluator};
use crate::encoders::dotpacking8::values::ScalarU8Values;

/// Bridged block-8 encoder for sparse vectors with `u32` components and scalar
/// quantized `u8` values.
///
/// Mirrors `OptimisticDotVByteScalarU8Encoder` (on `feature/quantization`) but emits the
/// block-8 packed format from [`crate::encoders::dotpacking8`] instead of the VByte format.
/// The block-8 encoding stores per-block gap bit-widths in a 4-bit selector field, so
/// individual gaps must fit in 16 bits (`b <= 16`). To support `u32` components, this encoder
/// injects synthetic zero-valued "bridge" coordinates whenever a gap (including the first,
/// relative to 0) would exceed `u16::MAX`, keeping the encoding on the fast SIMD path.
///
/// # Invariant: code 0 marks a bridge
///
/// A real coordinate never quantizes to code 0: every input value is assumed to be at least
/// its component's quantization step (`max / 255` over the training data), so its code is in
/// `1..=255`. Code 0 is therefore reserved for bridges, which score zero and are removed by
/// `decode_vector` by their code alone. This holds for learned sparse embeddings with
/// strictly positive values whose components all occur in the training data (checked on
/// SPLARE: no entry quantizes to 0); it is checked by a `debug_assert!` in `push_encoded`.
pub type BridgedDotPacking8U32ScalarU8Encoder = DotPacking8Encoder<u32, ScalarU8Values>;
pub type BridgedDotPacking8U32ScalarU8QueryEvaluator<'e> =
    DotPacking8QueryEvaluator<'e, u32, ScalarU8Values>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PlainSparseDatasetGrowable;
    use crate::SquaredEuclideanDistance;
    use crate::core::dataset::DatasetGrowable;
    use crate::core::vector::{PackedVectorView, SparseVectorView};
    use crate::vector_encoder::SparseDataEncoder;
    use crate::{PackedSparseVectorEncoder, PlainSparseDataset, QueryEvaluator, VectorEncoder};

    fn build_training_data(
        dim: usize,
        vectors: &[(&[u32], &[f32])],
    ) -> PlainSparseDataset<u32, f32, SquaredEuclideanDistance> {
        let encoder = crate::encoders::sparse_scalar::PlainSparseQuantizer::<
            u32,
            f32,
            SquaredEuclideanDistance,
        >::new(dim, dim);
        let mut growable = PlainSparseDatasetGrowable::new(encoder);
        for &(components, values) in vectors {
            growable.push(SparseVectorView::new(components, values));
        }
        growable.into()
    }

    #[test]
    fn new_creates_encoder() {
        let encoder = BridgedDotPacking8U32ScalarU8Encoder::new(100);
        assert_eq!(encoder.input_dim(), 100);
        assert_eq!(encoder.output_dim(), 100);
        assert!(encoder.component_mapping().is_none());
        assert!(encoder.inverse_component_mapping().is_none());
    }

    #[test]
    fn training_sets_component_mapping_and_quants() {
        let mut encoder = BridgedDotPacking8U32ScalarU8Encoder::new(5);
        let td = build_training_data(
            5,
            &[
                (
                    &[0_u32, 1, 2, 3, 4],
                    &[255.0_f32, 255.0, 255.0, 255.0, 255.0],
                ),
                (&[0_u32, 2], &[128.0_f32, 64.0]),
            ],
        );
        encoder.train(&td);

        let mapping = encoder.component_mapping().unwrap();
        let inverse = encoder.inverse_component_mapping().unwrap();
        assert_eq!(mapping.len(), 5);
        assert_eq!(inverse.len(), 5);
        for &v in mapping {
            assert!(v < 5);
        }
        for &v in inverse {
            assert!(v < 5);
        }
        assert_eq!(encoder.values().steps().len(), 5);
    }

    #[test]
    fn encode_decode_preserves_structure() {
        let mut encoder = BridgedDotPacking8U32ScalarU8Encoder::new(10);
        let td = build_training_data(
            10,
            &[
                (&[0_u32, 1, 2, 3], &[255.0_f32, 255.0, 255.0, 255.0]),
                (&[0_u32, 5], &[128.0_f32, 64.0]),
            ],
        );
        encoder.train(&td);

        let values = [1.0_f32, 2.0_f32];
        let input = SparseVectorView::new(&[0_u32, 5], &values);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);
        assert!(!buffer.is_empty());

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &[0_u32, 5]);
        assert_eq!(decoded.values().len(), 2);
        for (&decoded_v, &original_v) in decoded.values().iter().zip(values.iter()) {
            assert!(
                (decoded_v - original_v).abs() <= 1.01,
                "decoded {decoded_v} too far from original {original_v}"
            );
        }
    }

    /// dim=4, quants trained to 1.0 → encoded u8 == value, decoded == value.
    /// dot = 100*2 + 150*3 + 200*1 = 850.
    #[test]
    fn encode_decode_dot_product_exact() {
        let mut encoder = BridgedDotPacking8U32ScalarU8Encoder::new(4);
        let td = build_training_data(4, &[(&[0_u32, 1, 2, 3], &[255.0_f32, 255.0, 255.0, 255.0])]);
        encoder.train(&td);
        assert!(
            encoder
                .values()
                .steps()
                .iter()
                .all(|&q| (q - 1.0).abs() < 1e-6),
            "expected quants == 1.0 for all components"
        );

        let vec_values = [100.0_f32, 150.0, 200.0];
        let vec_input = SparseVectorView::new(&[0_u32, 1, 2], &vec_values);

        let mut buffer = Vec::new();
        encoder.push_encoded(vec_input, &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &[0_u32, 1, 2]);
        for (&decoded_v, &original_v) in decoded.values().iter().zip(vec_values.iter()) {
            assert!(
                (decoded_v - original_v).abs() < 1e-4,
                "decoded {decoded_v} != original {original_v}"
            );
        }

        let query_values = [2.0_f32, 3.0, 1.0];
        let query = SparseVectorView::new(&[0_u32, 1, 2], &query_values);
        let evaluator = encoder.query_evaluator(query, &());
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));
        assert!(
            (dist.0 - 850.0).abs() < 1e-2,
            "expected dot product 850.0, got {}",
            dist.0
        );
    }

    /// Exercise the bridging path: a `(0, u16::MAX + 100)` pair forces at least one
    /// synthetic zero bridge. The dot product must still match the analytical value
    /// because bridges contribute zero by construction.
    #[test]
    fn dot_product_with_bridged_large_gap() {
        let dim = (u16::MAX as usize) + 200;
        let mut encoder = BridgedDotPacking8U32ScalarU8Encoder::new(dim);
        let gap = u16::MAX as u32 + 100;
        let comps = [0_u32, gap];
        let td = build_training_data(dim, &[(&comps, &[255.0_f32, 255.0])]);
        encoder.train(&td);

        let values = [100.0_f32, 200.0];
        let input = SparseVectorView::new(&comps, &values);
        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        // Analytical: quants for both trained components are 1.0, so dot is exact.
        // dot = 100*2 + 200*5 = 200 + 1000 = 1200.
        let query_values = [2.0_f32, 5.0];
        let query = SparseVectorView::new(&comps, &query_values);
        let evaluator = encoder.query_evaluator(query, &());
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));
        assert!(
            (dist.0 - 1200.0).abs() < 1e-2,
            "expected dot product 1200.0, got {}",
            dist.0
        );
    }

    /// An encoder with no component mapping and a unit step for every component, so
    /// encoded codes equal the (integer) input values.
    fn unit_step_encoder(dim: usize) -> BridgedDotPacking8U32ScalarU8Encoder {
        BridgedDotPacking8U32ScalarU8Encoder::with_values(
            dim,
            ScalarU8Values::from_steps(vec![1.0_f32; dim].into_boxed_slice()),
        )
    }

    /// A vector whose smallest component exceeds `u16::MAX` must encode, decode to exactly
    /// its own components (no bridges), and score correctly.
    #[test]
    fn first_component_beyond_u16_roundtrip_and_dot() {
        let dim = 200_000;
        let encoder = unit_step_encoder(dim);
        let comps = [70_000_u32, 70_010, 140_000, 199_999];
        let values = [10.0_f32, 20.0, 30.0, 40.0];
        let mut buffer = Vec::new();
        encoder.push_encoded(SparseVectorView::new(&comps, &values), &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &comps);
        assert_eq!(decoded.values(), &values);

        let query_values = [1.0_f32, 2.0, 3.0, 4.0];
        let evaluator = encoder.query_evaluator(SparseVectorView::new(&comps, &query_values), &());
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));
        assert!(
            (dist.0 - 300.0).abs() < 1e-3,
            "expected 300, got {}",
            dist.0
        );
    }

    /// Bridges are an encoding artifact: decoding returns only the input components, also
    /// across a bulk block (> 8 entries after bridging) and with several bridges in a row.
    #[test]
    fn decode_drops_bridges() {
        let dim = 400_000;
        let encoder = unit_step_encoder(dim);
        let comps: Vec<u32> = vec![0, 3, 200_000, 200_001, 200_002, 200_003, 200_004, 399_999];
        let values: Vec<f32> = (1..=comps.len()).map(|i| i as f32).collect();
        let mut buffer = Vec::new();
        encoder.push_encoded(SparseVectorView::new(&comps, &values), &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), comps.as_slice());
        assert_eq!(decoded.values(), values.as_slice());
    }

    /// A genuine component exactly `u16::MAX` after the previous one needs no bridge and must
    /// be kept by the decoder.
    #[test]
    fn decode_keeps_real_component_at_max_gap() {
        let dim = 200_000;
        let encoder = unit_step_encoder(dim);
        let comps = [5_u32, 5 + u16::MAX as u32];
        let values = [3.0_f32, 4.0];
        let mut buffer = Vec::new();
        encoder.push_encoded(SparseVectorView::new(&comps, &values), &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &comps);
        assert_eq!(decoded.values(), &values);
    }

    /// Vector length not a multiple of 8 → exercises `decode_tail`.
    #[test]
    fn bulk_and_tail_blocks() {
        let dim = 256;
        let mut encoder = BridgedDotPacking8U32ScalarU8Encoder::new(dim);
        let comps: Vec<u32> = (0_u32..10).map(|i| i * 5).collect();
        let train_values: Vec<f32> = (0..10).map(|_| 255.0).collect();
        let td = build_training_data(dim, &[(&comps, &train_values)]);
        encoder.train(&td);

        let values: Vec<f32> = (0..10).map(|i| 10.0 + i as f32 * 7.0).collect();
        let input = SparseVectorView::new(&comps, &values);
        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let query_values: Vec<f32> = (0..10).map(|i| 1.0 + i as f32 * 0.5).collect();
        let query = SparseVectorView::new(&comps, &query_values);
        let evaluator = encoder.query_evaluator(query, &());
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        // Reference: dot product over the (quantized) values.
        let mut expected = 0.0_f32;
        for (i, &c) in comps.iter().enumerate() {
            let q = encoder.values().steps()[c as usize];
            let encoded_u8 = if q > 0.0 {
                (values[i] / q).clamp(0.0, 255.0) as u8
            } else {
                0u8
            };
            let decoded_v = encoded_u8 as f32 * q;
            expected += decoded_v * query_values[i];
        }
        assert!(
            (dist.0 - expected).abs() < 1e-2,
            "dot {} != expected {}",
            dist.0,
            expected
        );
    }
}
