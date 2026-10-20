use crate::encoders::dotpacking8::encoder::{DotPacking8Encoder, DotPacking8QueryEvaluator};
use crate::encoders::dotpacking8::values::ScalarU8Values;

pub type DotPacking8U16ScalarU8Encoder = DotPacking8Encoder<u16, ScalarU8Values>;
pub type DotPacking8U16ScalarU8QueryEvaluator<'e> =
    DotPacking8QueryEvaluator<'e, u16, ScalarU8Values>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::distances::Distance;
    use crate::core::vector::{PackedVectorView, SparseVectorView};
    use crate::encoders::dotpacking8::values::ValueCodec;
    use crate::encoders::sparse_scalar::PlainSparseQuantizer;
    use crate::vector_encoder::SparseDataEncoder;
    use crate::{
        DatasetGrowable, PackedSparseVectorEncoder, PlainSparseDatasetGrowable, QueryEvaluator,
        VectorEncoder,
    };
    use crate::{PlainSparseDataset, SquaredEuclideanDistance};

    fn build_training_data(
        dim: usize,
        components: &[u16],
    ) -> PlainSparseDataset<u16, f32, SquaredEuclideanDistance> {
        let quantizer = PlainSparseQuantizer::<u16, f32, SquaredEuclideanDistance>::new(dim, dim);
        let mut growable = PlainSparseDatasetGrowable::new(quantizer);
        let values0: Vec<f32> = components
            .iter()
            .enumerate()
            .map(|(idx, _)| 10.0 + (idx % 7) as f32 * 3.5)
            .collect();
        let values1: Vec<f32> = components
            .iter()
            .enumerate()
            .map(|(idx, _)| 25.0 + (idx % 5) as f32 * 2.25)
            .collect();
        growable.push(SparseVectorView::new(components, &values0));
        if components.len() > 1 {
            growable.push(SparseVectorView::new(components, &values1));
        }
        growable.into()
    }

    fn quantized_values(
        encoder: &DotPacking8U16ScalarU8Encoder,
        components: &[u16],
        values: &[f32],
    ) -> Vec<f32> {
        components
            .iter()
            .zip(values.iter())
            .map(|(&c, &v)| {
                let encoded = encoder.values().encode(c as usize, v);
                encoder.values().decode(c as usize, encoded)
            })
            .collect()
    }

    fn dot_with_query(
        components: &[u16],
        values: &[f32],
        query_components: &[u16],
        query_values: &[f32],
    ) -> f32 {
        let mut sum = 0.0f32;
        let mut i = 0usize;
        let mut j = 0usize;
        while i < components.len() && j < query_components.len() {
            match components[i].cmp(&query_components[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    sum += values[i] * query_values[j];
                    i += 1;
                    j += 1;
                }
            }
        }
        sum
    }

    #[test]
    fn compute_distance_only_bulk() {
        let comps = [2_u16, 4, 8, 24, 36, 48, 53, 90];
        let training = build_training_data(100, &comps);
        let mut encoder = DotPacking8U16ScalarU8Encoder::new(100);
        encoder.train(&training);

        let values = [1.0_f32, 3.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0];
        let input = SparseVectorView::new(&comps, &values);
        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let query_components = [2_u16, 4, 6, 8, 24, 70, 90];
        let query_values = [0.5_f32, 1.5, 2.5, 1.0, 2.0, 1.0, 2.0];
        let query = SparseVectorView::new(&query_components, &query_values);
        let evaluator = encoder.query_evaluator(query, &());
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        let q_values = quantized_values(&encoder, &comps, &values);
        let expected = dot_with_query(&comps, &q_values, &query_components, &query_values);
        assert!((dist.distance() - expected).abs() < 1e-5);
    }

    #[test]
    fn compute_distance_bulk_and_tail() {
        let comps = [2_u16, 4, 8, 24, 28, 36, 48, 53, 70, 90];
        let training = build_training_data(100, &comps);
        let mut encoder = DotPacking8U16ScalarU8Encoder::new(100);
        encoder.train(&training);

        let values = [1.0_f32, 3.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0];
        let input = SparseVectorView::new(&comps, &values);
        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let query_components = [2_u16, 4, 8, 24, 28, 70, 90];
        let query_values = [0.5_f32, 1.5, 2.5, 1.0, 2.0, 1.0, 2.0];
        let query = SparseVectorView::new(&query_components, &query_values);
        let evaluator = encoder.query_evaluator(query, &());
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        let q_values = quantized_values(&encoder, &comps, &values);
        let expected = dot_with_query(&comps, &q_values, &query_components, &query_values);
        assert!((dist.distance() - expected).abs() < 1e-5);
    }

    fn verify_gaps(gaps: &[u32]) {
        let num_vals = gaps.len();
        let mut components = Vec::new();
        let mut curr = 0u32;
        for &g in gaps {
            curr += g;
            components.push(curr as u16);
        }

        let training = build_training_data(u16::MAX as usize + 1, &components);
        let mut encoder = DotPacking8U16ScalarU8Encoder::new(u16::MAX as usize + 1);
        encoder.train(&training);

        let values: Vec<_> = (0..num_vals).map(|i| 1.0 + (i % 7) as f32).collect();
        let input = SparseVectorView::new(&components, &values);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let query_vals: Vec<_> = (0..num_vals).map(|i| 1.0 + (i % 5) as f32).collect();
        let query = SparseVectorView::new(&components, &query_vals);

        let evaluator = encoder.query_evaluator(query, &());
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        let q_values = quantized_values(&encoder, &components, &values);
        let mut expected = 0.0f32;
        for i in 0..num_vals {
            expected += q_values[i] * query_vals[i];
        }

        assert!(
            (dist.distance() - expected).abs() < 1e-3,
            "Failed: dist={}, expected={}, gaps_len={}",
            dist.distance(),
            expected,
            num_vals
        );
    }

    #[test]
    fn test_plain_multiple_blocks() {
        let gaps = vec![1u32; 32];
        verify_gaps(&gaps);
    }

    #[test]
    fn test_plain_small_gaps() {
        let gaps = vec![1u32; 1];
        verify_gaps(&gaps);
    }

    #[test]
    fn test_plain_with_tail() {
        let gaps = vec![1u32; 11];
        verify_gaps(&gaps);
    }

    #[test]
    fn test_plain_large_gaps() {
        let gaps = vec![1000u32, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000];
        verify_gaps(&gaps);
    }

    #[test]
    fn test_plain_u16_gaps() {
        let gaps = vec![60000, 1, 1, 2, 3, 200, 450, 2000, 3];
        verify_gaps(&gaps);
    }

    #[test]
    fn block8_decode_roundtrip() {
        let components = [0_u16, 4, 8, 24];
        let training = build_training_data(100, &components);
        let mut encoder = DotPacking8U16ScalarU8Encoder::new(100);
        encoder.train(&training);

        let values = [1.0_f32, 3.0, 2.0, 3.0];
        let input0 = SparseVectorView::new(&components, &values);
        let values1 = [3.0_f32, 2.0, 3.0];
        let input1 = SparseVectorView::new(&[4_u16, 8, 24], &values1);

        let mut buffer0 = Vec::new();
        encoder.push_encoded(input0, &mut buffer0);

        let mut buffer1 = Vec::new();
        encoder.push_encoded(input1, &mut buffer1);

        let decoded0 = encoder.decode_vector(PackedVectorView::new(&buffer0));
        let decoded1 = encoder.decode_vector(PackedVectorView::new(&buffer1));

        assert_eq!(decoded0.components(), &components);
        let decoded_vals0 = decoded0.values();
        let expected0 = quantized_values(&encoder, &components, &values);
        assert_eq!(decoded_vals0.len(), 4);
        assert!((decoded_vals0[0] - expected0[0]).abs() < 1e-6);
        assert!((decoded_vals0[1] - expected0[1]).abs() < 1e-6);
        assert!((decoded_vals0[2] - expected0[2]).abs() < 1e-6);
        assert!((decoded_vals0[3] - expected0[3]).abs() < 1e-6);

        assert_eq!(decoded1.components(), &[4_u16, 8, 24]);
        let decoded_vals1 = decoded1.values();
        let expected1 = quantized_values(&encoder, &[4_u16, 8, 24], &values1);
        assert_eq!(decoded_vals1.len(), 3);
        assert!((decoded_vals1[0] - expected1[0]).abs() < 1e-6);
        assert!((decoded_vals1[1] - expected1[1]).abs() < 1e-6);
        assert!((decoded_vals1[2] - expected1[2]).abs() < 1e-6);
    }
}
