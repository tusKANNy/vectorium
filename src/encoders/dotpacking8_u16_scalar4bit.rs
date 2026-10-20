//! `dotpacking8` component compression blended with 4-bit scalar value compression.
//!
//! This is the non-clustered counterpart of `CDotPacking8Scalar4BitEncoder` (not imported on
//! this branch yet): it keeps the component codec of [`crate::DotPacking8U16ScalarU8Encoder`]
//! verbatim (decreasing-frequency component permutation + a single variable-bit gap-coded stream over all components, with
//! a dense-query SIMD gather), but stores the per-component **values** as 4-bit nibbles
//! instead of full `u8` bytes, roughly halving the value footprint.
//!
//! Unlike the `cdotpacking8` fork there is **no** reference-list / mapped-residual tier —
//! every component lives in one gap stream, and the value stream is one nibble-packed block.
//! The 4-bit value math is identical: `quants[c] = max_value[c] / 15`,
//! `code = clamp(v / quants[c], 0, 15)`, scoring `Σ code · (v_query · quants[c])`.
//!
//! The nibble layout and the [`Scalar4BitQuantizer`] are shared with the `cdotpacking8`
//! fork via [`crate::encoders::dotpacking8::nibble4bit`]; only the single-stream component
//! skeleton lives here.
//!
//! ## Per-vector byte layout
//! ```text
//! [n u16][selectors][gap payloads...][pad][values: nibble-packed]  (nibble_bytes(n) bytes)
//! ```

use crate::encoders::dotpacking8::encoder::{DotPacking8Encoder, DotPacking8QueryEvaluator};
use crate::encoders::dotpacking8::values::Scalar4BitValues;

pub type DotPacking8U16Scalar4BitEncoder = DotPacking8Encoder<u16, Scalar4BitValues>;
pub type DotPacking8U16Scalar4BitQueryEvaluator<'e> =
    DotPacking8QueryEvaluator<'e, u16, Scalar4BitValues>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SquaredEuclideanDistance;
    use crate::core::distances::Distance;
    use crate::core::vector::{PackedVectorView, SparseVectorView};
    use crate::encoders::dotpacking8::values::ValueCodec;
    use crate::encoders::sparse_scalar::PlainSparseQuantizer;
    use crate::vector_encoder::SparseDataEncoder;
    use crate::{DatasetGrowable, PlainSparseDatasetGrowable, QueryEvaluator, VectorEncoder};
    use crate::{PackedSparseVectorEncoder, PlainSparseDataset};

    /// Training data scaled so test inputs (values ~0.5..4) map to a spread of non-trivial
    /// 4-bit codes: per-component max is ~4, so `quants[c] ≈ 4/15 ≈ 0.27`.
    fn build_training_data(
        dim: usize,
        components: &[u16],
    ) -> PlainSparseDataset<u16, f32, SquaredEuclideanDistance> {
        let quantizer = PlainSparseQuantizer::<u16, f32, SquaredEuclideanDistance>::new(dim, dim);
        let mut growable = PlainSparseDatasetGrowable::new(quantizer);
        let values0: Vec<f32> = components
            .iter()
            .enumerate()
            .map(|(idx, _)| 1.0 + (idx % 7) as f32 * 0.45)
            .collect();
        let values1: Vec<f32> = components
            .iter()
            .enumerate()
            .map(|(idx, _)| 2.0 + (idx % 5) as f32 * 0.4)
            .collect();
        growable.push(SparseVectorView::new(components, &values0));
        if components.len() > 1 {
            growable.push(SparseVectorView::new(components, &values1));
        }
        growable.into()
    }

    fn quantized_values(
        encoder: &DotPacking8U16Scalar4BitEncoder,
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
        let (mut i, mut j) = (0usize, 0usize);
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

    /// Encode `input`, score against `query`, and assert the SIMD score equals the
    /// brute-force dot over the encoder's own dequantized values.
    fn assert_parity(
        encoder: &DotPacking8U16Scalar4BitEncoder,
        comps: &[u16],
        in_vals: &[f32],
        query_comps: &[u16],
        query_vals: &[f32],
    ) {
        let input = SparseVectorView::new(comps, in_vals);
        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let query = SparseVectorView::new(query_comps, query_vals);
        let evaluator = encoder.query_evaluator(query, &());
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        let q_values = quantized_values(encoder, comps, in_vals);
        let expected = dot_with_query(comps, &q_values, query_comps, query_vals);

        let tol = 1e-4 * expected.abs().max(1.0);
        assert!(
            (dist.distance() - expected).abs() <= tol,
            "score {} != expected {} (n={})",
            dist.distance(),
            expected,
            comps.len()
        );
    }

    const NS: [usize; 12] = [1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 23, 40];

    #[test]
    fn parity_various_n() {
        const DIM: usize = 4000;
        let all: Vec<u16> = (0u16..3000).collect();
        let training = build_training_data(DIM, &all);
        let mut enc = DotPacking8U16Scalar4BitEncoder::new(DIM);
        enc.train(&training);

        for &n in &NS {
            let comps: Vec<u16> = (0..n as u16).map(|i| i * 3).collect();
            let in_vals: Vec<f32> = (0..n).map(|i| 0.5 + (i % 11) as f32 * 0.3).collect();
            let q_vals: Vec<f32> = (0..n).map(|i| 0.25 + (i % 7) as f32 * 0.2).collect();
            assert_parity(&enc, &comps, &in_vals, &comps, &q_vals);
        }
    }

    #[test]
    fn parity_query_partial_overlap() {
        const DIM: usize = 200;
        let all: Vec<u16> = (0u16..100).collect();
        let training = build_training_data(DIM, &all);
        let mut enc = DotPacking8U16Scalar4BitEncoder::new(DIM);
        enc.train(&training);

        let comps: Vec<u16> = vec![0, 4, 8, 24, 28, 36, 48, 53, 70, 90];
        let in_vals = [1.0, 3.0, 2.0, 3.5, 1.5, 2.0, 1.0, 2.0, 3.0, 2.5];
        let query_comps: Vec<u16> = vec![2, 4, 8, 24, 28, 70, 90];
        let query_vals = [0.5, 1.5, 2.5, 1.0, 2.0, 1.0, 2.0];
        assert_parity(&enc, &comps, &in_vals, &query_comps, &query_vals);
    }

    fn same_when_quantized(
        before_c: &[u16],
        before_v: &[f32],
        after: &SparseVectorView<u16, f32>,
        encoder: &DotPacking8U16Scalar4BitEncoder,
    ) {
        assert_eq!(before_c, after.components());
        for (c, (v1, v2)) in before_c
            .iter()
            .zip(before_v.iter().zip(after.values().iter()))
        {
            let encoded = encoder.values().encode(*c as usize, *v1);
            let quantized = encoder.values().decode(*c as usize, encoded);
            assert!(
                (quantized - v2).abs() < 1e-5,
                "component {c}: {quantized} != {v2}"
            );
        }
    }

    #[test]
    fn decode_roundtrip_various_n() {
        const DIM: usize = 4000;
        let all: Vec<u16> = (0u16..3000).collect();
        let training = build_training_data(DIM, &all);
        let mut enc = DotPacking8U16Scalar4BitEncoder::new(DIM);
        enc.train(&training);

        for &n in &NS {
            let comps: Vec<u16> = (0..n as u16).map(|i| i * 7).collect();
            let in_vals: Vec<f32> = (0..n).map(|i| 0.5 + (i % 13) as f32 * 0.28).collect();
            let input = SparseVectorView::new(&comps, &in_vals);
            let mut buffer = Vec::new();
            enc.push_encoded(input, &mut buffer);
            let decoded = enc.decode_vector(PackedVectorView::new(&buffer));
            same_when_quantized(&comps, &in_vals, &decoded.as_view(), &enc);
        }
    }
}
