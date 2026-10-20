//! Characterization tests: freeze the encoded bytes, scores and decoded vectors of every
//! DotPacking8 encoder. Any refactor of the dotpacking8 module must keep these hashes.
#![cfg(test)]

use crate::core::vector::{PackedVectorView, SparseVectorView};
use crate::vector_encoder::SparseDataEncoder;
use crate::{
    BridgedDotPacking8U32ScalarU8Encoder, BridgedDotPacking8U32ScalarU16Encoder, DatasetGrowable,
    DotPacking8U16F16Encoder, DotPacking8U16FixedU8Encoder, DotPacking8U16Scalar4BitEncoder,
    DotPacking8U16ScalarU8Encoder, DotPacking8U16ScalarU16Encoder, DotProduct,
    PackedSparseVectorEncoder, PlainSparseDataset, PlainSparseDatasetGrowable,
    PlainSparseQuantizer, QueryEvaluator, VectorEncoder,
};

/// Deterministic xorshift64*, so the data never depends on `rand` versions.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// FNV-1a over 64-bit words.
fn fnv(words: impl IntoIterator<Item = u64>) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for w in words {
        for b in w.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// Vectors of lengths 0, 1, 7, 8, 9, 15, 16, 17, 64, 130 (and repeats), components sorted and
/// unique in `0..dim`, values in (0.05, 3.9] so FixedU8 never saturates.
fn vectors(dim: u64, seed: u64) -> Vec<(Vec<u32>, Vec<f32>)> {
    let mut rng = Rng(seed);
    let lens = [
        0usize, 1, 7, 8, 9, 15, 16, 17, 64, 130, 3, 40, 8, 1, 33, 100,
    ];
    lens.iter()
        .map(|&len| {
            let mut c: Vec<u32> = Vec::new();
            while c.len() < len {
                // Skewed: half the draws hit the first 1% of ids, so frequency ordering matters.
                let x = if rng.below(2) == 0 {
                    rng.below(dim / 100 + 1)
                } else {
                    rng.below(dim)
                };
                c.push(x as u32);
                c.sort_unstable();
                c.dedup();
            }
            let v = c
                .iter()
                .map(|_| 0.05 + (rng.below(1000) as f32) * 0.00385)
                .collect();
            (c, v)
        })
        .collect()
}

macro_rules! golden_case {
    ($name:ident, $Enc:ty, $C:ty, $dim:expr, $uniform_training:expr, $bridged:expr, $expect_words:expr, $expect_scores:expr, $expect_decoded:expr) => {
        #[test]
        fn $name() {
            let dim: u64 = $dim;
            let docs = vectors(dim, 0x9E3779B97F4A7C15);
            let queries = vectors(dim, 0xD1B54A32D192ED03);
            let mut growable =
                PlainSparseDatasetGrowable::new(PlainSparseQuantizer::<$C, f32, DotProduct>::new(
                    dim as usize,
                    dim as usize,
                ));
            for (c, v) in &docs {
                let c: Vec<$C> = c.iter().map(|&x| x as $C).collect();
                growable.push(SparseVectorView::new(&c, v));
            }
            let plain: PlainSparseDataset<$C, f32, DotProduct> = growable.into();

            let mut enc = <$Enc>::new(dim as usize);
            if $uniform_training {
                // Every component once, value 3.9: the frequency order degenerates to "decreasing
                // id" (a reversal), so the docs keep their large gaps and need bridges, and every
                // step is 3.9/255 (codes >= 3 for values >= 0.05: no code 0).
                let mut g = PlainSparseDatasetGrowable::new(PlainSparseQuantizer::<
                    $C,
                    f32,
                    DotProduct,
                >::new(
                    dim as usize, dim as usize
                ));
                for start in (0..dim).step_by(1000) {
                    let c: Vec<$C> = (start..(start + 1000).min(dim)).map(|x| x as $C).collect();
                    let v = vec![3.9f32; c.len()];
                    g.push(SparseVectorView::new(&c, &v));
                }
                let uniform: PlainSparseDataset<$C, f32, DotProduct> = g.into();
                enc.train(&uniform);
            } else {
                enc.train(&plain);
            }

            let mut words: Vec<u64> = Vec::new();
            let mut blobs: Vec<Vec<u64>> = Vec::new();
            for (c, v) in &docs {
                let c: Vec<$C> = c.iter().map(|&x| x as $C).collect();
                let mut blob = Vec::new();
                enc.push_encoded(SparseVectorView::new(&c, v), &mut blob);
                words.extend_from_slice(&blob);
                words.push(u64::MAX); // separator, so empty vectors are visible
                blobs.push(blob);
            }

            // Vectors whose stored entry count (header `n`) exceeds their input length carry
            // bridges: the u32 case must exercise them, the u16 cases never have any.
            let bridged = blobs
                .iter()
                .zip(&docs)
                .filter(|(b, (c, _))| !b.is_empty() && (b[0] & 0xFFFF) as usize > c.len())
                .count();
            assert_eq!(bridged > 0, $bridged, "{} vectors carry bridges", bridged);

            // Empty vectors encode to zero words, and today's views index `bytes[0..2]`, so
            // scoring or decoding them panics (fixed in Task 5). Skip them here so the golden
            // values describe the current behavior on every non-empty vector.
            let mut scores: Vec<u64> = Vec::new();
            let mut decoded: Vec<u64> = Vec::new();
            for (qc, qv) in &queries {
                let qc: Vec<$C> = qc.iter().map(|&x| x as $C).collect();
                let eval = enc.query_evaluator(SparseVectorView::new(&qc, qv), &());
                for blob in blobs.iter().filter(|b| !b.is_empty()) {
                    scores.push(
                        eval.compute_distance(PackedVectorView::new(blob))
                            .0
                            .to_bits() as u64,
                    );
                }
            }
            for blob in blobs.iter().filter(|b| !b.is_empty()) {
                let d = enc.decode_vector(PackedVectorView::new(blob));
                decoded.extend(d.components().iter().map(|&c| c as u64));
                decoded.extend(d.values().iter().map(|v| v.to_bits() as u64));
                decoded.push(u64::MAX);
            }

            let got = (fnv(words), fnv(scores), fnv(decoded));
            println!(
                "{}: {:#018x}, {:#018x}, {:#018x}",
                stringify!($name),
                got.0,
                got.1,
                got.2
            );
            assert_eq!(got, ($expect_words, $expect_scores, $expect_decoded));
        }
    };
}

golden_case!(
    golden_u16_fixedu8,
    DotPacking8U16FixedU8Encoder,
    u16,
    30_000,
    false,
    false,
    0x2ccb388a976bdd22,
    0x0a9877ab2b229c3c,
    0x6c63d2b00472be0c
);
golden_case!(
    golden_u16_scalaru8,
    DotPacking8U16ScalarU8Encoder,
    u16,
    30_000,
    false,
    false,
    0x9f95210df2984c29,
    0xfcf83fa5b1bc00f0,
    0xccdf248c43ed7e3e
);
golden_case!(
    golden_u16_f16,
    DotPacking8U16F16Encoder,
    u16,
    30_000,
    false,
    false,
    0xb9a954a294c01605,
    0x66cb248bdc844c6d,
    0xa9542ca05e310ad5
);
golden_case!(
    golden_u16_scalar4bit,
    DotPacking8U16Scalar4BitEncoder,
    u16,
    30_000,
    false,
    false,
    0x55ccbc2ee021782a,
    0x14ba300fd4243399,
    0xb390822da2c0ed98
);
golden_case!(
    golden_u32_scalaru8_bridged,
    BridgedDotPacking8U32ScalarU8Encoder,
    u32,
    200_000,
    true,
    true,
    0x0cf5fe2aa6265bab,
    0xad451a79b24c1694,
    0xdde6d036a236261e
);
golden_case!(
    golden_u32_scalaru8_frequency,
    BridgedDotPacking8U32ScalarU8Encoder,
    u32,
    200_000,
    false,
    false,
    0xfd908050043d8260,
    0x4986e0fb155918f2,
    0xe521a1c4faf4c1a5
);
golden_case!(
    golden_u16_scalaru16,
    DotPacking8U16ScalarU16Encoder,
    u16,
    30_000,
    false,
    false,
    0x26ca32c3e4c97420,
    0x17186cc9bc9ecee3,
    0x7c871b8c444782c8
);
golden_case!(
    golden_u32_scalaru16_bridged,
    BridgedDotPacking8U32ScalarU16Encoder,
    u32,
    200_000,
    true,
    true,
    0x97cab9ef7885d719,
    0xbd3713f57d908b86,
    0x32e5277fc80d32f7
);
