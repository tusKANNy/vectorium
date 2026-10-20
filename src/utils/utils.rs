use std::collections::HashSet;
use std::hash::Hash;
use std::simd::Simd;

use crate::{ComponentType, Dataset, PlainSparseDataset, ScalarSparseSupportedDistance, ValueType};

/// Prefetch the first `max_bytes` of `data` (all of it if shorter), one cache line at a time.
///
/// The right `max_bytes` depends on the access pattern, so callers choose it.
#[inline]
pub fn prefetch_read_slice<T>(data: &[T], max_bytes: usize) {
    let ptr = data.as_ptr() as *const u8;

    // Cache line size on x86 and most ARM CPUs is 64 bytes.
    // This is only a reasonable heuristic, not a strict guarantee.
    const CACHE_LINE: usize = 64;

    let len = core::mem::size_of_val(data);

    // Looping with pointer arithmetic improves unrolling and avoids bounds checks.
    let len_to_prefetch = len.min(max_bytes);
    let mut i = 0;
    while i < len_to_prefetch {
        unsafe {
            // locality = 1: data will be used soon, but is not extremely hot.
            core::intrinsics::prefetch_read_data::<u8, 1>(ptr.add(i));
        }
        i += CACHE_LINE;
    }
}

#[inline]
pub fn is_strictly_sorted<T: Ord>(slice: &[T]) -> bool {
    slice.windows(2).all(|w| w[0] < w[1])
}

/// An [`Extend`] sink that fills a fixed, pre-allocated slice instead of growing a buffer.
///
/// The encoders' `push_encoded` primitive is written against `Extend` so it can append into
/// dataset-owned storage. This adapter extends that to callers that already own the exact
/// destination — one row of a dataset's backing slab — letting them encode in place with no
/// intermediate allocation and no reassembly copy.
///
/// Both failure modes are checked, in release as well as debug, because both corrupt data
/// silently: overrunning the slice would write into the *next* record, and under-filling it would
/// leave whatever the destination held before (zeros, for a freshly allocated slab) inside an
/// otherwise valid-looking encoded vector. Use [`Self::finish`] to assert the record is complete —
/// one comparison per vector, against an encode that touches `d` values.
pub struct SliceSink<'a, T> {
    out: &'a mut [T],
    written: usize,
}

impl<'a, T> SliceSink<'a, T> {
    #[inline]
    pub fn new(out: &'a mut [T]) -> Self {
        Self { out, written: 0 }
    }

    /// How many items have been written so far.
    #[inline]
    pub fn written(&self) -> usize {
        self.written
    }

    /// Assert that exactly the whole slice was filled, panicking otherwise.
    ///
    /// Call this once the producer is done: a short write leaves stale values in the tail of the
    /// destination, which no later check can distinguish from encoder output.
    #[inline]
    pub fn finish(self) {
        assert_eq!(
            self.written,
            self.out.len(),
            "SliceSink: producer wrote {} of {} values; the rest of the destination would keep \
             stale contents",
            self.written,
            self.out.len()
        );
    }
}

impl<T> Extend<T> for SliceSink<'_, T> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for value in iter {
            assert!(
                self.written < self.out.len(),
                "SliceSink: producer overran its {}-value destination",
                self.out.len()
            );
            self.out[self.written] = value;
            self.written += 1;
        }
    }
}

/// Computes the size of the intersection of two unsorted lists of integers.
pub fn intersection<T: Eq + Hash + Clone>(s: &[T], groundtruth: &[T]) -> usize {
    let s_set: HashSet<_> = s.iter().cloned().collect();
    let mut size = 0;
    for v in groundtruth {
        if s_set.contains(v) {
            size += 1;
        }
    }
    size
}

/// Compute a permutation of components ordered by decreasing document frequency: the component
/// that occurs in the most vectors gets the new id 0. Ties are broken by decreasing original id.
///
/// Gap-coded encoders benefit because frequent components end up packed together at low ids, so
/// each vector starts with small gaps and its few rare components come last. On MS MARCO
/// (cocondenser, splade_v3, inference_less) this is within 0.9% of the best ordering measured
/// (reversed graph bisection) and needs only one pass over the data.
///
/// Returns `permutation` with `permutation[original_id] = new_id`.
pub(crate) fn permute_components_by_frequency<C, Item>(
    dim: usize,
    vectors: impl Iterator<Item = Item>,
) -> Box<[usize]>
where
    C: ComponentType,
    Item: AsRef<[C]>,
{
    let mut frequency = vec![0usize; dim];
    for vector in vectors {
        for &component_id in vector.as_ref().iter() {
            let component_idx: usize = component_id.as_();
            frequency[component_idx] += 1;
        }
    }

    let mut order: Vec<usize> = (0..dim).collect();
    order.sort_unstable_by_key(|&c| (std::cmp::Reverse(frequency[c]), std::cmp::Reverse(c)));

    let mut permutation = vec![0usize; dim];
    for (new_id, &original_id) in order.iter().enumerate() {
        permutation[original_id] = new_id;
    }

    permutation.into_boxed_slice()
}

/// Train per-component scalar quantization steps from percentile bounds.
///
/// Returns one step per component of `training_data`'s output dimension: the value `v` of a
/// component `c` encodes to `(v / quants[c]).clamp(0, 255)`. A step of `0.0` marks a component
/// that never occurs in the training data.
///
/// `lower_percentile` must be in `[0.0, 1.0)` and `upper_percentile` in
/// `(lower_percentile, 1.0]`.
///
/// Ported from branch feature/quantization (135e42e, extended in 72c2395), originally by
/// CosimoRulli.
pub fn train_sparse_scalar_quantizer<C, D>(
    training_data: &PlainSparseDataset<C, f32, D>,
    lower_percentile: f32,
    upper_percentile: f32,
) -> Vec<f32>
where
    C: ComponentType,
    D: ScalarSparseSupportedDistance,
{
    train_sparse_scalar_quantizer_with_levels(
        training_data,
        lower_percentile,
        upper_percentile,
        255.0,
    )
}

/// Same as [`train_sparse_scalar_quantizer`] with `num_levels` quantization levels instead of
/// 255: the step of component `c` is its (percentile-clipped) span divided by `num_levels`,
/// so values encode to `(v / quants[c]).clamp(0, num_levels)` (e.g. 15 for 4-bit codes).
///
/// `num_levels` must be positive.
///
/// Ported from branch feature/quantization (80064a9), originally by CosimoRulli.
pub fn train_sparse_scalar_quantizer_with_levels<C, D>(
    training_data: &PlainSparseDataset<C, f32, D>,
    lower_percentile: f32,
    upper_percentile: f32,
    num_levels: f32,
) -> Vec<f32>
where
    C: ComponentType,
    D: ScalarSparseSupportedDistance,
{
    sparse_scalar_quantizer_steps(
        training_data.output_dim(),
        training_data.iter().map(|v| (v.components(), v.values())),
        lower_percentile,
        upper_percentile,
        num_levels,
    )
}

/// Core of [`train_sparse_scalar_quantizer_with_levels`] over any sparse vectors, given as
/// `(components, values)` pairs; values of any [`ValueType`] are widened to `f32`. One pass.
pub(crate) fn sparse_scalar_quantizer_steps<'a, C, W>(
    dim: usize,
    vectors: impl Iterator<Item = (&'a [C], &'a [W])>,
    lower_percentile: f32,
    upper_percentile: f32,
    num_levels: f32,
) -> Vec<f32>
where
    C: ComponentType,
    W: ValueType,
{
    assert!(
        (0.0..1.0).contains(&lower_percentile),
        "lower_percentile must be in [0.0, 1.0), got {lower_percentile}"
    );
    assert!(
        (0.0..=1.0).contains(&upper_percentile) && upper_percentile > lower_percentile,
        "upper_percentile must be in (lower_percentile, 1.0], got {upper_percentile}"
    );
    assert!(
        num_levels > 0.0,
        "num_levels must be positive, got {num_levels}"
    );

    // Fast path: classic min-max quantization. Sorting every per-component bucket
    // is wasteful when we only need the maximum, and on high-dim sparse datasets
    // (e.g. SPLADE on MS MARCO) that sort dominates training time. Single pass,
    // no per-component allocations.
    if lower_percentile == 0.0 && upper_percentile >= 1.0 {
        let mut maxes = vec![0.0f32; dim];
        for (components, values) in vectors {
            for (&c, v) in components.iter().zip(values) {
                let v = v.to_f32().unwrap();
                let idx: usize = c.as_();
                if v > maxes[idx] {
                    maxes[idx] = v;
                }
            }
        }
        for q in maxes.iter_mut() {
            if *q > 0.0 {
                *q /= num_levels;
            }
        }
        return maxes;
    }

    // Collect per-component values
    let mut per_component: Vec<Vec<f32>> = vec![Vec::new(); dim];
    for (components, values) in vectors {
        for (&c, v) in components.iter().zip(values) {
            let idx: usize = c.as_();
            per_component[idx].push(v.to_f32().unwrap());
        }
    }

    let mut quants = vec![0.0f32; dim];

    for i in 0..dim {
        let vals = &mut per_component[i];
        if vals.is_empty() {
            continue;
        }
        vals.sort_unstable_by(|a, b| a.total_cmp(b));

        let min = if lower_percentile <= 0.0 {
            vals[0]
        } else {
            let idx = ((vals.len() as f32) * lower_percentile) as usize;
            let idx = idx.min(vals.len() - 1);
            vals[idx]
        };

        let max = if upper_percentile >= 1.0 {
            *vals.last().unwrap()
        } else {
            let idx = ((vals.len() as f32) * upper_percentile) as usize;
            let idx = idx.min(vals.len() - 1);
            vals[idx]
        };

        // This quantizer stores only a per-component step (no per-component offset),
        // so we fit the step to the robust span [min, max].
        let span = (max - min).max(0.0);
        if span > 0.0 {
            quants[i] = span / num_levels;
        } else if max > 0.0 {
            // Degenerate bucket (or identical percentiles): keep previous behavior.
            quants[i] = max / num_levels;
        }
    }

    quants
}

/// Load 16 bytes from `ptr` and broadcast them into both halves of a 256-bit vector.
///
/// # Safety
///
/// `ptr` must be valid for reading 16 bytes. The load is unaligned, so no alignment is needed.
///
/// Ported from branch feature/quantization (7e5f5a0), originally by luca.rizzo.
#[inline(always)]
pub unsafe fn load128_and_broadcast_to_256(ptr: *const u8) -> Simd<u8, 32> {
    use std::simd::simd_swizzle;
    const BROADCAST_MASK: [usize; 32] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        12, 13, 14, 15,
    ];
    let raw = unsafe { ptr.cast::<Simd<u8, 16>>().read_unaligned() };
    simd_swizzle!(raw, BROADCAST_MASK)
}

#[cfg(test)]
mod tests {
    use super::{
        SliceSink, intersection, is_strictly_sorted, permute_components_by_frequency,
        train_sparse_scalar_quantizer,
    };
    use crate::core::dataset::DatasetGrowable;
    use crate::core::vector::SparseVectorView;
    use crate::{
        PlainSparseDataset, PlainSparseDatasetGrowable, PlainSparseQuantizer,
        SquaredEuclideanDistance,
    };

    fn build_1d_training_data(
        values: &[f32],
    ) -> PlainSparseDataset<u16, f32, SquaredEuclideanDistance> {
        let q = PlainSparseQuantizer::<u16, f32, SquaredEuclideanDistance>::new(1, 1);
        let mut g = PlainSparseDatasetGrowable::new(q);
        for &v in values {
            g.push(SparseVectorView::new(&[0_u16], &[v]));
        }
        g.into()
    }

    #[test]
    fn lower_percentile_affects_trained_quant_step() {
        let td = build_1d_training_data(&[1.0, 2.0, 3.0, 4.0]);

        let q0 = train_sparse_scalar_quantizer(&td, 0.0, 1.0);
        let q50 = train_sparse_scalar_quantizer(&td, 0.5, 1.0);
        assert!(
            q50[0] < q0[0],
            "expected lower_percentile=0.5 to reduce quant step: q0={} q50={}",
            q0[0],
            q50[0]
        );
    }

    #[test]
    fn permute_components_by_frequency_returns_permutation() {
        let vectors_components = vec![vec![0_u16, 2], vec![1_u16, 3]];

        let perm = permute_components_by_frequency(4, vectors_components.into_iter());
        assert_eq!(perm.len(), 4);

        let mut seen = vec![false; 4];
        for &p in perm.iter() {
            assert!(p < 4);
            assert!(!seen[p], "permutation contains duplicate value");
            seen[p] = true;
        }
        assert!(seen.into_iter().all(|v| v));
    }

    #[test]
    fn permute_components_by_frequency_puts_frequent_components_first() {
        // frequencies: c0 = 1, c1 = 3, c2 = 2, c3 = 0, c4 = 2
        let vectors_components = vec![vec![0_u16, 1, 2], vec![1_u16, 4], vec![1_u16, 2, 4]];

        let perm = permute_components_by_frequency(5, vectors_components.into_iter());

        assert_eq!(perm[1], 0, "most frequent component gets id 0");
        // c2 and c4 tie at frequency 2: decreasing original id breaks the tie.
        assert_eq!(perm[4], 1);
        assert_eq!(perm[2], 2);
        assert_eq!(perm[0], 3);
        assert_eq!(perm[3], 4, "unused component goes last");
    }

    #[test]
    fn permute_components_by_frequency_handles_empty_input() {
        let perm = permute_components_by_frequency::<u16, Vec<u16>>(3, std::iter::empty());
        // all frequencies are 0, so the order is decreasing original id
        assert_eq!(&*perm, &[2, 1, 0]);
    }

    #[test]
    fn is_strictly_sorted_handles_duplicates() {
        assert!(is_strictly_sorted(&[1u32, 2, 3]));
        assert!(!is_strictly_sorted(&[1u32, 1, 2]));
    }

    #[test]
    fn intersection_counts_shared_elements() {
        let a = vec![1i32, 2, 3, 5];
        let b = vec![2i32, 3, 4];
        assert_eq!(intersection(&a, &b), 2);
    }

    #[test]
    fn slice_sink_fills_its_destination() {
        let mut out = [0u64; 4];
        let mut sink = SliceSink::new(&mut out);
        sink.extend([1u64, 2]);
        sink.extend([3u64, 4]);
        assert_eq!(sink.written(), 4);
        sink.finish();
        assert_eq!(out, [1, 2, 3, 4]);
    }

    #[test]
    #[should_panic(expected = "overran")]
    fn slice_sink_rejects_an_overrun() {
        // Must panic in release too: the fifth value would land in the *next* record.
        let mut out = [0u64; 4];
        let mut sink = SliceSink::new(&mut out);
        sink.extend([1u64, 2, 3, 4, 5]);
    }

    #[test]
    #[should_panic(expected = "wrote 2 of 4")]
    fn slice_sink_rejects_a_short_record() {
        // The untouched tail keeps whatever the destination held, which is indistinguishable from
        // real output once the record is stored — so `finish` has to catch it.
        let mut out = [0u64; 4];
        let mut sink = SliceSink::new(&mut out);
        sink.extend([1u64, 2]);
        sink.finish();
    }
}
