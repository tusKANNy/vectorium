//! Bridge coordinates: zero-coded entries inserted so that every gap, including the first
//! (relative to 0), fits in 16 bits. Only used by encoders with `C::MAY_NEED_BRIDGES`.

const MAX_GAP: u32 = u16::MAX as u32;

/// Returns the bridged copy, or `None` when no gap exceeds `u16::MAX` (no allocation).
pub(crate) fn insert_bridges<T: Copy>(
    components: &[u32],
    codes: &[T],
    zero: T,
) -> Option<(Vec<u32>, Vec<T>)> {
    let mut prev = 0u32;
    if components.iter().all(|&c| {
        let ok = c - prev <= MAX_GAP;
        prev = c;
        ok
    }) {
        return None;
    }
    let mut out_c = Vec::with_capacity(components.len() + 4);
    let mut out_v = Vec::with_capacity(codes.len() + 4);
    let mut last = 0u32;
    for (&c, &v) in components.iter().zip(codes) {
        while c - last > MAX_GAP {
            last += MAX_GAP;
            out_c.push(last);
            out_v.push(zero);
        }
        out_c.push(c);
        out_v.push(v);
        last = c;
    }
    Some((out_c, out_v))
}

/// Removes the entries whose code is `zero` (bridges; real values never have it, see
/// `Assumptions.md` #11).
pub(crate) fn drop_bridges<T: Copy + PartialEq>(
    components: &mut Vec<u32>,
    codes: &mut Vec<T>,
    zero: T,
) {
    let mut kept = 0;
    for i in 0..components.len() {
        if codes[i] != zero {
            components[kept] = components[i];
            codes[kept] = codes[i];
            kept += 1;
        }
    }
    components.truncate(kept);
    codes.truncate(kept);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_gap_needs_no_bridges() {
        assert!(insert_bridges(&[0u32, 1, 2, 3], &[10u8, 20, 30, 40], 0).is_none());
    }

    #[test]
    fn large_gap_is_bridged() {
        let gap = u16::MAX as u32 + 100;
        let (bc, bv) = insert_bridges(&[0u32, gap], &[10u8, 20], 0).unwrap();
        assert!(bc.len() > 2);
        assert_eq!(*bc.last().unwrap(), gap);
        assert_eq!(*bv.last().unwrap(), 20);
        for &v in &bv[1..bv.len() - 1] {
            assert_eq!(v, 0);
        }
        for window in bc.windows(2) {
            assert!(window[1] - window[0] <= u16::MAX as u32);
        }
    }

    #[test]
    fn first_component_beyond_u16_is_bridged() {
        // The first gap is relative to 0, so it must be bridged as well.
        let c = u16::MAX as u32 * 2 + 10;
        let (bc, bv) = insert_bridges(&[c], &[7u8], 0).unwrap();
        assert_eq!(bc, vec![u16::MAX as u32, 2 * u16::MAX as u32, c]);
        assert_eq!(bv, vec![0, 0, 7]);
    }

    #[test]
    fn drop_bridges_removes_zero_codes() {
        let mut c = vec![u16::MAX as u32, 70_000, 70_001];
        let mut v = vec![0u8, 5, 6];
        drop_bridges(&mut c, &mut v, 0);
        assert_eq!(c, vec![70_000, 70_001]);
        assert_eq!(v, vec![5, 6]);
    }
}
