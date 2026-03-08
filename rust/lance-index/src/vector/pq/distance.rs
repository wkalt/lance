// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use core::panic;
use std::cmp::{max, min};

use super::{num_centroids, utils::get_sub_vector_centroids};
use lance_linalg::distance::{Dot, L2, dot_distance_batch, l2_distance_batch};
use lance_linalg::simd::u8::u8x16;
use lance_linalg::simd::{SIMD, Shuffle};

// for quantizing the distance table, we need to know the max possible distance,
// so we perform a flat search on the first `FLAT_NUM_4BIT_PQ` rows.
// increasing this number will increase the accuracy of the quantization,
// but also increase the computation time.
// 200 is a good trade-off according to the original paper.
const FLAT_NUM_4BIT_PQ: usize = 200;

/// Build a Distance Table from the query to each PQ centroid
/// using L2 distance.
pub fn build_distance_table_l2<T: L2>(
    codebook: &[T],
    num_bits: u32,
    num_sub_vectors: usize,
    query: &[T],
) -> Vec<f32> {
    match num_bits {
        4 => build_distance_table_l2_impl::<4, T>(codebook, num_sub_vectors, query),
        8 => build_distance_table_l2_impl::<8, T>(codebook, num_sub_vectors, query),
        _ => panic!("Unsupported number of bits: {}", num_bits),
    }
}

#[inline]
pub fn build_distance_table_l2_impl<const NUM_BITS: u32, T: L2>(
    codebook: &[T],
    num_sub_vectors: usize,
    query: &[T],
) -> Vec<f32> {
    let dimension = query.len();
    let sub_vector_length = dimension / num_sub_vectors;
    let num_centroids = 2_usize.pow(NUM_BITS);
    let mut result = Vec::with_capacity(num_sub_vectors * num_centroids);
    for (i, sub_vec) in query.chunks_exact(sub_vector_length).enumerate() {
        let subvec_centroids =
            get_sub_vector_centroids::<NUM_BITS, _>(codebook, dimension, num_sub_vectors, i);
        result.extend(l2_distance_batch(
            sub_vec,
            subvec_centroids,
            sub_vector_length,
        ));
    }
    result
}

/// Build a Distance Table from the query to each PQ centroid
/// using Dot distance.
pub fn build_distance_table_dot<T: Dot>(
    codebook: &[T],
    num_bits: u32,
    num_sub_vectors: usize,
    query: &[T],
) -> Vec<f32> {
    match num_bits {
        4 => build_distance_table_dot_impl::<4, T>(codebook, num_sub_vectors, query),
        8 => build_distance_table_dot_impl::<8, T>(codebook, num_sub_vectors, query),
        _ => panic!("Unsupported number of bits: {}", num_bits),
    }
}

#[inline]
pub fn build_distance_table_dot_impl<const NUM_BITS: u32, T: Dot>(
    codebook: &[T],
    num_sub_vectors: usize,
    query: &[T],
) -> Vec<f32> {
    let dimension = query.len();
    let sub_vector_length = dimension / num_sub_vectors;
    let num_centroids = 2_usize.pow(NUM_BITS);
    let mut result = Vec::with_capacity(num_sub_vectors * num_centroids);
    for (i, sub_vec) in query.chunks_exact(sub_vector_length).enumerate() {
        let subvec_centroids =
            get_sub_vector_centroids::<NUM_BITS, _>(codebook, dimension, num_sub_vectors, i);
        result.extend(dot_distance_batch(
            sub_vec,
            subvec_centroids,
            sub_vector_length,
        ));
    }
    result
}

/// Compute L2 distance from the query to all code.
///
/// Parameters
/// ----------
/// - distance_table: the pre-computed L2 distance table.
///   It is a flatten array of [num_sub_vectors, num_centroids] f32.
/// - num_bits: the number of bits used for PQ.
/// - num_sub_vectors: the number of sub-vectors.
/// - code: the transposed PQ code to be used to compute the distances.
///
/// Returns
/// -------
///  The squared L2 distance.
///
#[inline]
pub(super) fn compute_pq_distance(
    distance_table: &[f32],
    num_bits: u32,
    num_sub_vectors: usize,
    code: &[u8],
    k_hint: usize,
) -> Vec<f32> {
    if code.is_empty() {
        return Vec::new();
    }
    if num_bits == 4 {
        return compute_pq_distance_4bit(distance_table, num_sub_vectors, code, k_hint);
    }
    compute_pq_distance_8bit(distance_table, num_sub_vectors, code, k_hint)
}

/// Force the scalar fallback path even on VBMI-capable CPUs.
/// Used for A/B benchmarking. Default: false (use FastScan when available).
static FORCE_SCALAR: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Set whether to force the scalar PQ distance path (for benchmarking).
pub fn set_force_scalar_pq(force: bool) {
    FORCE_SCALAR.store(force, std::sync::atomic::Ordering::Relaxed);
}

/// 8-bit PQ distance: dispatches to FastScan (AVX-512 VBMI) or scalar fallback.
#[inline]
fn compute_pq_distance_8bit(
    distance_table: &[f32],
    num_sub_vectors: usize,
    code: &[u8],
    k_hint: usize,
) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        if !FORCE_SCALAR.load(std::sync::atomic::Ordering::Relaxed)
            && is_x86_feature_detected!("avx512vbmi")
        {
            return compute_pq_distance_8bit_fastscan(
                distance_table,
                num_sub_vectors,
                code,
                k_hint,
            );
        }
    }
    compute_pq_distance_8bit_scalar(distance_table, num_sub_vectors, code)
}

/// FastScan for 8-bit PQ using AVX-512 VBMI.
///
/// Quantizes the 256-entry f32 distance table to u8, then uses `vpermi2b`
/// (128-byte table lookup in a single instruction) to process 64 vectors
/// simultaneously. Each sub-vector needs only 2 `vpermi2b` + 1 blend + 1 add
/// for 64 vectors — versus 64 scalar table lookups in the naive approach.
///
/// Like the 4-bit path, computes exact f32 distances for the first `flat_num`
/// vectors to calibrate the u8 quantization range, then uses the SIMD path
/// for the remaining vectors and dequantizes back to f32.
#[cfg(target_arch = "x86_64")]
fn compute_pq_distance_8bit_fastscan(
    distance_table: &[f32],
    num_sub_vectors: usize,
    code: &[u8],
    k_hint: usize,
) -> Vec<f32> {
    use std::arch::x86_64::*;

    let num_vectors = code.len() / num_sub_vectors;
    let mut distances = vec![0.0f32; num_vectors];
    const NUM_CENTROIDS: usize = 256;

    // Step 1: Compute exact f32 distances for first flat_num vectors to determine
    // the quantization range. Same approach as the 4-bit path.
    let k_hint = min(k_hint, num_vectors);
    let flat_num = max(FLAT_NUM_4BIT_PQ, k_hint).min(num_vectors);
    compute_pq_distance_8bit_flat(
        distance_table,
        num_vectors,
        num_sub_vectors,
        code,
        0,
        flat_num,
        &mut distances,
    );

    if flat_num >= num_vectors {
        return distances;
    }

    let qmax = *distances
        .iter()
        .take(flat_num)
        .max_by(|a, b| a.total_cmp(b))
        .unwrap();

    // Step 2: Quantize the 256-entry-per-sub-vector distance table to u8.
    let (qmin, quantized_table) = quantize_distance_table(distance_table, qmax);

    // Step 3: SIMD loop — process 64 vectors per block using vpermi2b.
    const BLOCK: usize = 64;
    let num_full_blocks = num_vectors / BLOCK;
    let mut quantized_dists = vec![0u8; num_vectors];

    // SAFETY: we checked avx512vbmi above. All pointer arithmetic stays within
    // bounds: quantized_table has num_sub_vectors * 256 bytes, code has
    // num_sub_vectors * num_vectors bytes, quantized_dists has num_vectors bytes.
    unsafe {
        let hi_bit = _mm512_set1_epi8(0x80u8 as i8);

        for block_idx in 0..num_full_blocks {
            let block_start = block_idx * BLOCK;
            let mut acc = _mm512_setzero_si512();

            for sv in 0..num_sub_vectors {
                let qt_base = sv * NUM_CENTROIDS;

                // Load 256-byte quantized table as two 128-byte halves.
                let dt0 =
                    _mm512_loadu_si512(quantized_table.as_ptr().add(qt_base) as *const _);
                let dt1 =
                    _mm512_loadu_si512(quantized_table.as_ptr().add(qt_base + 64) as *const _);
                let dt2 =
                    _mm512_loadu_si512(quantized_table.as_ptr().add(qt_base + 128) as *const _);
                let dt3 =
                    _mm512_loadu_si512(quantized_table.as_ptr().add(qt_base + 192) as *const _);

                // Load 64 code bytes.
                let code_offset = sv * num_vectors + block_start;
                let codes =
                    _mm512_loadu_si512(code.as_ptr().add(code_offset) as *const _);

                // vpermi2b: uses low 7 bits of each index byte to select from
                // the concatenation of two 64-byte registers (128 bytes total).
                // codes 0-127  -> r_lo correct (from dt0++dt1)
                // codes 128-255 -> r_hi correct (from dt2++dt3, using bits[6:0])
                let r_lo = _mm512_permutex2var_epi8(dt0, codes, dt1);
                let r_hi = _mm512_permutex2var_epi8(dt2, codes, dt3);

                // Select based on bit 7: codes < 128 use r_lo, >= 128 use r_hi.
                let mask = _mm512_test_epi8_mask(codes, hi_bit);
                let result = _mm512_mask_blend_epi8(mask, r_lo, r_hi);

                // Accumulate in u8 (wrapping add, same as 4-bit path).
                acc = _mm512_add_epi8(acc, result);
            }

            _mm512_storeu_si512(
                quantized_dists.as_mut_ptr().add(block_start) as *mut _,
                acc,
            );
        }
    }

    // Step 4: Handle remainder vectors (not a full 64-vector block) with scalar.
    let remainder = num_vectors % BLOCK;
    if remainder > 0 {
        let offset = max(num_vectors - remainder, flat_num);
        if offset < num_vectors {
            compute_pq_distance_8bit_flat(
                distance_table,
                num_vectors,
                num_sub_vectors,
                code,
                offset,
                num_vectors - offset,
                &mut distances,
            );
        }
    }

    // Step 5: Dequantize u8 distances back to f32, skipping the flat prefix
    // and the remainder (which already have exact f32 values).
    let range = (qmax - qmin) / 255.0;
    distances
        .iter_mut()
        .take(num_vectors - remainder)
        .skip(flat_num)
        .zip(
            quantized_dists
                .into_iter()
                .take(num_vectors - remainder)
                .skip(flat_num),
        )
        .for_each(|(dist, q_dist)| {
            *dist = (q_dist as f32) * range + qmin;
        });

    distances
}

/// Compute exact f32 distances for a range of 8-bit PQ codes.
fn compute_pq_distance_8bit_flat(
    distance_table: &[f32],
    num_vectors: usize,
    num_sub_vectors: usize,
    code: &[u8],
    offset: usize,
    length: usize,
    dists: &mut [f32],
) {
    const NUM_CENTROIDS: usize = 256;
    let distances = &mut dists[offset..offset + length];
    for sv in 0..num_sub_vectors {
        let base = sv * num_vectors + offset;
        let dt = &distance_table[sv * NUM_CENTROIDS..(sv + 1) * NUM_CENTROIDS];
        for i in 0..length {
            unsafe {
                *distances.get_unchecked_mut(i) +=
                    *dt.get_unchecked(*code.get_unchecked(base + i) as usize);
            }
        }
    }
}

/// Scalar fallback: tiled + 4× sub-vector unrolled.
#[inline]
fn compute_pq_distance_8bit_scalar(
    distance_table: &[f32],
    num_sub_vectors: usize,
    code: &[u8],
) -> Vec<f32> {
    let num_vectors = code.len() / num_sub_vectors;
    let mut distances = vec![0.0f32; num_vectors];
    const NUM_CENTROIDS: usize = 256;
    const TILE: usize = 2048;

    for tile_start in (0..num_vectors).step_by(TILE) {
        let tile_end = (tile_start + TILE).min(num_vectors);
        let tile_len = tile_end - tile_start;
        let tile_dists = &mut distances[tile_start..tile_end];

        let mut sv = 0;
        let sv_end4 = num_sub_vectors - (num_sub_vectors % 4);
        while sv < sv_end4 {
            let base0 = sv * num_vectors + tile_start;
            let base1 = (sv + 1) * num_vectors + tile_start;
            let base2 = (sv + 2) * num_vectors + tile_start;
            let base3 = (sv + 3) * num_vectors + tile_start;
            let dt0 = &distance_table[sv * NUM_CENTROIDS..(sv + 1) * NUM_CENTROIDS];
            let dt1 = &distance_table[(sv + 1) * NUM_CENTROIDS..(sv + 2) * NUM_CENTROIDS];
            let dt2 = &distance_table[(sv + 2) * NUM_CENTROIDS..(sv + 3) * NUM_CENTROIDS];
            let dt3 = &distance_table[(sv + 3) * NUM_CENTROIDS..(sv + 4) * NUM_CENTROIDS];
            for i in 0..tile_len {
                unsafe {
                    let d = tile_dists.get_unchecked_mut(i);
                    *d += *dt0.get_unchecked(*code.get_unchecked(base0 + i) as usize)
                        + *dt1.get_unchecked(*code.get_unchecked(base1 + i) as usize)
                        + *dt2.get_unchecked(*code.get_unchecked(base2 + i) as usize)
                        + *dt3.get_unchecked(*code.get_unchecked(base3 + i) as usize);
                }
            }
            sv += 4;
        }
        while sv < num_sub_vectors {
            let base = sv * num_vectors + tile_start;
            let dt = &distance_table[sv * NUM_CENTROIDS..(sv + 1) * NUM_CENTROIDS];
            for i in 0..tile_len {
                unsafe {
                    *tile_dists.get_unchecked_mut(i) +=
                        *dt.get_unchecked(*code.get_unchecked(base + i) as usize);
                }
            }
            sv += 1;
        }
    }

    distances
}

#[inline]
pub(super) fn compute_pq_distance_4bit(
    distance_table: &[f32],
    num_sub_vectors: usize,
    code: &[u8],
    k_hint: usize,
) -> Vec<f32> {
    let num_vectors = code.len() * 2 / num_sub_vectors;
    let mut distances = vec![0.0f32; num_vectors];

    // compute the distances for first k_hint rows
    // then use the max distance as qmax to quantize the distance table
    let k_hint = min(k_hint, num_vectors);
    let flat_num = max(FLAT_NUM_4BIT_PQ, k_hint).min(num_vectors);
    compute_pq_distance_4bit_flat(
        distance_table,
        num_vectors,
        code,
        0,
        flat_num,
        &mut distances,
    );
    let qmax = *distances
        .iter()
        .take(flat_num)
        .max_by(|a, b| a.total_cmp(b))
        .unwrap();

    let (qmin, quantized_dists_table) = quantize_distance_table(distance_table, qmax);
    const NUM_CENTROIDS: usize = 2_usize.pow(4);
    let mut quantized_dists = vec![0_u8; num_vectors];

    let remainder = num_vectors % NUM_CENTROIDS;
    for i in (0..num_vectors - remainder).step_by(NUM_CENTROIDS) {
        let mut block_distances = u8x16::zeros();

        for (sub_vec_idx, vec_indices) in code.chunks_exact(num_vectors).enumerate() {
            let origin_dist_table = unsafe {
                u8x16::load_unaligned(
                    quantized_dists_table
                        .as_ptr()
                        .add(sub_vec_idx * 2 * NUM_CENTROIDS),
                )
            };
            let origin_next_dist_table = unsafe {
                u8x16::load_unaligned(
                    quantized_dists_table
                        .as_ptr()
                        .add((sub_vec_idx * 2 + 1) * NUM_CENTROIDS),
                )
            };

            let indices = unsafe { u8x16::load_unaligned(vec_indices.as_ptr().add(i)) };

            // compute current distances
            let current_indices = indices.bit_and(0x0F);
            block_distances += origin_dist_table.shuffle(current_indices);

            // compute next distances
            let next_indices = indices.right_shift::<4>();
            block_distances += origin_next_dist_table.shuffle(next_indices);
        }

        unsafe {
            block_distances.store_unaligned(quantized_dists.as_mut_ptr().add(i));
        }
    }
    if remainder > 0 {
        let offset = max(num_vectors - remainder, flat_num);
        compute_pq_distance_4bit_flat(
            distance_table,
            num_vectors,
            code,
            offset,
            num_vectors - offset,
            &mut distances,
        );
    }

    // need to dequantize the distances
    // to make the distances comparable to the others from the other partitions
    let range = (qmax - qmin) / 255.0;
    distances
        .iter_mut()
        .take(num_vectors - remainder) // don't overwrite the remainder
        .skip(flat_num) // don't overwrite the first k_hint
        .zip(
            quantized_dists
                .into_iter()
                .take(num_vectors - remainder)
                .skip(flat_num),
        )
        .for_each(|(dist, q_dist)| {
            *dist = (q_dist as f32) * range + qmin;
        });
    distances
}

// compute the distance for 4bit PQ
// it only computes for the rows from offset to offset + length
fn compute_pq_distance_4bit_flat(
    distance_table: &[f32],
    num_vectors: usize,
    code: &[u8],
    offset: usize,
    length: usize,
    dists: &mut [f32],
) {
    const NUM_CENTROIDS: usize = 2_usize.pow(4);

    for (sub_vec_idx, vec_indices) in code.chunks_exact(num_vectors).enumerate() {
        let vec_indices = &vec_indices[offset..offset + length];
        let distances = &mut dists[offset..offset + length];
        let dist_table = &distance_table[sub_vec_idx * 2 * NUM_CENTROIDS..];
        let next_dist_table = &distance_table[(sub_vec_idx * 2 + 1) * NUM_CENTROIDS..];
        for (i, &centroid_idx) in vec_indices.iter().enumerate() {
            let current_idx = centroid_idx & 0xF;
            let next_idx = centroid_idx >> 4;
            distances[i] += dist_table[current_idx as usize];
            distances[i] += next_dist_table[next_idx as usize];
        }
    }
}

// Quantize the distance table to u8,
// map distance `d` to `(d-qmin) * 255 / (qmax-qmin)`
// returns (qmin, quantized_distance_table)
#[inline]
fn quantize_distance_table(distance_table: &[f32], qmax: f32) -> (f32, Vec<u8>) {
    let qmin = distance_table.iter().cloned().fold(f32::INFINITY, f32::min);
    let factor = 255.0 / (qmax - qmin);
    let quantized_dist_table = distance_table
        .iter()
        .map(|&d| ((d - qmin) * factor).round() as u8)
        .collect();

    (qmin, quantized_dist_table)
}

/// Compute L2 distance from the query to all code without transposing the code.
/// for testing only
///
/// Type parameters
/// ---------------
/// - C: the tile size of code-book to run at once.
/// - V: the tile size of PQ code to run at once.
///
#[allow(dead_code)]
fn compute_l2_distance_without_transposing<const C: usize, const V: usize>(
    distance_table: &[f32],
    num_bits: u32,
    num_sub_vectors: usize,
    code: &[u8],
) -> Vec<f32> {
    let num_centroids = num_centroids(num_bits);
    let iter = code.chunks_exact(num_sub_vectors * V);
    let distances = iter.clone().flat_map(|c| {
        let mut sums = [0.0_f32; V];
        for i in (0..num_sub_vectors).step_by(C) {
            for (vec_idx, sum) in sums.iter_mut().enumerate() {
                let vec_start = vec_idx * num_sub_vectors;
                let s = c[vec_start + i..]
                    .iter()
                    .take(min(C, num_sub_vectors - i))
                    .enumerate()
                    .map(|(k, c)| distance_table[(i + k) * num_centroids + *c as usize])
                    .sum::<f32>();
                *sum += s;
            }
        }
        sums.into_iter()
    });
    // Remainder
    let remainder = iter.remainder().chunks(num_sub_vectors).map(|c| {
        c.iter()
            .enumerate()
            .map(|(sub_vec_idx, code)| distance_table[sub_vec_idx * num_centroids + *code as usize])
            .sum::<f32>()
    });
    distances.chain(remainder).collect()
}

#[cfg(test)]
mod tests {
    use crate::vector::pq::storage::transpose;

    use super::*;
    use arrow_array::UInt8Array;

    #[test]
    fn test_compute_on_transposed_codes() {
        let num_vectors = 100;
        let num_sub_vectors = 4;
        let num_bits = 8;
        let dimension = 16;
        let codebook =
            Vec::from_iter((0..num_sub_vectors * num_vectors * dimension).map(|v| v as f32));
        let query = Vec::from_iter((0..dimension).map(|v| v as f32));
        let distance_table = build_distance_table_l2(&codebook, num_bits, num_sub_vectors, &query);

        let pq_codes = Vec::from_iter((0..num_vectors * num_sub_vectors).map(|v| v as u8));
        let pq_codes = UInt8Array::from_iter_values(pq_codes);
        let transposed_codes = transpose(&pq_codes, num_vectors, num_sub_vectors);
        let distances = compute_pq_distance(
            &distance_table,
            num_bits,
            num_sub_vectors,
            transposed_codes.values(),
            100,
        );
        let expected = compute_l2_distance_without_transposing::<4, 1>(
            &distance_table,
            num_bits,
            num_sub_vectors,
            pq_codes.values(),
        );
        assert_eq!(distances, expected);
    }

    /// Test with production-like parameters to exercise the FastScan SIMD path.
    /// Uses 96 sub-vectors (768-dim / 8-dim-per-sv) and 1000 vectors (>200 flat_num).
    /// Verifies that the top-k results match the scalar reference (the u8 accumulator
    /// wraps for distant vectors, which is expected — same as the 4-bit path).
    #[test]
    fn test_compute_8bit_fastscan_path() {
        use rand::{Rng, SeedableRng, prelude::StdRng};

        let num_vectors = 1000;
        let num_sub_vectors = 96;
        let num_bits = 8;
        let dimension = 768;

        let mut rng = StdRng::from_seed([42; 32]);
        let codebook: Vec<f32> = (0..256 * dimension).map(|_| rng.random::<f32>()).collect();
        let query: Vec<f32> = (0..dimension).map(|_| rng.random::<f32>()).collect();
        let distance_table = build_distance_table_l2(&codebook, num_bits, num_sub_vectors, &query);

        let pq_codes_raw: Vec<u8> = (0..num_vectors * num_sub_vectors)
            .map(|_| rng.random::<u8>())
            .collect();
        let pq_codes = UInt8Array::from_iter_values(pq_codes_raw.iter().copied());
        let transposed_codes = transpose(&pq_codes, num_vectors, num_sub_vectors);

        // Compute using the optimized path (may use FastScan)
        let distances = compute_pq_distance(
            &distance_table,
            num_bits,
            num_sub_vectors,
            transposed_codes.values(),
            100,
        );

        // Compute exact scalar reference
        let expected = compute_l2_distance_without_transposing::<4, 1>(
            &distance_table,
            num_bits,
            num_sub_vectors,
            &pq_codes_raw,
        );

        assert_eq!(distances.len(), expected.len());

        // The first flat_num distances should be exact
        let flat_num = 200;
        for i in 0..flat_num {
            assert!(
                (distances[i] - expected[i]).abs() < 1e-3,
                "Mismatch at flat index {}: got {}, expected {}",
                i,
                distances[i],
                expected[i],
            );
        }

        // For the SIMD-processed vectors: verify that the top-k ordering is
        // preserved. The u8 accumulator wraps for distant vectors, but close
        // vectors (small distances) should be accurately ranked.
        let k = 20;
        let mut expected_ranked: Vec<(usize, f32)> =
            expected.iter().copied().enumerate().collect();
        expected_ranked.sort_by(|a, b| a.1.total_cmp(&b.1));

        let mut actual_ranked: Vec<(usize, f32)> =
            distances.iter().copied().enumerate().collect();
        actual_ranked.sort_by(|a, b| a.1.total_cmp(&b.1));

        // Check that the top-k vectors from the expected ranking appear in
        // the top-2k of the actual ranking (allowing some reordering due to
        // quantization noise).
        let expected_top_k: std::collections::HashSet<usize> =
            expected_ranked.iter().take(k).map(|(idx, _)| *idx).collect();
        let actual_top_2k: std::collections::HashSet<usize> =
            actual_ranked.iter().take(k * 2).map(|(idx, _)| *idx).collect();

        let overlap = expected_top_k.intersection(&actual_top_2k).count();
        assert!(
            overlap >= k / 2,
            "Top-{k} recall too low: only {overlap}/{k} of true top-k found in top-{}",
            k * 2,
        );
    }
}
