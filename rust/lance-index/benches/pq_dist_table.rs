// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark of building PQ distance table.

use std::iter::repeat_n;

use arrow_array::types::{Float16Type, Float32Type, Float64Type};
use arrow_array::{FixedSizeListArray, UInt8Array};
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use lance_arrow::{ArrowFloatType, FixedSizeListArrayExt, FloatArray};
use lance_index::vector::pq::ProductQuantizer;
use lance_index::vector::pq::distance::*;
use lance_linalg::distance::{DistanceType, Dot, L2};
use lance_testing::datagen::generate_random_array_with_seed;
use rand::{Rng, SeedableRng, prelude::StdRng};

#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

const DIM: usize = 128;
const PQ: usize = DIM / 8;
const TOTAL: usize = 16 * 1000;

// Production-scale parameters (mxbai-embed 768-dim, ~52K rows/partition)
const PROD_DIM: usize = 768;
const PROD_PQ: usize = PROD_DIM / 8; // 96 sub-vectors
const PROD_TOTAL: usize = 52_800;

fn construct_dist_table(c: &mut Criterion) {
    construct_dist_table_for_type::<Float16Type>(c, "f16");
    construct_dist_table_for_type::<Float32Type>(c, "f32");
    construct_dist_table_for_type::<Float64Type>(c, "f64");
}

fn construct_dist_table_for_type<T: ArrowFloatType>(c: &mut Criterion, type_name: &str)
where
    T::Native: L2 + Dot,
    T::ArrayType: FloatArray<T>,
{
    let codebook = generate_random_array_with_seed::<T>(256 * DIM, [88; 32]);
    let query = generate_random_array_with_seed::<T>(DIM, [32; 32]);

    c.bench_function(
        format!(
            "construct_dist_table: {},PQ={},DIM={},type={}",
            DistanceType::L2,
            PQ,
            DIM,
            type_name
        )
        .as_str(),
        |b| {
            b.iter(|| {
                black_box(build_distance_table_l2(
                    codebook.as_slice(),
                    8,
                    PQ,
                    query.as_slice(),
                ));
            })
        },
    );

    c.bench_function(
        format!(
            "construct_dist_table: {},PQ={},DIM={},type={}",
            DistanceType::Dot,
            PQ,
            DIM,
            type_name
        )
        .as_str(),
        |b| {
            b.iter(|| {
                black_box(build_distance_table_dot(
                    codebook.as_slice(),
                    8,
                    PQ,
                    query.as_slice(),
                ));
            })
        },
    );
}

fn compute_distances(c: &mut Criterion) {
    compute_distances_for_type::<Float16Type>(c, "f16");
    compute_distances_for_type::<Float32Type>(c, "f32");
    compute_distances_for_type::<Float64Type>(c, "f64");
}

fn compute_distances_for_type<T: ArrowFloatType>(c: &mut Criterion, type_name: &str)
where
    T::Native: L2 + Dot,
    T::ArrayType: FloatArray<T>,
{
    let codebook = generate_random_array_with_seed::<T>(256 * DIM, [88; 32]);
    let query = generate_random_array_with_seed::<T>(DIM, [32; 32]);

    let mut rnd = StdRng::from_seed([32; 32]);
    let code = UInt8Array::from_iter_values(repeat_n(rnd.random::<u8>(), TOTAL * PQ));

    for dt in [DistanceType::L2, DistanceType::Cosine, DistanceType::Dot] {
        let pq = ProductQuantizer::new(
            PQ,
            8,
            DIM,
            FixedSizeListArray::try_new_from_values(codebook.clone(), DIM as i32).unwrap(),
            dt,
        );

        c.bench_function(
            format!(
                "compute_distances: {},{},PQ={},DIM={},type={}",
                TOTAL, dt, PQ, DIM, type_name
            )
            .as_str(),
            |b| {
                b.iter(|| {
                    black_box(pq.compute_distances(&query, &code).unwrap());
                })
            },
        );
    }
}

/// Original (baseline) 8-bit PQ distance: simple sub-vector-major loop, no tiling.
fn compute_pq_distance_original(
    distance_table: &[f32],
    num_sub_vectors: usize,
    code: &[u8],
) -> Vec<f32> {
    let num_vectors = code.len() / num_sub_vectors;
    let mut distances = vec![0.0f32; num_vectors];
    const NUM_CENTROIDS: usize = 256;
    for (sub_vec_idx, vec_indices) in code.chunks_exact(num_vectors).enumerate() {
        let dist_table =
            &distance_table[sub_vec_idx * NUM_CENTROIDS..(sub_vec_idx + 1) * NUM_CENTROIDS];
        vec_indices
            .iter()
            .zip(distances.iter_mut())
            .for_each(|(&centroid_idx, sum)| {
                *sum += dist_table[centroid_idx as usize];
            });
    }
    distances
}

/// Benchmark PQ distance computation at production scale with transposed codes.
/// Compares 8-bit (current vs original) and 4-bit implementations.
fn compute_pq_distance_prod(c: &mut Criterion) {
    use lance_index::vector::pq::storage::transpose;

    let mut rnd = StdRng::from_seed([42; 32]);

    // --- 8-bit PQ ---
    let raw_codes =
        UInt8Array::from_iter_values((0..PROD_TOTAL * PROD_PQ).map(|_| rnd.random::<u8>()));
    let transposed = transpose(&raw_codes, PROD_TOTAL, PROD_PQ);

    let codebook_8bit = generate_random_array_with_seed::<Float32Type>(256 * PROD_DIM, [88; 32]);

    let pq_8bit = ProductQuantizer::new(
        PROD_PQ,
        8,
        PROD_DIM,
        FixedSizeListArray::try_new_from_values(codebook_8bit.clone(), PROD_DIM as i32).unwrap(),
        DistanceType::L2,
    );

    let query = generate_random_array_with_seed::<Float32Type>(PROD_DIM, [32; 32]);
    let dt = build_distance_table_l2(codebook_8bit.as_slice(), 8, PROD_PQ, query.as_slice());

    c.bench_function(
        format!(
            "compute_pq_distance: N={},PQ={},DIM={},8bit",
            PROD_TOTAL, PROD_PQ, PROD_DIM
        )
        .as_str(),
        |b| {
            b.iter(|| {
                black_box(pq_8bit.compute_distances(&query, &transposed).unwrap());
            })
        },
    );

    c.bench_function(
        format!(
            "compute_pq_distance_original: N={},PQ={},DIM={},8bit",
            PROD_TOTAL, PROD_PQ, PROD_DIM
        )
        .as_str(),
        |b| {
            b.iter(|| {
                black_box(compute_pq_distance_original(
                    &dt,
                    PROD_PQ,
                    transposed.values(),
                ));
            })
        },
    );

    // --- 4-bit PQ ---
    // 4-bit: 16 centroids per sub-vector, each byte packs 2 sub-vectors.
    // Codebook: 16 centroids * dim values.
    let codebook_4bit = generate_random_array_with_seed::<Float32Type>(16 * PROD_DIM, [77; 32]);

    let pq_4bit = ProductQuantizer::new(
        PROD_PQ,
        4,
        PROD_DIM,
        FixedSizeListArray::try_new_from_values(codebook_4bit.clone(), PROD_DIM as i32).unwrap(),
        DistanceType::L2,
    );

    // 4-bit codes: each byte holds 2 sub-vectors, so num_sub_vectors_in_byte = PQ/2.
    // Raw codes: PROD_TOTAL rows * (PROD_PQ/2) bytes per row.
    let raw_codes_4bit = UInt8Array::from_iter_values(
        (0..PROD_TOTAL * PROD_PQ / 2).map(|_| rnd.random::<u8>()),
    );
    let transposed_4bit = transpose(&raw_codes_4bit, PROD_TOTAL, PROD_PQ / 2);

    c.bench_function(
        format!(
            "compute_pq_distance: N={},PQ={},DIM={},4bit",
            PROD_TOTAL, PROD_PQ, PROD_DIM
        )
        .as_str(),
        |b| {
            b.iter(|| {
                black_box(pq_4bit.compute_distances(&query, &transposed_4bit).unwrap());
            })
        },
    );
}

/// Benchmark FastScan vs original across a range of embedding dimensions.
fn compute_pq_distance_dims(c: &mut Criterion) {
    use lance_index::vector::pq::storage::transpose;

    let num_vectors = 52_800;
    let sub_vector_length = 8; // each sub-vector is 8 floats

    for dim in [256, 512, 768, 1024, 1536, 2048, 3072, 4096] {
        let num_sub_vectors = dim / sub_vector_length;
        let mut rnd = StdRng::from_seed([42; 32]);

        let raw_codes = UInt8Array::from_iter_values(
            (0..num_vectors * num_sub_vectors).map(|_| rnd.random::<u8>()),
        );
        let transposed = transpose(&raw_codes, num_vectors, num_sub_vectors);

        let codebook =
            generate_random_array_with_seed::<Float32Type>(256 * dim, [88; 32]);

        let pq = ProductQuantizer::new(
            num_sub_vectors,
            8,
            dim,
            FixedSizeListArray::try_new_from_values(codebook.clone(), dim as i32).unwrap(),
            DistanceType::L2,
        );

        let query = generate_random_array_with_seed::<Float32Type>(dim, [32; 32]);
        let dt = build_distance_table_l2(codebook.as_slice(), 8, num_sub_vectors, query.as_slice());

        c.bench_function(
            format!(
                "pq_dist_dims: N={},DIM={},PQ={},fastscan",
                num_vectors, dim, num_sub_vectors
            )
            .as_str(),
            |b| {
                b.iter(|| {
                    black_box(pq.compute_distances(&query, &transposed).unwrap());
                })
            },
        );

        c.bench_function(
            format!(
                "pq_dist_dims: N={},DIM={},PQ={},original",
                num_vectors, dim, num_sub_vectors
            )
            .as_str(),
            |b| {
                b.iter(|| {
                    black_box(compute_pq_distance_original(
                        &dt,
                        num_sub_vectors,
                        transposed.values(),
                    ));
                })
            },
        );
    }
}

/// Benchmark PQ distance under cache contention from concurrent threads.
///
/// Spawns background threads that continuously compute PQ distances (thrashing
/// shared L2/L3 cache), then measures the latency of the main thread's
/// computation.  This simulates the production scenario where many concurrent
/// queries compete for cache.
fn compute_pq_distance_contended(c: &mut Criterion) {
    use lance_index::vector::pq::storage::transpose;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let num_bg_threads = 128;

    let mut rnd = StdRng::from_seed([42; 32]);
    let raw_codes =
        UInt8Array::from_iter_values((0..PROD_TOTAL * PROD_PQ).map(|_| rnd.random::<u8>()));
    let transposed = transpose(&raw_codes, PROD_TOTAL, PROD_PQ);
    let codebook = generate_random_array_with_seed::<Float32Type>(256 * PROD_DIM, [88; 32]);

    let pq = ProductQuantizer::new(
        PROD_PQ,
        8,
        PROD_DIM,
        FixedSizeListArray::try_new_from_values(codebook.clone(), PROD_DIM as i32).unwrap(),
        DistanceType::L2,
    );
    let query = generate_random_array_with_seed::<Float32Type>(PROD_DIM, [32; 32]);
    let dt = build_distance_table_l2(codebook.as_slice(), 8, PROD_PQ, query.as_slice());

    // Each background thread gets its own codes and PQ to avoid false sharing.
    let stop = Arc::new(AtomicBool::new(false));
    let handles: Vec<_> = (0..num_bg_threads)
        .map(|i| {
            let stop = stop.clone();
            let codebook = codebook.clone();
            let seed = [i as u8 + 1; 32];
            std::thread::spawn(move || {
                let mut rnd = StdRng::from_seed(seed);
                let raw = UInt8Array::from_iter_values(
                    (0..PROD_TOTAL * PROD_PQ).map(|_| rnd.random::<u8>()),
                );
                let codes = transpose(&raw, PROD_TOTAL, PROD_PQ);
                let pq = ProductQuantizer::new(
                    PROD_PQ,
                    8,
                    PROD_DIM,
                    FixedSizeListArray::try_new_from_values(codebook, PROD_DIM as i32).unwrap(),
                    DistanceType::L2,
                );
                let q = generate_random_array_with_seed::<Float32Type>(PROD_DIM, seed);
                while !stop.load(Ordering::Relaxed) {
                    black_box(pq.compute_distances(&q, &codes).unwrap());
                }
            })
        })
        .collect();

    c.bench_function(
        format!(
            "compute_pq_distance_contended: N={},PQ={},DIM={},8bit,bg_threads={}",
            PROD_TOTAL, PROD_PQ, PROD_DIM, num_bg_threads
        )
        .as_str(),
        |b| {
            b.iter(|| {
                black_box(pq.compute_distances(&query, &transposed).unwrap());
            })
        },
    );

    c.bench_function(
        format!(
            "compute_pq_distance_original_contended: N={},PQ={},DIM={},8bit,bg_threads={}",
            PROD_TOTAL, PROD_PQ, PROD_DIM, num_bg_threads
        )
        .as_str(),
        |b| {
            b.iter(|| {
                black_box(compute_pq_distance_original(
                    &dt,
                    PROD_PQ,
                    transposed.values(),
                ));
            })
        },
    );

    stop.store(true, Ordering::Relaxed);
    for h in handles {
        let _ = h.join();
    }
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = construct_dist_table, compute_distances, compute_pq_distance_prod, compute_pq_distance_dims, compute_pq_distance_contended);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = construct_dist_table, compute_distances, compute_pq_distance_prod, compute_pq_distance_dims, compute_pq_distance_contended);

criterion_main!(benches);
