// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! End-to-end IVF-PQ vector search benchmark comparing FastScan vs scalar
//! 8-bit PQ distance computation.
//!
//! Creates a dataset sized so each IVF partition holds ~sqrt(1B) ≈ 31K rows,
//! matching production 1B-scale index geometry. Runs each nprobes config
//! with both FastScan (AVX-512 VBMI) and the scalar fallback.
//!
//! Usage:
//!   RUSTFLAGS="-C target-cpu=native" cargo bench -p lance --bench pq_fastscan

#![allow(clippy::print_stdout)]

use std::sync::Arc;

use arrow_array::{
    FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, cast::as_primitive_array,
};
use arrow_schema::{DataType, Field, FieldRef, Schema as ArrowSchema};
use criterion::{Criterion, criterion_group, criterion_main};
use futures::TryStreamExt;
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance::index::vector::VectorIndexParams;
use lance_arrow::{FixedSizeListArrayExt, as_fixed_size_list_array};
use lance_index::{
    DatasetIndexExt, IndexType,
    vector::{
        ivf::IvfBuildParams,
        pq::{PQBuildParams, distance::set_force_scalar_pq},
    },
};
use lance_linalg::distance::MetricType;
use rand::Rng;

#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

const DIM: usize = 768;
const NUM_PARTITIONS: usize = 32;
// ~31K rows per partition (sqrt(1B) scale), 32 partitions → ~1M total rows.
const ROWS_PER_PARTITION: usize = 31_250;
const NUM_ROWS: usize = ROWS_PER_PARTITION * NUM_PARTITIONS;
const BATCH_SIZE: usize = 10_000;

fn bench_pq_fastscan(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let path = "./pq_fastscan_bench.lance";
    println!(
        "Creating dataset: {} rows, {}d, {} partitions (~{} rows/partition)",
        NUM_ROWS, DIM, NUM_PARTITIONS, ROWS_PER_PARTITION
    );
    rt.block_on(create_dataset(std::path::Path::new(path)));

    let dataset = rt.block_on(async { Dataset::open(path).await.unwrap() });

    // Pick a random query vector from the dataset.
    let first_batch = rt.block_on(async {
        dataset
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_next()
            .await
            .unwrap()
            .unwrap()
    });
    let mut rng = rand::rng();
    let vector_column = first_batch.column_by_name("vector").unwrap();
    let value =
        as_fixed_size_list_array(&vector_column).value(rng.random_range(0..vector_column.len()));
    let q: &Float32Array = as_primitive_array(&value);

    for nprobes in [1, 10, 20] {
        // FastScan path
        set_force_scalar_pq(false);
        c.bench_function(
            format!(
                "e2e_fastscan: d={},N={},nprobes={},k=10",
                DIM, NUM_ROWS, nprobes
            )
            .as_str(),
            |b| {
                b.to_async(&rt).iter(|| async {
                    let results = dataset
                        .scan()
                        .nearest("vector", q, 10)
                        .unwrap()
                        .minimum_nprobes(nprobes)
                        .try_into_stream()
                        .await
                        .unwrap()
                        .try_collect::<Vec<_>>()
                        .await
                        .unwrap();
                    assert!(!results.is_empty());
                })
            },
        );

        // Scalar path
        set_force_scalar_pq(true);
        c.bench_function(
            format!(
                "e2e_scalar: d={},N={},nprobes={},k=10",
                DIM, NUM_ROWS, nprobes
            )
            .as_str(),
            |b| {
                b.to_async(&rt).iter(|| async {
                    let results = dataset
                        .scan()
                        .nearest("vector", q, 10)
                        .unwrap()
                        .minimum_nprobes(nprobes)
                        .try_into_stream()
                        .await
                        .unwrap()
                        .try_collect::<Vec<_>>()
                        .await
                        .unwrap();
                    assert!(!results.is_empty());
                })
            },
        );
    }

    // Reset to default
    set_force_scalar_pq(false);

    // Keep the dataset so pq_fastscan_rps example can reuse it.
    // Delete manually: rm -rf ./pq_fastscan_bench.lance
}

async fn create_dataset(path: &std::path::Path) {
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "vector",
        DataType::FixedSizeList(
            FieldRef::new(Field::new("item", DataType::Float32, true)),
            DIM as i32,
        ),
        false,
    )]));

    let _ = std::fs::remove_dir_all(path);

    let batches: Vec<RecordBatch> = (0..NUM_ROWS / BATCH_SIZE)
        .map(|_| {
            let mut rng = rand::rng();
            let values: Vec<f32> = (0..BATCH_SIZE * DIM)
                .map(|_| rng.random_range(0.0..1.0))
                .collect();
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(
                    FixedSizeListArray::try_new_from_values(
                        Float32Array::from(values),
                        DIM as i32,
                    )
                    .unwrap(),
                )],
            )
            .unwrap()
        })
        .collect();

    let write_params = WriteParams {
        max_rows_per_file: NUM_ROWS,
        max_rows_per_group: BATCH_SIZE,
        mode: WriteMode::Create,
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let mut dataset = Dataset::write(reader, path.to_str().unwrap(), Some(write_params))
        .await
        .unwrap();

    let ivf_params = IvfBuildParams {
        num_partitions: Some(NUM_PARTITIONS),
        ..Default::default()
    };
    let pq_params = PQBuildParams {
        num_bits: 8,
        num_sub_vectors: DIM / 8, // 96 sub-vectors, 8 floats each
        ..Default::default()
    };
    let params = VectorIndexParams::with_ivf_pq_params(MetricType::L2, ivf_params, pq_params);
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_pq_fastscan);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_pq_fastscan);

criterion_main!(benches);
