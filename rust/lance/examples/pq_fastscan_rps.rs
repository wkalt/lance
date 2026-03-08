//! Measure vector search RPS on an existing Lance dataset, comparing
//! FastScan (AVX-512 VBMI) vs scalar PQ distance computation.
//!
//! Usage:
//!   RUSTFLAGS="-C target-cpu=native" cargo run -p lance --release \
//!     --example pq_fastscan_rps -- <dataset_path> [options]
//!
//! Arguments:
//!   <dataset_path>              Path to a Lance dataset with an IVF-PQ index
//!   --column <name>             Vector column name (default: "vector")
//!   --nprobes <n>               Number of probes (default: 20)
//!   --k <n>                     Top-k results (default: 10)
//!   --threads <n>               Concurrent query threads (default: 1)
//!   --duration <secs>           Duration per run in seconds (default: 10)
//!   --refine <n>                Refine factor (default: 0, disabled)
//!   --index-cache-gb <n>        Index cache size in GB (default: 16)

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use arrow_array::{Array, Float32Array};
use futures::TryStreamExt;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::Dataset;
use lance_arrow::as_fixed_size_list_array;
use lance_index::DatasetIndexExt;
use lance_index::vector::pq::distance::set_force_scalar_pq;
use rand::Rng;

struct Config {
    dataset_path: String,
    column: String,
    nprobes: usize,
    k: usize,
    threads: usize,
    duration_secs: u64,
    refine: usize,
    index_cache_gb: usize,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
        eprintln!(
            "Usage: {} <dataset_path> [--column name] [--nprobes n] [--k n] \
             [--threads n] [--duration secs] [--refine n] [--index-cache-gb n]",
            args[0]
        );
        std::process::exit(1);
    }

    let mut cfg = Config {
        dataset_path: args[1].clone(),
        column: "vector".to_string(),
        nprobes: 20,
        k: 10,
        threads: 1,
        duration_secs: 10,
        refine: 0,
        index_cache_gb: 16,
    };

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--column" => {
                cfg.column = args[i + 1].clone();
                i += 2;
            }
            "--nprobes" => {
                cfg.nprobes = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--k" => {
                cfg.k = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--threads" => {
                cfg.threads = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--duration" => {
                cfg.duration_secs = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--refine" => {
                cfg.refine = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--index-cache-gb" => {
                cfg.index_cache_gb = args[i + 1].parse().unwrap();
                i += 2;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
    }
    cfg
}

fn run_bench(
    rt: &tokio::runtime::Runtime,
    dataset: Arc<Dataset>,
    cfg: &Config,
    queries: &[Vec<f32>],
    label: &str,
) -> f64 {
    let stop = Arc::new(AtomicBool::new(false));
    let total_queries = Arc::new(AtomicU64::new(0));
    let duration = Duration::from_secs(cfg.duration_secs);

    let handles: Vec<_> = (0..cfg.threads)
        .map(|thread_id| {
            let stop = stop.clone();
            let total_queries = total_queries.clone();
            let dataset = dataset.clone();
            let column = cfg.column.clone();
            let nprobes = cfg.nprobes;
            let k = cfg.k;
            let refine = cfg.refine;
            let queries = queries.to_vec();
            let handle = rt.handle().clone();

            std::thread::spawn(move || {
                handle.block_on(async {
                    let mut idx = thread_id % queries.len();
                    let mut count = 0u64;

                    while !stop.load(Ordering::Relaxed) {
                        let q = Float32Array::from(queries[idx].clone());
                        let mut scan = dataset.scan();
                        let mut nearest = scan
                            .nearest(&column, &q, k)
                            .unwrap()
                            .minimum_nprobes(nprobes);
                        if refine > 0 {
                            nearest = nearest.refine(refine as u32);
                        }
                        let results = nearest
                            .try_into_stream()
                            .await
                            .unwrap()
                            .try_collect::<Vec<_>>()
                            .await
                            .unwrap();
                        assert!(!results.is_empty());
                        count += 1;
                        idx = (idx + 1) % queries.len();
                    }

                    total_queries.fetch_add(count, Ordering::Relaxed);
                });
            })
        })
        .collect();

    std::thread::sleep(duration);
    stop.store(true, Ordering::Relaxed);

    for h in handles {
        h.join().unwrap();
    }

    let total = total_queries.load(Ordering::Relaxed);
    let rps = total as f64 / cfg.duration_secs as f64;
    let avg_ms = if rps > 0.0 {
        1000.0 * cfg.threads as f64 / rps
    } else {
        f64::INFINITY
    };

    println!(
        "  {:<12} {:>8} queries in {}s | {:>8.1} RPS | {:>6.2} ms/query avg",
        label, total, cfg.duration_secs, rps, avg_ms
    );
    rps
}

fn main() {
    let cfg = parse_args();
    let rt = tokio::runtime::Runtime::new().unwrap();

    let cache_bytes = cfg.index_cache_gb * 1024 * 1024 * 1024;
    println!("Loading dataset: {}", cfg.dataset_path);
    println!("  index cache: {} GB", cfg.index_cache_gb);

    let (dataset, queries, dim) = rt.block_on(async {
        let dataset = DatasetBuilder::from_uri(&cfg.dataset_path)
            .with_index_cache_size_bytes(cache_bytes)
            .load()
            .await
            .unwrap();

        let num_rows = dataset.count_rows(None).await.unwrap();
        let num_indices = dataset.load_indices().await.unwrap().len();
        println!("  rows: {}, indices: {}", num_rows, num_indices);

        // Sample random query vectors from the dataset.
        let batch = dataset
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_next()
            .await
            .unwrap()
            .unwrap();

        let schema = batch.schema();
        let vector_column = batch.column_by_name(&cfg.column).unwrap_or_else(|| {
            let cols: Vec<_> = schema.fields().iter().map(|f| f.name().as_str()).collect();
            panic!(
                "Column '{}' not found. Available columns: {:?}\nUse --column <name> to specify.",
                cfg.column, cols
            );
        });
        let fsl = as_fixed_size_list_array(&vector_column);
        let dim = fsl.value_length() as usize;
        let mut rng = rand::rng();
        let num_queries = 100;

        let inner_type = match fsl.value_type() {
            arrow_schema::DataType::Float32 => "f32",
            arrow_schema::DataType::Float16 => "f16",
            other => panic!("Unsupported vector element type: {:?}", other),
        };
        println!("  vector type: {}x{}", inner_type, dim);

        let queries: Vec<Vec<f32>> = (0..num_queries)
            .map(|_| {
                let idx = rng.random_range(0..fsl.len());
                let value = fsl.value(idx);
                match inner_type {
                    "f32" => {
                        let arr = value
                            .as_any()
                            .downcast_ref::<arrow_array::Float32Array>()
                            .unwrap();
                        arr.values().to_vec()
                    }
                    "f16" => {
                        let arr = value
                            .as_any()
                            .downcast_ref::<arrow_array::Float16Array>()
                            .unwrap();
                        arr.values().iter().map(|v| v.to_f32()).collect()
                    }
                    _ => unreachable!(),
                }
            })
            .collect();

        (dataset, queries, dim)
    });

    println!(
        "\nConfig: d={}, nprobes={}, k={}, threads={}, duration={}s, refine={}",
        dim, cfg.nprobes, cfg.k, cfg.threads, cfg.duration_secs, cfg.refine
    );

    let dataset = Arc::new(dataset);

    // Warmup — load all partitions into the index cache by running queries
    // with a very high nprobes, then run queries at the actual nprobes.
    println!("\nWarming up (populating index cache)...");
    rt.block_on(async {
        // Probe all partitions: nprobes=max ensures every partition is loaded.
        // We use usize::MAX and let lance clamp to the actual partition count.
        println!("  loading all partitions...");
        let q = Float32Array::from(queries[0].clone());
        let _ = dataset
            .scan()
            .nearest(&cfg.column, &q, cfg.k)
            .unwrap()
            .minimum_nprobes(usize::MAX)
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        // Run queries at actual nprobes to warm remaining caches.
        println!("  running warmup queries...");
        for q_vec in queries.iter().take(50) {
            let q = Float32Array::from(q_vec.clone());
            let _ = dataset
                .scan()
                .nearest(&cfg.column, &q, cfg.k)
                .unwrap()
                .minimum_nprobes(cfg.nprobes)
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
        }
    });
    println!("Warmup done.\n");

    // FastScan
    println!("--- FastScan (AVX-512 VBMI) ---");
    set_force_scalar_pq(false);
    let rps_fast = run_bench(&rt, dataset.clone(), &cfg, &queries, "fastscan");

    // Scalar
    println!("\n--- Scalar (baseline) ---");
    set_force_scalar_pq(true);
    let rps_scalar = run_bench(&rt, dataset.clone(), &cfg, &queries, "scalar");

    // Summary
    set_force_scalar_pq(false);
    println!("\n--- Summary ---");
    println!("  FastScan: {:.1} RPS", rps_fast);
    println!("  Scalar:   {:.1} RPS", rps_scalar);
    if rps_scalar > 0.0 {
        println!("  Speedup:  {:.2}x", rps_fast / rps_scalar);
    }
}
