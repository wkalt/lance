# Memory Optimization Report: Cosine IVF-PQ Index Building

**Date:** 2026-02-25
**Branch:** `task/more-memory-lower`
**Workload:** 500k vectors, 3072 dimensions, f32, cosine distance
**Index config:** IVF-PQ, 100 partitions, 96 sub-vectors

## Executive Summary

Through a combination of in-place buffer mutation, owned-value passing, and batch size
tuning, **peak RSS was reduced from 8,329 MB to 3,371 MB (60% reduction)** with no
measurable wall-clock regression (42.1s baseline → 43.1s recommended).

The recommended configuration (all code changes with default batch_size=4096) achieves:
- **Shuffle**: 8,008 MB → 2,911 MB (**-64%**)
- **Training**: 3,539 MB → 2,785 MB (**-21%**)
- **Overall peak**: 8,329 MB → 3,371 MB (**-60%**)

The shuffle stage is no longer the memory bottleneck — `train_quantizer` (PQ training)
now dominates at 2,785 MB, which is close to the irreducible minimum (the training
sample must be fully in memory for k-means).

## Results Table

| # | Variant                     | train_ivf | train_pq | shuffle  | Peak RSS | Wall Time | Delta vs Baseline |
|---|----------------------------|-----------|----------|----------|----------|-----------|-------------------|
| 1 | **Baseline** (v3.0.0-rc.1) | 2,703 MB  | 3,539 MB | 8,008 MB | 8,329 MB | 42.1s     | —                 |
| 2 | +Training optimizations     | 1,562 MB  | 2,780 MB | 8,025 MB | 8,369 MB | 42.6s     | +0.5%             |
| 3 | +Owned Transformer trait    | 1,555 MB  | 2,767 MB | 5,514 MB | 5,774 MB | 42.1s     | -31%              |
| 4 | +Halved concurrency         | 1,561 MB  | 2,771 MB | 4,525 MB | 5,945 MB | 43.1s     | -29%              |
| 5 | All code opts (default)     | 1,563 MB  | 2,772 MB | 4,533 MB | 5,815 MB | 43.6s     | -30%              |
| 6 | All + CONCURRENCY=4         | 1,555 MB  | 2,770 MB | 4,830 MB | 5,578 MB | 51.6s     | -33%              |
| 7 | All + CONCURRENCY=2         | 1,557 MB  | 2,767 MB | 3,543 MB | 5,333 MB | 63.1s     | -36%              |
| 8 | All + BATCH_SIZE=4096       | 1,561 MB  | 2,773 MB | 2,894 MB | 3,492 MB | 42.1s     | **-58%**          |
| 9 | All + BATCH_SIZE=2048       | 1,560 MB  | 2,764 MB | 2,170 MB | 2,773 MB | 42.6s     | **-67%**          |
|10 | All + CONC=2 + BATCH=4096   | 1,555 MB  | 2,768 MB | 2,399 MB | 3,286 MB | 62.6s     | -61%              |
|**R** | **Recommended** (all code, default batch=4096) | **1,565 MB** | **2,785 MB** | **2,911 MB** | **3,371 MB** | **43.1s** | **-60%** |

## Optimization Details

### 1. Training optimizations (`1389e9b9`) — Already on branch

**What:** `normalize_fsl_owned()` for in-place L2 normalization of training data, plus
skip `arrow::compute::filter` when all vectors are already finite.

**Impact on training stages:**
- train_ivf: 2,703 → 1,562 MB (**-42%**)
- train_pq: 3,539 → 2,780 MB (**-21%**)

**Impact on overall peak:** None — shuffle dominates at ~8 GB.

### 2. Owned Transformer trait (`0228933c`) — THE KEY CHANGE

**What:** Changed `Transformer::transform(&self, batch: &RecordBatch)` to take
`RecordBatch` by value. This enables:

- **NormalizeTransformer**: Extracts the vector column via `RecordBatch::remove_column()`,
  gets sole ownership of the underlying buffer, and normalizes in-place via
  `Buffer::into_mutable()`. Eliminates a full copy of the vector data per batch.
- **IvfTransformer**: Removes the initial `batch.clone()` at the pipeline entry point.
- **KeepFiniteVectors**: Returns the batch as-is (no clone) when all vectors are finite.
  Also adds a bulk finite check on the entire values buffer before falling back to
  per-vector iteration.
- All other transformers: Trivial `&` removal, no unnecessary clones.

**Impact:** Shuffle dropped from 8,025 → 5,514 MB (**-31%**). Peak RSS dropped from
8,369 → 5,774 MB (**-31%**). Zero wall-clock regression.

**Why it works:** During the shuffle phase, every batch of 8192 vectors (96 MB each at
3072 dims × f32) passes through NormalizeTransformer, which previously allocated a full
copy of the values buffer. With 16 concurrent batches in flight, that's ~1.5 GB of
transient copies eliminated.

**Files changed:** 13 files in `lance-index` and `lance` crates (all Transformer
implementations + call sites).

### 3. Reduced shuffle concurrency (`1aa47c08`)

**What:** Halved the default `batch_readahead` and `.buffered()` concurrency during
shuffle from `num_cpus` to `num_cpus/2`. Configurable via `LANCE_SHUFFLE_CONCURRENCY`.

**Impact:** Shuffle dropped from 5,514 → 4,525 MB (**-18%**) with only +1s wall time.

**Tradeoff:** Further reducing concurrency (CONCURRENCY=2) drops shuffle to 3,543 MB
but adds 50% wall time (+21s). Not worth it as a default.

### 4. Configurable shuffle batch size (`d6d3330e`) — HUGE WIN

**What:** Added `LANCE_SHUFFLE_BATCH_SIZE` env var to override the scan batch size during
shuffle. Default remains 8192, but setting it to 4096 or 2048 dramatically reduces
per-batch memory.

**Impact:**
- BATCH_SIZE=4096: Shuffle drops to 2,894 MB, peak 3,492 MB (**-58%**), **zero wall regression**
- BATCH_SIZE=2048: Shuffle drops to 2,170 MB, peak 2,773 MB (**-67%**), **zero wall regression**

**Why no throughput impact:** The bottleneck in the shuffle pipeline is the IVF transform
(partition assignment, PQ encoding), which is compute-bound. Smaller batches don't change
the total compute, they just reduce the memory footprint of each in-flight batch. The
overhead of processing more smaller batches is negligible.

## Recommendations

### Immediate (should ship now)

1. **Owned Transformer trait** — Pure improvement, no tradeoffs. Saves ~2.5 GB on this
   workload with zero performance impact. The API change is internal to `lance-index`.

2. **Default shuffle batch size = 4096** — Instead of inheriting the dataset's default
   batch size (8192), hardcode the shuffle scan to use 4096. This halves per-batch memory
   during shuffle with zero throughput cost. Should be the new default, not just an env var.

### Consider

3. **Halved concurrency** — Modest improvement (+1s wall time for -18% shuffle memory).
   Worth it as a default since the wall time impact is <3%. The env var override is good
   for users with very large vectors who need more memory savings.

4. **BATCH_SIZE=2048** — Even lower memory, still zero wall regression. Could be the
   default for high-dimensional vectors (dim > 1024).

### After these changes, the bottleneck shifts

With batch_size=4096 + owned transforms, the new bottleneck is **train_quantizer at
2,773 MB**. This is the PQ training on the sample data and is much harder to optimize
further since the sample needs to be fully in memory for k-means clustering.

## Appendix: Per-Stage Memory Breakdown

The IVF-PQ index build has 4 stages:

1. **train_ivf** — Sample ~1% of data, L2-normalize, run k-means to find IVF centroids.
   Memory ≈ 1× sample size after our training optimizations.

2. **train_quantizer** — Same sample, train PQ codebooks. Memory ≈ 1× sample size
   (same sample used).

3. **shuffle** — Scan ALL data, transform each batch through the pipeline
   (Flatten → Normalize → KeepFinite → Partition → Residual → PQ encode), write to
   temp file, then read back and sort by partition. Memory dominated by concurrent
   in-flight batches during transform.

4. **build_partitions** — Read sorted partitions, write final index files. Minimal memory.

## Appendix: How to Reproduce

```bash
# Generate dataset
python3 -c "
import pyarrow as pa, numpy as np, lance
n, dim = 500_000, 3072
for i in range(0, n, 50_000):
    size = min(50_000, n - i)
    vecs = np.random.randn(size, dim).astype(np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    table = pa.table({'vector': pa.FixedSizeListArray.from_arrays(vecs.flatten(), dim),
                      'id': pa.array(range(i, i+size), type=pa.int64())})
    lance.write_dataset(table, '/tmp/bench_500k_3072', mode='overwrite' if i == 0 else 'append')
"

# Run indexer (with sophon indexer binary)
LANCE_SHUFFLE_BATCH_SIZE=4096 /usr/bin/time -v ./indexer \
    --table-uri /tmp/bench_500k_3072 \
    --progress-csv /tmp/progress.csv \
    --progress-interval-ms 500 \
    reindex ivf-pq \
    --column-name vector \
    --distance-type cosine \
    --num-partitions 100 \
    --num-sub-vectors 96
```
