// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::LazyLock;

use crossbeam_queue::SegQueue;
use futures::{future::BoxFuture, FutureExt};
use lance_encoding::EncodingsIo;
use lance_io::scheduler::FileScheduler;
use tracing::warn;

use super::reader::DEFAULT_READ_CHUNK_SIZE;

/// Maximum size of individual buffers to retain in the pool (16MB).
const MAX_POOLED_BUFFER_SIZE: usize = 16 * 1024 * 1024;

/// Soft limit on total pooled buffer capacity in bytes (256MB).
const MAX_POOL_TOTAL_BYTES: usize = 256 * 1024 * 1024;

/// Global buffer pool for reassembling split ranges.
static BUFFER_POOL: LazyLock<BufferPool> = LazyLock::new(BufferPool::new);

struct BufferPool {
    buffers: SegQueue<Vec<u8>>,
    total_capacity: AtomicU64,
}

impl BufferPool {
    fn new() -> Self {
        Self {
            buffers: SegQueue::new(),
            total_capacity: AtomicU64::new(0),
        }
    }

    fn get(&self, size: usize) -> Vec<u8> {
        if let Some(mut buf) = self.buffers.pop() {
            self.total_capacity
                .fetch_sub(buf.capacity() as u64, Ordering::Relaxed);
            buf.clear();
            if buf.capacity() < size {
                buf.reserve(size - buf.capacity());
            }
            buf
        } else {
            Vec::with_capacity(size)
        }
    }

    fn return_buffer(&self, buf: Vec<u8>) {
        let cap = buf.capacity();
        if cap > MAX_POOLED_BUFFER_SIZE {
            return;
        }
        let current_total = self.total_capacity.load(Ordering::Relaxed) as usize;
        if current_total + cap > MAX_POOL_TOTAL_BYTES {
            return;
        }
        self.total_capacity.fetch_add(cap as u64, Ordering::Relaxed);
        self.buffers.push(buf);
    }
}

/// A buffer that returns itself to the pool when dropped.
struct PooledBuffer {
    vec: Vec<u8>,
}

impl AsRef<[u8]> for PooledBuffer {
    fn as_ref(&self) -> &[u8] {
        &self.vec
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        BUFFER_POOL.return_buffer(std::mem::take(&mut self.vec));
    }
}

#[derive(Debug)]
pub struct LanceEncodingsIo {
    scheduler: FileScheduler,
    /// Size of chunks when reading large pages
    read_chunk_size: u64,
}

impl LanceEncodingsIo {
    pub fn new(scheduler: FileScheduler) -> Self {
        Self {
            scheduler,
            read_chunk_size: DEFAULT_READ_CHUNK_SIZE,
        }
    }

    pub fn with_read_chunk_size(mut self, read_chunk_size: u64) -> Self {
        self.read_chunk_size = read_chunk_size;
        self
    }
}

impl EncodingsIo for LanceEncodingsIo {
    fn submit_request(
        &self,
        ranges: Vec<std::ops::Range<u64>>,
        priority: u64,
    ) -> BoxFuture<'static, lance_core::Result<Vec<bytes::Bytes>>> {
        let mut split_ranges = Vec::new();
        let mut split_indices = Vec::new(); // Track which original range each split came from

        // Split large ranges into smaller chunks
        //
        // TODO: consider read_chunk_size before submitting requests.
        for (idx, range) in ranges.iter().enumerate() {
            let range_size = range.end - range.start;

            if range_size > self.read_chunk_size {
                let num_chunks = range_size.div_ceil(self.read_chunk_size);
                let chunk_size = range_size / num_chunks;

                warn!(
                    range_size = range_size,
                    num_chunks = num_chunks,
                    chunk_size = self.read_chunk_size,
                    "Splitting large I/O range"
                );

                for i in 0..num_chunks {
                    let start = range.start + i * chunk_size;
                    let end = if i == num_chunks - 1 {
                        range.end // Last chunk gets any remaining bytes
                    } else {
                        start + chunk_size
                    };
                    split_ranges.push(start..end);
                    split_indices.push(idx);
                }
            } else {
                split_ranges.push(range.clone());
                split_indices.push(idx);
            }
        }

        let fut = self.scheduler.submit_request(split_ranges, priority);

        async move {
            let split_results = fut.await?;

            // Fast path: if no splitting occurred, return results directly
            if split_results.len() == ranges.len() {
                return Ok(split_results);
            }

            // Slow path: reassemble split results
            let mut results = vec![Vec::new(); ranges.len()];

            for (split_result, &orig_idx) in split_results.iter().zip(split_indices.iter()) {
                results[orig_idx].push(split_result.clone());
            }

            Ok(results
                .into_iter()
                .map(|chunks| {
                    if chunks.len() == 1 {
                        chunks.into_iter().next().unwrap()
                    } else {
                        // Concatenate multiple chunks using pooled buffer
                        let total_size: usize = chunks.iter().map(|c| c.len()).sum();
                        let mut combined = BUFFER_POOL.get(total_size);
                        for chunk in chunks {
                            combined.extend_from_slice(&chunk);
                        }
                        bytes::Bytes::from_owner(PooledBuffer { vec: combined })
                    }
                })
                .collect())
        }
        .boxed()
    }
}
