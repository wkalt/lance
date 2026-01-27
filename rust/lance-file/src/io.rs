// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use futures::{future::BoxFuture, FutureExt};
use lance_encoding::EncodingsIo;
use lance_io::scheduler::FileScheduler;

#[derive(Debug)]
pub struct LanceEncodingsIo {
    scheduler: FileScheduler,
}

impl LanceEncodingsIo {
    pub fn new(scheduler: FileScheduler) -> Self {
        Self { scheduler }
    }

    /// Deprecated: read_chunk_size is no longer used. Range splitting is handled
    /// by FileScheduler via max_iop_size.
    pub fn with_read_chunk_size(self, _read_chunk_size: u64) -> Self {
        self
    }
}

impl EncodingsIo for LanceEncodingsIo {
    fn submit_request(
        &self,
        ranges: Vec<std::ops::Range<u64>>,
        priority: u64,
    ) -> BoxFuture<'static, lance_core::Result<Vec<Vec<bytes::Bytes>>>> {
        // Range splitting and reassembly is handled by FileScheduler, which
        // splits by max_iop_size and reassembles results. No need to duplicate
        // that here.
        self.scheduler.submit_request(ranges, priority).boxed()
    }
}
