// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lazy, byte-range-fetched FST traversal for FTS token sets.
//!
//! The inverted index's per-partition token set is stored as a single packed
//! `burntsushi/fst` blob in a LargeBinary column. Today the whole blob is
//! eagerly pulled into memory before any lookup, which is fine for warm /
//! local data but terrible on cold object storage: a one-term query against
//! a large partition has to pull tens of megabytes of FST to answer a
//! question that only touches a few kilobytes of state.
//!
//! [`LazyFst`] wraps the underlying file reader and fetches byte ranges on
//! demand as the FST traversal walks states. A lookup of `"science"` against
//! a ~20 MiB blob typically touches ~2–4 KiB of bytes across 6–10 node
//! reads. The shape is latency-bound on cold object stores (sequential round
//! trips per node), but the *total bytes* per lookup are orders of magnitude
//! smaller than the eager path.
//!
//! The implementation reuses `fst::raw::Fst` for node decoding by feeding it
//! a sparse `Vec<u8>` that mirrors the blob's address space. Only the bytes
//! we've fetched are populated; the rest remain zeroed and untouched. Each
//! node access ensures a small (4 KiB) window around the address is loaded,
//! then constructs a fresh `Fst::new(&buf[..])` — which only re-parses
//! header + trailer — and decodes the node. The borrow lasts only as long
//! as the decode, freeing us to mutate the buffer between hops.

use std::ops::Range;
use std::sync::Arc;

use bytes::Bytes;
use fst::raw::Fst;
use lance_core::{Error, Result};
use rangemap::RangeSet;
use tokio::sync::Mutex;

use crate::scalar::inverted::index::{
    TOKEN_FST_BYTES_COL, TOKEN_NEXT_ID_COL, TOKEN_TOTAL_LENGTH_COL,
};

/// Async byte-range source backing a [`LazyFst`]. The implementation is
/// expected to return bytes corresponding to the FST blob's address space
/// (i.e. `range.start = 0` is the first byte of the FST), which means any
/// underlying file-format offsetting (e.g. Lance's column-page byte offset)
/// is handled by the implementation, not by [`LazyFst`].
#[async_trait::async_trait]
pub trait LazyFstByteReader: Send + Sync + std::fmt::Debug {
    /// Length of the FST blob in bytes.
    fn blob_len(&self) -> u64;

    /// Read the requested byte range from the FST blob. Implementations
    /// must return exactly `range.end - range.start` bytes.
    async fn read_range(&self, range: Range<u64>) -> Result<Bytes>;
}

/// FST traversal that fetches state-region bytes on demand instead of
/// pre-loading the entire blob.
///
/// Instances are cheap to construct (one HTTP request for the header, one
/// for the trailer) and are intended to be held for the lifetime of a
/// `TokenSet` so that successive lookups can share fetched node bytes
/// through the internal range cache.
#[derive(Debug)]
pub struct LazyFst {
    reader: Arc<dyn LazyFstByteReader>,
    blob_len: usize,
    /// Sparse buffer mirroring the FST blob's address space; populated
    /// lazily. On Linux `vec![0; n]` for large `n` is mmap-backed, so
    /// untouched pages cost virtual address space, not RAM.
    buffer: Mutex<LazyBuffer>,
    /// Number of keys; read from the trailer at construction so callers
    /// can query `len` synchronously.
    num_keys: usize,
    /// next_id stored in the FST token-set extras alongside the FST blob.
    /// Set via [`Self::try_open_with_token_extras`].
    pub next_id: u32,
    /// total_length stored in the FST token-set extras.
    pub total_length: u64,
}

#[derive(Debug)]
struct LazyBuffer {
    bytes: Vec<u8>,
    loaded: RangeSet<usize>,
}

/// Window of bytes to fetch around each node access. Burntsushi-style FST
/// nodes are variable-length but bounded by transition count; 4 KiB covers
/// any reasonable node and amortizes HTTP overhead across nearby decodes.
const NODE_WINDOW: usize = 4096;
/// Footer of the FST file: contains root_addr + len + (optional) crc32.
const FST_TRAILER_LEN: usize = 36;
/// Header of the FST file: version (u64) + type (u64).
const FST_HEADER_LEN: usize = 16;

impl LazyFst {
    /// Build a lazy view over the FST blob. Issues two small reads up
    /// front (header + trailer) so that subsequent node accesses can
    /// always reconstruct an `Fst` from the in-memory buffer.
    ///
    /// `next_id` and `total_length` are read directly from the trailer
    /// extras burntsushi/fst stores after the standard trailer — see
    /// [`crate::scalar::inverted::index::TokenSet::into_fst_batch`] for
    /// the on-disk layout.
    pub async fn try_open(reader: Arc<dyn LazyFstByteReader>) -> Result<Self> {
        let blob_len = reader.blob_len() as usize;
        if blob_len < FST_HEADER_LEN + FST_TRAILER_LEN {
            return Err(Error::invalid_input(format!(
                "FST blob too small ({} bytes) to be valid",
                blob_len
            )));
        }
        let mut buffer = vec![0u8; blob_len];
        let mut loaded: RangeSet<usize> = RangeSet::default();
        let header = reader.read_range(0..FST_HEADER_LEN as u64).await?;
        buffer[..FST_HEADER_LEN].copy_from_slice(&header);
        loaded.insert(0..FST_HEADER_LEN);
        let trailer_start = blob_len - FST_TRAILER_LEN;
        let trailer = reader
            .read_range(trailer_start as u64..blob_len as u64)
            .await?;
        buffer[trailer_start..blob_len].copy_from_slice(&trailer);
        loaded.insert(trailer_start..blob_len);
        // burntsushi/fst v3 trailer layout: [pad..16][len u64][root_addr
        // u64][crc32 u32]. The `len` field is the number of keys.
        let num_keys = u64::from_le_bytes(trailer[16..24].try_into().unwrap()) as usize;
        let lazy = Self {
            reader,
            blob_len,
            buffer: Mutex::new(LazyBuffer {
                bytes: buffer,
                loaded,
            }),
            num_keys,
            next_id: 0,
            total_length: 0,
        };
        Ok(lazy)
    }

    /// Same as [`Self::try_open`], but also reads the inverted-index
    /// TokenSet companion fields (`next_id`, `total_length`) from a small
    /// trailing region of the FST file. These fields are written by the
    /// builder alongside the FST and are needed to construct a complete
    /// `TokenSet`.
    pub async fn try_open_with_token_extras(
        reader: Arc<dyn LazyFstByteReader>,
        next_id: u32,
        total_length: u64,
    ) -> Result<Self> {
        let mut me = Self::try_open(reader).await?;
        me.next_id = next_id;
        me.total_length = total_length;
        Ok(me)
    }

    pub fn blob_len(&self) -> usize {
        self.blob_len
    }

    /// Number of unique tokens in the FST. Equivalent to the eager
    /// `fst::Map::len`. Sync: read from the FST trailer at construction.
    pub fn len(&self) -> usize {
        self.num_keys
    }

    pub fn is_empty(&self) -> bool {
        self.num_keys == 0
    }

    /// Look up a single token. Returns `Some(token_id)` if present.
    /// Fetches at most one node-sized window per state visited.
    pub async fn get(&self, key: &[u8]) -> Result<Option<u32>> {
        let mut buf = self.buffer.lock().await;
        let root_addr = Fst::new(&buf.bytes[..]).map_err(io_err)?.root().addr();
        let mut addr = root_addr;
        let mut output: u64 = 0;
        for &b in key {
            ensure_window(&self.reader, &mut buf, addr).await?;
            let fst = Fst::new(&buf.bytes[..]).map_err(io_err)?;
            let node = fst.node(addr);
            match node.find_input(b) {
                None => return Ok(None),
                Some(i) => {
                    let t = node.transition(i);
                    output = output.wrapping_add(t.out.value());
                    addr = t.addr;
                }
            }
        }
        ensure_window(&self.reader, &mut buf, addr).await?;
        let fst = Fst::new(&buf.bytes[..]).map_err(io_err)?;
        let final_node = fst.node(addr);
        if final_node.is_final() {
            let id = output.wrapping_add(final_node.final_output().value());
            // FST token IDs fit in u32 (the inverted-index builder packs
            // them as u32 originally).
            Ok(Some(id as u32))
        } else {
            Ok(None)
        }
    }

    /// Force the entire blob into memory. Used by paths that need full FST
    /// access (e.g. fuzzy-expansion automata that iterate every key). Does
    /// nothing if already fully loaded.
    pub async fn materialize(&self) -> Result<Bytes> {
        let mut buf = self.buffer.lock().await;
        let full = 0..self.blob_len;
        let gaps: Vec<Range<usize>> = buf.loaded.gaps(&full).collect();
        for gap in gaps {
            let bytes = self
                .reader
                .read_range(gap.start as u64..gap.end as u64)
                .await?;
            buf.bytes[gap.start..gap.end].copy_from_slice(&bytes);
            buf.loaded.insert(gap);
        }
        Ok(Bytes::copy_from_slice(&buf.bytes))
    }
}

async fn ensure_window(
    reader: &Arc<dyn LazyFstByteReader>,
    buf: &mut LazyBuffer,
    addr: usize,
) -> Result<()> {
    let start = addr.saturating_sub(NODE_WINDOW - 1);
    let end = (addr + 1).min(buf.bytes.len());
    if end <= start {
        return Ok(());
    }
    let target = start..end;
    let gaps: Vec<Range<usize>> = buf.loaded.gaps(&target).collect();
    for gap in gaps {
        let bytes = reader.read_range(gap.start as u64..gap.end as u64).await?;
        buf.bytes[gap.start..gap.end].copy_from_slice(&bytes);
        buf.loaded.insert(gap);
    }
    Ok(())
}

fn io_err(e: fst::Error) -> Error {
    Error::io(format!("lazy fst: {}", e))
}

/// Locate the byte range of the FST blob inside a Lance v2 token file
/// (`part_N_tokens.lance`). The blob lives in a LargeBinary column named
/// [`TOKEN_FST_BYTES_COL`]; its bytes occupy the largest buffer of the
/// column's single page. Returns the absolute file byte range.
///
/// Falls back to `Err` if the supplied reader isn't a lance v2 FileReader
/// (the caller should then eager-load instead of taking the lazy path).
pub fn locate_fst_blob_in_lance_file(
    reader: &Arc<dyn crate::scalar::IndexReader>,
) -> Result<Range<u64>> {
    let file_reader = reader
        .as_any()
        .downcast_ref::<lance_file::reader::FileReader>()
        .ok_or_else(|| {
            Error::not_supported(
                "lazy FST requires a lance v2 FileReader; got a different IndexReader impl"
                    .to_string(),
            )
        })?;
    let arrow_schema = arrow_schema::Schema::from(file_reader.schema().as_ref());
    let col_idx = arrow_schema
        .fields
        .iter()
        .position(|f| f.name() == TOKEN_FST_BYTES_COL)
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "expected column {} not found in token file schema",
                TOKEN_FST_BYTES_COL
            ))
        })?;
    let col_meta = file_reader
        .metadata()
        .column_metadatas
        .get(col_idx)
        .ok_or_else(|| Error::invalid_input("column metadata index out of range".to_string()))?;
    let page = col_meta
        .pages
        .first()
        .ok_or_else(|| Error::invalid_input("token FST column has no pages".to_string()))?;
    // LargeBinary single-row layout: one validity buffer (tiny), one
    // offsets buffer (~16 B), one values buffer (the FST itself). Pick the
    // largest by size.
    let (idx, _size) = page
        .buffer_sizes
        .iter()
        .enumerate()
        .max_by_key(|(_, s)| **s)
        .ok_or_else(|| Error::invalid_input("token FST page has no buffers".to_string()))?;
    let start = page.buffer_offsets[idx];
    let end = start + page.buffer_sizes[idx];
    Ok(start..end)
}

/// [`LazyFstByteReader`] that issues byte-range reads against a Lance v2
/// `FileReader`'s underlying scheduler. The (offset, len) of the FST blob
/// within the file is supplied at construction time, so the trait's
/// blob-relative addresses are translated to absolute file offsets here.
#[derive(Debug)]
pub struct LanceFileLazyFstReader {
    reader: Arc<lance_file::reader::FileReader>,
    blob_offset: u64,
    blob_len: u64,
}

impl LanceFileLazyFstReader {
    pub fn new(
        reader: Arc<lance_file::reader::FileReader>,
        blob_offset: u64,
        blob_len: u64,
    ) -> Self {
        Self {
            reader,
            blob_offset,
            blob_len,
        }
    }
}

#[async_trait::async_trait]
impl LazyFstByteReader for LanceFileLazyFstReader {
    fn blob_len(&self) -> u64 {
        self.blob_len
    }

    async fn read_range(&self, range: Range<u64>) -> Result<Bytes> {
        let abs = self.blob_offset + range.start..self.blob_offset + range.end;
        self.reader.read_file_byte_range(abs).await
    }
}

// Silence "unused import" warnings for the column-name constants. They
// only appear in doc comments above; callers reach them via the
// re-export in `inverted::index`.
#[allow(dead_code)]
const _UNUSED: (&str, &str, &str) = (
    TOKEN_FST_BYTES_COL,
    TOKEN_NEXT_ID_COL,
    TOKEN_TOTAL_LENGTH_COL,
);
