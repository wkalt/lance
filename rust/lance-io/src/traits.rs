// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Range;

use async_trait::async_trait;
use bytes::Bytes;
use deepsize::DeepSizeOf;
use object_store::path::Path;
use prost::Message;
use tokio::io::{AsyncWrite, AsyncWriteExt};

use lance_core::Result;

/// A trait for checking if data is cached without reading it.
///
/// This trait can be implemented by caching object stores to indicate
/// whether a range of bytes is already in cache. When data is cached,
/// the I/O scheduler can return lazy futures that read on demand rather
/// than eagerly spawning I/O tasks.
#[async_trait]
pub trait CacheChecker: std::fmt::Debug + Send + Sync {
    /// Check if a range of bytes for a given path is cached.
    ///
    /// Returns true if all bytes in the range are cached and can be
    /// read without going through the I/O loop.
    async fn is_range_cached(&self, location: &Path, range: Range<u64>) -> bool;
}

pub trait ProtoStruct {
    type Proto: Message;
}

/// A trait for writing to a file on local file system or object store.
#[async_trait]
pub trait Writer: AsyncWrite + Unpin + Send {
    /// Tell the current offset.
    async fn tell(&mut self) -> Result<usize>;
}

/// Lance Write Extension.
#[async_trait]
pub trait WriteExt {
    /// Write a Protobuf message to the [Writer], and returns the file position
    /// where the protobuf is written.
    async fn write_protobuf(&mut self, msg: &impl Message) -> Result<usize>;

    async fn write_struct<
        'b,
        M: Message + From<&'b T>,
        T: ProtoStruct<Proto = M> + Send + Sync + 'b,
    >(
        &mut self,
        obj: &'b T,
    ) -> Result<usize> {
        let msg: M = M::from(obj);
        self.write_protobuf(&msg).await
    }
    /// Write magics to the tail of a file before closing the file.
    async fn write_magics(
        &mut self,
        pos: usize,
        major_version: i16,
        minor_version: i16,
        magic: &[u8],
    ) -> Result<()>;
}

#[async_trait]
impl<W: Writer + ?Sized> WriteExt for W {
    async fn write_protobuf(&mut self, msg: &impl Message) -> Result<usize> {
        let offset = self.tell().await?;

        let len = msg.encoded_len();

        self.write_u32_le(len as u32).await?;
        self.write_all(&msg.encode_to_vec()).await?;

        Ok(offset)
    }

    async fn write_magics(
        &mut self,
        pos: usize,
        major_version: i16,
        minor_version: i16,
        magic: &[u8],
    ) -> Result<()> {
        self.write_i64_le(pos as i64).await?;
        self.write_i16_le(major_version).await?;
        self.write_i16_le(minor_version).await?;
        self.write_all(magic).await?;
        Ok(())
    }
}

#[async_trait]
pub trait Reader: std::fmt::Debug + Send + Sync + DeepSizeOf {
    fn path(&self) -> &Path;

    /// Suggest optimal I/O size per storage device.
    fn block_size(&self) -> usize;

    /// Suggest optimal I/O parallelism per storage device.
    fn io_parallelism(&self) -> usize;

    /// Object/File Size.
    async fn size(&self) -> object_store::Result<usize>;

    /// Read a range of bytes from the object.
    ///
    /// TODO: change to read_at()?
    async fn get_range(&self, range: Range<usize>) -> object_store::Result<Bytes>;

    /// Read all bytes from the object.
    ///
    /// By default this reads the size in a separate IOP but some implementations
    /// may not need the size beforehand.
    async fn get_all(&self) -> object_store::Result<Bytes>;

    /// Check if a range of bytes is cached and can be read lazily.
    ///
    /// When this returns true, the scheduler may bypass the I/O queue and
    /// return a lazy future that reads directly from the cache on demand.
    ///
    /// By default, returns false (assume not cached).
    async fn is_range_cached(&self, _range: Range<usize>) -> bool {
        false
    }
}
