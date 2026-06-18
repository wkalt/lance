// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Sparse column overlays (delta / patch columns).
//!
//! An overlay stores `(row_offset -> new_value)` for a subset of a fragment's
//! rows for ONE field, written as a small sidecar file and merged over the base
//! column at scan time (overlay value wins). It is the read/write dual of a
//! [`crate::format::DeletionFile`]: a deletion vector stores sparse offsets to
//! *drop*, an overlay stores sparse offsets to *replace*. This makes a filtered
//! column update's write cost proportional to the number of changed rows rather
//! than the whole fragment.
//!
//! This module is the format + I/O + merge core (slice 1). Fragment-metadata
//! persistence, a commit op, conflict resolution, and compaction materialization
//! are separate slices.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch, UInt32Array};
use arrow_ipc::CompressionType;
use arrow_ipc::reader::FileReader as ArrowFileReader;
use arrow_ipc::writer::{FileWriter as ArrowFileWriter, IpcWriteOptions};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use arrow_select::concat::concat;
use arrow_select::take::take;
use lance_core::error::{CorruptFileSnafu, box_error};
use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use rand::Rng;
use snafu::ResultExt;

use crate::format::ColumnOverlayFile;

pub const OVERLAYS_DIR: &str = "_overlays";

/// Path of an overlay sidecar, relative to the dataset root.
pub fn overlay_file_path(base: &Path, fragment_id: u64, overlay: &ColumnOverlayFile) -> Path {
    base.clone().join(OVERLAYS_DIR).join(format!(
        "{fragment_id}-{}-{}-{}.lance",
        overlay.field_id, overlay.read_version, overlay.id
    ))
}

pub fn relative_overlay_file_path(fragment_id: u64, overlay: &ColumnOverlayFile) -> String {
    format!(
        "{OVERLAYS_DIR}/{fragment_id}-{}-{}-{}.lance",
        overlay.field_id, overlay.read_version, overlay.id
    )
}

fn overlay_arrow_schema(value_type: &DataType) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("offset", DataType::UInt32, false),
        Field::new("value", value_type.clone(), true),
    ]))
}

/// Serialize `(offsets, values)` to the overlay sidecar's Arrow IPC bytes.
///
/// `offsets` are fragment-local row offsets; `values[i]` is the new value at
/// `offsets[i]`. Both must have equal length.
pub fn encode_overlay(offsets: &[u32], values: ArrayRef) -> Result<Vec<u8>> {
    if offsets.len() != values.len() {
        return Err(Error::invalid_input(format!(
            "overlay offsets ({}) and values ({}) length mismatch",
            offsets.len(),
            values.len()
        )));
    }
    let schema = overlay_arrow_schema(values.data_type());
    let offset_arr = Arc::new(UInt32Array::from_iter_values(offsets.iter().copied()));
    let batch = RecordBatch::try_new(schema.clone(), vec![offset_arr, values])?;

    let mut out: Vec<u8> = Vec::new();
    let write_options =
        IpcWriteOptions::default().try_with_compression(Some(CompressionType::ZSTD))?;
    {
        let mut writer =
            ArrowFileWriter::try_new_with_options(&mut out, schema.as_ref(), write_options)?;
        writer.write(&batch)?;
        writer.finish()?;
    }
    Ok(out)
}

/// Inverse of [`encode_overlay`]: returns `(offsets, values)`.
pub fn decode_overlay(bytes: Vec<u8>, path: &Path) -> Result<(UInt32Array, ArrayRef)> {
    let cursor = std::io::Cursor::new(bytes);
    let mut batches: Vec<RecordBatch> = ArrowFileReader::try_new(cursor, None)?
        .collect::<std::result::Result<_, ArrowError>>()
        .map_err(box_error)
        .context(CorruptFileSnafu { path: path.clone() })?;
    if batches.len() != 1 {
        return Err(Error::corrupt_file(
            path.clone(),
            format!("Expected one batch in overlay file, got {}", batches.len()),
        ));
    }
    let batch = batches.pop().unwrap();
    let offsets = batch
        .column(0)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| Error::corrupt_file(path.clone(), "overlay offset column not u32"))?
        .clone();
    let values = batch.column(1).clone();
    Ok((offsets, values))
}

/// Write a sparse overlay sidecar for `field_id` of `fragment_id`.
pub async fn write_column_overlay_file(
    base: &Path,
    fragment_id: u64,
    field_id: i32,
    read_version: u64,
    offsets: &[u32],
    values: ArrayRef,
    object_store: &ObjectStore,
) -> Result<ColumnOverlayFile> {
    let id = rand::rng().random::<u64>();
    let overlay = ColumnOverlayFile {
        field_id,
        read_version,
        id,
        num_overlaid_rows: Some(offsets.len()),
        base_id: None,
    };
    let path = overlay_file_path(base, fragment_id, &overlay);
    let bytes = encode_overlay(offsets, values)?;
    object_store.put(&path, &bytes).await?;
    Ok(overlay)
}

/// Read an overlay sidecar back into `(offsets, values)`.
pub async fn read_column_overlay_file(
    base: &Path,
    fragment_id: u64,
    overlay: &ColumnOverlayFile,
    object_store: &ObjectStore,
) -> Result<(UInt32Array, ArrayRef)> {
    let path = overlay_file_path(base, fragment_id, overlay);
    let data = object_store.read_one_all(&path).await?;
    decode_overlay(data.to_vec(), &path)
}

/// Patch `column_name` of `batch` with an overlay.
///
/// `batch` covers fragment-local offsets `[batch_start_offset, batch_start_offset
/// plus batch.num_rows())`. For every overlay offset in that range, the column's
/// value at the corresponding position is replaced by the overlay value (overlay
/// wins). Type-generic via concat + take, so it works for any field type.
pub fn apply_overlay_to_batch(
    batch: &RecordBatch,
    column_name: &str,
    batch_start_offset: u64,
    overlay_offsets: &UInt32Array,
    overlay_values: &ArrayRef,
) -> Result<RecordBatch> {
    let col_idx = batch.schema().index_of(column_name)?;
    let base = batch.column(col_idx);
    let n = base.len();

    // combined = [base ... | overlay_values ...]; take picks overlay where present.
    let combined = concat(&[base.as_ref(), overlay_values.as_ref()])?;
    let base_len = n as u32;
    let mut idx: Vec<u32> = (0..base_len).collect();
    let end = batch_start_offset + n as u64;
    for j in 0..overlay_offsets.len() {
        let off = overlay_offsets.value(j) as u64;
        if off >= batch_start_offset && off < end {
            let local = (off - batch_start_offset) as usize;
            idx[local] = base_len + j as u32;
        }
    }
    let take_idx = UInt32Array::from(idx);
    let patched = take(combined.as_ref(), &take_idx, None)?;

    let mut cols = batch.columns().to_vec();
    cols[col_idx] = patched;
    Ok(RecordBatch::try_new(batch.schema(), cols)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int64Array, StringArray};

    fn batch_i64(name: &str, vals: Vec<i64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Int64, true)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vals))]).unwrap()
    }

    #[test]
    fn test_apply_overlay_basic() {
        let batch = batch_i64("v", vec![10, 20, 30, 40, 50]);
        let offsets = UInt32Array::from(vec![1u32, 3]);
        let values = Arc::new(Int64Array::from(vec![200i64, 400])) as ArrayRef;
        let out = apply_overlay_to_batch(&batch, "v", 0, &offsets, &values).unwrap();
        let got = out.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(got.values(), &[10, 200, 30, 400, 50]);
    }

    #[test]
    fn test_apply_overlay_batch_offset_window() {
        // Batch covers fragment offsets [2, 5): base [30, 40, 50].
        // Overlay at global offset 0 (out of window -> ignored) and 3 (-> local 1).
        let batch = batch_i64("v", vec![30, 40, 50]);
        let offsets = UInt32Array::from(vec![0u32, 3]);
        let values = Arc::new(Int64Array::from(vec![999i64, 400])) as ArrayRef;
        let out = apply_overlay_to_batch(&batch, "v", 2, &offsets, &values).unwrap();
        let got = out.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(got.values(), &[30, 400, 50]);
    }

    #[test]
    fn test_apply_overlay_string_and_null() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec![
                Some("a"),
                Some("b"),
                Some("c"),
            ]))],
        )
        .unwrap();
        let offsets = UInt32Array::from(vec![1u32]);
        let values = Arc::new(StringArray::from(vec![None as Option<&str>])) as ArrayRef;
        let out = apply_overlay_to_batch(&batch, "s", 0, &offsets, &values).unwrap();
        let got = out
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(got.value(0), "a");
        assert!(got.is_null(1)); // overlay can set null
        assert_eq!(got.value(2), "c");
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let offsets = vec![1u32, 5, 9];
        let values = Arc::new(Int64Array::from(vec![100i64, 500, 900])) as ArrayRef;
        let bytes = encode_overlay(&offsets, values).unwrap();
        let (got_off, got_val) =
            decode_overlay(bytes, &Path::from("_overlays/test.lance")).unwrap();
        assert_eq!(got_off.values(), &[1, 5, 9]);
        let gv = got_val.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(gv.values(), &[100, 500, 900]);
    }

    #[test]
    fn test_encode_length_mismatch_errors() {
        let values = Arc::new(Int64Array::from(vec![1i64, 2])) as ArrayRef;
        assert!(encode_overlay(&[0u32], values).is_err());
    }

    #[tokio::test]
    async fn test_object_store_roundtrip() {
        let (store, base) = ObjectStore::from_uri("memory://").await.unwrap();
        let values = Arc::new(Int64Array::from(vec![7i64, 8])) as ArrayRef;
        let overlay = write_column_overlay_file(&base, 3, 10, 1, &[2u32, 4], values, &store)
            .await
            .unwrap();
        assert_eq!(overlay.field_id, 10);
        assert_eq!(overlay.num_overlaid_rows, Some(2));
        let (off, val) = read_column_overlay_file(&base, 3, &overlay, &store)
            .await
            .unwrap();
        assert_eq!(off.values(), &[2, 4]);
        let v = val.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(v.values(), &[7, 8]);
    }
}
