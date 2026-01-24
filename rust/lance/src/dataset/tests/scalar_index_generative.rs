// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Generative (property-based) testing for scalar indexes.
//!
//! This module tests that scalar indexes (BTree and Bitmap) maintain correctness
//! through arbitrary sequences of operations: writes, deletes, index creation,
//! compaction, and index optimization.
//!
//! The core approach is model-based testing: we maintain a simple reference model
//! (a HashMap) alongside the real Lance dataset. After operations, we verify that
//! queries against the real dataset match the model.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use arrow_array::{
    Array, BooleanArray, Float64Array, Int32Array, Int64Array, RecordBatch, RecordBatchIterator,
    StringArray,
};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use futures::TryStreamExt;
use proptest::prelude::*;

use crate::dataset::optimize::compact_files;
use crate::dataset::{MergeInsertBuilder, WhenNotMatched};
use crate::Dataset;
use lance_datafusion::utils::reader_to_stream;
use lance_index::scalar::ScalarIndexParams;
use lance_index::{DatasetIndexExt, IndexType};

// ============================================================================
// Schema and Row Types
// ============================================================================

/// The columns in our test schema, each suitable for different index types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(dead_code)] // Some variants reserved for future test expansion
pub enum Column {
    /// Primary key (i64) - always unique, used for tracking
    Id,
    /// Integer column suitable for BTree index
    IntCol,
    /// Float column suitable for BTree index
    FloatCol,
    /// String column suitable for BTree index
    StringCol,
    /// Low-cardinality string column suitable for Bitmap index
    Category,
    /// Boolean column suitable for Bitmap index
    BoolCol,
    /// Nullable integer for testing NULL handling
    NullableInt,
}

#[allow(dead_code)] // Reserved for future test expansion
impl Column {
    fn name(&self) -> &'static str {
        match self {
            Column::Id => "id",
            Column::IntCol => "int_col",
            Column::FloatCol => "float_col",
            Column::StringCol => "string_col",
            Column::Category => "category",
            Column::BoolCol => "bool_col",
            Column::NullableInt => "nullable_int",
        }
    }

    fn all_indexable() -> Vec<Column> {
        vec![
            Column::IntCol,
            Column::FloatCol,
            Column::StringCol,
            Column::Category,
            Column::BoolCol,
            Column::NullableInt,
        ]
    }
}

/// Row data without ID (for generation).
#[derive(Debug, Clone, PartialEq)]
pub struct RowData {
    pub int_col: i32,
    pub float_col: f64,
    pub string_col: String,
    pub category: String,    // One of: "A", "B", "C", "D", "E"
    pub bool_col: bool,
    pub nullable_int: Option<i32>,
}

/// A single row of test data (with ID assigned).
#[derive(Debug, Clone, PartialEq)]
pub struct TestRow {
    pub id: i64,
    pub int_col: i32,
    pub float_col: f64,
    pub string_col: String,
    pub category: String,    // One of: "A", "B", "C", "D", "E"
    pub bool_col: bool,
    pub nullable_int: Option<i32>,
}

impl TestRow {
    fn from_data(id: i64, data: RowData) -> Self {
        Self {
            id,
            int_col: data.int_col,
            float_col: data.float_col,
            string_col: data.string_col,
            category: data.category,
            bool_col: data.bool_col,
            nullable_int: data.nullable_int,
        }
    }
}

impl TestRow {
    /// Evaluate a predicate against this row.
    fn matches(&self, pred: &Predicate) -> bool {
        match pred {
            Predicate::True => true,
            Predicate::False => false,
            Predicate::Eq(col, val) => self.get_value(col) == *val,
            Predicate::Ne(col, val) => self.get_value(col) != *val,
            Predicate::Lt(col, val) => self.get_value(col) < *val,
            Predicate::Le(col, val) => self.get_value(col) <= *val,
            Predicate::Gt(col, val) => self.get_value(col) > *val,
            Predicate::Ge(col, val) => self.get_value(col) >= *val,
            Predicate::In(col, vals) => vals.contains(&self.get_value(col)),
            Predicate::IsNull(col) => self.is_null(col),
            Predicate::IsNotNull(col) => !self.is_null(col),
            Predicate::And(a, b) => self.matches(a) && self.matches(b),
            Predicate::Or(a, b) => self.matches(a) || self.matches(b),
            Predicate::Not(p) => !self.matches(p),
        }
    }

    fn get_value(&self, col: &Column) -> Value {
        match col {
            Column::Id => Value::Int64(self.id),
            Column::IntCol => Value::Int32(self.int_col),
            Column::FloatCol => Value::Float64(self.float_col),
            Column::StringCol => Value::String(self.string_col.clone()),
            Column::Category => Value::String(self.category.clone()),
            Column::BoolCol => Value::Bool(self.bool_col),
            Column::NullableInt => match self.nullable_int {
                Some(v) => Value::Int32(v),
                None => Value::Null,
            },
        }
    }

    fn is_null(&self, col: &Column) -> bool {
        match col {
            Column::NullableInt => self.nullable_int.is_none(),
            _ => false,
        }
    }
}

/// Values that can appear in predicates.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Value {
    Null,
    Bool(bool),
    Int32(i32),
    Int64(i64),
    Float64(f64),
    String(String),
}

// ============================================================================
// Predicates
// ============================================================================

/// A filter predicate that can be evaluated against rows.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Some variants reserved for future test expansion
pub enum Predicate {
    True,
    False,
    Eq(Column, Value),
    Ne(Column, Value),
    Lt(Column, Value),
    Le(Column, Value),
    Gt(Column, Value),
    Ge(Column, Value),
    In(Column, Vec<Value>),
    IsNull(Column),
    IsNotNull(Column),
    And(Box<Predicate>, Box<Predicate>),
    Or(Box<Predicate>, Box<Predicate>),
    Not(Box<Predicate>),
}

impl Predicate {
    /// Convert to a SQL filter string for Lance.
    fn to_sql(&self) -> String {
        match self {
            Predicate::True => "true".to_string(),
            Predicate::False => "false".to_string(),
            Predicate::Eq(col, val) => format!("{} = {}", col.name(), val.to_sql()),
            Predicate::Ne(col, val) => format!("{} != {}", col.name(), val.to_sql()),
            Predicate::Lt(col, val) => format!("{} < {}", col.name(), val.to_sql()),
            Predicate::Le(col, val) => format!("{} <= {}", col.name(), val.to_sql()),
            Predicate::Gt(col, val) => format!("{} > {}", col.name(), val.to_sql()),
            Predicate::Ge(col, val) => format!("{} >= {}", col.name(), val.to_sql()),
            Predicate::In(col, vals) => {
                let vals_sql: Vec<String> = vals.iter().map(|v| v.to_sql()).collect();
                format!("{} IN ({})", col.name(), vals_sql.join(", "))
            }
            Predicate::IsNull(col) => format!("{} IS NULL", col.name()),
            Predicate::IsNotNull(col) => format!("{} IS NOT NULL", col.name()),
            Predicate::And(a, b) => format!("({}) AND ({})", a.to_sql(), b.to_sql()),
            Predicate::Or(a, b) => format!("({}) OR ({})", a.to_sql(), b.to_sql()),
            Predicate::Not(p) => format!("NOT ({})", p.to_sql()),
        }
    }
}

impl Value {
    fn to_sql(&self) -> String {
        match self {
            Value::Null => "NULL".to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Int32(i) => i.to_string(),
            Value::Int64(i) => i.to_string(),
            Value::Float64(f) => format!("{:?}", f), // Use debug to preserve precision
            Value::String(s) => format!("'{}'", s.replace('\'', "''")),
        }
    }
}

// ============================================================================
// Operations
// ============================================================================

/// Index type for scalar indexes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarIndexType {
    BTree,
    Bitmap,
}

/// Operations that can be performed on the dataset.
///
/// For generated operations, `WriteData` holds row data without IDs.
/// During execution, IDs are assigned and converted to `WriteRows`.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Some variants reserved for future test expansion
pub enum Op {
    /// Write new rows (with IDs already assigned) - used during execution.
    WriteRows(Vec<TestRow>),

    /// Write new row data (without IDs) - used during generation.
    WriteData(Vec<RowData>),

    /// Delete rows matching a predicate.
    Delete(Predicate),

    /// Create a scalar index on a column.
    CreateIndex {
        column: Column,
        index_type: ScalarIndexType,
    },

    /// Drop an index on a column.
    DropIndex { column: Column },

    /// Compact the dataset files.
    Compact,

    /// Optimize indices (merge delta indices).
    OptimizeIndices,

    /// Close and reopen the dataset from URI.
    /// Tests that index state is correctly serialized/deserialized.
    Reload,

    /// Upsert rows - insert new rows or update existing ones by ID.
    /// This is a complex operation that exercises index update paths.
    MergeInsert(Vec<TestRow>),

    /// Upsert with data (IDs assigned at runtime, mix of updates and inserts).
    MergeInsertData {
        /// New rows (will be assigned fresh IDs - inserts)
        new_rows: Vec<RowData>,
        /// Updates to existing rows (will pick random existing IDs at runtime)
        num_updates: usize,
    },
}

// ============================================================================
// Reference Model
// ============================================================================

/// Simple reference model that tracks expected state.
pub struct Model {
    /// Live rows by ID.
    rows: BTreeMap<i64, TestRow>,
    /// Next ID to assign.
    next_id: i64,
    /// Which columns have BTree indexes.
    btree_indexes: HashSet<Column>,
    /// Which columns have Bitmap indexes.
    bitmap_indexes: HashSet<Column>,
}

impl Model {
    pub fn new() -> Self {
        Self {
            rows: BTreeMap::new(),
            next_id: 0,
            btree_indexes: HashSet::new(),
            bitmap_indexes: HashSet::new(),
        }
    }

    /// Apply an operation to the model.
    pub fn apply(&mut self, op: &Op) {
        match op {
            Op::WriteRows(rows) => {
                for row in rows {
                    self.rows.insert(row.id, row.clone());
                }
            }
            Op::WriteData(_) => {
                panic!("WriteData should be converted to WriteRows before applying to model");
            }
            Op::Delete(pred) => {
                self.rows.retain(|_, row| !row.matches(pred));
            }
            Op::CreateIndex { column, index_type } => match index_type {
                ScalarIndexType::BTree => {
                    self.btree_indexes.insert(*column);
                }
                ScalarIndexType::Bitmap => {
                    self.bitmap_indexes.insert(*column);
                }
            },
            Op::DropIndex { column } => {
                self.btree_indexes.remove(column);
                self.bitmap_indexes.remove(column);
            }
            Op::Compact | Op::OptimizeIndices | Op::Reload => {
                // No effect on logical state
            }
            Op::MergeInsert(rows) => {
                // Upsert: insert new rows or replace existing by ID
                for row in rows {
                    self.rows.insert(row.id, row.clone());
                }
            }
            Op::MergeInsertData { .. } => {
                panic!("MergeInsertData should be converted to MergeInsert before applying");
            }
        }
    }

    /// Query the model with a predicate.
    pub fn query(&self, pred: &Predicate) -> Vec<TestRow> {
        self.rows
            .values()
            .filter(|row| row.matches(pred))
            .cloned()
            .collect()
    }

    /// Get all rows.
    pub fn all_rows(&self) -> Vec<TestRow> {
        self.rows.values().cloned().collect()
    }

    /// Get the row count.
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Allocate IDs for new rows.
    pub fn allocate_ids(&mut self, count: usize) -> Vec<i64> {
        let ids: Vec<i64> = (self.next_id..self.next_id + count as i64).collect();
        self.next_id += count as i64;
        ids
    }
}

impl Default for Model {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Dataset Operations
// ============================================================================

fn test_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int64, false),
        ArrowField::new("int_col", DataType::Int32, false),
        ArrowField::new("float_col", DataType::Float64, false),
        ArrowField::new("string_col", DataType::Utf8, false),
        ArrowField::new("category", DataType::Utf8, false),
        ArrowField::new("bool_col", DataType::Boolean, false),
        ArrowField::new("nullable_int", DataType::Int32, true),
    ]))
}

fn rows_to_batch(rows: &[TestRow]) -> RecordBatch {
    let schema = test_schema();

    let id_array: Int64Array = rows.iter().map(|r| r.id).collect();
    let int_array: Int32Array = rows.iter().map(|r| r.int_col).collect();
    let float_array: Float64Array = rows.iter().map(|r| r.float_col).collect();
    let string_array = StringArray::from_iter_values(rows.iter().map(|r| r.string_col.as_str()));
    let category_array = StringArray::from_iter_values(rows.iter().map(|r| r.category.as_str()));
    let bool_array: BooleanArray = rows.iter().map(|r| r.bool_col).collect();
    let nullable_int_array: Int32Array = rows.iter().map(|r| r.nullable_int).collect();

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(id_array),
            Arc::new(int_array),
            Arc::new(float_array),
            Arc::new(string_array),
            Arc::new(category_array),
            Arc::new(bool_array),
            Arc::new(nullable_int_array),
        ],
    )
    .unwrap()
}

/// Apply an operation to the real dataset.
///
/// Note: For memory:// URIs, we cannot use Dataset::open() to reload the dataset
/// since memory storage doesn't persist across different Dataset instances.
/// Instead, we rely on the mutating methods properly updating the dataset handle.
async fn apply_op(dataset: &mut Dataset, op: &Op) -> crate::Result<()> {
    match op {
        Op::WriteRows(rows) => {
            if rows.is_empty() {
                return Ok(());
            }
            let batch = rows_to_batch(rows);
            let schema = batch.schema();
            let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
            dataset.append(reader, None).await?;
        }
        Op::WriteData(_) => {
            panic!("WriteData should be converted to WriteRows before applying");
        }
        Op::Delete(pred) => {
            let sql = pred.to_sql();
            dataset.delete(&sql).await?;
        }
        Op::CreateIndex { column, index_type } => {
            let index_type_lance = match index_type {
                ScalarIndexType::BTree => IndexType::BTree,
                ScalarIndexType::Bitmap => IndexType::Bitmap,
            };
            dataset
                .create_index(
                    &[column.name()],
                    index_type_lance,
                    None,
                    &ScalarIndexParams::default(),
                    true, // replace if exists
                )
                .await?;
        }
        Op::DropIndex { column } => {
            // Lance doesn't have explicit drop index - we just track in model
            // In practice, creating a new index replaces the old one
            let _ = column;
        }
        Op::Compact => {
            compact_files(dataset, Default::default(), None).await?;
        }
        Op::OptimizeIndices => {
            dataset.optimize_indices(&Default::default()).await?;
        }
        Op::Reload => {
            // Reload must be handled by the caller since it requires
            // dropping and reopening the dataset reference
            panic!("Reload should be handled by run_test_sequence, not apply_op");
        }
        Op::MergeInsert(rows) => {
            if rows.is_empty() {
                return Ok(());
            }
            let batch = rows_to_batch(rows);
            let schema = batch.schema();
            let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
            let stream = reader_to_stream(Box::new(reader));

            // Use id column as the merge key
            let (new_dataset, _stats) =
                MergeInsertBuilder::try_new(Arc::new(dataset.clone()), vec!["id".to_string()])?
                    .when_not_matched(WhenNotMatched::InsertAll)
                    .try_build()?
                    .execute(stream)
                    .await?;

            *dataset = new_dataset.as_ref().clone();
        }
        Op::MergeInsertData { .. } => {
            panic!("MergeInsertData should be converted to MergeInsert before apply_op");
        }
    }
    Ok(())
}

// ============================================================================
// Invariant Checking
// ============================================================================

/// Violation of an expected invariant.
#[derive(Debug)]
pub struct InvariantViolation {
    pub description: String,
    pub expected: String,
    pub actual: String,
}

impl std::fmt::Display for InvariantViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Invariant violation: {}\n  Expected: {}\n  Actual: {}",
            self.description, self.expected, self.actual
        )
    }
}

/// Extract rows from a dataset scan result.
async fn scan_to_rows(dataset: &Dataset, filter: Option<&str>) -> crate::Result<Vec<TestRow>> {
    let mut scanner = dataset.scan();
    if let Some(f) = filter {
        scanner.filter(f)?;
    }

    let batches: Vec<RecordBatch> = scanner.try_into_stream().await?.try_collect().await?;

    let mut rows = Vec::new();
    for batch in batches {
        let id_col = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let int_col = batch
            .column_by_name("int_col")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let float_col = batch
            .column_by_name("float_col")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let string_col = batch
            .column_by_name("string_col")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let category_col = batch
            .column_by_name("category")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let bool_col = batch
            .column_by_name("bool_col")
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        let nullable_int_col = batch
            .column_by_name("nullable_int")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();

        for i in 0..batch.num_rows() {
            rows.push(TestRow {
                id: id_col.value(i),
                int_col: int_col.value(i),
                float_col: float_col.value(i),
                string_col: string_col.value(i).to_string(),
                category: category_col.value(i).to_string(),
                bool_col: bool_col.value(i),
                nullable_int: if nullable_int_col.is_null(i) {
                    None
                } else {
                    Some(nullable_int_col.value(i))
                },
            });
        }
    }

    Ok(rows)
}

/// Compare two sets of rows (order-independent).
fn rows_equal(mut a: Vec<TestRow>, mut b: Vec<TestRow>) -> bool {
    a.sort_by_key(|r| r.id);
    b.sort_by_key(|r| r.id);
    a == b
}

/// Check all invariants between the dataset and model.
async fn check_invariants(
    dataset: &Dataset,
    model: &Model,
    predicates_to_check: &[Predicate],
) -> Result<(), InvariantViolation> {
    // 1. Row count matches
    let real_count = dataset.count_rows(None).await.map_err(|e| InvariantViolation {
        description: "Failed to count rows".to_string(),
        expected: model.row_count().to_string(),
        actual: format!("Error: {}", e),
    })?;

    if real_count != model.row_count() {
        return Err(InvariantViolation {
            description: "Row count mismatch".to_string(),
            expected: model.row_count().to_string(),
            actual: real_count.to_string(),
        });
    }

    // 2. Full scan matches
    let real_rows = scan_to_rows(dataset, None)
        .await
        .map_err(|e| InvariantViolation {
            description: "Failed to scan dataset".to_string(),
            expected: "Success".to_string(),
            actual: format!("Error: {}", e),
        })?;

    let model_rows = model.all_rows();
    if !rows_equal(real_rows.clone(), model_rows.clone()) {
        return Err(InvariantViolation {
            description: "Full scan mismatch".to_string(),
            expected: format!("{} rows: {:?}", model_rows.len(), model_rows.iter().map(|r| r.id).collect::<Vec<_>>()),
            actual: format!("{} rows: {:?}", real_rows.len(), real_rows.iter().map(|r| r.id).collect::<Vec<_>>()),
        });
    }

    // 3. Filtered scans match for each predicate
    for pred in predicates_to_check {
        let sql = pred.to_sql();
        let real_filtered = scan_to_rows(dataset, Some(&sql))
            .await
            .map_err(|e| InvariantViolation {
                description: format!("Failed to scan with filter: {}", sql),
                expected: "Success".to_string(),
                actual: format!("Error: {}", e),
            })?;

        let model_filtered = model.query(pred);
        if !rows_equal(real_filtered.clone(), model_filtered.clone()) {
            return Err(InvariantViolation {
                description: format!("Filtered scan mismatch for: {}", sql),
                expected: format!(
                    "{} rows: {:?}",
                    model_filtered.len(),
                    model_filtered.iter().map(|r| r.id).collect::<Vec<_>>()
                ),
                actual: format!(
                    "{} rows: {:?}",
                    real_filtered.len(),
                    real_filtered.iter().map(|r| r.id).collect::<Vec<_>>()
                ),
            });
        }
    }

    Ok(())
}

// ============================================================================
// Proptest Strategies
// ============================================================================

/// Strategy for generating categories (low cardinality).
fn category_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("A".to_string()),
        Just("B".to_string()),
        Just("C".to_string()),
        Just("D".to_string()),
        Just("E".to_string()),
    ]
}

/// Strategy for generating a single RowData (without ID - ID is assigned later).
fn row_data_strategy() -> impl Strategy<Value = RowData> {
    (
        any::<i32>(),                           // int_col
        -1e6f64..1e6f64,                        // float_col (bounded to avoid precision issues)
        "[a-z]{1,10}",                          // string_col
        category_strategy(),                    // category
        any::<bool>(),                          // bool_col
        proptest::option::of(any::<i32>()),     // nullable_int
    )
        .prop_map(|(int_col, float_col, string_col, category, bool_col, nullable_int)| RowData {
            int_col,
            float_col,
            string_col,
            category,
            bool_col,
            nullable_int,
        })
}

/// Strategy for generating a batch of row data.
fn rows_strategy(max_count: usize) -> impl Strategy<Value = Vec<RowData>> {
    proptest::collection::vec(row_data_strategy(), 1..=max_count)
}

/// Strategy for generating a column.
fn column_strategy() -> impl Strategy<Value = Column> {
    prop_oneof![
        Just(Column::IntCol),
        Just(Column::FloatCol),
        Just(Column::StringCol),
        Just(Column::Category),
        Just(Column::BoolCol),
        Just(Column::NullableInt),
    ]
}

/// Strategy for generating a value appropriate for a column.
#[allow(dead_code)] // Reserved for future test expansion
fn value_for_column_strategy(col: Column) -> BoxedStrategy<Value> {
    match col {
        Column::Id => any::<i64>().prop_map(Value::Int64).boxed(),
        Column::IntCol => any::<i32>().prop_map(Value::Int32).boxed(),
        Column::FloatCol => (-1e6f64..1e6f64).prop_map(Value::Float64).boxed(),
        Column::StringCol => "[a-z]{1,10}".prop_map(Value::String).boxed(),
        Column::Category => category_strategy().prop_map(Value::String).boxed(),
        Column::BoolCol => any::<bool>().prop_map(Value::Bool).boxed(),
        Column::NullableInt => prop_oneof![
            Just(Value::Null),
            any::<i32>().prop_map(Value::Int32),
        ]
        .boxed(),
    }
}

/// Strategy for generating a leaf predicate (non-recursive).
#[allow(dead_code)] // Reserved for future test expansion
fn leaf_predicate_strategy() -> impl Strategy<Value = Predicate> {
    column_strategy().prop_flat_map(|col| {
        let col_clone = col;
        value_for_column_strategy(col).prop_flat_map(move |val| {
            let col = col_clone;
            let val_clone = val.clone();
            prop_oneof![
                Just(Predicate::Eq(col, val.clone())),
                Just(Predicate::Ne(col, val.clone())),
                Just(Predicate::Lt(col, val.clone())),
                Just(Predicate::Le(col, val.clone())),
                Just(Predicate::Gt(col, val.clone())),
                Just(Predicate::Ge(col, val_clone)),
                Just(Predicate::IsNull(col)),
                Just(Predicate::IsNotNull(col)),
            ]
        })
    })
}

/// Strategy for generating predicates with limited recursion.
#[allow(dead_code)] // Reserved for future test expansion
fn predicate_strategy() -> impl Strategy<Value = Predicate> {
    leaf_predicate_strategy().prop_recursive(
        2,   // depth
        8,   // max nodes
        3,   // items per collection
        |inner| {
            prop_oneof![
                // Leaf nodes are weighted higher to keep trees small
                8 => leaf_predicate_strategy(),
                1 => (inner.clone(), inner.clone()).prop_map(|(a, b)| Predicate::And(Box::new(a), Box::new(b))),
                1 => (inner.clone(), inner.clone()).prop_map(|(a, b)| Predicate::Or(Box::new(a), Box::new(b))),
                1 => inner.prop_map(|p| Predicate::Not(Box::new(p))),
            ]
        },
    )
}

/// Strategy for delete predicates (simpler, to ensure we actually delete something).
fn delete_predicate_strategy() -> impl Strategy<Value = Predicate> {
    prop_oneof![
        // Simple equality on category (likely to match some rows)
        category_strategy().prop_map(|cat| Predicate::Eq(Column::Category, Value::String(cat))),
        // Range on int_col
        any::<i32>().prop_map(|v| Predicate::Lt(Column::IntCol, Value::Int32(v))),
        any::<i32>().prop_map(|v| Predicate::Gt(Column::IntCol, Value::Int32(v))),
        // Boolean
        any::<bool>().prop_map(|v| Predicate::Eq(Column::BoolCol, Value::Bool(v))),
        // Null check
        Just(Predicate::IsNull(Column::NullableInt)),
        Just(Predicate::IsNotNull(Column::NullableInt)),
    ]
}

/// Strategy for generating an operation.
fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        // Write operations (most common) - small batches for speed
        40 => rows_strategy(10).prop_map(Op::WriteData),
        // Delete operations
        20 => delete_predicate_strategy().prop_map(Op::Delete),
        // Create BTree index (less frequent - expensive)
        5 => column_strategy().prop_map(|col| Op::CreateIndex {
            column: col,
            index_type: ScalarIndexType::BTree,
        }),
        // Create Bitmap index (less frequent - expensive)
        5 => prop_oneof![
            Just(Column::Category),
            Just(Column::BoolCol),
        ].prop_map(|col| Op::CreateIndex {
            column: col,
            index_type: ScalarIndexType::Bitmap,
        }),
        // Compact (rare - very expensive)
        2 => Just(Op::Compact),
        // Optimize indices (rare - expensive)
        2 => Just(Op::OptimizeIndices),
        // Reload dataset (tests serialization of index state)
        5 => Just(Op::Reload),
        // MergeInsert (upsert) - mix of inserts and updates
        5 => (rows_strategy(5), 0usize..=3).prop_map(|(new_rows, num_updates)| {
            Op::MergeInsertData { new_rows, num_updates }
        }),
    ]
}

/// Strategy for generating a sequence of operations.
fn op_sequence_strategy(min_ops: usize, max_ops: usize) -> impl Strategy<Value = Vec<Op>> {
    proptest::collection::vec(op_strategy(), min_ops..=max_ops)
}

/// Strategy for stress testing with larger data volumes.
/// Creates many fragments with larger batches to stress index structures.
fn stress_op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        // Large writes - 100-500 rows per batch
        30 => (100usize..=500).prop_flat_map(|size| rows_strategy(size).prop_map(Op::WriteData)),
        // Medium writes - 20-100 rows (creates more fragments)
        30 => (20usize..=100).prop_flat_map(|size| rows_strategy(size).prop_map(Op::WriteData)),
        // Small writes - 5-20 rows (fragment proliferation)
        20 => (5usize..=20).prop_flat_map(|size| rows_strategy(size).prop_map(Op::WriteData)),
        // Delete operations
        10 => delete_predicate_strategy().prop_map(Op::Delete),
        // Create BTree index
        3 => column_strategy().prop_map(|col| Op::CreateIndex {
            column: col,
            index_type: ScalarIndexType::BTree,
        }),
        // Create Bitmap index
        3 => prop_oneof![
            Just(Column::Category),
            Just(Column::BoolCol),
        ].prop_map(|col| Op::CreateIndex {
            column: col,
            index_type: ScalarIndexType::Bitmap,
        }),
        // Compact (important for stress testing - merges fragments)
        2 => Just(Op::Compact),
        // Optimize indices
        1 => Just(Op::OptimizeIndices),
        // Reload
        1 => Just(Op::Reload),
        // MergeInsert with larger batches
        5 => ((50usize..=200).prop_flat_map(|size| rows_strategy(size)), 5usize..=20)
            .prop_map(|(new_rows, num_updates)| Op::MergeInsertData { new_rows, num_updates }),
    ]
}

/// Strategy for generating a stress test sequence.
/// Uses larger batches and more operations to create significant data volume.
fn stress_sequence_strategy(min_ops: usize, max_ops: usize) -> impl Strategy<Value = Vec<Op>> {
    proptest::collection::vec(stress_op_strategy(), min_ops..=max_ops)
}

// ============================================================================
// Test Execution
// ============================================================================

/// Error type for test sequence failures.
#[derive(Debug)]
struct TestSequenceError(String);

impl std::fmt::Display for TestSequenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for TestSequenceError {}

/// Execute a sequence of operations and verify invariants.
/// With incremental=true, verifies after each operation (slower but catches bugs earlier).
async fn run_test_sequence_impl(ops: Vec<Op>, incremental: bool) -> Result<(), TestSequenceError> {
    // Use file:// URI so Reload operations work (memory:// doesn't persist across Dataset instances)
    let temp_dir = tempfile::tempdir()
        .map_err(|e| TestSequenceError(format!("Failed to create temp dir: {}", e)))?;
    let uri = format!("file://{}/test_dataset", temp_dir.path().display());

    // Create initial empty dataset
    let schema = test_schema();
    let empty_batch = RecordBatch::new_empty(schema.clone());
    let reader = RecordBatchIterator::new(vec![Ok(empty_batch)], schema);
    let mut dataset = Dataset::write(reader, &uri, None)
        .await
        .map_err(|e| TestSequenceError(format!("Failed to create dataset: {}", e)))?;

    let mut model = Model::new();

    // Track predicates we've used for comprehensive checking
    let mut predicates_used: Vec<Predicate> = Vec::new();

    // Standard predicates to check
    let standard_predicates: Vec<Predicate> = vec![
        Predicate::True, // Full scan - always check
        Predicate::Eq(Column::Category, Value::String("A".to_string())),
        Predicate::Eq(Column::BoolCol, Value::Bool(true)),
        Predicate::IsNull(Column::NullableInt),
    ];

    for (op_idx, op) in ops.into_iter().enumerate() {
        // For WriteData/MergeInsertData operations, we need to assign IDs
        let op = match op {
            Op::WriteData(row_data) => {
                let ids = model.allocate_ids(row_data.len());
                let rows: Vec<TestRow> = row_data
                    .into_iter()
                    .zip(ids)
                    .map(|(data, id)| TestRow::from_data(id, data))
                    .collect();
                Op::WriteRows(rows)
            }
            Op::MergeInsertData { new_rows, num_updates } => {
                // Generate rows for merge insert: mix of inserts and updates
                let mut rows = Vec::new();

                // Add new rows with fresh IDs (inserts)
                let new_ids = model.allocate_ids(new_rows.len());
                for (data, id) in new_rows.into_iter().zip(new_ids) {
                    rows.push(TestRow::from_data(id, data));
                }

                // Add updates for existing rows (pick random existing IDs)
                let existing_ids: Vec<i64> = model.rows.keys().copied().collect();
                if !existing_ids.is_empty() && num_updates > 0 {
                    // Use a simple deterministic selection based on op_idx
                    for i in 0..num_updates.min(existing_ids.len()) {
                        let idx = (op_idx + i) % existing_ids.len();
                        let existing_id = existing_ids[idx];
                        // Create updated row data with the existing ID
                        let updated = RowData {
                            int_col: (op_idx as i32 * 100 + i as i32) % 1000,
                            float_col: (op_idx as f64 + i as f64) * 0.1,
                            string_col: format!("updated_{}", op_idx),
                            category: ["A", "B", "C", "D", "E"][i % 5].to_string(),
                            bool_col: i % 2 == 0,
                            nullable_int: if i % 3 == 0 { None } else { Some((op_idx * 10 + i) as i32) },
                        };
                        rows.push(TestRow::from_data(existing_id, updated));
                    }
                }

                Op::MergeInsert(rows)
            }
            Op::Delete(ref pred) => {
                predicates_used.push(pred.clone());
                op
            }
            other => other,
        };

        let op_desc = format!("{:?}", op);

        // Apply to model
        model.apply(&op);

        // Handle Reload specially - close and reopen the dataset
        if matches!(op, Op::Reload) {
            drop(dataset);
            dataset = Dataset::open(&uri)
                .await
                .map_err(|e| TestSequenceError(format!(
                    "Op {}: Failed to reload dataset: {}",
                    op_idx, e
                )))?;
        } else {
            // Apply to dataset
            apply_op(&mut dataset, &op)
                .await
                .map_err(|e| TestSequenceError(format!(
                    "Op {}: Failed to apply {}: {}",
                    op_idx, op_desc, e
                )))?;
        }

        // Incremental verification after each operation
        if incremental {
            check_invariants(&dataset, &model, &standard_predicates)
                .await
                .map_err(|e| TestSequenceError(format!(
                    "Op {}: Invariant violation after {}: {}",
                    op_idx, op_desc, e
                )))?;
        }
    }

    // Final verification with all predicates
    let all_predicates: Vec<Predicate> = standard_predicates
        .into_iter()
        .chain(predicates_used.into_iter().take(2))
        .collect();

    check_invariants(&dataset, &model, &all_predicates)
        .await
        .map_err(|e| TestSequenceError(format!("Final invariant violation: {}", e)))?;

    Ok(())
}

/// Execute a sequence of operations and verify invariants at the end.
async fn run_test_sequence(ops: Vec<Op>) -> Result<(), TestSequenceError> {
    run_test_sequence_impl(ops, false).await
}

/// Execute a sequence of operations with incremental verification after each op.
/// This is slower but catches the exact operation that causes a bug.
async fn run_test_sequence_incremental(ops: Vec<Op>) -> Result<(), TestSequenceError> {
    run_test_sequence_impl(ops, true).await
}

// ============================================================================
// Tests
// ============================================================================

/// Main generative test for scalar index integrity.
///
/// Runs many random operation sequences in parallel and verifies invariants.
#[tokio::test(flavor = "multi_thread")]
async fn test_scalar_index_integrity() {
    use proptest::test_runner::{TestRunner, Config};
    use proptest::strategy::ValueTree;
    use std::sync::atomic::{AtomicU32, AtomicBool, Ordering};
    use std::time::Instant;
    use tokio::sync::Semaphore;

    let total_cases: u32 = 1_000;
    let parallelism = 128; // High parallelism since ops are I/O bound
    let report_interval = 100;

    let completed = Arc::new(AtomicU32::new(0));
    let failed = Arc::new(AtomicBool::new(false));
    let semaphore = Arc::new(Semaphore::new(parallelism));
    let start = Instant::now();

    // Generate and run in batches to balance parallelism with memory usage
    let batch_size = 1000usize;
    let mut runner = TestRunner::new(Config {
        cases: total_cases,
        ..Config::default()
    });
    let strategy = op_sequence_strategy(2, 8); // Shorter sequences for speed

    for batch_start in (0..total_cases).step_by(batch_size) {
        if failed.load(Ordering::Relaxed) {
            break;
        }

        let batch_end = (batch_start + batch_size as u32).min(total_cases);
        let mut batch_handles = Vec::with_capacity(batch_size);

        for i in batch_start..batch_end {
            let case = strategy.new_tree(&mut runner).unwrap();
            let ops = case.current();

            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let completed = completed.clone();
            let failed = failed.clone();

            batch_handles.push(tokio::spawn(async move {
                let _permit = permit;

                if failed.load(Ordering::Relaxed) {
                    return;
                }

                let result = run_test_sequence(ops).await;

                let count = completed.fetch_add(1, Ordering::Relaxed) + 1;
                if count % report_interval == 0 {
                    let elapsed = start.elapsed();
                    let rate = count as f64 / elapsed.as_secs_f64();
                    let remaining = (total_cases - count) as f64 / rate;
                    eprintln!(
                        "[{:>7}/{} ({:>5.1}%)] {:.0} cases/sec, ETA: {:.0}s",
                        count, total_cases,
                        count as f64 / total_cases as f64 * 100.0,
                        rate, remaining
                    );
                }

                if let Err(e) = result {
                    failed.store(true, Ordering::Relaxed);
                    eprintln!("Case {} failed: {}", i, e);
                }
            }));
        }

        // Wait for batch to complete before generating more
        for handle in batch_handles {
            let _ = handle.await;
        }
    }

    let elapsed = start.elapsed();
    eprintln!(
        "Completed {} cases in {:.1}s ({:.0} cases/sec)",
        completed.load(Ordering::Relaxed),
        elapsed.as_secs_f64(),
        completed.load(Ordering::Relaxed) as f64 / elapsed.as_secs_f64()
    );

    if failed.load(Ordering::Relaxed) {
        panic!("One or more test cases failed");
    }
}

/// High-value scenario: Write -> CreateIndex -> Delete -> Query
/// This catches bugs where indexes return deleted rows.
#[tokio::test]
async fn test_index_after_delete() {
    let ops = vec![
        // Write initial data
        Op::WriteRows(vec![
            TestRow { id: 0, int_col: 1, float_col: 1.0, string_col: "a".to_string(), category: "A".to_string(), bool_col: true, nullable_int: Some(1) },
            TestRow { id: 1, int_col: 2, float_col: 2.0, string_col: "b".to_string(), category: "A".to_string(), bool_col: false, nullable_int: Some(2) },
            TestRow { id: 2, int_col: 3, float_col: 3.0, string_col: "c".to_string(), category: "B".to_string(), bool_col: true, nullable_int: None },
        ]),
        // Create indexes
        Op::CreateIndex { column: Column::Category, index_type: ScalarIndexType::BTree },
        Op::CreateIndex { column: Column::BoolCol, index_type: ScalarIndexType::Bitmap },
        // Delete some rows
        Op::Delete(Predicate::Eq(Column::Category, Value::String("A".to_string()))),
    ];

    run_test_sequence(ops).await.unwrap();
}

/// High-value scenario: CreateIndex -> Write -> Query
/// This catches bugs where new data isn't visible through the index.
#[tokio::test]
async fn test_write_after_index() {
    let ops = vec![
        // Write initial data
        Op::WriteRows(vec![
            TestRow { id: 0, int_col: 1, float_col: 1.0, string_col: "a".to_string(), category: "A".to_string(), bool_col: true, nullable_int: Some(1) },
        ]),
        // Create index
        Op::CreateIndex { column: Column::Category, index_type: ScalarIndexType::BTree },
        // Write more data
        Op::WriteRows(vec![
            TestRow { id: 1, int_col: 2, float_col: 2.0, string_col: "b".to_string(), category: "A".to_string(), bool_col: false, nullable_int: Some(2) },
            TestRow { id: 2, int_col: 3, float_col: 3.0, string_col: "c".to_string(), category: "B".to_string(), bool_col: true, nullable_int: None },
        ]),
    ];

    run_test_sequence(ops).await.unwrap();
}

/// High-value scenario: Write -> CreateIndex -> Compact -> Query
/// This catches bugs where compaction invalidates index pointers.
#[tokio::test]
async fn test_compact_after_index() {
    let ops = vec![
        // Write data in multiple batches to create fragments
        Op::WriteRows(vec![
            TestRow { id: 0, int_col: 1, float_col: 1.0, string_col: "a".to_string(), category: "A".to_string(), bool_col: true, nullable_int: Some(1) },
        ]),
        Op::WriteRows(vec![
            TestRow { id: 1, int_col: 2, float_col: 2.0, string_col: "b".to_string(), category: "A".to_string(), bool_col: false, nullable_int: Some(2) },
        ]),
        Op::WriteRows(vec![
            TestRow { id: 2, int_col: 3, float_col: 3.0, string_col: "c".to_string(), category: "B".to_string(), bool_col: true, nullable_int: None },
        ]),
        // Create index
        Op::CreateIndex { column: Column::Category, index_type: ScalarIndexType::BTree },
        // Compact
        Op::Compact,
    ];

    run_test_sequence(ops).await.unwrap();
}

/// High-value scenario: Multiple index types on different columns.
#[tokio::test]
async fn test_multiple_index_types() {
    let ops = vec![
        Op::WriteRows(vec![
            TestRow { id: 0, int_col: 1, float_col: 1.0, string_col: "a".to_string(), category: "A".to_string(), bool_col: true, nullable_int: Some(1) },
            TestRow { id: 1, int_col: 2, float_col: 2.0, string_col: "b".to_string(), category: "B".to_string(), bool_col: false, nullable_int: Some(2) },
            TestRow { id: 2, int_col: 3, float_col: 3.0, string_col: "c".to_string(), category: "C".to_string(), bool_col: true, nullable_int: None },
            TestRow { id: 3, int_col: 4, float_col: 4.0, string_col: "d".to_string(), category: "A".to_string(), bool_col: false, nullable_int: Some(4) },
        ]),
        // Create BTree indexes on numeric/string columns
        Op::CreateIndex { column: Column::IntCol, index_type: ScalarIndexType::BTree },
        Op::CreateIndex { column: Column::StringCol, index_type: ScalarIndexType::BTree },
        // Create Bitmap indexes on low-cardinality columns
        Op::CreateIndex { column: Column::Category, index_type: ScalarIndexType::Bitmap },
        Op::CreateIndex { column: Column::BoolCol, index_type: ScalarIndexType::Bitmap },
        // Write more data
        Op::WriteRows(vec![
            TestRow { id: 4, int_col: 5, float_col: 5.0, string_col: "e".to_string(), category: "D".to_string(), bool_col: true, nullable_int: Some(5) },
        ]),
        // Delete some data
        Op::Delete(Predicate::Eq(Column::Category, Value::String("A".to_string()))),
        // Optimize indices
        Op::OptimizeIndices,
    ];

    run_test_sequence(ops).await.unwrap();
}

/// Test NULL handling with indexes.
#[tokio::test]
async fn test_null_handling() {
    let ops = vec![
        Op::WriteRows(vec![
            TestRow { id: 0, int_col: 1, float_col: 1.0, string_col: "a".to_string(), category: "A".to_string(), bool_col: true, nullable_int: Some(1) },
            TestRow { id: 1, int_col: 2, float_col: 2.0, string_col: "b".to_string(), category: "B".to_string(), bool_col: false, nullable_int: None },
            TestRow { id: 2, int_col: 3, float_col: 3.0, string_col: "c".to_string(), category: "C".to_string(), bool_col: true, nullable_int: Some(3) },
            TestRow { id: 3, int_col: 4, float_col: 4.0, string_col: "d".to_string(), category: "D".to_string(), bool_col: false, nullable_int: None },
        ]),
        Op::CreateIndex { column: Column::NullableInt, index_type: ScalarIndexType::BTree },
        // Delete rows with NULL
        Op::Delete(Predicate::IsNull(Column::NullableInt)),
    ];

    run_test_sequence(ops).await.unwrap();
}

/// Test that index state survives dataset reload.
/// This catches bugs in index serialization/deserialization.
#[tokio::test]
async fn test_index_survives_reload() {
    let ops = vec![
        // Write initial data
        Op::WriteRows(vec![
            TestRow { id: 0, int_col: 1, float_col: 1.0, string_col: "a".to_string(), category: "A".to_string(), bool_col: true, nullable_int: Some(1) },
            TestRow { id: 1, int_col: 2, float_col: 2.0, string_col: "b".to_string(), category: "A".to_string(), bool_col: false, nullable_int: Some(2) },
            TestRow { id: 2, int_col: 3, float_col: 3.0, string_col: "c".to_string(), category: "B".to_string(), bool_col: true, nullable_int: None },
        ]),
        // Create indexes
        Op::CreateIndex { column: Column::Category, index_type: ScalarIndexType::BTree },
        Op::CreateIndex { column: Column::BoolCol, index_type: ScalarIndexType::Bitmap },
        // Reload the dataset - index state should persist
        Op::Reload,
        // Write more data after reload
        Op::WriteRows(vec![
            TestRow { id: 3, int_col: 4, float_col: 4.0, string_col: "d".to_string(), category: "C".to_string(), bool_col: false, nullable_int: Some(4) },
        ]),
        // Reload again
        Op::Reload,
        // Delete some rows
        Op::Delete(Predicate::Eq(Column::Category, Value::String("A".to_string()))),
        // Final reload
        Op::Reload,
    ];

    run_test_sequence(ops).await.unwrap();
}

/// Test reload with incremental verification.
/// Verifies index correctness after each reload.
#[tokio::test]
async fn test_reload_incremental() {
    let ops = vec![
        Op::WriteRows(vec![
            TestRow { id: 0, int_col: 10, float_col: 1.0, string_col: "x".to_string(), category: "A".to_string(), bool_col: true, nullable_int: Some(100) },
            TestRow { id: 1, int_col: 20, float_col: 2.0, string_col: "y".to_string(), category: "B".to_string(), bool_col: false, nullable_int: None },
        ]),
        Op::CreateIndex { column: Column::IntCol, index_type: ScalarIndexType::BTree },
        Op::Reload,
        Op::WriteRows(vec![
            TestRow { id: 2, int_col: 30, float_col: 3.0, string_col: "z".to_string(), category: "A".to_string(), bool_col: true, nullable_int: Some(200) },
        ]),
        Op::CreateIndex { column: Column::Category, index_type: ScalarIndexType::Bitmap },
        Op::Reload,
        Op::Delete(Predicate::Lt(Column::IntCol, Value::Int32(25))),
        Op::Reload,
        Op::Compact,
        Op::Reload,
        Op::OptimizeIndices,
        Op::Reload,
    ];

    run_test_sequence_incremental(ops).await.unwrap();
}

// ============================================================================
// Concurrent Testing
// ============================================================================

/// Operations that can be performed concurrently by different "threads".
/// Inspired by DuckDB's concurrentloop and CockroachDB's KV Nemesis.
#[derive(Debug, Clone)]
pub enum ConcurrentOp {
    /// Read operation - scan with optional filter.
    /// Readers should always see a consistent snapshot.
    Read { filter: Option<Predicate> },

    /// Write new rows.
    Append(Vec<RowData>),

    /// Delete rows matching a predicate.
    Delete(Predicate),

    /// Create an index.
    CreateIndex { column: Column, index_type: ScalarIndexType },

    /// Compact files.
    Compact,

    /// Optimize indices.
    OptimizeIndices,
}

/// Result of a concurrent operation.
#[derive(Debug)]
pub enum ConcurrentOpResult {
    /// Read succeeded, captured row IDs for verification.
    ReadSuccess { row_ids: Vec<i64> },

    /// Write succeeded.
    WriteSuccess,

    /// Operation failed due to conflict (expected in concurrent scenarios).
    Conflict(String),

    /// Operation failed unexpectedly.
    Error(String),
}

/// Specification for a concurrent test scenario.
/// Each "thread" has a sequence of operations to perform.
#[derive(Debug, Clone)]
pub struct ConcurrentScenario {
    /// Operations per thread. threads[i] = list of ops for thread i.
    pub threads: Vec<Vec<ConcurrentOp>>,
}

/// State shared between concurrent test threads.
struct SharedState {
    /// URI of the dataset.
    uri: String,
    /// Atomic counter for assigning unique IDs.
    next_id: std::sync::atomic::AtomicI64,
}

impl SharedState {
    fn new(uri: String) -> Self {
        Self {
            uri,
            next_id: std::sync::atomic::AtomicI64::new(0),
        }
    }

    fn allocate_ids(&self, count: usize) -> Vec<i64> {
        use std::sync::atomic::Ordering;
        let start = self.next_id.fetch_add(count as i64, Ordering::SeqCst);
        (start..start + count as i64).collect()
    }
}

/// Verify that a read is internally consistent within its snapshot.
/// This checks that filtered scans return a subset of the full scan that
/// matches the predicate, using the same dataset handle (same snapshot).
async fn verify_read_consistency(
    dataset: &Dataset,
    filter: &Option<Predicate>,
) -> Result<Vec<i64>, String> {
    // First, get all rows from the snapshot
    let all_rows = scan_to_rows(dataset, None)
        .await
        .map_err(|e| format!("Full scan failed: {}", e))?;

    // Check for duplicate IDs within the snapshot (corruption)
    let mut seen_ids = std::collections::HashSet::new();
    for row in &all_rows {
        if !seen_ids.insert(row.id) {
            return Err(format!(
                "Duplicate row ID {} found in snapshot. Data corruption detected.",
                row.id
            ));
        }
    }

    // If there's a filter, verify filtered results match
    if let Some(pred) = filter {
        let filter_sql = pred.to_sql();
        let filtered_rows = scan_to_rows(dataset, Some(&filter_sql))
            .await
            .map_err(|e| format!("Filtered scan failed for '{}': {}", filter_sql, e))?;

        // Compute expected filtered rows from full scan
        let expected_ids: std::collections::HashSet<i64> = all_rows
            .iter()
            .filter(|r| r.matches(pred))
            .map(|r| r.id)
            .collect();

        let actual_ids: std::collections::HashSet<i64> =
            filtered_rows.iter().map(|r| r.id).collect();

        if expected_ids != actual_ids {
            let missing: Vec<_> = expected_ids.difference(&actual_ids).collect();
            let extra: Vec<_> = actual_ids.difference(&expected_ids).collect();
            return Err(format!(
                "Filtered scan inconsistent for '{}': missing {:?}, extra {:?}",
                filter_sql, missing, extra
            ));
        }

        Ok(filtered_rows.iter().map(|r| r.id).collect())
    } else {
        Ok(all_rows.iter().map(|r| r.id).collect())
    }
}

/// Execute a single concurrent operation with optional incremental verification.
/// Returns the result.
async fn execute_concurrent_op(
    state: &SharedState,
    op: &ConcurrentOp,
    verify_reads: bool,
) -> ConcurrentOpResult {
    // Open a fresh dataset handle for this operation (simulates different connections)
    let dataset_result = Dataset::open(&state.uri).await;
    let mut dataset = match dataset_result {
        Ok(ds) => ds,
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("conflict") || msg.contains("Conflict") {
                return ConcurrentOpResult::Conflict(msg);
            }
            return ConcurrentOpResult::Error(format!("Failed to open dataset: {}", e));
        }
    };

    match op {
        ConcurrentOp::Read { filter } => {
            if verify_reads {
                // Full snapshot-consistent verification
                match verify_read_consistency(&dataset, filter).await {
                    Ok(row_ids) => ConcurrentOpResult::ReadSuccess { row_ids },
                    Err(e) => ConcurrentOpResult::Error(e),
                }
            } else {
                // Simple read without verification
                let filter_sql = filter.as_ref().map(|p| p.to_sql());
                match scan_to_rows(&dataset, filter_sql.as_deref()).await {
                    Ok(rows) => ConcurrentOpResult::ReadSuccess {
                        row_ids: rows.iter().map(|r| r.id).collect(),
                    },
                    Err(e) => ConcurrentOpResult::Error(format!("Read failed: {}", e)),
                }
            }
        }

        ConcurrentOp::Append(row_data) => {
            if row_data.is_empty() {
                return ConcurrentOpResult::WriteSuccess;
            }

            let ids = state.allocate_ids(row_data.len());
            let rows: Vec<TestRow> = row_data
                .iter()
                .cloned()
                .zip(ids)
                .map(|(data, id)| TestRow::from_data(id, data))
                .collect();

            let batch = rows_to_batch(&rows);
            let schema = batch.schema();
            let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);

            match dataset.append(reader, None).await {
                Ok(_) => ConcurrentOpResult::WriteSuccess,
                Err(e) => {
                    let msg = e.to_string();
                    if msg.contains("conflict") || msg.contains("Conflict") ||
                       msg.contains("CommitConflict") || msg.contains("version") {
                        ConcurrentOpResult::Conflict(msg)
                    } else {
                        ConcurrentOpResult::Error(format!("Append failed: {}", e))
                    }
                }
            }
        }

        ConcurrentOp::Delete(pred) => {
            let sql = pred.to_sql();
            match dataset.delete(&sql).await {
                Ok(_) => ConcurrentOpResult::WriteSuccess,
                Err(e) => {
                    let msg = e.to_string();
                    if msg.contains("conflict") || msg.contains("Conflict") ||
                       msg.contains("CommitConflict") || msg.contains("version") {
                        ConcurrentOpResult::Conflict(msg)
                    } else {
                        ConcurrentOpResult::Error(format!("Delete failed: {}", e))
                    }
                }
            }
        }

        ConcurrentOp::CreateIndex { column, index_type } => {
            let index_type_lance = match index_type {
                ScalarIndexType::BTree => IndexType::BTree,
                ScalarIndexType::Bitmap => IndexType::Bitmap,
            };

            match dataset
                .create_index(
                    &[column.name()],
                    index_type_lance,
                    None,
                    &ScalarIndexParams::default(),
                    true,
                )
                .await
            {
                Ok(_) => ConcurrentOpResult::WriteSuccess,
                Err(e) => {
                    let msg = e.to_string();
                    if msg.contains("conflict") || msg.contains("Conflict") ||
                       msg.contains("CommitConflict") || msg.contains("version") {
                        ConcurrentOpResult::Conflict(msg)
                    } else {
                        ConcurrentOpResult::Error(format!("CreateIndex failed: {}", e))
                    }
                }
            }
        }

        ConcurrentOp::Compact => {
            match compact_files(&mut dataset, Default::default(), None).await {
                Ok(_) => ConcurrentOpResult::WriteSuccess,
                Err(e) => {
                    let msg = e.to_string();
                    if msg.contains("conflict") || msg.contains("Conflict") ||
                       msg.contains("CommitConflict") || msg.contains("version") {
                        ConcurrentOpResult::Conflict(msg)
                    } else {
                        ConcurrentOpResult::Error(format!("Compact failed: {}", e))
                    }
                }
            }
        }

        ConcurrentOp::OptimizeIndices => {
            match dataset.optimize_indices(&Default::default()).await {
                Ok(_) => ConcurrentOpResult::WriteSuccess,
                Err(e) => {
                    let msg = e.to_string();
                    if msg.contains("conflict") || msg.contains("Conflict") ||
                       msg.contains("CommitConflict") || msg.contains("version") {
                        ConcurrentOpResult::Conflict(msg)
                    } else {
                        ConcurrentOpResult::Error(format!("OptimizeIndices failed: {}", e))
                    }
                }
            }
        }
    }
}

/// Run a concurrent test scenario.
///
/// This spawns multiple tasks that execute operations on the same dataset
/// simultaneously, similar to DuckDB's `concurrentloop` pattern.
///
/// Key invariants to verify after all operations complete:
/// 1. Dataset is readable and consistent
/// 2. All committed writes are visible
/// 3. No duplicate or phantom rows
///
/// If `incremental_verify` is true, each read operation also verifies that
/// filtered scans are consistent with full scans within the same snapshot.
async fn run_concurrent_scenario_impl(
    scenario: ConcurrentScenario,
    incremental_verify: bool,
) -> Result<(), TestSequenceError> {
    use std::sync::atomic::Ordering;

    // Create dataset with file:// URI for persistence across opens
    let temp_dir = tempfile::tempdir()
        .map_err(|e| TestSequenceError(format!("Failed to create temp dir: {}", e)))?;
    let uri = format!("file://{}/test_dataset", temp_dir.path().display());

    // Create initial empty dataset
    let schema = test_schema();
    let empty_batch = RecordBatch::new_empty(schema.clone());
    let reader = RecordBatchIterator::new(vec![Ok(empty_batch)], schema);
    Dataset::write(reader, &uri, None)
        .await
        .map_err(|e| TestSequenceError(format!("Failed to create dataset: {}", e)))?;

    let state = Arc::new(SharedState::new(uri.clone()));

    // Track results from all threads
    let all_results: Arc<tokio::sync::Mutex<Vec<Vec<ConcurrentOpResult>>>> =
        Arc::new(tokio::sync::Mutex::new(Vec::new()));

    // Spawn all threads simultaneously
    let mut handles = Vec::new();

    for (thread_id, ops) in scenario.threads.into_iter().enumerate() {
        let state = state.clone();
        let all_results = all_results.clone();

        handles.push(tokio::spawn(async move {
            let mut thread_results = Vec::new();

            for op in ops {
                let result = execute_concurrent_op(&state, &op, incremental_verify).await;

                // Log conflicts for debugging but don't fail
                if let ConcurrentOpResult::Conflict(ref msg) = result {
                    eprintln!("Thread {} conflict on {:?}: {}", thread_id, op, msg);
                }

                // Fail fast on unexpected errors
                if let ConcurrentOpResult::Error(ref msg) = result {
                    eprintln!("Thread {} error on {:?}: {}", thread_id, op, msg);
                }

                thread_results.push(result);
            }

            all_results.lock().await.push(thread_results);
        }));
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.await.map_err(|e| TestSequenceError(format!("Task panicked: {}", e)))?;
    }

    // Verify final state is consistent
    let final_dataset = Dataset::open(&uri)
        .await
        .map_err(|e| TestSequenceError(format!("Failed to open final dataset: {}", e)))?;

    // Basic consistency check: we can read all rows and they have unique IDs
    let all_rows = scan_to_rows(&final_dataset, None)
        .await
        .map_err(|e| TestSequenceError(format!("Failed to scan final dataset: {}", e)))?;

    // Check for duplicate IDs (would indicate corruption)
    let mut seen_ids = std::collections::HashSet::new();
    for row in &all_rows {
        if !seen_ids.insert(row.id) {
            return Err(TestSequenceError(format!(
                "Duplicate row ID detected: {}. This indicates index/data corruption.",
                row.id
            )));
        }
    }

    // Check ID range is valid (all IDs should be < next_id)
    let max_allocated = state.next_id.load(Ordering::SeqCst);
    for row in &all_rows {
        if row.id >= max_allocated {
            return Err(TestSequenceError(format!(
                "Row ID {} >= max allocated {}. Phantom row detected.",
                row.id, max_allocated
            )));
        }
    }

    // Check that filtered scans are consistent with full scan
    let predicates = vec![
        Predicate::Eq(Column::Category, Value::String("A".to_string())),
        Predicate::Eq(Column::BoolCol, Value::Bool(true)),
        Predicate::IsNull(Column::NullableInt),
    ];

    for pred in predicates {
        let filtered = scan_to_rows(&final_dataset, Some(&pred.to_sql()))
            .await
            .map_err(|e| TestSequenceError(format!("Failed filtered scan: {}", e)))?;

        // Verify filtered results are subset of full scan with matching predicate
        let expected: Vec<i64> = all_rows
            .iter()
            .filter(|r| r.matches(&pred))
            .map(|r| r.id)
            .collect();

        let actual: Vec<i64> = filtered.iter().map(|r| r.id).collect();

        let expected_set: std::collections::HashSet<_> = expected.iter().collect();
        let actual_set: std::collections::HashSet<_> = actual.iter().collect();

        if expected_set != actual_set {
            return Err(TestSequenceError(format!(
                "Filtered scan mismatch for '{}': expected {:?}, got {:?}",
                pred.to_sql(),
                expected,
                actual
            )));
        }
    }

    // Check results for any unexpected errors
    let results = all_results.lock().await;
    for (thread_id, thread_results) in results.iter().enumerate() {
        for (op_id, result) in thread_results.iter().enumerate() {
            if let ConcurrentOpResult::Error(msg) = result {
                return Err(TestSequenceError(format!(
                    "Thread {} op {} failed: {}",
                    thread_id, op_id, msg
                )));
            }
        }
    }

    Ok(())
}

/// Run a concurrent scenario without incremental verification (faster).
async fn run_concurrent_scenario(scenario: ConcurrentScenario) -> Result<(), TestSequenceError> {
    run_concurrent_scenario_impl(scenario, false).await
}

/// Run a concurrent scenario with incremental verification on each read.
/// This is slower but catches bugs where filtered scans don't match full scans.
#[allow(dead_code)]
async fn run_concurrent_scenario_incremental(
    scenario: ConcurrentScenario,
) -> Result<(), TestSequenceError> {
    run_concurrent_scenario_impl(scenario, true).await
}

// ============================================================================
// Concurrent Test Strategies
// ============================================================================

/// Strategy for generating concurrent read operations.
fn concurrent_read_strategy() -> impl Strategy<Value = ConcurrentOp> {
    prop_oneof![
        Just(ConcurrentOp::Read { filter: None }),
        category_strategy().prop_map(|cat| ConcurrentOp::Read {
            filter: Some(Predicate::Eq(Column::Category, Value::String(cat))),
        }),
        any::<bool>().prop_map(|b| ConcurrentOp::Read {
            filter: Some(Predicate::Eq(Column::BoolCol, Value::Bool(b))),
        }),
        Just(ConcurrentOp::Read {
            filter: Some(Predicate::IsNull(Column::NullableInt)),
        }),
    ]
}

/// Strategy for generating concurrent write operations.
fn concurrent_write_strategy() -> impl Strategy<Value = ConcurrentOp> {
    prop_oneof![
        // Small appends
        rows_strategy(5).prop_map(ConcurrentOp::Append),
        // Deletes
        delete_predicate_strategy().prop_map(ConcurrentOp::Delete),
    ]
}

/// Strategy for generating concurrent index operations.
fn concurrent_index_strategy() -> impl Strategy<Value = ConcurrentOp> {
    prop_oneof![
        column_strategy().prop_map(|col| ConcurrentOp::CreateIndex {
            column: col,
            index_type: ScalarIndexType::BTree,
        }),
        prop_oneof![Just(Column::Category), Just(Column::BoolCol),]
            .prop_map(|col| ConcurrentOp::CreateIndex {
                column: col,
                index_type: ScalarIndexType::Bitmap,
            }),
    ]
}

/// Strategy for a thread that primarily reads.
fn reader_thread_strategy(num_ops: usize) -> impl Strategy<Value = Vec<ConcurrentOp>> {
    proptest::collection::vec(concurrent_read_strategy(), num_ops..=num_ops)
}

/// Strategy for a thread that primarily writes.
fn writer_thread_strategy(num_ops: usize) -> impl Strategy<Value = Vec<ConcurrentOp>> {
    proptest::collection::vec(concurrent_write_strategy(), num_ops..=num_ops)
}

/// Strategy for a thread that does mixed operations.
fn mixed_thread_strategy(num_ops: usize) -> impl Strategy<Value = Vec<ConcurrentOp>> {
    proptest::collection::vec(
        prop_oneof![
            50 => concurrent_read_strategy(),
            30 => concurrent_write_strategy(),
            10 => concurrent_index_strategy(),
            5 => Just(ConcurrentOp::Compact),
            5 => Just(ConcurrentOp::OptimizeIndices),
        ],
        num_ops..=num_ops,
    )
}

/// Strategy for generating a concurrent scenario with readers and writers.
fn reader_writer_scenario_strategy(
    num_readers: usize,
    num_writers: usize,
    ops_per_thread: usize,
) -> impl Strategy<Value = ConcurrentScenario> {
    let readers = proptest::collection::vec(
        reader_thread_strategy(ops_per_thread),
        num_readers..=num_readers,
    );
    let writers = proptest::collection::vec(
        writer_thread_strategy(ops_per_thread),
        num_writers..=num_writers,
    );

    (readers, writers).prop_map(|(readers, writers)| {
        let mut threads = readers;
        threads.extend(writers);
        ConcurrentScenario { threads }
    })
}

/// Strategy for generating a fully mixed concurrent scenario.
fn mixed_concurrent_scenario_strategy(
    num_threads: usize,
    ops_per_thread: usize,
) -> impl Strategy<Value = ConcurrentScenario> {
    proptest::collection::vec(mixed_thread_strategy(ops_per_thread), num_threads..=num_threads)
        .prop_map(|threads| ConcurrentScenario { threads })
}

// ============================================================================
// Concurrent Tests
// ============================================================================

/// Test concurrent reads and writes.
/// Inspired by DuckDB's concurrent_checkpoint_deletes_index.test_slow
#[tokio::test(flavor = "multi_thread")]
async fn test_concurrent_readers_writers() {
    use proptest::test_runner::{Config, TestRunner};
    use proptest::strategy::ValueTree;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::time::Instant;

    let total_cases: u32 = 1000;
    let parallelism = 32;
    let report_interval = 100;

    let completed = Arc::new(AtomicU32::new(0));
    let failed = Arc::new(AtomicBool::new(false));
    let semaphore = Arc::new(tokio::sync::Semaphore::new(parallelism));
    let start = Instant::now();

    let mut runner = TestRunner::new(Config {
        cases: total_cases,
        ..Config::default()
    });

    // 4 readers, 2 writers, 3 ops each
    let strategy = reader_writer_scenario_strategy(4, 2, 3);

    let batch_size = 100usize;
    for batch_start in (0..total_cases).step_by(batch_size) {
        if failed.load(Ordering::Relaxed) {
            break;
        }

        let batch_end = (batch_start + batch_size as u32).min(total_cases);
        let mut batch_handles = Vec::with_capacity(batch_size);

        for i in batch_start..batch_end {
            let case = strategy.new_tree(&mut runner).unwrap();
            let scenario = case.current();

            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let completed = completed.clone();
            let failed = failed.clone();

            batch_handles.push(tokio::spawn(async move {
                let _permit = permit;

                if failed.load(Ordering::Relaxed) {
                    return;
                }

                let result = run_concurrent_scenario(scenario).await;

                let count = completed.fetch_add(1, Ordering::Relaxed) + 1;
                if count % report_interval == 0 {
                    let elapsed = start.elapsed();
                    let rate = count as f64 / elapsed.as_secs_f64();
                    eprintln!(
                        "[Concurrent R/W {:>5}/{} ({:>5.1}%)] {:.0} cases/sec",
                        count,
                        total_cases,
                        count as f64 / total_cases as f64 * 100.0,
                        rate
                    );
                }

                if let Err(e) = result {
                    failed.store(true, Ordering::Relaxed);
                    eprintln!("Concurrent case {} failed: {}", i, e);
                }
            }));
        }

        for handle in batch_handles {
            let _ = handle.await;
        }
    }

    let elapsed = start.elapsed();
    eprintln!(
        "Completed {} concurrent R/W cases in {:.1}s ({:.0} cases/sec)",
        completed.load(Ordering::Relaxed),
        elapsed.as_secs_f64(),
        completed.load(Ordering::Relaxed) as f64 / elapsed.as_secs_f64()
    );

    if failed.load(Ordering::Relaxed) {
        panic!("One or more concurrent test cases failed");
    }
}

/// Test concurrent mixed operations (reads, writes, index ops, compaction).
/// Inspired by CockroachDB's KV Nemesis.
#[tokio::test(flavor = "multi_thread")]
async fn test_concurrent_mixed_operations() {
    use proptest::test_runner::{Config, TestRunner};
    use proptest::strategy::ValueTree;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::time::Instant;

    let total_cases: u32 = 500;
    let parallelism = 16; // Lower parallelism due to more expensive ops
    let report_interval = 50;

    let completed = Arc::new(AtomicU32::new(0));
    let failed = Arc::new(AtomicBool::new(false));
    let semaphore = Arc::new(tokio::sync::Semaphore::new(parallelism));
    let start = Instant::now();

    let mut runner = TestRunner::new(Config {
        cases: total_cases,
        ..Config::default()
    });

    // 6 threads, 4 ops each, all doing mixed operations
    let strategy = mixed_concurrent_scenario_strategy(6, 4);

    let batch_size = 50usize;
    for batch_start in (0..total_cases).step_by(batch_size) {
        if failed.load(Ordering::Relaxed) {
            break;
        }

        let batch_end = (batch_start + batch_size as u32).min(total_cases);
        let mut batch_handles = Vec::with_capacity(batch_size);

        for i in batch_start..batch_end {
            let case = strategy.new_tree(&mut runner).unwrap();
            let scenario = case.current();

            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let completed = completed.clone();
            let failed = failed.clone();

            batch_handles.push(tokio::spawn(async move {
                let _permit = permit;

                if failed.load(Ordering::Relaxed) {
                    return;
                }

                let result = run_concurrent_scenario(scenario).await;

                let count = completed.fetch_add(1, Ordering::Relaxed) + 1;
                if count % report_interval == 0 {
                    let elapsed = start.elapsed();
                    let rate = count as f64 / elapsed.as_secs_f64();
                    eprintln!(
                        "[Concurrent Mixed {:>4}/{} ({:>5.1}%)] {:.0} cases/sec",
                        count,
                        total_cases,
                        count as f64 / total_cases as f64 * 100.0,
                        rate
                    );
                }

                if let Err(e) = result {
                    failed.store(true, Ordering::Relaxed);
                    eprintln!("Mixed concurrent case {} failed: {}", i, e);
                }
            }));
        }

        for handle in batch_handles {
            let _ = handle.await;
        }
    }

    let elapsed = start.elapsed();
    eprintln!(
        "Completed {} mixed concurrent cases in {:.1}s ({:.0} cases/sec)",
        completed.load(Ordering::Relaxed),
        elapsed.as_secs_f64(),
        completed.load(Ordering::Relaxed) as f64 / elapsed.as_secs_f64()
    );

    if failed.load(Ordering::Relaxed) {
        panic!("One or more mixed concurrent test cases failed");
    }
}

/// High-value scenario: Index creation during concurrent writes.
/// This is a known problematic pattern in many databases.
#[tokio::test]
async fn test_index_creation_during_writes() {
    let scenario = ConcurrentScenario {
        threads: vec![
            // Writer thread 1
            vec![
                ConcurrentOp::Append(vec![
                    RowData { int_col: 1, float_col: 1.0, string_col: "a".to_string(), category: "A".to_string(), bool_col: true, nullable_int: Some(1) },
                    RowData { int_col: 2, float_col: 2.0, string_col: "b".to_string(), category: "B".to_string(), bool_col: false, nullable_int: Some(2) },
                ]),
                ConcurrentOp::Append(vec![
                    RowData { int_col: 3, float_col: 3.0, string_col: "c".to_string(), category: "A".to_string(), bool_col: true, nullable_int: None },
                ]),
            ],
            // Writer thread 2
            vec![
                ConcurrentOp::Append(vec![
                    RowData { int_col: 4, float_col: 4.0, string_col: "d".to_string(), category: "C".to_string(), bool_col: false, nullable_int: Some(4) },
                ]),
                ConcurrentOp::Delete(Predicate::Eq(Column::Category, Value::String("B".to_string()))),
            ],
            // Index creation thread
            vec![
                ConcurrentOp::CreateIndex { column: Column::Category, index_type: ScalarIndexType::BTree },
                ConcurrentOp::CreateIndex { column: Column::BoolCol, index_type: ScalarIndexType::Bitmap },
            ],
            // Reader thread (should always see consistent state)
            vec![
                ConcurrentOp::Read { filter: None },
                ConcurrentOp::Read { filter: Some(Predicate::Eq(Column::Category, Value::String("A".to_string()))) },
                ConcurrentOp::Read { filter: Some(Predicate::Eq(Column::BoolCol, Value::Bool(true))) },
            ],
        ],
    };

    run_concurrent_scenario(scenario).await.unwrap();
}

/// High-value scenario: Compaction during concurrent reads and writes.
#[tokio::test]
async fn test_compaction_during_operations() {
    let scenario = ConcurrentScenario {
        threads: vec![
            // Writer thread - creates multiple fragments
            vec![
                ConcurrentOp::Append(vec![
                    RowData { int_col: 1, float_col: 1.0, string_col: "a".to_string(), category: "A".to_string(), bool_col: true, nullable_int: Some(1) },
                ]),
                ConcurrentOp::Append(vec![
                    RowData { int_col: 2, float_col: 2.0, string_col: "b".to_string(), category: "B".to_string(), bool_col: false, nullable_int: Some(2) },
                ]),
                ConcurrentOp::Append(vec![
                    RowData { int_col: 3, float_col: 3.0, string_col: "c".to_string(), category: "C".to_string(), bool_col: true, nullable_int: None },
                ]),
            ],
            // Compaction thread
            vec![
                ConcurrentOp::Compact,
            ],
            // Index + optimize thread
            vec![
                ConcurrentOp::CreateIndex { column: Column::IntCol, index_type: ScalarIndexType::BTree },
                ConcurrentOp::OptimizeIndices,
            ],
            // Reader thread
            vec![
                ConcurrentOp::Read { filter: None },
                ConcurrentOp::Read { filter: Some(Predicate::Lt(Column::IntCol, Value::Int32(3))) },
            ],
        ],
    };

    run_concurrent_scenario(scenario).await.unwrap();
}

/// Test many concurrent readers with a single writer.
/// Verifies MVCC snapshot isolation.
#[tokio::test]
async fn test_many_readers_one_writer() {
    let scenario = ConcurrentScenario {
        threads: vec![
            // Single writer doing multiple operations
            vec![
                ConcurrentOp::Append(vec![
                    RowData { int_col: 1, float_col: 1.0, string_col: "a".to_string(), category: "A".to_string(), bool_col: true, nullable_int: Some(1) },
                    RowData { int_col: 2, float_col: 2.0, string_col: "b".to_string(), category: "A".to_string(), bool_col: false, nullable_int: Some(2) },
                ]),
                ConcurrentOp::CreateIndex { column: Column::Category, index_type: ScalarIndexType::BTree },
                ConcurrentOp::Append(vec![
                    RowData { int_col: 3, float_col: 3.0, string_col: "c".to_string(), category: "B".to_string(), bool_col: true, nullable_int: None },
                ]),
                ConcurrentOp::Delete(Predicate::Eq(Column::BoolCol, Value::Bool(false))),
            ],
            // Reader 1
            vec![
                ConcurrentOp::Read { filter: None },
                ConcurrentOp::Read { filter: Some(Predicate::Eq(Column::Category, Value::String("A".to_string()))) },
            ],
            // Reader 2
            vec![
                ConcurrentOp::Read { filter: Some(Predicate::Eq(Column::BoolCol, Value::Bool(true))) },
                ConcurrentOp::Read { filter: None },
            ],
            // Reader 3
            vec![
                ConcurrentOp::Read { filter: Some(Predicate::IsNull(Column::NullableInt)) },
                ConcurrentOp::Read { filter: Some(Predicate::IsNotNull(Column::NullableInt)) },
            ],
            // Reader 4
            vec![
                ConcurrentOp::Read { filter: None },
                ConcurrentOp::Read { filter: Some(Predicate::Lt(Column::IntCol, Value::Int32(2))) },
            ],
            // Reader 5
            vec![
                ConcurrentOp::Read { filter: Some(Predicate::Gt(Column::IntCol, Value::Int32(1))) },
                ConcurrentOp::Read { filter: None },
            ],
        ],
    };

    run_concurrent_scenario(scenario).await.unwrap();
}

// ============================================================================
// Incremental Verification Tests
// ============================================================================

/// Sequential test with incremental verification after each operation.
/// This is slower but pinpoints exactly which operation causes a failure.
#[tokio::test(flavor = "multi_thread")]
async fn test_scalar_index_incremental_verification() {
    use proptest::strategy::ValueTree;
    use proptest::test_runner::{Config, TestRunner};
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::time::Instant;
    use tokio::sync::Semaphore;

    let total_cases: u32 = 500; // Fewer cases since each is more expensive
    let parallelism = 64;
    let report_interval = 50;

    let completed = Arc::new(AtomicU32::new(0));
    let failed = Arc::new(AtomicBool::new(false));
    let semaphore = Arc::new(Semaphore::new(parallelism));
    let start = Instant::now();

    let batch_size = 100usize;
    let mut runner = TestRunner::new(Config {
        cases: total_cases,
        ..Config::default()
    });
    let strategy = op_sequence_strategy(3, 10); // Longer sequences to test more states

    for batch_start in (0..total_cases).step_by(batch_size) {
        if failed.load(Ordering::Relaxed) {
            break;
        }

        let batch_end = (batch_start + batch_size as u32).min(total_cases);
        let mut batch_handles = Vec::with_capacity(batch_size);

        for i in batch_start..batch_end {
            let case = strategy.new_tree(&mut runner).unwrap();
            let ops = case.current();

            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let completed = completed.clone();
            let failed = failed.clone();

            batch_handles.push(tokio::spawn(async move {
                let _permit = permit;

                if failed.load(Ordering::Relaxed) {
                    return;
                }

                // Use incremental verification
                let result = run_test_sequence_incremental(ops).await;

                let count = completed.fetch_add(1, Ordering::Relaxed) + 1;
                if count % report_interval == 0 {
                    let elapsed = start.elapsed();
                    let rate = count as f64 / elapsed.as_secs_f64();
                    eprintln!(
                        "[Incremental {:>4}/{} ({:>5.1}%)] {:.0} cases/sec",
                        count,
                        total_cases,
                        count as f64 / total_cases as f64 * 100.0,
                        rate
                    );
                }

                if let Err(e) = result {
                    failed.store(true, Ordering::Relaxed);
                    eprintln!("Incremental case {} failed: {}", i, e);
                }
            }));
        }

        for handle in batch_handles {
            let _ = handle.await;
        }
    }

    let elapsed = start.elapsed();
    eprintln!(
        "Completed {} incremental cases in {:.1}s ({:.0} cases/sec)",
        completed.load(Ordering::Relaxed),
        elapsed.as_secs_f64(),
        completed.load(Ordering::Relaxed) as f64 / elapsed.as_secs_f64()
    );

    if failed.load(Ordering::Relaxed) {
        panic!("One or more incremental test cases failed");
    }
}

/// Concurrent test with incremental verification on each read.
/// Each read verifies that filtered results match the full scan within the same snapshot.
#[tokio::test(flavor = "multi_thread")]
async fn test_concurrent_incremental_verification() {
    use proptest::strategy::ValueTree;
    use proptest::test_runner::{Config, TestRunner};
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::time::Instant;

    let total_cases: u32 = 200; // Fewer cases - very expensive
    let parallelism = 16;
    let report_interval = 20;

    let completed = Arc::new(AtomicU32::new(0));
    let failed = Arc::new(AtomicBool::new(false));
    let semaphore = Arc::new(tokio::sync::Semaphore::new(parallelism));
    let start = Instant::now();

    let mut runner = TestRunner::new(Config {
        cases: total_cases,
        ..Config::default()
    });

    // Use reader-heavy scenarios for maximum verification coverage
    let strategy = reader_writer_scenario_strategy(6, 2, 4);

    let batch_size = 50usize;
    for batch_start in (0..total_cases).step_by(batch_size) {
        if failed.load(Ordering::Relaxed) {
            break;
        }

        let batch_end = (batch_start + batch_size as u32).min(total_cases);
        let mut batch_handles = Vec::with_capacity(batch_size);

        for i in batch_start..batch_end {
            let case = strategy.new_tree(&mut runner).unwrap();
            let scenario = case.current();

            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let completed = completed.clone();
            let failed = failed.clone();

            batch_handles.push(tokio::spawn(async move {
                let _permit = permit;

                if failed.load(Ordering::Relaxed) {
                    return;
                }

                // Use incremental verification for concurrent reads
                let result = run_concurrent_scenario_incremental(scenario).await;

                let count = completed.fetch_add(1, Ordering::Relaxed) + 1;
                if count % report_interval == 0 {
                    let elapsed = start.elapsed();
                    let rate = count as f64 / elapsed.as_secs_f64();
                    eprintln!(
                        "[Conc+Incr {:>4}/{} ({:>5.1}%)] {:.0} cases/sec",
                        count,
                        total_cases,
                        count as f64 / total_cases as f64 * 100.0,
                        rate
                    );
                }

                if let Err(e) = result {
                    failed.store(true, Ordering::Relaxed);
                    eprintln!("Concurrent incremental case {} failed: {}", i, e);
                }
            }));
        }

        for handle in batch_handles {
            let _ = handle.await;
        }
    }

    let elapsed = start.elapsed();
    eprintln!(
        "Completed {} concurrent incremental cases in {:.1}s ({:.0} cases/sec)",
        completed.load(Ordering::Relaxed),
        elapsed.as_secs_f64(),
        completed.load(Ordering::Relaxed) as f64 / elapsed.as_secs_f64()
    );

    if failed.load(Ordering::Relaxed) {
        panic!("One or more concurrent incremental test cases failed");
    }
}

// ============================================================================
// Stress Tests (Large Data Volumes)
// ============================================================================

/// Stress test with larger data volumes and more fragments.
/// Each test case creates 10-50k rows across 20-50 fragments.
/// Runs fewer cases since each is much more expensive.
#[tokio::test(flavor = "multi_thread")]
async fn test_stress_large_data() {
    use proptest::strategy::ValueTree;
    use proptest::test_runner::{Config, TestRunner};
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::time::Instant;
    use tokio::sync::Semaphore;

    let total_cases: u32 = 50; // Fewer cases - each has 10-50k rows
    let parallelism = 8; // Lower parallelism - each case is memory intensive
    let report_interval = 5;

    let completed = Arc::new(AtomicU32::new(0));
    let failed = Arc::new(AtomicBool::new(false));
    let semaphore = Arc::new(Semaphore::new(parallelism));
    let start = Instant::now();

    let batch_size = 10usize;
    let mut runner = TestRunner::new(Config {
        cases: total_cases,
        ..Config::default()
    });

    // 20-40 operations with large batch sizes
    let strategy = stress_sequence_strategy(20, 40);

    for batch_start in (0..total_cases).step_by(batch_size) {
        if failed.load(Ordering::Relaxed) {
            break;
        }

        let batch_end = (batch_start + batch_size as u32).min(total_cases);
        let mut batch_handles = Vec::with_capacity(batch_size);

        for i in batch_start..batch_end {
            let case = strategy.new_tree(&mut runner).unwrap();
            let ops = case.current();

            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let completed = completed.clone();
            let failed = failed.clone();

            batch_handles.push(tokio::spawn(async move {
                let _permit = permit;

                if failed.load(Ordering::Relaxed) {
                    return;
                }

                // Use incremental verification for stress tests
                let result = run_test_sequence_incremental(ops).await;

                let count = completed.fetch_add(1, Ordering::Relaxed) + 1;
                if count % report_interval == 0 {
                    let elapsed = start.elapsed();
                    let rate = count as f64 / elapsed.as_secs_f64();
                    eprintln!(
                        "[Stress {:>3}/{} ({:>5.1}%)] {:.1} cases/sec",
                        count,
                        total_cases,
                        count as f64 / total_cases as f64 * 100.0,
                        rate
                    );
                }

                if let Err(e) = result {
                    failed.store(true, Ordering::Relaxed);
                    eprintln!("Stress case {} failed: {}", i, e);
                }
            }));
        }

        for handle in batch_handles {
            let _ = handle.await;
        }
    }

    let elapsed = start.elapsed();
    eprintln!(
        "Completed {} stress cases in {:.1}s ({:.2} cases/sec)",
        completed.load(Ordering::Relaxed),
        elapsed.as_secs_f64(),
        completed.load(Ordering::Relaxed) as f64 / elapsed.as_secs_f64()
    );

    if failed.load(Ordering::Relaxed) {
        panic!("One or more stress test cases failed");
    }
}
