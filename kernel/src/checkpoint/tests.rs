use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{from_slice, json, Value};
use tempfile::tempdir;
use test_utils::{actions_to_string, add_commit, delta_path_for_version, TestAction};
use url::Url;

use crate::action_reconciliation::{
    deleted_file_retention_timestamp_with_time, ActionReconciliationIterator,
    ActionReconciliationIteratorState, DEFAULT_RETENTION_SECS,
};
use crate::actions::{Add, Metadata, Protocol, Remove};
use crate::arrow::array::{create_array, Array, AsArray, RecordBatch, StructArray};
use crate::arrow::datatypes::{DataType, Field, Schema};
use crate::checkpoint::{
    create_last_checkpoint_data, CheckpointWriter, LastCheckpointHintStats,
    CHECKPOINT_ACTIONS_SCHEMA_V2,
};
use crate::checkpoint::create_last_checkpoint_data;
use crate::committer::FileSystemCommitter;
use crate::engine::arrow_data::ArrowEngineData;
use crate::engine::default::executor::tokio::TokioMultiThreadExecutor;
use crate::engine::default::DefaultEngine;
use crate::engine_data::FilteredEngineData;
use crate::log_replay::HasSelectionVector;
use crate::object_store::memory::InMemory;
use crate::object_store::path::Path;
use crate::object_store::ObjectStoreExt as _;
use crate::schema::{DataType as KernelDataType, StructField, StructType};
use crate::table_features::TableFeature;
use crate::transaction::create_table::create_table;
use crate::utils::test_utils::Action;
use crate::{DeltaResult, Engine, EngineData, FileMeta, LogPath, Snapshot, SnapshotRef};

use object_store::local::LocalFileSystem;
use object_store::{memory::InMemory, path::Path, ObjectStore};
use serde_json::{from_slice, json, Value};
use tempfile::tempdir;
use test_utils::delta_path_for_version;
use url::Url;

#[test]
fn test_deleted_file_retention_timestamp() -> DeltaResult<()> {
    const MILLIS_PER_SECOND: i64 = 1_000;

    let reference_time_secs = 10_000;
    let reference_time = Duration::from_secs(reference_time_secs);

    let result = deleted_file_retention_timestamp_with_time(retention, reference_time)?;
    assert_eq!(result, expected_timestamp);

    Ok(())
}

#[tokio::test]
async fn test_create_checkpoint_metadata_batch() -> DeltaResult<()> {
    let (store, _) = new_in_memory_store();
    let engine = SyncEngine::new_with_store(store.clone());

    // 1st commit (version 0) - metadata and protocol actions
    // Protocol action includes the v2Checkpoint reader/writer feature.
    write_commit_to_store(
        &store,
        vec![
            create_v2_checkpoint_protocol_action(),
            create_metadata_action(),
        ],
        0,
    )
    .await?;

    let table_root = Url::parse("memory:///")?;
    let snapshot = Snapshot::builder_for(table_root).build(&engine)?;
    let writer = snapshot.create_checkpoint_writer()?;

    // Use V2 schema for the checkpoint metadata batch
    let checkpoint_batch =
        writer.create_checkpoint_metadata_batch(&engine, &CHECKPOINT_ACTIONS_SCHEMA_V2)?;
    assert!(checkpoint_batch.filtered_data.has_selected_rows());

    // Verify the underlying EngineData contains the expected fields
    let (underlying_data, _) = checkpoint_batch.filtered_data.into_parts();
    let arrow_engine_data = ArrowEngineData::try_from_engine_data(underlying_data)?;
    let record_batch = arrow_engine_data.record_batch();

    // Verify the schema has the expected fields
    let schema = record_batch.schema();
    assert!(
        schema.field_with_name("checkpointMetadata").is_ok(),
        "Schema should have checkpointMetadata field"
    );
    assert!(
        schema.field_with_name("add").is_ok(),
        "Schema should have add field"
    );
    assert!(
        schema.field_with_name("remove").is_ok(),
        "Schema should have remove field"
    );

    // Verify we have one row
    assert_eq!(record_batch.num_rows(), 1);

    // Verify action counts
    assert_eq!(checkpoint_batch.actions_count, 1);
    assert_eq!(checkpoint_batch.add_actions_count, 0);

    Ok(())
}

#[test]
fn test_create_last_checkpoint_data() -> DeltaResult<()> {
    let version = 10;
    let total_actions_counter = 100;
    let add_actions_counter = 75;
    let size_in_bytes: i64 = 1024 * 1024; // 1MB
    let (store, _) = new_in_memory_store();
    let engine = SyncEngine::new_with_store(store.clone());

    // Create last checkpoint metadata
    let last_checkpoint_batch = create_last_checkpoint_data(
        &engine,
        version,
        total_actions_counter,
        add_actions_counter,
        size_in_bytes,
    )?;

    // Verify the underlying EngineData contains the expected `LastCheckpointInfo` schema and data
    let arrow_engine_data = ArrowEngineData::try_from_engine_data(last_checkpoint_batch)?;
    let record_batch = arrow_engine_data.record_batch();

    // Build the expected RecordBatch
    let expected_schema = Arc::new(Schema::new(vec![
        Field::new("version", DataType::Int64, false),
        Field::new("size", DataType::Int64, false),
        Field::new("parts", DataType::Int64, true),
        Field::new("sizeInBytes", DataType::Int64, true),
        Field::new("numOfAddFiles", DataType::Int64, true),
    ]));
    let expected = RecordBatch::try_new(
        expected_schema,
        vec![
            create_array!(Int64, [version]),
            create_array!(Int64, [total_actions_counter]),
            create_array!(Int64, [None]),
            create_array!(Int64, [size_in_bytes]),
            create_array!(Int64, [add_actions_counter]),
        ],
    )
    .unwrap();

    assert_eq!(*record_batch, expected);
    Ok(())
}

/// TODO(#855): Merge copies and move to `test_utils`
/// Create an in-memory store and return the store and the URL for the store's _delta_log directory.
pub(super) fn new_in_memory_store() -> (Arc<InMemory>, Url) {
    (
        Arc::new(InMemory::new()),
        Url::parse("memory:///")
            .unwrap()
            .join("_delta_log/")
            .unwrap(),
    )
}

/// TODO(#855): Merge copies and move to `test_utils`
/// Writes all actions to a _delta_log json commit file in the store.
/// This function formats the provided filename into the _delta_log directory.
pub(super) async fn write_commit_to_store(
    store: &Arc<InMemory>,
    actions: Vec<Action>,
    version: u64,
) -> DeltaResult<()> {
    let json_lines: Vec<String> = actions
        .into_iter()
        .map(|action| serde_json::to_string(&action).expect("action to string"))
        .collect();
    let content = json_lines.join("\n");
    let commit_path = delta_path_for_version(version, "json");
    store.put(&commit_path, content.into()).await?;
    Ok(())
}

/// Create a Protocol action without v2Checkpoint feature support
fn create_basic_protocol_action() -> Action {
    Action::Protocol(
        Protocol::try_new_modern(TableFeature::EMPTY_LIST, TableFeature::EMPTY_LIST).unwrap(),
    )
}

/// Create a Protocol action with catalogManaged feature support. Per the Delta protocol,
/// catalogManaged depends on inCommitTimestamp.
fn create_catalog_managed_protocol_action() -> Action {
    Action::Protocol(
        Protocol::try_new_modern(["catalogManaged"], ["catalogManaged", "inCommitTimestamp"])
            .unwrap(),
    )
}

/// Create a Protocol action with v2Checkpoint feature support
pub(super) fn create_v2_checkpoint_protocol_action() -> Action {
    Action::Protocol(Protocol::try_new_modern(vec!["v2Checkpoint"], vec!["v2Checkpoint"]).unwrap())
}

/// Create a Metadata action with the given table configuration.
fn create_metadata_action_with_config(configuration: HashMap<String, String>) -> Action {
    Action::Metadata(
        Metadata::try_new(
            Some("test-table".into()),
            None,
            Arc::new(StructType::new_unchecked([StructField::nullable(
                "value",
                KernelDataType::INTEGER,
            )])),
            vec![],
            0,
            configuration,
        )
        .unwrap(),
    )
}

/// Create a Metadata action with no configuration.
pub(super) fn create_metadata_action() -> Action {
    create_metadata_action_with_config(HashMap::new())
}

/// Create a simple Add action with the specified path (no stats)
pub(super) fn create_add_action(path: &str) -> Action {
    Action::Add(Add {
        path: path.into(),
        data_change: true,
        ..Default::default()
    })
}

/// Create a Remove action with the specified path
///
/// The remove action has deletion_timestamp set to i64::MAX to ensure the
/// remove action is not considered expired during testing.
pub(super) fn create_remove_action(path: &str) -> Action {
    Action::Remove(Remove {
        path: path.into(),
        data_change: true,
        deletion_timestamp: Some(i64::MAX), // Ensure the remove action is not expired
        ..Default::default()
    })
}

fn try_finalize_checkpoint(
    writer: CheckpointWriter,
    engine: &dyn crate::Engine,
    metadata: &FileMeta,
    data_iter: ActionReconciliationIterator,
) -> DeltaResult<()> {
    let state = data_iter.state();
    drop(data_iter);
    let state = Arc::into_inner(state).expect("no other Arc references");
    let last_checkpoint_stats = LastCheckpointHintStats::from_reconciliation_state(
        state,
        metadata.size,
        0, /* num_sidecars */
    )?;
    writer.finalize(engine, &last_checkpoint_stats)
}

/// Helper to verify the contents of the `_last_checkpoint` file
async fn assert_last_checkpoint_contents(
    store: &Arc<InMemory>,
    expected_version: u64,
    expected_size: u64,
    expected_num_add_files: u64,
    expected_size_in_bytes: u64,
) -> DeltaResult<()> {
    let last_checkpoint_data = read_last_checkpoint_file(store).await?;
    let expected_data = json!({
        "version": expected_version,
        "size": expected_size,
        "sizeInBytes": expected_size_in_bytes,
        "numOfAddFiles": expected_num_add_files,
    });
    assert_eq!(last_checkpoint_data, expected_data);
    Ok(())
}

/// Reads the `_last_checkpoint` file from storage
async fn read_last_checkpoint_file(store: &Arc<InMemory>) -> DeltaResult<Value> {
    let path = Path::from("_delta_log/_last_checkpoint");
    let data = store.get(&path).await?;
    let byte_data = data.bytes().await?;
    Ok(from_slice(&byte_data)?)
}

/// Performs a full checkpoint write for the given snapshot.
fn do_checkpoint<E: Engine>(snapshot: SnapshotRef, engine: &E) -> DeltaResult<()> {
    let writer = snapshot.checkpoint()?;
    let checkpoint_path = writer.checkpoint_path()?;

    // Get checkpoint data iterator and consume it while collecting filtered batches
    let mut data_iter = writer.checkpoint_data(engine)?;
    let mut filtered_batches: Vec<Box<dyn EngineData>> = Vec::new();
    for batch_result in data_iter.by_ref() {
        let filtered_data: FilteredEngineData = batch_result?;
        if filtered_data.has_selected_rows() {
            filtered_batches.push(filtered_data.apply_selection_vector()?);
        }
    }

    // Write the checkpoint data to parquet
    let batches_iter: Box<dyn Iterator<Item = DeltaResult<Box<dyn EngineData>>> + Send> =
        Box::new(filtered_batches.into_iter().map(Ok));
    engine
        .parquet_handler()
        .write_parquet_file(checkpoint_path.clone(), batches_iter)?;

    // Get file metadata (size) from storage and finalize
    let metadata = engine.storage_handler().head(&checkpoint_path)?;
    writer.finalize(engine, &metadata, data_iter)?;

    Ok(())
}

/// Tests the `checkpoint()` API with:
/// - A table that does not support v2Checkpoint
/// - No version specified (latest version is used)
#[tokio::test]
async fn test_v1_checkpoint_latest_version_by_default() -> DeltaResult<()> {
    let (store, _) = new_in_memory_store();
    let engine = SyncEngine::new_with_store(store.clone());

    // 1st commit: adds `fake_path_1`
    write_commit_to_store(
        &store,
        vec![create_add_action_with_stats("fake_path_1", 10)],
        0,
    )
    .await?;

    // 2nd commit: adds `fake_path_2` & removes `fake_path_1`
    write_commit_to_store(
        &store,
        vec![
            create_add_action_with_stats("fake_path_2", 20),
            create_remove_action("fake_path_1"),
        ],
        1,
    )
    .await?;

    // 3rd commit: metadata & protocol actions
    // Protocol action does not include the v2Checkpoint reader/writer feature.
    write_commit_to_store(
        &store,
        vec![create_metadata_action(), create_basic_protocol_action()],
        2,
    )
    .await?;

    let table_root = Url::parse("memory:///")?;
    let snapshot = Snapshot::builder_for(table_root).build(&engine)?;
    let writer = snapshot.create_checkpoint_writer()?;

    // Verify the checkpoint file path is the latest version by default.
    assert_eq!(
        writer.checkpoint_path()?,
        Url::parse("memory:///_delta_log/00000000000000000002.checkpoint.parquet")?
    );

    let result = writer.checkpoint_data(&engine)?;
    let mut data_iter = result;
    // The first batch should be the metadata and protocol actions.
    let batch = data_iter.next().unwrap()?;
    assert_eq!(batch.selection_vector(), &[true, true]);

    // The second batch should include both the add action and the remove action
    let batch = data_iter.next().unwrap()?;
    assert_eq!(batch.selection_vector(), &[true, true]);

    // The third batch should not be included as the selection vector does not
    // contain any true values, as the file added is removed in a following commit.
    assert!(data_iter.next().is_none());

    // Finalize and verify checkpoint metadata
    let size_in_bytes = 10;
    let metadata = FileMeta {
        location: Url::parse("memory:///fake_path_2")?,
        last_modified: 0,
        size: size_in_bytes,
    };
    try_finalize_checkpoint(writer, &engine, &metadata, data_iter)?;
    // Asserts the checkpoint file contents:
    // - version: latest version (2)
    // - size: 1 metadata + 1 protocol + 1 add action + 1 remove action
    // - numOfAddFiles: 1 add file from 2nd commit (fake_path_2)
    // - sizeInBytes: passed to finalize (10)
    assert_last_checkpoint_contents(&store, 2, 4, 1, size_in_bytes).await?;

    Ok(())
}

/// Tests the `checkpoint()` API with:
/// - A table that does not support v2Checkpoint
/// - A specific version specified (version 0)
#[tokio::test]
async fn test_v1_checkpoint_specific_version() -> DeltaResult<()> {
    let (store, _) = new_in_memory_store();
    let engine = SyncEngine::new_with_store(store.clone());

    // 1st commit (version 0) - metadata and protocol actions
    // Protocol action does not include the v2Checkpoint reader/writer feature.
    write_commit_to_store(
        &store,
        vec![create_basic_protocol_action(), create_metadata_action()],
        0,
    )
    .await?;

    // 2nd commit (version 1) - add actions
    write_commit_to_store(
        &store,
        vec![
            create_add_action_with_stats("file1.parquet", 100),
            create_add_action_with_stats("file2.parquet", 200),
        ],
        1,
    )
    .await?;

    let table_root = Url::parse("memory:///")?;
    // Specify version 0 for checkpoint
    let snapshot = Snapshot::builder_for(table_root)
        .at_version(0)
        .build(&engine)?;
    let writer = snapshot.create_checkpoint_writer()?;

    // Verify the checkpoint file path is the specified version.
    assert_eq!(
        writer.checkpoint_path()?,
        Url::parse("memory:///_delta_log/00000000000000000000.checkpoint.parquet")?
    );

    let result = writer.checkpoint_data(&engine)?;
    let mut data_iter = result;
    // The first batch should be the metadata and protocol actions.
    let batch = data_iter.next().unwrap()?;
    assert_eq!(batch.selection_vector(), &[true, true]);

    // No more data should exist because we only requested version 0
    assert!(data_iter.next().is_none());

    // Finalize and verify checkpoint metadata
    let size_in_bytes = 10;
    let metadata = FileMeta {
        location: Url::parse("memory:///fake_path_2")?,
        last_modified: 0,
        size: size_in_bytes,
    };
    try_finalize_checkpoint(writer, &engine, &metadata, data_iter)?;
    // Asserts the checkpoint file contents:
    // - version: specified version (0)
    // - size: 1 metadata + 1 protocol
    // - numOfAddFiles: no add files in version 0
    // - sizeInBytes: passed to finalize (10)
    assert_last_checkpoint_contents(&store, 0, 2, 0, size_in_bytes).await?;

    Ok(())
}

#[tokio::test]
async fn test_finalize_errors_if_checkpoint_data_iterator_is_not_exhausted() -> DeltaResult<()> {
    let (store, _) = new_in_memory_store();
    let engine = SyncEngine::new_with_store(store.clone());

    // 1st commit (version 0) - metadata and protocol actions
    write_commit_to_store(
        &store,
        vec![create_basic_protocol_action(), create_metadata_action()],
        0,
    )
    .await?;

    let table_root = Url::parse("memory:///")?;
    let snapshot = Snapshot::builder_for(table_root)
        .at_version(0)
        .build(&engine)?;
    let writer = snapshot.create_checkpoint_writer()?;
    let data_iter = writer.checkpoint_data(&engine)?;

    /* The returned data iterator has batches that we do not consume */

    // Attempting to build LastCheckpointHintStats from a non-exhausted state should fail
    let state = data_iter.state();
    drop(data_iter);
    let state = Arc::into_inner(state).expect("no other Arc references");
    let err = LastCheckpointHintStats::from_reconciliation_state(
        state, 0, /* size_in_bytes */
        0, /* num_sidecars */
    )
    .expect_err("from_reconciliation_state should fail on non-exhausted iterator");
    assert!(err
        .to_string()
        .contains("reconciliation iterator must be fully consumed"));

    Ok(())
}

#[test]
fn test_last_checkpoint_hint_stats_with_nonzero_num_sidecars() -> DeltaResult<()> {
    let state = ActionReconciliationIteratorState::new_exhausted(5, 2);
    let stats = LastCheckpointHintStats::from_reconciliation_state(state, 100, 3)?;
    assert_eq!(stats.num_actions, 8); // 5 reconciled + 3 sidecar actions
    assert_eq!(stats.size_in_bytes, 100);
    assert_eq!(stats.num_of_add_files, 2); // sidecar actions do not bump this
    Ok(())
}

#[rstest::rstest]
#[case::num_sidecars_exceeds_i64(
    0,
    0,
    0,
    u64::MAX,
    "num_sidecars 18446744073709551615 exceeds i64"
)]
#[case::actions_count_overflow(
    i64::MAX,
    0,
    0,
    1,
    "checkpoint action count overflowed i64: 9223372036854775807 + 1"
)]
#[case::size_in_bytes_exceeds_i64(
    0,
    0,
    u64::MAX,
    0,
    "size_in_bytes 18446744073709551615 exceeds i64"
)]
fn test_last_checkpoint_hint_stats_rejects_invalid_input(
    #[case] actions_count: i64,
    #[case] add_actions_count: i64,
    #[case] size_in_bytes: u64,
    #[case] num_sidecars: u64,
    #[case] expected_err_substring: &str,
) {
    let state = ActionReconciliationIteratorState::new_exhausted(actions_count, add_actions_count);
    let err =
        LastCheckpointHintStats::from_reconciliation_state(state, size_in_bytes, num_sidecars)
            .expect_err("invalid input must error");
    assert!(
        err.to_string().contains(expected_err_substring),
        "error should mention {expected_err_substring}, got: {err}"
    );
}

/// Tests the `checkpoint()` API with:
/// - A table that does supports v2Checkpoint
/// - No version specified (latest version is used)
#[tokio::test]
async fn test_v2_checkpoint_supported_table() -> DeltaResult<()> {
    let (store, _) = new_in_memory_store();
    let engine = SyncEngine::new_with_store(store.clone());

    // 1st commit: adds `fake_path_2` & removes `fake_path_1`
    write_commit_to_store(
        &store,
        vec![
            create_add_action_with_stats("fake_path_2", 50),
            create_remove_action("fake_path_1"),
        ],
        0,
    )
    .await?;

    // 2nd commit: metadata & protocol actions
    // Protocol action includes the v2Checkpoint reader/writer feature.
    write_commit_to_store(
        &store,
        vec![
            create_metadata_action(),
            create_v2_checkpoint_protocol_action(),
        ],
        1,
    )
    .await?;

    let table_root = Url::parse("memory:///")?;
    let snapshot = Snapshot::builder_for(table_root).build(&engine)?;
    let writer = snapshot.create_checkpoint_writer()?;

    // Verify the checkpoint file path is the latest version by default.
    assert_eq!(
        writer.checkpoint_path()?,
        Url::parse("memory:///_delta_log/00000000000000000001.checkpoint.parquet")?
    );

    let result = writer.checkpoint_data(&engine)?;
    let mut data_iter = result;
    // The first batch should be the metadata and protocol actions.
    let batch = data_iter.next().unwrap()?;
    assert_eq!(batch.selection_vector(), &[true, true]);

    // The second batch should include both the add action and the remove action
    let batch = data_iter.next().unwrap()?;
    assert_eq!(batch.selection_vector(), &[true, true]);

    // The third batch should be the CheckpointMetaData action.
    let batch = data_iter.next().unwrap()?;
    // According to the new contract, with_all_rows_selected creates an empty selection vector
    assert_eq!(batch.selection_vector(), &[] as &[bool]);
    assert!(batch.has_selected_rows());

    // No more data should exist
    assert!(data_iter.next().is_none());

    // Finalize and verify checkpoint metadata
    let size_in_bytes = 10;
    let metadata = FileMeta {
        location: Url::parse("memory:///fake_path_2")?,
        last_modified: 0,
        size: size_in_bytes,
    };
    try_finalize_checkpoint(writer, &engine, &metadata, data_iter)?;
    // Asserts the checkpoint file contents:
    // - version: latest version (1)
    // - size: 1 metadata + 1 protocol + 1 add action + 1 remove action + 1 checkpointMetadata
    // - numOfAddFiles: 1 add file from version 0
    // - sizeInBytes: passed to finalize (10)
    assert_last_checkpoint_contents(&store, 1, 5, 1, size_in_bytes).await?;

    Ok(())
}

#[tokio::test]
async fn test_no_checkpoint_on_unpublished_snapshot() -> DeltaResult<()> {
    let (store, _) = new_in_memory_store();
    let engine = SyncEngine::new_with_store(store.clone());

    // normal commit with catalog-managed protocol
    write_commit_to_store(
        &store,
        vec![
            create_metadata_action_with_config(HashMap::from([(
                "delta.enableInCommitTimestamps".to_string(),
                "true".to_string(),
            )])),
            create_catalog_managed_protocol_action(),
        ],
        0,
    )
    .await?;

    // staged commit
    let staged_commit_path = Path::from(
        "_delta_log/_staged_commits/00000000000000000001.3a0d65cd-4056-49b8-937b-95f9e3ee90e5.json",
    );
    let add_action = Action::Add(Add::default());
    store
        .put(
            &staged_commit_path,
            serde_json::to_string(&add_action).unwrap().into(),
        )
        .await
        .unwrap();

    let table_root = Url::parse("memory:///")?;
    let staged_commit = FileMeta {
        location: Url::parse("memory:///_delta_log/_staged_commits/00000000000000000001.3a0d65cd-4056-49b8-937b-95f9e3ee90e5.json")?,
        last_modified: 0,
        size: 100,
    };
    let snapshot = Snapshot::builder_for(table_root.clone())
        .with_log_tail(vec![LogPath::try_new(staged_commit).unwrap()])
        .with_max_catalog_version(1)
        .build(&engine)?;

    assert!(matches!(
        snapshot.create_checkpoint_writer().unwrap_err(),
        crate::Error::Generic(e) if e == "Log segment is not published"
    ));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_checkpoint_preserves_domain_metadata() -> DeltaResult<()> {
    // ===== Setup =====
    let tmp_dir = tempdir().unwrap();
    let table_path = tmp_dir.path();
    let table_url = Url::from_directory_path(table_path).unwrap();
    std::fs::create_dir_all(table_path.join("_delta_log")).unwrap();

    // ===== Create Table =====
    let commit0 = [
        json!({
            "protocol": {
                "minReaderVersion": 3,
                "minWriterVersion": 7,
                "readerFeatures": [],
                "writerFeatures": ["domainMetadata"]
            }
        }),
        json!({
            "metaData": {
                "id": "test-table-id",
                "format": { "provider": "parquet", "options": {} },
                "schemaString": "{\"type\":\"struct\",\"fields\":[{\"name\":\"value\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}}]}",
                "partitionColumns": [],
                "configuration": {},
                "createdTime": 1587968585495i64
            }
        }),
    ]
    .map(|j| j.to_string())
    .join("\n");
    std::fs::write(
        table_path.join("_delta_log/00000000000000000000.json"),
        commit0,
    )
    .unwrap();

    // ===== Create Engine =====
    let store = Arc::new(LocalFileSystem::new());
    let executor = Arc::new(TokioMultiThreadExecutor::new(
        tokio::runtime::Handle::current(),
    ));
    let engine = DefaultEngine::new_with_executor(store.clone(), executor);

    let commit_domain_metadata = |domain: &str, value: &str| -> DeltaResult<()> {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let txn = snapshot.transaction(Box::new(FileSystemCommitter::new()))?;
        let result = txn
            .with_domain_metadata(domain.to_string(), value.to_string())
            .commit(&engine)?;
        assert!(result.is_committed());
        Ok(())
    };

    // ===== Commit Domain Metadata =====
    commit_domain_metadata("foo", "bar1")?;
    commit_domain_metadata("foo", "bar2")?;

    // ===== Case 1: Verify domain metadata is preserved *before* checkpoint =====
    let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
    assert_eq!(snapshot.version(), 2);
    let domain_value = snapshot.get_domain_metadata("foo", &engine)?;
    assert_eq!(domain_value, Some("bar2".to_string()));

    // Trigger checkpoint
    do_checkpoint(snapshot, &engine)?;

    // ===== Case 2: Verify domain metadata is preserved *after* checkpoint =====
    let snapshot = Snapshot::builder_for(table_url)
        .at_version(2)
        .build(&engine)?;
    let domain_value = snapshot.get_domain_metadata("foo", &engine)?;
    assert_eq!(domain_value, Some("bar2".to_string()));

    Ok(())
}

// TODO: Add test that checkpoint does not contain tombstoned domain metadata.
