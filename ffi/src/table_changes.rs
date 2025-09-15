//! TableChanges related ffi code

use std::sync::{Arc, Mutex};

use delta_kernel::arrow::array::{
    ffi::{FFI_ArrowArray, FFI_ArrowSchema},
    ArrayData, RecordBatch, StructArray,
};
use delta_kernel::arrow::compute::filter_record_batch;
use delta_kernel::engine::arrow_data::ArrowEngineData;
use delta_kernel::scan::ScanResult;
use delta_kernel::table_changes::scan::TableChangesScan;
use delta_kernel::table_changes::TableChanges;
use delta_kernel::{DeltaResult, Error, Version};
use delta_kernel_ffi_macros::handle_descriptor;
use tracing::debug;

use super::handle::Handle;
use url::Url;

use crate::engine_data::ArrowFFIData;
use crate::expressions::kernel_visitor::{unwrap_kernel_predicate, KernelExpressionVisitorState};
use crate::scan::EnginePredicate;
use crate::{
    kernel_string_slice, unwrap_and_parse_path_as_url, AllocateStringFn, ExternEngine,
    ExternResult, IntoExternResult, KernelStringSlice, NullableCvoid, SharedExternEngine,
    SharedSchema,
};

#[handle_descriptor(target=TableChanges, mutable=true, sized=true)]
pub struct ExclusiveTableChanges;

/// Get the table changes from the specified table at a specific version
///
/// - `table_root`: url pointing at the table root (where `_delta_log` folder is located)
/// - `engine`: Implementation of [`Engine`] apis.
/// - `start_version`: The start version of the change data feed
/// End version will be the newest table version.
///
/// # Safety
///
/// Caller is responsible for passing valid handles and path pointer.
#[no_mangle]
pub unsafe extern "C" fn table_changes_from_version(
    path: KernelStringSlice,
    engine: Handle<SharedExternEngine>,
    start_version: Version,
) -> ExternResult<Handle<ExclusiveTableChanges>> {
    let url = unsafe { unwrap_and_parse_path_as_url(path) };
    let engine = unsafe { engine.as_ref() };
    table_changes_impl(url, engine, start_version, None).into_extern_result(&engine)
}

/// Get the table changes from the specified table between two versions
///
/// - `table_root`: url pointing at the table root (where `_delta_log` folder is located)
/// - `engine`: Implementation of [`Engine`] apis.
/// - `start_version`: The start version of the change data feed
/// - `end_version`: The end version (inclusive) of the change data feed.
///
/// # Safety
///
/// Caller is responsible for passing valid handles and path pointer.
#[no_mangle]
pub unsafe extern "C" fn table_changes_between_versions(
    path: KernelStringSlice,
    engine: Handle<SharedExternEngine>,
    start_version: Version,
    end_version: Version,
) -> ExternResult<Handle<ExclusiveTableChanges>> {
    let url = unsafe { unwrap_and_parse_path_as_url(path) };
    let engine = unsafe { engine.as_ref() };
    table_changes_impl(url, engine, start_version, end_version.into()).into_extern_result(&engine)
}

fn table_changes_impl(
    url: DeltaResult<Url>,
    extern_engine: &dyn ExternEngine,
    start_version: Version,
    end_version: Option<Version>,
) -> DeltaResult<Handle<ExclusiveTableChanges>> {
    let table_changes = TableChanges::try_new(
        url?,
        extern_engine.engine().as_ref(),
        start_version,
        end_version,
    );
    Ok(Box::new(table_changes?).into())
}

/// Drops table changes.
///
/// # Safety
/// Caller is responsible for passing a valid table changes handle.
#[no_mangle]
pub unsafe extern "C" fn free_table_changes(table_changes: Handle<ExclusiveTableChanges>) {
    table_changes.drop_handle();
}

/// Get schema from the specified TableChanges.
///
/// # Safety
///
/// Caller is responsible for passing a valid table changes handle.
#[no_mangle]
pub unsafe extern "C" fn schema(
    table_changes: Handle<ExclusiveTableChanges>,
) -> Handle<SharedSchema> {
    let table_changes = unsafe { table_changes.as_ref() };
    Arc::new(table_changes.schema().clone()).into()
}

/// Get start version from the specified TableChanges.
///
/// # Safety
///
/// Caller is responsible for passing a valid table changes handle.
#[no_mangle]
pub unsafe extern "C" fn start_version(table_changes: Handle<ExclusiveTableChanges>) -> u64 {
    let table_changes = unsafe { table_changes.as_ref() };
    table_changes.start_version()
}

/// Get end version from the specified TableChanges.
///
/// # Safety
///
/// Caller is responsible for passing a valid table changes handle.
#[no_mangle]
pub unsafe extern "C" fn end_version(table_changes: Handle<ExclusiveTableChanges>) -> u64 {
    let table_changes = unsafe { table_changes.as_ref() };
    table_changes.end_version()
}

#[handle_descriptor(target=TableChangesScan, mutable=false, sized=true)]
pub struct SharedTableChangesScan;

/// Get a [`TableChangesScan`] over the table specified by the passed table changes.
/// It is the responsibility of the _engine_ to free this scan when complete by calling [`free_table_changes_scan`].
/// Consumes TableChanges.
///
/// # Safety
///
/// Caller is responsible for passing a valid table changes pointer, and engine pointer
#[no_mangle]
pub unsafe extern "C" fn table_changes_scan(
    table_changes: Handle<ExclusiveTableChanges>,
    engine: Handle<SharedExternEngine>,
    predicate: Option<&mut EnginePredicate>,
) -> ExternResult<Handle<SharedTableChangesScan>> {
    let table_changes = unsafe { table_changes.into_inner() };
    table_changes_scan_impl(*table_changes, predicate).into_extern_result(&engine.as_ref())
}

fn table_changes_scan_impl(
    table_changes: TableChanges,
    predicate: Option<&mut EnginePredicate>,
) -> DeltaResult<Handle<SharedTableChangesScan>> {
    let mut scan_builder = table_changes.into_scan_builder();
    if let Some(predicate) = predicate {
        let mut visitor_state = KernelExpressionVisitorState::default();
        let pred_id = (predicate.visitor)(predicate.predicate, &mut visitor_state);
        let predicate = unwrap_kernel_predicate(&mut visitor_state, pred_id);
        debug!("Got predicate: {:#?}", predicate);
        scan_builder = scan_builder.with_predicate(predicate.map(Arc::new));
    }
    Ok(Arc::new(scan_builder.build()?).into())
}

/// Get the table root of a TableChangesScan.
///
/// # Safety
/// Engine is responsible for providing a valid scan pointer and allocate_fn (for allocating the
/// string)
#[no_mangle]
pub unsafe extern "C" fn table_changes_scan_table_root(
    table_changes_scan: Handle<SharedTableChangesScan>,
    allocate_fn: AllocateStringFn,
) -> NullableCvoid {
    let table_changes_scan = unsafe { table_changes_scan.as_ref() };
    let table_root = table_changes_scan.table_root().to_string();
    allocate_fn(kernel_string_slice!(table_root))
}
/// Get the logical schema of the specified TableChangesScan.
///
/// # Safety
///
/// Caller is responsible for passing a valid snapshot handle.
#[no_mangle]
pub unsafe extern "C" fn table_changes_scan_logical_schema(
    table_changes_scan: Handle<SharedTableChangesScan>,
) -> Handle<SharedSchema> {
    let table_changes_scan = unsafe { table_changes_scan.as_ref() };
    table_changes_scan.logical_schema().clone().into()
}

/// Get the physical schema of the specified TableChangesScan.
///
/// # Safety
///
/// Caller is responsible for passing a valid snapshot handle.
#[no_mangle]
pub unsafe extern "C" fn table_changes_scan_physical_schema(
    table_changes_scan: Handle<SharedTableChangesScan>,
) -> Handle<SharedSchema> {
    let table_changes_scan = unsafe { table_changes_scan.as_ref() };
    table_changes_scan.physical_schema().clone().into()
}

//#[no_mangle]
//pub unsafe extern "C" fn table_changes_scan_execute(
//    table_changes_scan: Handle<SharedTableChangesScan>,
//    engine: Handle<SharedExternEngine>,
//) -> ExternResult<Handle<ExclusiveFileReadResultIterator>> {
//    let table_changes_scan = unsafe { table_changes_scan.clone_as_arc() };
//    let engine = unsafe { engine.clone_as_arc() };
//    table_changes_scan_execute_impl(table_changes_scan, engine.clone()).into_extern_result(&engine.as_ref())
//}
//
//fn table_changes_scan_execute_impl(
//    table_changes_scan: Arc<TableChangesScan>,
//    extern_engine: Arc<dyn ExternEngine>,
//) -> DeltaResult<Handle<ExclusiveFileReadResultIterator>> {
//    //let table_changes_iter = table_changes_scan.execute(engine.engine().clone())?;
//    let data: Vec<_> = table_changes_scan.execute(engine.clone()).unwrap().try_collect().unwrap();
//    //let owned_iter = table_changes_iter.collect::<Vec<_>>().into_iter();
//    //let data = ScanTableChangesIterator {
//    //    data: Mutex::new(Box::new(owned_iter)),
//    //    engine: engine.clone(),
//    //};
//    //Ok(Arc::new(data).into())
//
//    let res = Box::new(FileReadResultIterator {
//        data,
//        engine: extern_engine,
//    });
//    Ok(res.into())
//}

pub struct ScanTableChangesIterator {
    data: Mutex<Box<dyn Iterator<Item = DeltaResult<ScanResult>> + Send>>,
    engine: Arc<dyn ExternEngine>,
}

#[handle_descriptor(target=ScanTableChangesIterator, mutable=false, sized=true)]
pub struct SharedScanTableChangesIterator;

impl Drop for ScanTableChangesIterator {
    fn drop(&mut self) {
        debug!("dropping ScanTableChangesIterator");
    }
}

#[no_mangle]
pub unsafe extern "C" fn table_changes_scan_iter_init(
    table_changes_scan: Handle<SharedTableChangesScan>,
    engine: Handle<SharedExternEngine>,
) -> ExternResult<Handle<SharedScanTableChangesIterator>> {
    let table_changes_scan = unsafe { table_changes_scan.clone_as_arc() };
    let engine = unsafe { engine.clone_as_arc() };
    table_changes_scan_iter_init_impl(table_changes_scan, engine.clone())
        .into_extern_result(&engine.as_ref())
}

fn table_changes_scan_iter_init_impl(
    table_changes_scan: Arc<TableChangesScan>,
    engine: Arc<dyn ExternEngine>,
) -> DeltaResult<Handle<SharedScanTableChangesIterator>> {
    let table_changes_iter = table_changes_scan.execute(engine.engine().clone())?;
    let owned_iter = table_changes_iter.collect::<Vec<_>>().into_iter();
    let data = ScanTableChangesIterator {
        data: Mutex::new(Box::new(owned_iter)),
        engine: engine.clone(),
    };
    Ok(Arc::new(data).into())
}

#[no_mangle]
pub unsafe extern "C" fn scan_table_changes_next(
    data: Handle<SharedScanTableChangesIterator>,
    engine_context: NullableCvoid,
) -> ExternResult<*mut ArrowFFIData> {
    let data = unsafe { data.as_ref() };
    scan_table_changes_next_impl(data, engine_context).into_extern_result(&data.engine.as_ref())
}

fn scan_table_changes_next_impl(
    data: &ScanTableChangesIterator,
    engine_context: NullableCvoid,
) -> DeltaResult<*mut ArrowFFIData> {
    let mut data = data
        .data
        .lock()
        .map_err(|_| Error::generic("poisoned mutex"))?;
    if let Some(scan_result) = data.next().transpose()? {
        let mask = scan_result.full_mask();
        let data = scan_result.raw_data?;
        let mut record_batch: RecordBatch = data
            .into_any()
            .downcast::<ArrowEngineData>()
            .map_err(|_| delta_kernel::Error::EngineDataType("ArrowEngineData".to_string()))?
            .into();

        if let Some(mask) = mask {
            record_batch = filter_record_batch(&record_batch, &mask.into())?;
        }

        let sa: StructArray = record_batch.into();
        let array_data: ArrayData = sa.into();
        // these call `clone`. is there a way to not copy anything and what exactly are they cloning?
        let array = FFI_ArrowArray::new(&array_data);
        let schema = FFI_ArrowSchema::try_from(array_data.data_type())?;
        let ret_data = Box::new(ArrowFFIData { array, schema });
        Ok(Box::leak(ret_data))
    } else {
        Ok(std::ptr::null_mut())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    //use super::{free_expression_evaluator, new_expression_evaluator};
    //use crate::{free_engine, handle::Handle, tests::get_default_engine, SharedSchema};
    use crate::engine_to_handle;
    use crate::ffi_test_utils::{
        allocate_err, allocate_str, assert_extern_result_error_with_message, ok_or_panic,
        recover_string, EngineErrorWithMessage,
    };
    use crate::{
        kernel_string_slice, snapshot, snapshot_at_version, version, KernelStringSlice,
        SharedSchema,
    };
    use delta_kernel::engine::default::{executor::tokio::TokioBackgroundExecutor, DefaultEngine};
    use delta_kernel::{
        schema::{DataType, StructField, StructType},
        Expression,
    };
    use object_store::memory::InMemory;
    use std::sync::Arc;
    use test_utils::{actions_to_string, actions_to_string_partitioned, add_commit, TestAction};

    #[tokio::test]
    async fn test_table_changes() -> Result<(), Box<dyn std::error::Error>> {
        let storage = Arc::new(InMemory::new());
        add_commit(
            storage.as_ref(),
            0,
            actions_to_string(vec![TestAction::Metadata]),
        )
        .await?;
        let engine = DefaultEngine::new(storage.clone(), Arc::new(TokioBackgroundExecutor::new()));
        let engine = engine_to_handle(Arc::new(engine), allocate_err);
        let path = "memory:///";

        //let snapshot1 =
        //    unsafe { ok_or_panic(snapshot(kernel_string_slice!(path), engine.shallow_copy())) };
        //let version1 = unsafe { version(snapshot1.shallow_copy()) };
        //assert_eq!(version1, 0);

        //// Test getting snapshot at version
        //let snapshot2 = unsafe {
        //    ok_or_panic(snapshot_at_version(
        //        kernel_string_slice!(path),
        //        engine.shallow_copy(),
        //        0,
        //    ))
        //};
        //let version2 = unsafe { version(snapshot2.shallow_copy()) };
        //assert_eq!(version2, 0);

        let table_changes =
            unsafe { table_changes_from_version(kernel_string_slice!(path), engine, 1) };
        match table_changes {
            ExternResult::Ok(handle) => {
                assert_eq!(unsafe { start_version(handle) }, 0);
            }
            ExternResult::Err(e) => unsafe {
                let err_with_msg: &EngineErrorWithMessage = &*(e as *mut EngineErrorWithMessage);
                eprintln!(
                    "Error type: {:?}, message: {:?}",
                    (*err_with_msg).etype,
                    (*err_with_msg).message
                );
            },
        }

        //unsafe { free_snapshot(snapshot1) }
        //unsafe { free_snapshot(snapshot2) }
        //unsafe { free_engine(engine) }
        Ok(())
    }
}
