pub mod config;
pub mod llm_bool;

use crate::llm::async_func::AsyncScalarFunctionArgs;
use async_trait::async_trait;
use datafusion::arrow::array::{ArrayRef, RecordBatch};
use datafusion::arrow::datatypes::DataType;
use datafusion::common::internal_err;
use datafusion::common::Result;
use datafusion::config::ConfigOptions;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
};
use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

/// A scalar UDF that can invoke using async methods
///
/// Note this is less efficient than the ScalarUDFImpl, but it can be used
/// to register remote functions in the context.
///
/// The name is chosen to  mirror ScalarUDFImpl
#[async_trait]
pub trait AsyncScalarUDFImpl: Debug + Send + Sync {
    /// the function cast as any
    fn as_any(&self) -> &dyn Any;

    /// The name of the function
    fn name(&self) -> &str;

    /// The signature of the function
    fn signature(&self) -> &Signature;

    /// The return type of the function
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType>;

    /// The ideal batch size for this function
    /// If there are multiple async functions in a plan, the batch size will be the max of all of them
    fn ideal_batch_size(&self) -> Option<usize> {
        None
    }

    /// Invoke the function asynchronously with the async arguments
    async fn invoke_async(&self, args: &RecordBatch) -> Result<ArrayRef>;

    async fn invoke_async_with_args(
        &self,
        args: AsyncScalarFunctionArgs,
        option: &ConfigOptions,
    ) -> Result<ColumnarValue>;
}

/// A scalar UDF that must be invoked using async methods
///
/// Note this is not meant to be used directly, but is meant to be an implementation detail
/// for AsyncUDFImpl.
///
/// This is used to register remote functions in the context. The function
/// should not be invoked by DataFusion. It's only used to generate the logical
/// plan and unparsed them to SQL.
#[derive(Debug)]
pub struct AsyncScalarUDF {
    inner: Arc<dyn AsyncScalarUDFImpl>,
}

impl AsyncScalarUDF {
    pub fn new(inner: Arc<dyn AsyncScalarUDFImpl>) -> Self {
        Self { inner }
    }

    pub fn ideal_batch_size(&self) -> Option<usize> {
        self.inner.ideal_batch_size()
    }

    /// Turn this AsyncUDF into a ScalarUDF, suitable for
    /// registering in the context
    pub fn into_scalar_udf(self) -> Arc<ScalarUDF> {
        Arc::new(ScalarUDF::new_from_impl(self))
    }

    /// Invoke the function asynchronously with the record batch
    pub async fn invoke_async(&self, batch: &RecordBatch) -> Result<ArrayRef> {
        self.inner.invoke_async(batch).await
    }

    /// Invoke the function asynchronously with the async arguments
    pub async fn invoke_async_with_args(
        &self,
        args: AsyncScalarFunctionArgs,
        option: &ConfigOptions,
    ) -> Result<ColumnarValue> {
        self.inner.invoke_async_with_args(args, option).await
    }
}

impl ScalarUDFImpl for AsyncScalarUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn signature(&self) -> &Signature {
        self.inner.signature()
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        self.inner.return_type(_arg_types)
    }

    fn invoke(&self, _args: &[ColumnarValue]) -> Result<ColumnarValue> {
        internal_err!("This function should not be called")
    }
}
