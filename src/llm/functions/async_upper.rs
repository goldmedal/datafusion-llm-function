use crate::llm::async_func::AsyncScalarFunctionArgs;
use crate::llm::functions::AsyncScalarUDFImpl;
use async_trait::async_trait;
use datafusion::arrow::array::{ArrayIter, ArrayRef, AsArray, RecordBatch, StringArray};
use datafusion::arrow::datatypes::DataType;
use datafusion::common::internal_err;
use datafusion::common::types::{logical_int64, logical_string};
use datafusion::common::Result;
use datafusion::config::ConfigOptions;
use datafusion::logical_expr::{
    ColumnarValue, Signature, TypeSignature, TypeSignatureClass, Volatility,
};
use log::trace;
use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

#[derive(Debug)]
pub struct AsyncUpper {
    signature: Signature,
}

impl Default for AsyncUpper {
    fn default() -> Self {
        Self::new()
    }
}

impl AsyncUpper {
    pub fn new() -> Self {
        Self {
            signature: Signature::new(
                TypeSignature::Coercible(vec![TypeSignatureClass::Native(logical_string())]),
                Volatility::Volatile,
            ),
        }
    }
}

#[async_trait]
impl AsyncScalarUDFImpl for AsyncUpper {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "async_upper"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Utf8)
    }

    fn ideal_batch_size(&self) -> Option<usize> {
        Some(10)
    }

    async fn invoke_async_with_args(
        &self,
        args: AsyncScalarFunctionArgs,
        _option: &ConfigOptions,
    ) -> Result<ArrayRef> {
        trace!("Invoking async_upper with args: {:?}", args);
        let value = &args.args[0];
        let result = match value {
            ColumnarValue::Array(array) => {
                let string_array = array.as_string::<i32>();
                let iter = ArrayIter::new(string_array);
                let result = iter
                    .map(|string| string.map(|s| s.to_uppercase()))
                    .collect::<StringArray>();
                Arc::new(result) as ArrayRef
            }
            _ => return internal_err!("Expected a string argument, got {:?}", value),
        };
        Ok(result)
    }
}
