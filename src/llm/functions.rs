use datafusion::arrow::datatypes::DataType;
use datafusion::common::internal_err;
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, ScalarUDFImpl, Signature, TypeSignature, Volatility,
};
use std::any::Any;

/// A scalar UDF that will be bypassed when planning logical plan.
/// This is used to register the remote function to the context. The function should not be
/// invoked by DataFusion. It's only used to generate the logical plan and unparsed them to SQL.
#[derive(Debug)]
pub struct ByPassScalarUDF {
    name: String,
    return_type: DataType,
    signature: Signature,
}

impl ByPassScalarUDF {
    pub fn new(name: &str, return_type: DataType) -> Self {
        Self {
            name: name.to_string(),
            return_type,
            signature: Signature::one_of(
                vec![TypeSignature::VariadicAny, TypeSignature::Nullary],
                Volatility::Volatile,
            ),
        }
    }
}

impl ScalarUDFImpl for ByPassScalarUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(self.return_type.clone())
    }

    fn invoke(&self, _args: &[ColumnarValue]) -> Result<ColumnarValue> {
        internal_err!("This function should not be called")
    }
}
