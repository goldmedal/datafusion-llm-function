use crate::llm::functions::{AsyncUDF, AsyncUDFImpl};
use crate::llm::physical_optimizer::AsyncFuncRule;
use async_trait::async_trait;
use datafusion::arrow::array::{ArrayRef, AsArray, BooleanArray, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Int32Type};
use datafusion::common::Result;
use datafusion::execution::{FunctionRegistry, SessionStateBuilder};
use datafusion::functions_aggregate::min_max::max_udaf;
use datafusion::logical_expr::{Signature, TypeSignature, Volatility};
use datafusion::prelude::SessionContext;
use std::any::Any;
use std::sync::Arc;

mod llm;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let mut state = SessionStateBuilder::default()
        .with_physical_optimizer_rule(Arc::new(AsyncFuncRule {}))
        .build();

    let llm_bool = LLMBool::new();
    let udf = AsyncUDF::new(Arc::new(llm_bool));
    state.register_udf(udf.into_scalar_udf())?;
    state.register_udaf(max_udaf())?;
    let ctx = SessionContext::new_with_state(state);
    ctx.sql("create table t1 (c1 int, c2 int, c3 int)")
        .await?
        .show()
        .await?;
    ctx.sql("insert into t1 values (1, 2, 3), (11, 2, 3), (1, 2, 3)")
        .await?
        .show()
        .await?;
    ctx.sql("insert into t1 values (1, 2, 3), (1, 2, 3), (21, 2, 3)")
        .await?
        .show()
        .await?;
    ctx.sql("insert into t1 values (31, 2, 3), (1, 2, 3), (1, 2, 3)")
        .await?
        .show()
        .await?;

    ctx.sql("explain select llm_bool('If all of them are Aisa countries: {}, {}, {}', t1.c1, t1.c2, t1.c3) from t1")
        .await?.show().await?;

    ctx.sql("select llm_bool('If all of them are Aisa countries: {}, {}, {}', t1.c1, t1.c2, t1.c3) from t1")
        .await?.show().await?;

    Ok(())
}

/// This is a simple example of a UDF that takes a string, invokes a (remote) LLM function
/// and returns a boolean
#[derive(Debug)]
struct LLMBool {
    signature: Signature,
}

impl LLMBool {
    fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![TypeSignature::VariadicAny, TypeSignature::Nullary],
                Volatility::Volatile,
            ),
        }
    }
}

#[async_trait]
impl AsyncUDFImpl for LLMBool {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "llm_bool"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    async fn invoke_async(&self, args: &RecordBatch) -> Result<ArrayRef> {
        // TODO make the actual async call to open AI
        //
        // Note this is an async function

        // example calls the function with three integer args. We return true if
        // the first argument is greater than 10
        let first_arg = args.columns()[0].as_primitive::<Int32Type>();

        let output: BooleanArray = first_arg
            .iter()
            .map(|arg| arg.map(|arg| arg > 10))
            .collect();

        Ok(Arc::new(output))
    }
}

// TODO:
//  pretty batch and concat with promotion
//  as the input of the LLM function
// async fn ask_openai() -> Result<Option<String>> {
//     let messages = vec![Message {
//         role: "user".to_string(),
//         content: "Say this is a test!".to_string(),
//     }];
//     let body = MySendBody {
//         model: "gpt-4o-mini".to_string(),
//         messages,
//         temperature: 0.7,
//     };
//
//     let key = match env::var("OPENAI_API_KEY") {
//         Ok(key) => key,
//         Err(e) => panic!("OPENAI_API_KEY is not set: {}", e),
//     };
//     let key = format!("Bearer {}", key);
//     let client = reqwest::Client::new();
//     let response = client.post("https://api.openai.com/v1/chat/completions")
//         .header("Authorization", key)
//         .json(&body)
//         .send().await
//         .map_err(|e| {
//             internal_err!("Failed to send request to OpenAI: {}", e)
//         })?;
//     if response.status().is_success() {
//         response.text()
//     }
//     else {
//         internal_err!("Failed to send request to OpenAI: {}", response.status())
//     }
// }
/*
#[derive(Serialize)]
struct MySendBody {
    model: String,
    messages: Vec<Message>,
    temperature: f32,
}

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}
*/
