use crate::llm::async_func::AsyncScalarFunctionArgs;
use crate::llm::functions::config::LLMConfig;
use crate::llm::functions::AsyncScalarUDFImpl;
use async_trait::async_trait;
use comfy_table::{Cell, Table};
use datafusion::arrow::array::{ArrayRef, BooleanArray, RecordBatch};
use datafusion::arrow::datatypes::{DataType, SchemaRef};
use datafusion::arrow::error::ArrowError;
use datafusion::arrow::util::display::{ArrayFormatter, FormatOptions};
use datafusion::common::{exec_err, internal_err, not_impl_err, plan_err, ScalarValue};
use datafusion::config::ConfigOptions;
use datafusion::logical_expr::{ColumnarValue, Signature, TypeSignature, Volatility};
use log::debug;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

/// This is a simple example of a UDF that takes a string, invokes a (remote) LLM function
/// and returns a boolean
#[derive(Debug)]
pub struct LLMBool {
    signature: Signature,
}

impl LLMBool {
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![TypeSignature::VariadicAny, TypeSignature::Nullary],
                Volatility::Volatile,
            ),
        }
    }
}

#[async_trait]
impl AsyncScalarUDFImpl for LLMBool {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "llm_bool"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> datafusion::common::Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn ideal_batch_size(&self) -> Option<usize> {
        Some(6)
    }

    async fn invoke_async(&self, _args: &RecordBatch) -> datafusion::common::Result<ArrayRef> {
        not_impl_err!("This function should not be called")
    }

    async fn invoke_async_with_args(
        &self,
        args: AsyncScalarFunctionArgs,
        option: &ConfigOptions,
    ) -> datafusion::common::Result<ArrayRef> {
        let ColumnarValue::Scalar(question) = &args.args[0] else {
            return plan_err!("Expected a scalar argument");
        };
        let question = match question {
            ScalarValue::Utf8(Some(question)) => question,
            _ => return plan_err!("Expected a string argument"),
        };

        let data = args.args[1..]
            .iter()
            .map(|arg| arg.to_array(args.number_rows))
            .collect::<datafusion::common::Result<Vec<_>>>()?;
        let table = pretty_format_column(
            &data,
            args.schema,
            args.number_rows,
            &FormatOptions::default(),
        )?;
        let prompt = format!(
            r#"
Given the following data:
{}

Follow the rules for the response strictly to answer questions based on the data:
- Placeholders `{{colum name}}` are used in the question to represent the column values.
- The output must be strictly lowercase `true` or `false`.
- You should only say 'true' or 'false' based on the question.
- You should strictly only return {} rows.

Evaluate each row based on the following question and return only the boolean result (`true` or `false`) for each row, without any explanation:
<question>{}</question>
 "#,
            table, args.number_rows, question
        );

        debug!("Generated prompt: {}", prompt);

        let Some(llm_config) = option.extensions.get::<LLMConfig>() else {
            return internal_err!("LLM configuration not found");
        };
        let result = ask_llm(prompt, llm_config).await?;
        if result.len() != args.number_rows {
            return internal_err!(
                "Expected {} results, got {}",
                args.number_rows,
                result.len()
            );
        }
        let array_ref = Arc::new(BooleanArray::from(result));
        Ok(array_ref)
    }
}

pub fn pretty_format_column(
    columns: &[ArrayRef],
    schema: SchemaRef,
    number_rows: usize,
    options: &FormatOptions,
) -> datafusion::common::Result<Table> {
    let mut table = Table::new();
    table.load_preset(comfy_table::presets::UTF8_FULL_CONDENSED);

    if columns.is_empty() {
        return Ok(table);
    }

    let header = schema
        .fields()
        .iter()
        .map(|f| Cell::new(f.name()))
        .collect::<Vec<_>>();
    table.set_header(header);

    let formatters = columns
        .iter()
        .map(|c| ArrayFormatter::try_new(c.as_ref(), options))
        .collect::<std::result::Result<Vec<_>, ArrowError>>()?;

    for row in 0..number_rows {
        let mut cells = Vec::new();
        for formatter in &formatters {
            cells.push(Cell::new(formatter.value(row)));
        }
        table.add_row(cells);
    }
    Ok(table)
}

async fn ask_llm(
    content: impl Into<String>,
    llm_config: &LLMConfig,
) -> datafusion::common::Result<Vec<bool>> {
    let messages = vec![Message {
        role: "user".to_string(),
        content: content.into(),
    }];

    // For the structured result, we expect an array of booleans.
    // See ollama https://ollama.com/blog/structured-outputs
    let mut properties = HashMap::new();
    properties.insert(
        "result".to_string(),
        Property {
            r#type: "array".to_string(),
            item: Some(Box::new(Property {
                r#type: "boolean".to_string(),
                item: None,
            })),
        },
    );

    let body = RequestBody {
        model: llm_config.model.clone(),
        messages,
        stream: false,
        format: OutputFormat {
            r#type: "object".to_string(),
            properties,
            required: vec!["result".to_string()],
        },
    };

    debug!("ask_llm request body: {body:?}");

    let response = if llm_config.api_key.is_empty() {
        let client = reqwest::Client::new();
        client
            .post(&llm_config.chat_endpoint)
            .json(&body)
            .send()
            .await
            .unwrap()
    } else {
        let key = format!("Bearer {}", llm_config.api_key);
        dbg!(&key);
        let client = reqwest::Client::new();
        client
            .post(&llm_config.chat_endpoint)
            .header("Authorization", key)
            .json(&body)
            .send()
            .await
            .unwrap()
    };
    if response.status().is_success() {
        let response = response.json::<Response>().await.unwrap();
        let structured_result =
            match serde_json::from_str::<StructuredResult>(&response.message.content) {
                Ok(result) => result,
                Err(e) => return exec_err!("Failed to parse response from LLM: {}", e),
            };
        let result = structured_result.result;
        debug!("ask_llm response: {response:?}");
        Ok(result)
    } else {
        exec_err!("Failed to send request to LLM: {}", response.status())
    }
}

#[derive(Serialize, Debug)]
struct RequestBody {
    model: String,
    messages: Vec<Message>,
    stream: bool,
    format: OutputFormat,
}

#[derive(Serialize, Deserialize, Debug)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize, Debug)]
struct OutputFormat {
    #[serde(rename = "type")]
    r#type: String,
    properties: HashMap<String, Property>,
    required: Vec<String>,
}

#[derive(Serialize, Debug)]
struct Property {
    #[serde(rename = "type")]
    r#type: String,
    item: Option<Box<Property>>,
}

#[derive(Serialize, Deserialize, Debug)]
struct Response {
    model: String,
    created_at: String,
    message: Message,
    done_reason: String,
    done: bool,
    total_duration: i64,
    load_duration: i64,
    prompt_eval_count: i64,
    prompt_eval_duration: i64,
    eval_count: i32,
    eval_duration: i64,
}

#[derive(Deserialize, Debug)]
struct StructuredResult {
    #[serde(default, with = "bool_from_int")]
    result: Vec<bool>,
}

mod bool_from_int {
    use serde::{self, Deserialize, Deserializer, Serialize, Serializer};

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<bool>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value: serde_json::Value = Deserialize::deserialize(deserializer)?;
        match value {
            serde_json::Value::Array(values) => {
                let result = values
                    .into_iter()
                    .map(|v| match v {
                        serde_json::Value::Bool(b) => Ok(b),
                        serde_json::Value::String(s) => Ok(s == "true"),
                        serde_json::Value::Number(n) if n.is_u64() => Ok(n.as_u64().unwrap() != 0),
                        _ => Err(serde::de::Error::custom(format!(
                            "invalid type for boolean {}",
                            v
                        ))),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(result)
            }
            _ => Err(serde::de::Error::custom(format!(
                "expected an array but got {}",
                value
            ))),
        }
    }

    pub fn serialize<S>(value: &bool, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Serialize::serialize(value, serializer)
    }
}
