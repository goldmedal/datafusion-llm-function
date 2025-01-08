use datafusion::arrow::array::{record_batch, BooleanArray, RecordBatchIterator};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::{internal_err, Column, Result};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::ColumnarValue;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_optimizer::pruning::RequiredColumns;
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PlanProperties, RecordBatchStream,
};
use futures::stream::{Stream, StreamExt};
use log::trace;
use serde::Serialize;
use std::any::Any;
use std::env;
use std::fmt::{Display, Formatter};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

#[derive(Debug)]
pub struct LLMExec {
    input: Arc<dyn ExecutionPlan>,
    /// Cache holding plan properties like equivalences, output partitioning etc.
    cache: PlanProperties,
    metrics: ExecutionPlanMetricsSet,
}

impl LLMExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, metrics: ExecutionPlanMetricsSet) -> Self {
        let cache = LLMExec::compute_properties(&input, input.schema().clone()).unwrap();
        Self {
            input,
            cache,
            metrics,
        }
    }

    /// This function creates the cache object that stores the plan properties such as schema, equivalence properties, ordering, partitioning, etc.
    fn compute_properties(
        input: &Arc<dyn ExecutionPlan>,
        schema: SchemaRef,
    ) -> Result<PlanProperties> {
        let eq_properties = EquivalenceProperties::new(schema);

        // TODO: This is a dummy partitioning. We need to figure out the actual partitioning.
        let output_partitioning = Partitioning::RoundRobinBatch(1);

        Ok(PlanProperties::new(
            eq_properties,
            output_partitioning,
            input.pipeline_behavior(),
            input.boundedness(),
        ))
    }
}

impl DisplayAs for LLMExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "LLM()")
            }
        }
    }
}

impl ExecutionPlan for LLMExec {
    fn name(&self) -> &str {
        "llm"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(LLMExec::new(
            children[0].clone(),
            self.metrics.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        trace!("Start ProjectionExec::execute for partition {} of context session_id {} and task_id {:?}", partition, context.session_id(), context.task_id());
        Ok(Box::pin(LLMStream {
            required_columns: vec![],
            promotion: "".to_string(),
            input: self.input.execute(partition, context)?,
            baseline_metrics: BaselineMetrics::new(&self.metrics, partition),
        }))
    }
}

pub struct LLMStream {
    required_columns: Vec<Column>,
    promotion: String,
    input: SendableRecordBatchStream,
    baseline_metrics: BaselineMetrics,
}

impl Stream for LLMStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let poll = self.input.poll_next_unpin(cx).map(|x| match x {
            Some(Ok(batch)) => {
                let result = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(self.mock_llm_bool(&batch))
                });
                Some(result)
            }
            other => other,
        });
        self.baseline_metrics.record_poll(poll)
    }
}

impl RecordBatchStream for LLMStream {
    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }
}

impl LLMStream {
    async fn mock_llm_bool(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let shared_result = Arc::new(Mutex::new(None));
        let result_clone = Arc::clone(&shared_result);
        let num = batch.num_rows();

        // try to simulate the LLM async call
        let handle = tokio::spawn(async move {
            println!("Called LLM mock function with {} rows", num);
            let record = vec![true; num];
            let bool_array = BooleanArray::from(record);
            let schema = Schema::new(vec![Field::new("llm_bool", DataType::Boolean, false)]);
            let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(bool_array)]).unwrap();
            let mut result = result_clone.lock().unwrap();
            *result = Some(batch);
        });

        handle.await.unwrap();
        let result = Arc::clone(&shared_result).lock().unwrap().take().unwrap();
        Ok(result)
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
