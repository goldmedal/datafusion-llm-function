use crate::llm::async_func::AsyncFuncExpr;
use datafusion::arrow::datatypes::{Fields, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::Result;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PlanProperties,
};
use futures::stream::StreamExt;
use log::trace;
use std::any::Any;
use std::sync::Arc;

/// This structure evaluates  a set of async expressions on a record
/// batch producing a new record batch
///
/// This is similar to a ProjectionExec except that the functions can be async
///
/// The schema of the output of the AsyncFuncExec is:
/// Input columns followed by one column for each async expression
#[derive(Debug)]
pub struct AsyncFuncExec {
    /// The async expressions to evaluate
    async_exprs: Vec<AsyncFuncExpr>,
    input: Arc<dyn ExecutionPlan>,
    /// Cache holding plan properties like equivalences, output partitioning etc.
    cache: PlanProperties,
    metrics: ExecutionPlanMetricsSet,
}

impl AsyncFuncExec {
    pub fn new(async_exprs: Vec<AsyncFuncExpr>, input: Arc<dyn ExecutionPlan>) -> Self {
        // compute the output schema: input schema then async expressions
        let fields: Fields = input
            .schema()
            .fields()
            .iter()
            .cloned()
            .chain(
                async_exprs
                    .iter()
                    .map(|async_expr| Arc::new(async_expr.field(input.schema().as_ref()))),
            )
            .collect();
        let schema = Arc::new(Schema::new(fields));
        let cache = AsyncFuncExec::compute_properties(&input, schema).unwrap();

        Self {
            input,
            async_exprs,
            cache,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// This function creates the cache object that stores the plan properties
    /// such as schema, equivalence properties, ordering, partitioning, etc.
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

impl DisplayAs for AsyncFuncExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let expr: Vec<String> = self
                    .async_exprs
                    .iter()
                    .map(|async_expr| async_expr.to_string())
                    .collect();

                write!(f, "AsyncFuncExec: async_expr=[{}]", expr.join(", "))
            }
        }
    }
}

impl ExecutionPlan for AsyncFuncExec {
    fn name(&self) -> &str {
        "async_func"
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
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(AsyncFuncExec::new(
            self.async_exprs.clone(),
            Arc::clone(&self.input),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        trace!(
            "Start AsyncFuncExpr::execute for partition {} of context session_id {} and task_id {:?}",
            partition,
            context.session_id(),
            context.task_id()
        );
        // TODO figure out how to record metrics

        // first execute the input stream
        let input_stream = self.input.execute(partition, context.clone())?;

        // now, for each record batch, evaluate the async expressions and add the columns to the result
        let async_exprs_captured = Arc::new(self.async_exprs.clone());
        let schema_captured = self.schema();

        let stream_with_async_functions = input_stream.then(move |batch| {
            // need to clone *again* to capture the async_exprs and schema in the
            // stream and satisfy lifetime requirements.
            let async_exprs_captured = Arc::clone(&async_exprs_captured);
            let schema_captured = schema_captured.clone();
            let config_option = context.session_config().options().clone();

            async move {
                let batch = batch?;
                // append the result of evaluating the async expressions to the output
                let mut output_arrays = batch.columns().to_vec();
                for async_expr in async_exprs_captured.iter() {
                    let output = async_expr.invoke_with_args(&batch, &config_option).await?;
                    output_arrays.push(output.to_array(batch.num_rows())?);
                }
                let batch = RecordBatch::try_new(schema_captured, output_arrays)?;
                Ok(batch)
            }
        });

        // Adapt the stream with the output schema
        let adapter = RecordBatchStreamAdapter::new(self.schema(), stream_with_async_functions);
        Ok(Box::pin(adapter))
    }
}
