use crate::llm::functions::ByPassScalarUDF;
use crate::llm::logical::LLMFunctionAnalyzeRule;
use crate::llm::physical_planner::LLMQueryPlanner;
use datafusion::common::Result;
use datafusion::execution::{FunctionRegistry, SessionStateBuilder};
use datafusion::functions_aggregate::min_max::max_udaf;
use datafusion::logical_expr::ScalarUDF;
use datafusion::prelude::SessionContext;
use std::sync::Arc;

mod llm;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let mut state = SessionStateBuilder::default()
        .with_analyzer_rules(vec![Arc::new(LLMFunctionAnalyzeRule {})])
        .with_optimizer_rules(vec![])
        .with_query_planner(Arc::new(LLMQueryPlanner {}))
        .with_physical_optimizer_rules(vec![])
        .build();
    let udf = ByPassScalarUDF::new("llm_bool", datafusion::arrow::datatypes::DataType::Boolean);
    state.register_udf(Arc::new(ScalarUDF::new_from_impl(udf)))?;
    state.register_udaf(max_udaf())?;
    let ctx = SessionContext::new_with_state(state);
    ctx.sql("create table t1 (c1 int, c2 int, c3 int)")
        .await?
        .show()
        .await?;
    ctx.sql("insert into t1 values (1, 2, 3), (1, 2, 3), (1, 2, 3)")
        .await?
        .show()
        .await?;
    ctx.sql("insert into t1 values (1, 2, 3), (1, 2, 3), (1, 2, 3)")
        .await?
        .show()
        .await?;
    ctx.sql("insert into t1 values (1, 2, 3), (1, 2, 3), (1, 2, 3)")
        .await?
        .show()
        .await?;

    ctx.sql("explain select llm_bool('If all of them are Aisa countries: {}, {}, {}', t1.c1, t1.c2, t1.c3) from t1")
        .await?.show().await?;
    Ok(())
}
