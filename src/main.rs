use crate::llm::functions::async_upper::AsyncUpper;
use crate::llm::functions::config::LLMConfig;
use crate::llm::functions::llm_bool::LLMBool;
use crate::llm::functions::AsyncScalarUDF;
use crate::llm::physical_optimizer::AsyncFuncRule;
use datafusion::common::Result;
use datafusion::config::{ConfigOptions, Extensions};
use datafusion::execution::{FunctionRegistry, SessionStateBuilder};
use datafusion::functions_aggregate::min_max::max_udaf;
use datafusion::prelude::{SessionConfig, SessionContext};
use std::sync::Arc;

pub mod llm;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let mut state = SessionStateBuilder::default()
        .with_physical_optimizer_rule(Arc::new(AsyncFuncRule {}))
        .build();

    let async_upper = AsyncUpper::new();
    let udf = AsyncScalarUDF::new(Arc::new(async_upper));
    state.register_udf(udf.into_scalar_udf())?;
    state.register_udaf(max_udaf())?;
    let ctx = SessionContext::new_with_state(state);
    ctx.sql("create table country (name string, region string)")
        .await?
        .show()
        .await?;
    ctx.sql("insert into country values ('taiwan', 'asia'), ('usa', 'north america')")
        .await?
        .show()
        .await?;
    ctx.sql("insert into country values  ('china', 'asia'), ('french', 'asia')")
        .await?
        .show()
        .await?;
    ctx.sql("insert into country values  ('french', 'europe'), ('japan', 'south america')")
        .await?
        .show()
        .await?;

    ctx.sql("explain select async_upper(name) from country c")
        .await?
        .show()
        .await?;

    match ctx
        .sql("select async_upper(name) from country c")
        .await?
        .show()
        .await
    {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}
