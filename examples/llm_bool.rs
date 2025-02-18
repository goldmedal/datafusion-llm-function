use datafusion::common::Result;
use datafusion::config::{ConfigOptions, Extensions};
use datafusion::execution::{FunctionRegistry, SessionStateBuilder};
use datafusion::functions_aggregate::min_max::max_udaf;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_llm_function::llm::functions::config::LLMConfig;
use datafusion_llm_function::llm::functions::llm_bool::LLMBool;
use datafusion_llm_function::llm::functions::AsyncScalarUDF;
use datafusion_llm_function::llm::physical_optimizer::AsyncFuncRule;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let mut extensions = Extensions::new();
    extensions.insert(LLMConfig::default());

    let mut config = ConfigOptions::new().with_extensions(extensions);
    config.set("llm.model", "llama3.2:3b")?;
    config.set("llm.api_key", "")?;
    config.set("llm.chat_endpoint", "http://localhost:11434/api/chat")?;

    let session_config = SessionConfig::from(config);

    let mut state = SessionStateBuilder::default()
        .with_physical_optimizer_rule(Arc::new(AsyncFuncRule {}))
        .with_config(session_config)
        .build();

    let llm_bool = LLMBool::new();
    let udf = AsyncScalarUDF::new(Arc::new(llm_bool));
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

    ctx.sql("explain select llm_bool('Does {name} locates at {region}?', c.name, c.region) from country c")
        .await?.show().await?;

    match ctx
        .sql("select llm_bool('Does {name} locate at {region}?', c.name, c.region) from country c")
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
