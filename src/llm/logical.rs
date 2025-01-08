/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

use datafusion::arrow::datatypes::{DataType, Field};
use datafusion::common::alias::AliasGenerator;
use datafusion::common::tree_node::{Transformed, TransformedResult};
use datafusion::common::{not_impl_err, plan_err, Column, DFSchema, DFSchemaRef, Result};
use datafusion::config::ConfigOptions;
use datafusion::logical_expr::{
    col, Aggregate, Expr, Extension, Filter, LogicalPlan, LogicalPlanBuilder, Partitioning,
    Projection, UserDefinedLogicalNodeCore,
};
use datafusion::optimizer::AnalyzerRule;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::{format, Debug, Formatter};
use std::sync::Arc;

#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub struct LLMPlan {
    pub schema: DFSchemaRef,
    pub input: Arc<LogicalPlan>,
    pub required_columns: Vec<Column>,
}

impl PartialOrd for LLMPlan {
    fn partial_cmp(&self, _other: &Self) -> Option<Ordering> {
        None
    }
}

impl UserDefinedLogicalNodeCore for LLMPlan {
    fn name(&self) -> &str {
        "LLM"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![self.input.as_ref()]
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<Expr> {
        self.schema
            .fields()
            .iter()
            .map(|field| col(field.name()))
            .collect()
    }

    fn fmt_for_explain(&self, f: &mut Formatter) -> std::fmt::Result {
        let columns: Vec<String> = self
            .required_columns
            .iter()
            .map(|c| c.name().to_string())
            .collect();
        write!(f, "LLM({})", columns.join(", "))
    }

    fn with_exprs_and_inputs(&self, _exprs: Vec<Expr>, inputs: Vec<LogicalPlan>) -> Result<Self> {
        Ok(LLMPlan {
            schema: self.schema.clone(),
            input: Arc::new(inputs[0].clone()),
            required_columns: self.required_columns.clone(),
        })
    }
}

pub struct LLMFunctionAnalyzeRule {}

impl Debug for LLMFunctionAnalyzeRule {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        f.debug_struct("LLMFunctionAnalyzeRule").finish()
    }
}

impl AnalyzerRule for LLMFunctionAnalyzeRule {
    fn analyze(&self, plan: LogicalPlan, config: &ConfigOptions) -> Result<LogicalPlan> {
        plan.transform_down_with_subqueries(|plan| self.analyze_llm_function(plan))
            .data()
    }

    fn name(&self) -> &str {
        "LLMFunctionAnalyzeRule"
    }
}

impl LLMFunctionAnalyzeRule {
    fn analyze_llm_function(&self, plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
        let alias_generator = AliasGenerator::new();
        let mut collected: HashMap<String, Vec<Expr>> = HashMap::new();

        // Find the scalar functions that are semantic functions and collect the required columns
        // The first argument is the promotion, the rest are the columns
        //  e.g. `llm_bool("If all of them are Aisa countries: {}, {}, {}", column1, column2, column3)`
        let plan = plan
            .map_expressions(|expr| {
                if let Expr::ScalarFunction(scalar_function) = &expr {
                    if scalar_function.name().starts_with("llm_") {
                        if scalar_function.args.len() == 0 {
                            return plan_err!("LLM function must have at least one argument");
                        }
                        // let promotion = scalar_function.args[0].clone();
                        let columns: Vec<_> = scalar_function.args[1..].iter().cloned().collect();
                        if !columns.is_empty() {
                            collected.insert(alias_generator.next("__llm"), columns);
                        }
                    }
                }
                Ok(Transformed::no(expr))
            })?
            .data;

        if collected.is_empty() {
            return Ok(Transformed::no(plan));
        }

        if collected.len() > 1 {
            return not_impl_err!("Only one LLM function is allowed in a query");
        }
        // TODO: currently support one LLM function
        let first = collected
            .get("__llm_1")
            .map(|v| v.clone())
            .unwrap_or_default();
        let schema: DFSchemaRef = Arc::new(DFSchema::from_unqualified_fields(
            vec![Field::new("c1", DataType::Boolean, false)].into(),
            HashMap::new(),
        )?);

        let input = input(&plan)?;
        let projection = LogicalPlanBuilder::from(input)
            .project(first.clone())?
            .build()?;

        let first = first
            .iter()
            .map(|e| {
                if let Expr::Column(c) = e {
                    Ok(c.clone())
                } else {
                    return not_impl_err!("Only column references are allowed in LLM functions");
                }
            })
            .collect::<Result<_>>()?;

        let node = LLMPlan {
            schema,
            input: Arc::new(projection),
            required_columns: first,
        };
        let llm_plan = LogicalPlan::Extension(Extension {
            node: Arc::new(node),
        });
        Ok(Transformed::yes(llm_plan))
    }
}

fn input(plan: &LogicalPlan) -> Result<Arc<LogicalPlan>> {
    Ok(match plan {
        LogicalPlan::Projection(Projection { input, .. }) => Arc::clone(input),
        LogicalPlan::Aggregate(Aggregate { input, .. }) => Arc::clone(input),
        LogicalPlan::Filter(Filter { input, .. }) => Arc::clone(input),
        _ => return not_impl_err!("Unsupported plan: {:?}", plan),
    })
}

#[cfg(test)]
mod test {
    use crate::llm::functions::ByPassScalarUDF;
    use crate::llm::logical::LLMFunctionAnalyzeRule;
    use datafusion::common::Result;
    use datafusion::execution::{FunctionRegistry, SessionStateBuilder};
    use datafusion::logical_expr::ScalarUDF;
    use datafusion::prelude::SessionContext;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_simple() -> Result<()> {
        let mut state = SessionStateBuilder::default()
            .with_analyzer_rules(vec![Arc::new(LLMFunctionAnalyzeRule {})])
            .with_optimizer_rules(vec![])
            .build();
        let udf = ByPassScalarUDF::new("llm_bool", datafusion::arrow::datatypes::DataType::Boolean);
        state.register_udf(Arc::new(ScalarUDF::new_from_impl(udf)))?;
        let ctx = SessionContext::new_with_state(state);
        ctx.sql("create table t1 (c1 int, c2 int, c3 int)")
            .await?
            .show()
            .await?;
        let plan = ctx.sql("select llm_bool('If all of them are Aisa countries: {}, {}, {}', c1, c2, c3) from t1")
            .await?.into_optimized_plan()?;
        println!("{}", plan);
        Ok(())
    }
}
