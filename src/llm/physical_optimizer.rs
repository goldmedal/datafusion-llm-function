use crate::llm::async_func::AsyncFuncExpr;
use crate::llm::exec::AsyncFuncExec;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::config::ConfigOptions;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{PhysicalExpr, ScalarFunctionExpr};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::coalesce_batches::CoalesceBatchesExec;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::ExecutionPlan;
use std::sync::Arc;
use datafusion::error::DataFusionError;

#[derive(Debug)]
pub struct AsyncFuncRule {}

impl PhysicalOptimizerRule for AsyncFuncRule {
    /// Insert a AsyncFunctionNode node in front of this projection if there are any async functions in it
    ///
    /// For example, if the projection is:
    /// ```text
    ///   ProjectionExec(["A", "B", llm_func('foo', "C") + 1])
    /// ```
    ///
    /// Rewrite to
    ///   ProjectionExec(["A", "B", "__async_fn_1" + 1]) <-- note here that the async function is not evaluated and instead is a new column
    ///     AsyncFunctionNode(["A", "B", llm_func('foo', "C")])
    ///
    /// If the async function has an ideal batch size, coalesce the batches to that size. For example, if the ideal batch size is 64:
    /// ```
    ///   ProjectionExec(["A", "B", "__async_fn_1" + 1])
    ///     AsyncFunctionNode(["A", "B", llm_func('foo', "C")])
    ///       CoalesceBatchesExec(target_batch_size=64)
    /// ```
    ///
    /// If there are multiple async functions, the batch size will be the max of all of them.
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        // replace ProjectionExec with async exec there are any async functions
        // TODO: handle other types of ExecutionPlans (like Filter)
        let Some(proj_exec) = plan.as_any().downcast_ref::<ProjectionExec>() else {
            return Ok(plan);
        };

        // find any instances of async functions in the expressions
        let num_input_columns = proj_exec.input().schema().fields().len();
        let mut async_map = AsyncMapper::new(num_input_columns);
        proj_exec.expr().iter().for_each(|(expr, _column_name)| {
            async_map.find_references(expr);
        });

        if async_map.is_empty() {
            return Ok(plan);
        }

        // rewrite the projection's expressions in terms of the columns with the result of async evaluation
        let new_exprs = proj_exec
            .expr()
            .iter()
            .map(|(expr, column_name)| {
                let new_expr = Arc::clone(expr)
                    .transform_up(|e| Ok(async_map.map_expr(e)))
                    .expect("no failures as closure is infallible");
                (new_expr.data, column_name.to_string())
            })
            .collect::<Vec<_>>();

        let max_ideal_size = async_map
            .async_exprs
            .iter()
            .map(|expr| expr.ideal_batch_size())
            .collect::<Result<Vec<_>, DataFusionError>>()?
            .into_iter()
            .flatten()
            .max();

        // If any of the async functions have an ideal batch size, coalesce the batches
        let async_exec = if max_ideal_size.is_some() {
            let new_ideal_size = max_ideal_size.unwrap();
            let coal_batch =
                CoalesceBatchesExec::new(Arc::clone(proj_exec.input()), new_ideal_size);
            AsyncFuncExec::new(async_map.async_exprs, Arc::new(coal_batch))
        } else {
            AsyncFuncExec::new(async_map.async_exprs, Arc::clone(proj_exec.input()))
        };

        let new_proj_exec = ProjectionExec::try_new(new_exprs, Arc::new(async_exec))?;

        Ok(Arc::new(new_proj_exec) as _)
    }

    fn name(&self) -> &str {
        "async_func_rule"
    }

    /// verify the schema has not changed
    fn schema_check(&self) -> bool {
        true
    }
}

/// Maps async_expressions to new columns
///
/// The output of the async functions are appended, in order, to the end of the input schema
#[derive(Debug)]
struct AsyncMapper {
    /// the number of columns in the input plan
    /// used to generate the output column names.
    /// the first async expr is `__async_fn_0`, the second is `__async_fn_1`, etc
    num_input_columns: usize,
    /// the expressions to map
    async_exprs: Vec<AsyncFuncExpr>,
}

impl AsyncMapper {
    pub fn new(num_input_columns: usize) -> Self {
        Self {
            num_input_columns,
            async_exprs: Vec::new(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.async_exprs.is_empty()
    }

    /// Finds any references to async functions in the expression and adds them to the map
    pub fn find_references(&mut self, proj_expr: &Arc<dyn PhysicalExpr>) {
        // recursively look for references to async functions
        proj_expr
            .apply(|expr| {
                if let Some(func) = expr.as_any().downcast_ref::<ScalarFunctionExpr>() {
                    if AsyncFuncExpr::is_async_func(func.fun()) {
                        let next_name = format!("__async_fn_{}", self.async_exprs.len());
                        self.async_exprs
                            .push(AsyncFuncExpr::new(next_name, Arc::clone(expr)));
                    }
                }
                Ok(TreeNodeRecursion::Continue)
            })
            .expect("no failures as closure is infallible");
    }

    /// If the expression matches any of the async functions, return the new column
    pub fn map_expr(&self, expr: Arc<dyn PhysicalExpr>) -> Transformed<Arc<dyn PhysicalExpr>> {
        // find the first matching async function if any
        let Some(idx) = self
            .async_exprs
            .iter()
            .enumerate()
            .find_map(
                |(idx, async_expr)| {
                    if async_expr == &expr {
                        Some(idx)
                    } else {
                        None
                    }
                },
            )
        else {
            return Transformed::no(expr);
        };
        // rewrite in terms of the output column
        Transformed::yes(self.output_column(idx))
    }

    /// return the output column for the async function at index idx
    pub fn output_column(&self, idx: usize) -> Arc<dyn PhysicalExpr> {
        let async_expr = &self.async_exprs[idx];
        let output_idx = self.num_input_columns + idx;
        Arc::new(Column::new(async_expr.name(), output_idx))
    }
}
