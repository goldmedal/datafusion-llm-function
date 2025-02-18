pub mod llm;

mod test {
    use crate::llm::functions::async_upper::AsyncUpper;
    use crate::llm::functions::AsyncScalarUDF;
    use crate::llm::physical_optimizer::AsyncFuncRule;
    use datafusion::arrow::array::{RecordBatch, StringArray};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::assert_batches_eq;
    use datafusion::common::Result;
    use datafusion::execution::{FunctionRegistry, SessionStateBuilder};
    use datafusion::prelude::SessionContext;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_project() -> Result<()> {
        let ctx = init_ctx()?;
        let sql = "select async_upper(c1) from test_table";
        let df = ctx.sql(sql).await?.collect().await?;
        let expected = vec![
            "+----------------------------+",
            "| async_upper(test_table.c1) |",
            "+----------------------------+",
            "| A                          |",
            "| B                          |",
            "| C                          |",
            "+----------------------------+",
        ];

        assert_batches_eq!(expected, &df);

        let sql = "select * from (select async_upper(c1) as c1 from test_table)";
        let df = ctx.sql(sql).await?.collect().await?;
        let expected = vec![
            "+----+", "| c1 |", "+----+", "| A  |", "| B  |", "| C  |", "+----+",
        ];

        assert_batches_eq!(expected, &df);
        Ok(())
    }

    #[tokio::test]
    async fn test_filter() -> Result<()> {
        let ctx = init_ctx()?;
        let sql = "select * from test_table where async_upper(c1) = async_upper(c2)";
        let df = ctx.sql(sql).await?.collect().await?;
        let expected = vec![
            "+----+----+",
            "| c1 | c2 |",
            "+----+----+",
            "| a  | a  |",
            "+----+----+",
        ];

        assert_batches_eq!(expected, &df);
        Ok(())
    }

    fn init_ctx() -> Result<SessionContext> {
        let mut state = SessionStateBuilder::default()
            .with_physical_optimizer_rule(Arc::new(AsyncFuncRule {}))
            .build();

        let async_upper = AsyncUpper::new();
        let udf = AsyncScalarUDF::new(Arc::new(async_upper));
        state.register_udf(udf.into_scalar_udf())?;
        let ctx = SessionContext::new_with_state(state);
        ctx.register_batch("test_table", test_table()?)?;
        Ok(ctx)
    }

    fn test_table() -> Result<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("c1", DataType::Utf8, false),
            Field::new("c2", DataType::Utf8, false),
        ]));

        let c1_array = StringArray::from(vec!["a", "b", "c"]);
        let c2_array = StringArray::from(vec!["a", "e", "f"]);

        Ok(RecordBatch::try_new(
            schema,
            vec![Arc::new(c1_array), Arc::new(c2_array)],
        )?)
    }
}
