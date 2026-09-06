"""Manual test for executor.execute_sql()."""

from sqlalchemy import create_engine
from app.agents.executor import execute_sql

engine = create_engine("sqlite:///data/sample.db")

sql = (
    "SELECT dim_customers.country, SUM(fact_sales.sales_amount) AS total_sales_amount "
    "FROM dim_customers JOIN fact_sales ON dim_customers.customer_key = fact_sales.customer_key "
    "GROUP BY dim_customers.country"
)

result = execute_sql(sql, engine)

print("success:", result["success"])
if result["success"]:
    print("columns:", result["columns"])
    print("row_count:", len(result["rows"]))
    for row in result["rows"]:
        print(row)
else:
    print("error:", result["error"])
