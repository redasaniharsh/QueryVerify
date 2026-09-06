"""Manual test for verifier.verify_result() - fixed 7-row result for determinism."""

from app.agents.verifier import verify_result

question = "What is the total sales amount by country?"
sql = (
    "SELECT dim_customers.country, SUM(fact_sales.sales_amount) AS total_sales_amount "
    "FROM dim_customers JOIN fact_sales ON dim_customers.customer_key = fact_sales.customer_key "
    "GROUP BY dim_customers.country"
)
result = {
    "success": True,
    "columns": ["country", "total_sales_amount"],
    "rows": [
        {"country": None, "total_sales_amount": 226820},
        {"country": "Australia", "total_sales_amount": 9060172},
        {"country": "Canada", "total_sales_amount": 1977738},
        {"country": "France", "total_sales_amount": 2643751},
        {"country": "Germany", "total_sales_amount": 2894066},
        {"country": "United Kingdom", "total_sales_amount": 3391376},
        {"country": "United States", "total_sales_amount": 9162327},
    ],
}

verification = verify_result(question, sql, result)
print(verification)