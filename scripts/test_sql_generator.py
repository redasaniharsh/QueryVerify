"""Manual test for sql_generator.generate_sql()."""

from sqlalchemy import create_engine
from app.db.schema_introspector import get_schema_context
from app.agents.sql_generator import generate_sql

engine = create_engine("sqlite:///data/sample.db")
schema = get_schema_context(engine)

sql = generate_sql(
    question="What is the total sales amount by country?",
    schema_context=schema,
)

print(sql)
