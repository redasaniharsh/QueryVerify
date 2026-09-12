"""
Loads the same 3 CSVs (fact_sales, dim_customers, dim_products) into a
Microsoft SQL Server database, one table per CSV, using pandas + SQLAlchemy.

Just like scripts/setup_db.py does for SQLite, this creates the tables and
declares the real primary/foreign key constraints that we rely on for schema
introspection (the SQL generator sees these join rules in its prompt context).

Usage:
    py -3.11 scripts/setup_db_sqlserver.py [database_name]
        database_name defaults to "QueryVerifyTest" on .\\SQLEXPRESS.

The connection string may be overridden with the MssqlURL environment
variable; the default is:

    mssql+pyodbc://.\\SQLEXPRESS/<database>?driver=ODBC+Driver+17+for+SQL+Server
                               &trusted_connection=yes
"""

import os
import sys

import pandas as pd
from sqlalchemy import NVARCHAR, create_engine, text

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

CSV_TABLE_MAP = [
    ("dim_customers.csv", "dim_customers", "customer_key"),
    ("dim_products.csv", "dim_products", "product_key"),
    ("fact_sales.csv", "fact_sales", None),
]

FKS = [
    (
        "fact_sales",
        ["customer_key"],
        "dim_customers",
        ["customer_key"],
        "fk_fact_customer",
    ),
    (
        "fact_sales",
        ["product_key"],
        "dim_products",
        ["product_key"],
        "fk_fact_product",
    ),
]


def main():
    database = sys.argv[1] if len(sys.argv) > 1 else "QueryVerifyTest"
    url = os.environ.get(
        "MSSQL_URL",
        (
            "mssql+pyodbc://.\\SQLEXPRESS/"
            f"{database}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
        ),
    )
    engine = create_engine(url)

    # Non-date text columns keep their inferred NVARCHAR; date-like string
    # columns are explicitly NVARCHAR(10) 'YYYY-MM-DD' so the generated SQL's
    # string date comparisons behave identically to the SQLite sample.
    date_cols = {
        "dim_customers": ["birthdate", "create_date"],
        "dim_products": ["start_date"],
        "fact_sales": ["order_date", "shipping_date", "due_date"],
    }
    dtype_overrides = {
        "maintenance": NVARCHAR(50),
        "category_id": NVARCHAR(50),
        "category": NVARCHAR(50),
        "subcategory": NVARCHAR(50),
        "product_line": NVARCHAR(50),
    }

    with engine.begin() as conn:
        for csv_file, table_name, pk in CSV_TABLE_MAP:
            csv_path = os.path.join(DATA_DIR, csv_file)
            df = pd.read_csv(csv_path)

            dtypes = {}
            for c in date_cols.get(table_name, []):
                dtypes[c] = NVARCHAR(10)
            for c, t in dtype_overrides.items():
                if c in df.columns:
                    dtypes[c] = t

            df.to_sql(
                table_name,
                conn,
                if_exists="replace",
                index=False,
                dtype=dtypes,
            )
            print(f"Loaded {len(df)} rows into {table_name}")

            if pk:
                row = conn.execute(
                    text(
                        "SELECT DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS "
                        "WHERE TABLE_NAME = :t AND COLUMN_NAME = :c"
                    ),
                    {"t": table_name, "c": pk},
                ).fetchone()
                conn.execute(
                    text(
                        f"ALTER TABLE [{table_name}] ALTER COLUMN [{pk}] "
                        f"{row[0]} NOT NULL"
                    )
                )
                conn.execute(
                    text(
                        f"ALTER TABLE [{table_name}] ADD CONSTRAINT "
                        f"PK_{table_name} PRIMARY KEY ([{pk}])"
                    )
                )
                print(f"  primary key {pk} on {table_name}")

        for table, cols, ref_table, ref_cols, name in FKS:
            col_list = ", ".join(f"[{c}]" for c in cols)
            ref_list = ", ".join(f"[{c}]" for c in ref_cols)
            conn.execute(
                text(
                    f"ALTER TABLE [{table}] WITH NOCHECK ADD CONSTRAINT [{name}] "
                    f"FOREIGN KEY ({col_list}) REFERENCES [{ref_table}] ({ref_list})"
                )
            )
            print(f"  foreign key {name}: {table}({', '.join(cols)}) -> "
                  f"{ref_table}({', '.join(ref_cols)})")

    print(f"\nDatabase '{database}' populated on SQL Server.")


if __name__ == "__main__":
    main()