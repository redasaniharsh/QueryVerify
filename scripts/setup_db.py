"""
Creates data/sample.db from the CSVs already sitting in data/:
  - dim_customers.csv  (customer_key, customer_id, customer_number, first_name,
                         last_name, country, marital_status, gender, birthdate,
                         create_date)
  - dim_products.csv   (product_key, product_id, product_number, product_name,
                         category_id, category, subcategory, maintenance, cost,
                         product_line, start_date)
  - fact_sales.csv      (order_number, product_key, customer_key, order_date,
                         shipping_date, due_date, sales_amount, quantity, price)

fact_sales.product_key -> dim_products.product_key
fact_sales.customer_key -> dim_customers.customer_key
"""

import os
import pandas as pd
from sqlalchemy import create_engine

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_PATH = os.path.join(DATA_DIR, "sample.db")

CSV_TABLE_MAP = {
    "dim_customers.csv": "dim_customers",
    "dim_products.csv": "dim_products",
    "fact_sales.csv": "fact_sales",
}


def main():
    engine = create_engine(f"sqlite:///{DB_PATH}")

    for csv_file, table_name in CSV_TABLE_MAP.items():
        csv_path = os.path.join(DATA_DIR, csv_file)
        df = pd.read_csv(csv_path)
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        print(f"Loaded {len(df)} rows into {table_name}")

    print(f"\nDatabase written to {DB_PATH}")


if __name__ == "__main__":
    main()
