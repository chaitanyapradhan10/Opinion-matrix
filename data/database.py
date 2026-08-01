from dotenv import load_dotenv
import os
import pandas as pd
from sqlalchemy import create_engine

load_dotenv()

engine = create_engine(
    f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}"
    f"@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/{os.getenv('MYSQL_DATABASE')}"
)

table_name = os.getenv("MYSQL_TABLE")

if not table_name:
    raise ValueError("MYSQL_TABLE is missing from the .env file.")

os.makedirs("data", exist_ok=True)

try:
    with engine.connect():
        print("Database connected successfully!")

    df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
    print(df.head())

    df.to_csv("data/reviews.csv", index=False)
    print("CSV saved successfully!")

except Exception as e:
    print(f"Error: {e}")