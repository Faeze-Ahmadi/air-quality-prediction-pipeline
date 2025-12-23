import sqlite3
import pandas as pd

conn = sqlite3.connect("data/aqi_history.sqlite")
df = pd.read_sql_query("SELECT * FROM aqi_readings", conn)
conn.close()

print(df.head())
print("Total rows:", len(df))
