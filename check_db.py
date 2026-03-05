import sqlite3, pandas as pd
conn = sqlite3.connect('data/ncaab.db')
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
print('Tables:', tables['name'].tolist())
df = pd.read_sql("SELECT date, COUNT(*) as total, SUM(CASE WHEN actual_margin IS NOT NULL THEN 1 ELSE 0 END) as graded FROM predictions GROUP BY date ORDER BY date", conn)
print(df.to_string())
conn.close()
