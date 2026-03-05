import sqlite3
import pandas as pd

conn = sqlite3.connect('data/ncaab.db')
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print('Tables:', [r[0] for r in cur.fetchall()])

cur.execute('PRAGMA table_info(odds)')
cols = [r[1] for r in cur.fetchall()]
print('Odds columns:', cols)

# Show sample odds rows
df = pd.read_sql('SELECT * FROM odds LIMIT 5', conn)
print('\nSample odds rows:')
print(df.to_string())

# Check for the bad lines
date_col = next((c for c in cols if 'date' in c.lower() or 'time' in c.lower() or 'game' in c.lower()), cols[0])
print(f'\nUsing column "{date_col}" for sorting')
df2 = pd.read_sql(f'SELECT * FROM odds ORDER BY {date_col} DESC LIMIT 20', conn)
spread_col = next((c for c in cols if 'spread' in c.lower()), None)
total_col = next((c for c in cols if 'total' in c.lower()), None)
home_col = next((c for c in cols if 'home' in c.lower() and 'team' in c.lower()), None)
away_col = next((c for c in cols if 'away' in c.lower() and 'team' in c.lower()), None)

show_cols = [c for c in [home_col, away_col, spread_col, total_col, date_col] if c]
print('\nRecent odds:')
print(df2[show_cols].to_string())
