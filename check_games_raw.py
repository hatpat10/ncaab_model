import sqlite3

conn = sqlite3.connect('data/ncaab.db')
cur = conn.cursor()

cur.execute('SELECT COUNT(*) FROM games_raw')
print('games_raw total:', cur.fetchone()[0])

cur.execute('PRAGMA table_info(games_raw)')
cols = [r[1] for r in cur.fetchall()]
print('games_raw columns:', cols)

date_col = next((c for c in cols if 'date' in c.lower()), None)
if date_col:
    cur.execute(f'SELECT MAX({date_col}) FROM games_raw')
    print('Latest date:', cur.fetchone()[0])
    cur.execute(f'SELECT COUNT(*) FROM games_raw WHERE {date_col} >= "2026-02-22"')
    print('Games since 2/22:', cur.fetchone()[0])
else:
    print('No date column found — showing first 3 rows:')
    cur.execute('SELECT * FROM games_raw LIMIT 3')
    for row in cur.fetchall():
        print(row)