import pandas as pd, json

with open(r'D:\ncaab_model\data\team_aliases.json') as f:
    aliases = json.load(f)

def norm(name):
    if name in aliases: return aliases[name]
    lower = name.lower()
    for k,v in aliases.items():
        if k.lower() == lower: return v
    words = name.split()
    for n in range(1, min(4, len(words))):
        s = ' '.join(words[:-n])
        if s in aliases: return aliases[s]
        for k,v in aliases.items():
            if k.lower() == s.lower(): return v
    return lower.replace(' ','_')

df = pd.read_parquet(r'D:\ncaab_model\data\processed\feature_matrix_full.parquet')
df['date'] = pd.to_datetime(df['date'])
past = df[df['date'] < '2026-02-24']

check = ['Iowa State Cyclones','Utah Utes','Auburn Tigers','Oklahoma Sooners','Duke Blue Devils','Notre Dame Fighting Irish']
for team_name in check:
    n = norm(team_name)
    rows = past[past['team_id'] == n]
    if len(rows):
        r = rows.sort_values('date').iloc[-1]
        adj_o = r.get('t_adj_o', float('nan'))
        adj_d = r.get('t_adj_d', float('nan'))
        print(n + ': adj_o=' + str(round(float(adj_o),1)) + ' adj_d=' + str(round(float(adj_d),1)) + ' date=' + str(r['date'].date()))
    else:
        print(n + ': NOT FOUND')

print('---all team_ids sample---')
print(sorted(past['team_id'].dropna().unique())[:40])
