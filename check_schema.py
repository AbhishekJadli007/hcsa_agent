import pandas as pd
import re
import os

def normalize(name):
    name = str(name).strip().lower()
    name = re.sub(r'[^a-z0-9]+', '_', name)
    return name.strip('_')

folder = 'data/Structured Datasets'
for fname in sorted(os.listdir(folder)):
    if fname.endswith('.xlsx') and not fname.startswith('~$'):
        df = pd.read_excel(os.path.join(folder, fname))
        print(f'=== {fname} ({len(df)} rows) ===')
        for c in df.columns:
            print(f'  {c!r:40} -> {normalize(c)!r}')
        print()