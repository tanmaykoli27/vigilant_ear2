import pandas as pd
import shutil
from pathlib import Path

df = pd.read_csv('../data/raw/metadata.csv')
filtered = Path('../data/raw/filtered/')

for label in df['label'].unique():
    Path(f'../data/raw/sorted/{label}').mkdir(parents=True, exist_ok=True)

for _, row in df.iterrows():
    src = filtered / row['filename']
    label = row['label']
    filename = row['filename']
    dst = Path(f'../data/raw/sorted/{label}/{filename}')
    if src.exists():
        shutil.copy2(src, dst)

print('Done! Files sorted into:')
for label in df['label'].unique():
    count = len(list(Path(f'../data/raw/sorted/{label}').glob('*')))
    print(f'  {label}: {count} files')