import pandas as pd
import shutil
from pathlib import Path

def main():
    print("Starting dataset filtering...")
    
    base_dir = Path('../data/raw')
    
    # ESC-50
    print("Processing ESC-50...")
    esc_dir = base_dir / 'ESC-50-master'
    esc_meta_path = esc_dir / 'meta' / 'esc50.csv'
    esc_audio_dir = esc_dir / 'audio'
    
    df_esc = pd.read_csv(esc_meta_path)
    esc_categories = ['scream', 'glass_breaking']
    df_esc_filtered = df_esc[df_esc['category'].isin(esc_categories)]
    print(f"Found {len(df_esc_filtered)} ESC-50 files.")
    
    # UrbanSound8K
    print("Processing UrbanSound8K...")
    us_dir = base_dir / 'UrbanSound8K'
    us_meta_path = us_dir / 'UrbanSound8K.csv'
    
    df_us = pd.read_csv(us_meta_path)
    us_categories = ['scream', 'glass_break', 'siren']
    df_us_filtered = df_us[df_us['class'].isin(us_categories)]
    print(f"Found {len(df_us_filtered)} UrbanSound8K files.")
    
    # Filtered dir
    filtered_dir = base_dir / 'filtered'
    filtered_dir.mkdir(exist_ok=True)
    
    metadata = []
    
    # Copy ESC-50
    print("Copying ESC-50 files...")
    for _, row in df_esc_filtered.iterrows():
        src_path = esc_audio_dir / row['filename']
        if src_path.exists():
            label = row['category'].replace('_', ' ')
            dest_filename = f"ESC50_{row['filename']}"
            dest_path = filtered_dir / dest_filename
            shutil.copy2(src_path, dest_path)
            metadata.append({'filename': dest_filename, 'label': label, 'source': 'ESC-50'})
            print(f"Copied ESC-50: {dest_filename}")
    
    # Copy UrbanSound8K
    print("Copying UrbanSound8K files...")
    for _, row in df_us_filtered.iterrows():
        fold_dir = us_dir / f'fold{int(row["fold"])}'
        src_path = fold_dir / row['slice_file_name']
        if src_path.exists():
            label = row['class'].replace('_', ' ')
            dest_filename = f"US8K_{row['slice_file_name']}"
            dest_path = filtered_dir / dest_filename
            shutil.copy2(src_path, dest_path)
            metadata.append({'filename': dest_filename, 'label': label, 'source': 'UrbanSound8K'})
            print(f"Copied US8K: {dest_filename}")
    
    # Metadata CSV
    print("Creating metadata.csv...")
    df_meta = pd.DataFrame(metadata)
    meta_path = base_dir / 'metadata.csv'
    df_meta.to_csv(meta_path, index=False)
    print(f"Completed! Filtered {len(metadata)} files to ../data/raw/filtered/. Metadata saved.")

if __name__ == "__main__":
    main()

