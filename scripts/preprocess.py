import pandas as pd
import numpy as np
import librosa
from pathlib import Path

def preprocess_audio(audio_path, sr=16000, target_length=32000, top_db=20):
    """Load, trim, pad/cut, return waveform."""
    try:
        y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
        # Trim silence
        y_trim, _ = librosa.effects.trim(y, top_db=top_db)
        # Pad or truncate to 2s (32000 samples)
        if len(y_trim) < target_length:
            y_pad = np.pad(y_trim, (0, target_length - len(y_trim)), mode='constant')
        else:
            y_pad = y_trim[:target_length]
        return y_pad
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def audio_to_melspec(y, sr=16000, n_mels=128, n_fft=2048, hop_length=512):
    """Extract log mel-spectrogram."""
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel

def main():
    print("Starting audio preprocessing...")
    
    raw_dir = Path('../data/raw')
    filtered_dir = raw_dir / 'filtered'
    processed_dir = Path('../data/processed')
    processed_dir.mkdir(exist_ok=True)
    
    # Read metadata
    meta_path = raw_dir / 'metadata.csv'
    if not meta_path.exists():
        print("Run scripts/download.py first!")
        return
    
    df_meta = pd.read_csv(meta_path)
    print(f"Processing {len(df_meta)} files...")
    
    processed_meta = []
    success_count = 0
    
    for idx, row in df_meta.iterrows():
        audio_path = filtered_dir / row['filename']
        if not audio_path.exists():
            print(f"Missing: {audio_path}")
            continue
        
        y = preprocess_audio(audio_path)
        if y is None:
            continue
        
        mel = audio_to_melspec(y)
        
        # Save npy (replace .wav with .npy)
        spec_filename = audio_path.stem + '.npy'
        spec_path = processed_dir / spec_filename
        np.save(spec_path, mel)
        
        processed_meta.append({
            'filename': spec_filename,
            'label': row['label'],
            'source': row['source']
        })
        success_count += 1
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(df_meta)} files ({success_count} successful).")
    
    # Save processed metadata
    df_processed = pd.DataFrame(processed_meta)
    df_processed.to_csv(processed_dir / 'metadata.csv', index=False)
    print(f"Preprocessing complete! {success_count} spectrograms saved to ../data/processed/.")

if __name__ == "__main__":
    main()

