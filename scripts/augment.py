import pandas as pd
import numpy as np
from pathlib import Path
import librosa
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

def audio_to_melspec(y, sr=16000, n_mels=128, n_fft=2048, hop_length=512):
    """Extract log mel-spectrogram."""
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel

def main():
    print("Starting data augmentation...")
    
    processed_dir = Path('../data/processed')
    raw_dir = Path('../data/raw')
    filtered_dir = raw_dir / 'filtered'
    augmented_dir = Path('../data/augmented')
    augmented_dir.mkdir(exist_ok=True)
    
    # Read processed metadata
    meta_path = processed_dir / 'metadata.csv'
    if not meta_path.exists():
        print("Run scripts/preprocess.py first!")
        return
    
    df_meta = pd.read_csv(meta_path)
    print(f"Augmenting {len(df_meta)} files (3 augs each)...")
    
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0),
        TimeStretch(min_rate=0.8, max_rate=1.2, p=1.0),
        PitchShift(min_semitones=-2, max_semitones=2, p=1.0)
    ])
    
    aug_meta = []
    success_count = 0
    sr = 16000
    
    for idx, row in df_meta.iterrows():
        wav_path = filtered_dir / row['filename'].replace('.npy', '.wav')
        if not wav_path.exists():
            print(f"Missing WAV: {wav_path}")
            continue
        
        try:
            y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
            
            # Apply each augmentation separately
            aug_names = ['noise', 'timestretch', 'pitchshift']
            for aug_name in aug_names:
                # Note: Compose applies all, but to get individual, we create separate pipelines
                if aug_name == 'noise':
                    pipeline = Compose([AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0)])
                elif aug_name == 'timestretch':
                    pipeline = Compose([TimeStretch(min_rate=0.8, max_rate=1.2, p=1.0)])
                else:
                    pipeline = Compose([PitchShift(min_semitones=-2, max_semitones=2, p=1.0)])
                
                y_aug = pipeline(samples=y.astype(np.float32), sample_rate=sr)
                
                # Ensure length 32000
                if len(y_aug) > 32000:
                    y_aug = y_aug[:32000]
                elif len(y_aug) < 32000:
                    y_aug = np.pad(y_aug, (0, 32000 - len(y_aug)), mode='constant')
                
                mel = audio_to_melspec(y_aug, sr)
                
                # Save
                orig_stem = Path(row['filename']).stem
                spec_filename = f"{orig_stem}_{aug_name}.npy"
                spec_path = augmented_dir / spec_filename
                np.save(spec_path, mel)
                
                aug_meta.append({
                    'filename': spec_filename,
                    'label': row['label'],
                    'source': f"{row['source']}_{aug_name}"
                })
                success_count += 1
            
        except Exception as e:
            print(f"Error augmenting {row['filename']}: {e}")
            continue
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(df_meta)} files ({success_count} augs successful).")
    
    # Save augmented metadata
    df_aug = pd.DataFrame(aug_meta)
    df_aug.to_csv(augmented_dir / 'metadata.csv', index=False)
    print(f"Augmentation complete! {success_count} augmented spectrograms saved to ../data/augmented/.")

if __name__ == "__main__":
    main()

