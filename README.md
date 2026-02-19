# Music-Genre-Classification
Classifies the genre of music files (.wav) using ML models. Trained using the GTZAN Dataset: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification


## Steps to execute

1. Download the dataset from Kaggle: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download

2. From the downloaded dataset, move the contents of genres_original (contains folders of .wav files) folder to Music-Genre-Classification/dataset folder. 

3. Run MATLAB script:
   - `scripts/generate_mel_feature_csv.m`

## Feature Extraction Pipeline (MATLAB)

The MATLAB pipeline is modular and organized as:

- `scripts/generate_mel_feature_csv.m` : main entry point.
- `src/pipeline/processDatasetToMelFeatureTable.m` : dataset traversal + orchestration.
- `src/io/listGenreAudioFiles.m` : class/file discovery from folder structure.
- `src/audio/splitAudioIntoFixedSegments.m` : 30s audio -> 3s segments.
- `src/features/extractMelSpectrogramFeatureVector.m` : extract 60 important features (Mel + harmonic).
- `src/features/extractHarmonicFeatureVector.m` : compute harmonic feature block.
- `src/features/getImportantMelFeatureNames.m` : fixed names for the 60 features.
- `src/io/buildFeatureRowTable.m` : metadata + feature row construction.
- `src/io/alignColumnsToExampleCsv.m` : optional schema alignment with example CSV.

Output CSV:
- `data/features/mel_spectrogram_features.csv`

Included metadata columns:
- `class_name` (from folder name)
- `wav_file_name` (from WAV file name)
- `segment_index` (1-based segment id within each WAV file)

Feature columns:
- Mel features (45 total):
  - `mean_log_mel_energy_band_01 ... mean_log_mel_energy_band_15`
  - `std_log_mel_energy_band_01 ... std_log_mel_energy_band_15`
  - `mean_abs_delta_log_mel_energy_band_01 ... mean_abs_delta_log_mel_energy_band_15`
- Harmonic features (15 total):
  - `harmonic_voiced_frame_ratio`
  - `harmonic_f0_mean_hz`
  - `harmonic_f0_std_hz`
  - `harmonic_f0_median_hz`
  - `harmonic_f0_iqr_hz`
  - `harmonic_f0_min_hz`
  - `harmonic_f0_max_hz`
  - `harmonic_f0_range_hz`
  - `harmonic_f0_slope_hz_per_frame`
  - `harmonic_autocorr_peak_mean`
  - `harmonic_autocorr_peak_std`
  - `harmonic_energy_ratio_mean`
  - `harmonic_energy_ratio_std`
  - `harmonic_peak_prominence_mean`
  - `harmonic_peak_prominence_std`
