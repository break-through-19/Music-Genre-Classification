# Music-Genre-Classification
Classifies the genre of music files (.wav) using ML models. Trained using the GTZAN Dataset: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

Repo link: https://github.com/break-through-19/Music-Genre-Classification

## Steps to execute

1. Download the dataset from Kaggle: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download

2. From the downloaded dataset, move the contents of genres_original (contains folders of .wav files) folder to Music-Genre-Classification/dataset folder. 

3. Navigate to Home -> Add-Ons -> Search for "Audio Toolbox" -> Install.

4. Run MATLAB script:
   - `scripts/generate_mel_feature_csv.m`

5. To normalize the extracted feature CSV and generate 3D scatter plots:
   - `scripts/normalize_and_plot_feature_space.m`

## Feature Extraction Pipeline (MATLAB)

The MATLAB pipeline is modular and organized as:

- `scripts/generate_mel_feature_csv.m` : main entry point.
- `src/pipeline/processDatasetToMelFeatureTable.m` : dataset traversal + orchestration.
- `src/io/listGenreAudioFiles.m` : class/file discovery from folder structure.
- `src/audio/splitAudioIntoFixedSegments.m` : 30s audio -> 3s segments.
- `src/features/extractMelSpectrogramFeatureVector.m` : extract 16 important features (4 per category).
- `src/features/extractHarmonicFeatureVector.m` : compute harmonic feature block.
- `src/features/getImportantMelFeatureNames.m` : fixed names for the 16 features.
- `src/io/buildFeatureRowTable.m` : metadata + feature row construction.
- `src/io/alignColumnsToExampleCsv.m` : optional schema alignment with example CSV.
- `src/preprocessing/readFeatureDataset.m` : read and validate the feature CSV.
- `src/preprocessing/meanCenterAndNormalizeFeatureTable.m` : mean-center and z-score normalize feature columns.
- `src/visualization/getFeatureCategoryDefinitions.m` : define the four feature categories.
- `src/visualization/generateCategory3DScatterPlots.m` : write category-wise 3D scatter plots.

Output CSV:
- `data/features/mel_spectrogram_features.csv`
- `data/features/mel_spectrogram_features_normalized.csv`

Generated plot folder:
- `data/features/plots_3d_scatter`

Included metadata columns:
- `class_name` (from folder name)
- `wav_file_name` (from WAV file name)
- `segment_index` (1-based segment id within each WAV file)

Feature columns:
- Spectral/timbral level features (4):
  - `timbral_log_mel_mean`
  - `timbral_log_mel_median`
  - `timbral_band_energy_spread`
  - `timbral_frame_peak_mean`
- Spectral variability features (4):
  - `variability_log_mel_std_global`
  - `variability_band_std_mean`
  - `variability_frame_energy_std`
  - `variability_band_range_mean`
- Temporal dynamics features (4):
  - `temporal_frame_energy_delta_abs_mean`
  - `temporal_frame_energy_delta_std`
  - `temporal_zero_crossing_rate_mean`
  - `temporal_rms_std`
- Harmonic features (4):
  - `harmonic_voiced_frame_ratio`
  - `harmonic_f0_mean_hz`
  - `harmonic_f0_std_hz`
  - `harmonic_energy_ratio_mean`
