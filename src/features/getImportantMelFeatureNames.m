function featureColumnNames = getImportantMelFeatureNames()
% getImportantMelFeatureNames
% Return descriptive names for the 60 selected features:
% - 45 Mel-band statistical features (15 bands x 3 metrics)
% - 15 harmonic features

numImportantBands = 15;
numHarmonicFeatures = 15;
featureColumnNames = cell(1, 3 * numImportantBands + numHarmonicFeatures);

for bandIndex = 1:numImportantBands
    featureColumnNames{bandIndex} = sprintf('mean_log_mel_energy_band_%02d', bandIndex);
    featureColumnNames{numImportantBands + bandIndex} = sprintf('std_log_mel_energy_band_%02d', bandIndex);
    featureColumnNames{2 * numImportantBands + bandIndex} = sprintf('mean_abs_delta_log_mel_energy_band_%02d', bandIndex);
end

harmonicStartIndex = 3 * numImportantBands;
featureColumnNames{harmonicStartIndex + 1} = 'harmonic_voiced_frame_ratio';
featureColumnNames{harmonicStartIndex + 2} = 'harmonic_f0_mean_hz';
featureColumnNames{harmonicStartIndex + 3} = 'harmonic_f0_std_hz';
featureColumnNames{harmonicStartIndex + 4} = 'harmonic_f0_median_hz';
featureColumnNames{harmonicStartIndex + 5} = 'harmonic_f0_iqr_hz';
featureColumnNames{harmonicStartIndex + 6} = 'harmonic_f0_min_hz';
featureColumnNames{harmonicStartIndex + 7} = 'harmonic_f0_max_hz';
featureColumnNames{harmonicStartIndex + 8} = 'harmonic_f0_range_hz';
featureColumnNames{harmonicStartIndex + 9} = 'harmonic_f0_slope_hz_per_frame';
featureColumnNames{harmonicStartIndex + 10} = 'harmonic_autocorr_peak_mean';
featureColumnNames{harmonicStartIndex + 11} = 'harmonic_autocorr_peak_std';
featureColumnNames{harmonicStartIndex + 12} = 'harmonic_energy_ratio_mean';
featureColumnNames{harmonicStartIndex + 13} = 'harmonic_energy_ratio_std';
featureColumnNames{harmonicStartIndex + 14} = 'harmonic_peak_prominence_mean';
featureColumnNames{harmonicStartIndex + 15} = 'harmonic_peak_prominence_std';
end
