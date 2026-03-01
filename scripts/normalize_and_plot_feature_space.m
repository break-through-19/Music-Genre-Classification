% normalize_and_plot_feature_space.m
% Read the extracted feature CSV, apply mean-centering + z-score
% normalization to the numeric feature columns, save the normalized table,
% and generate category-wise 3D scatter plots from the original features.

clear;
clc;
close all;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(projectRoot, 'src')));

inputCsvPath = fullfile(projectRoot, 'data', 'features', 'mel_spectrogram_features.csv');
outputCsvPath = fullfile(projectRoot, 'data', 'features', 'mel_spectrogram_features_normalized.csv');
plotOutputRoot = fullfile(projectRoot, 'data', 'features', 'plots_3d_scatter');

featureCategories = getFeatureCategoryDefinitions();
featureColumnNames = collectFeatureColumnNames(featureCategories);

fprintf('Reading feature dataset: %s\n', inputCsvPath);
featureTable = readFeatureDataset(inputCsvPath, featureColumnNames);

fprintf('Applying mean-centering and z-score normalization...\n');
normalizedFeatureTable = meanCenterAndNormalizeFeatureTable(featureTable, featureColumnNames);

outputDir = fileparts(outputCsvPath);
if ~isfolder(outputDir)
    mkdir(outputDir);
end

writetable(normalizedFeatureTable, outputCsvPath);
fprintf('Wrote normalized dataset: %s\n', outputCsvPath);

fprintf('Generating 3D scatter plots from original feature values...\n');
generateCategory3DScatterPlots(featureTable, featureCategories, plotOutputRoot);
fprintf('Saved 3D scatter plots under: %s\n', plotOutputRoot);

function featureColumnNames = collectFeatureColumnNames(featureCategories)
% collectFeatureColumnNames
% Flatten the feature-name definitions from all feature categories.

featureColumnNames = {};

for categoryIndex = 1:numel(featureCategories)
    featureColumnNames = [featureColumnNames, featureCategories(categoryIndex).featureNames]; %#ok<AGROW>
end
end
