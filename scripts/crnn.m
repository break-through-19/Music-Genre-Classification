%% CRNN Classification on Original, PCA, and Curated Feature Sets
% This script trains and evaluates a Convolutional Recurrent Neural Network
% (CRNN) on three dataset variants:
% 1) Original normalized features (16 features)
% 2) PCA-transformed features (4 components)
% 3) Curated normalized features (14 features; drop columns 2 and 8)
%
% For each variant:
% - Hyperparameters are tuned using randomized search on a validation split.
% - Final model is trained on full training data.
% - Accuracy is reported on the held-out test set.
% - Confusion matrix is plotted and saved.

clear;
clc;
close all;

if exist('convolution1dLayer', 'file') ~= 2
    error('convolution1dLayer is not available. Please use a MATLAB release with 1D CNN support.');
end

projectRoot = fileparts(fileparts(mfilename('fullpath')));
normalizedCsvPath = fullfile(projectRoot, 'data', 'features', 'mel_spectrogram_features_normalized.csv');
pcaCsvPath = fullfile(projectRoot, 'data', 'transformed_features', 'mel_spectrogram_features_pca_r4.csv');

modelOutputRoot = fullfile(projectRoot, 'data', 'model_outputs', 'crnn');
confusionOutputDir = fullfile(modelOutputRoot, 'confusion_matrices');

if ~exist(modelOutputRoot, 'dir')
    mkdir(modelOutputRoot);
end

if ~exist(confusionOutputDir, 'dir')
    mkdir(confusionOutputDir);
end

rng(42);

%% 1) Load datasets
originalTable = readTablePreserveVariableNames(normalizedCsvPath);
pcaTable = readTablePreserveVariableNames(pcaCsvPath);

requiredMetadataColumns = {'class_name', 'wav_file_name', 'segment_index'};
validateRequiredColumns(originalTable, requiredMetadataColumns);
validateRequiredColumns(pcaTable, requiredMetadataColumns);

% Align PCA rows with original rows by metadata key, so train/test indices
% correspond exactly across all three dataset variants.
originalRowKeys = buildRowKeys(originalTable);
pcaRowKeys = buildRowKeys(pcaTable);
[isFoundInPca, pcaRowIndices] = ismember(originalRowKeys, pcaRowKeys);

if ~all(isFoundInPca)
    error('PCA dataset rows do not fully match the original dataset rows.');
end

pcaTableAligned = pcaTable(pcaRowIndices, :);

labelsAll = convertToCategorical(originalTable.class_name);
classNames = categories(labelsAll);
numClasses = numel(classNames);

%% 2) Build a shared train/test split
holdoutPartition = cvpartition(labelsAll, 'HoldOut', 0.2);
isTrain = training(holdoutPartition);
isTest = test(holdoutPartition);

labelsTrain = labelsAll(isTrain);
labelsTest = labelsAll(isTest);

%% 3) Define dataset variants
allOriginalFeatureNames = originalTable.Properties.VariableNames(4:end);
curatedColumnsToDrop = {'timbral_log_mel_median', 'variability_band_range_mean'};
curatedFeatureNames = setdiff(allOriginalFeatureNames, curatedColumnsToDrop, 'stable');
pcaFeatureNames = pcaTableAligned.Properties.VariableNames(4:end);

datasetVariants = struct( ...
    'variantName', {}, ...
    'variantTag', {}, ...
    'sourceTable', {}, ...
    'featureNames', {});

datasetVariants(1).variantName = 'Original (16 Features)';
datasetVariants(1).variantTag = 'original_16';
datasetVariants(1).sourceTable = originalTable;
datasetVariants(1).featureNames = allOriginalFeatureNames;

datasetVariants(2).variantName = 'PCA Transformed (4 Components)';
datasetVariants(2).variantTag = 'pca_r4';
datasetVariants(2).sourceTable = pcaTableAligned;
datasetVariants(2).featureNames = pcaFeatureNames;

datasetVariants(3).variantName = 'Curated Original (14 Features)';
datasetVariants(3).variantTag = 'curated_14';
datasetVariants(3).sourceTable = originalTable;
datasetVariants(3).featureNames = curatedFeatureNames;

%% 4) Configure hyperparameter search
searchConfig = struct();
searchConfig.maxTrials = 10;
searchConfig.tuningEpochs = 20;
searchConfig.finalTrainingEpochs = 60;
searchConfig.validationHoldout = 0.2;

%% 5) Train/evaluate CRNN for each variant
resultsRows = cell(numel(datasetVariants), 6);

for variantIndex = 1:numel(datasetVariants)
    variantDefinition = datasetVariants(variantIndex);
    fprintf('\n=== Variant: %s ===\n', variantDefinition.variantName);

    featureMatrixAll = table2array(variantDefinition.sourceTable(:, variantDefinition.featureNames));
    sequenceLength = size(featureMatrixAll, 2);

    featureTrain = featureMatrixAll(isTrain, :);
    featureTest = featureMatrixAll(isTest, :);

    sequenceTrain = convertFeatureMatrixToSequenceCell(featureTrain);
    sequenceTest = convertFeatureMatrixToSequenceCell(featureTest);

    bestHyperparameters = tuneCrnnHyperparameters( ...
        sequenceTrain, ...
        labelsTrain, ...
        numClasses, ...
        sequenceLength, ...
        searchConfig);

    fprintf('Best hyperparameters: filters=%d, hidden=%d, dropout=%.2f, lr=%.4f, batch=%d\n', ...
        bestHyperparameters.numFilters, ...
        bestHyperparameters.numHiddenUnits, ...
        bestHyperparameters.dropoutRate, ...
        bestHyperparameters.initialLearnRate, ...
        bestHyperparameters.miniBatchSize);

    trainedNet = trainCrnnClassifier( ...
        sequenceTrain, ...
        labelsTrain, ...
        numClasses, ...
        sequenceLength, ...
        bestHyperparameters, ...
        searchConfig.finalTrainingEpochs);

    predictedLabels = classify(trainedNet, sequenceTest, ...
        'MiniBatchSize', bestHyperparameters.miniBatchSize);

    testAccuracy = mean(predictedLabels == labelsTest) * 100;
    fprintf('Test accuracy (%s): %.2f%%\n', variantDefinition.variantName, testAccuracy);

    confusionImagePath = fullfile(confusionOutputDir, ...
        sprintf('CRNN_%s_confusion_matrix.png', variantDefinition.variantTag));
    saveConfusionMatrixFigure(labelsTest, predictedLabels, variantDefinition.variantName, testAccuracy, confusionImagePath);

    resultsRows{variantIndex, 1} = variantDefinition.variantName;
    resultsRows{variantIndex, 2} = size(featureMatrixAll, 2);
    resultsRows{variantIndex, 3} = bestHyperparameters.numFilters;
    resultsRows{variantIndex, 4} = bestHyperparameters.numHiddenUnits;
    resultsRows{variantIndex, 5} = bestHyperparameters.dropoutRate;
    resultsRows{variantIndex, 6} = testAccuracy;
end

%% 6) Save summary table
resultsSummaryTable = cell2table(resultsRows, ...
    'VariableNames', {'dataset_variant', 'num_features', 'num_filters', 'num_hidden_units', 'dropout_rate', 'test_accuracy_percent'});

resultsSummaryPath = fullfile(modelOutputRoot, 'crnn_results_summary.csv');
writetable(resultsSummaryTable, resultsSummaryPath);
fprintf('\nSaved summary: %s\n', resultsSummaryPath);
fprintf('Saved confusion matrices in: %s\n', confusionOutputDir);

%% Local functions
function dataTable = readTablePreserveVariableNames(csvPath)
% Read table while preserving column names exactly as present in CSV.

if exist(csvPath, 'file') ~= 2
    error('CSV file not found: %s', csvPath);
end

importOptions = detectImportOptions(csvPath);
importOptions.VariableNamingRule = 'preserve';
dataTable = readtable(csvPath, importOptions);
end

function validateRequiredColumns(dataTable, requiredColumns)
% Ensure mandatory columns exist in the table.

availableColumns = dataTable.Properties.VariableNames;
for columnIndex = 1:numel(requiredColumns)
    if ~ismember(requiredColumns{columnIndex}, availableColumns)
        error('Required column missing: %s', requiredColumns{columnIndex});
    end
end
end

function labelsCategorical = convertToCategorical(rawLabels)
% Convert any common label type to categorical.

if isa(rawLabels, 'categorical')
    labelsCategorical = rawLabels;
elseif iscell(rawLabels)
    labelsCategorical = categorical(rawLabels);
elseif isa(rawLabels, 'string')
    labelsCategorical = categorical(cellstr(rawLabels));
elseif ischar(rawLabels)
    labelsCategorical = categorical(cellstr(rawLabels));
else
    error('Unsupported label type for conversion to categorical.');
end
end

function rowKeys = buildRowKeys(dataTable)
% Build stable unique row keys using class, file, and segment index.

classColumn = convertColumnToCellstr(dataTable.class_name);
fileColumn = convertColumnToCellstr(dataTable.wav_file_name);
segmentColumn = dataTable.segment_index;
segmentAsCell = arrayfun(@(value) num2str(value), segmentColumn, 'UniformOutput', false);

rowKeys = strcat(classColumn, '|', fileColumn, '|', segmentAsCell);
end

function textCell = convertColumnToCellstr(columnData)
% Convert a table column to cell array of character vectors.

if iscell(columnData)
    textCell = columnData;
elseif isa(columnData, 'categorical')
    textCell = cellstr(columnData);
elseif isa(columnData, 'string')
    textCell = cellstr(columnData);
elseif ischar(columnData)
    textCell = cellstr(columnData);
else
    error('Unsupported text column type.');
end
end

function sequenceCell = convertFeatureMatrixToSequenceCell(featureMatrix)
% Convert N x F numeric matrix into N cell sequences of size 1 x F.

numSamples = size(featureMatrix, 1);
sequenceCell = cell(numSamples, 1);

for sampleIndex = 1:numSamples
    sequenceCell{sampleIndex} = reshape(featureMatrix(sampleIndex, :), 1, []);
end
end

function bestHyperparameters = tuneCrnnHyperparameters(sequenceTrain, labelsTrain, numClasses, sequenceLength, searchConfig)
% Randomized search over CRNN hyperparameter candidates using validation accuracy.

innerPartition = cvpartition(labelsTrain, 'HoldOut', searchConfig.validationHoldout);
isInnerTrain = training(innerPartition);
isInnerValidation = test(innerPartition);

sequenceInnerTrain = sequenceTrain(isInnerTrain);
labelInnerTrain = labelsTrain(isInnerTrain);
sequenceValidation = sequenceTrain(isInnerValidation);
labelValidation = labelsTrain(isInnerValidation);

candidateFilters = [16, 32, 64];
candidateHiddenUnits = [32, 64, 96];
candidateDropout = [0.2, 0.3, 0.4];
candidateLearnRate = [1e-3, 5e-4];
candidateBatchSize = [32, 64, 128];

allCombinations = [];
for filterValue = candidateFilters
    for hiddenValue = candidateHiddenUnits
        for dropoutValue = candidateDropout
            for learnRateValue = candidateLearnRate
                for batchValue = candidateBatchSize
                    allCombinations = [allCombinations; filterValue, hiddenValue, dropoutValue, learnRateValue, batchValue]; %#ok<AGROW>
                end
            end
        end
    end
end

numCombinations = size(allCombinations, 1);
numTrials = min(searchConfig.maxTrials, numCombinations);
trialOrder = randperm(numCombinations, numTrials);

bestValidationAccuracy = -inf;
bestHyperparameters = struct();

for trialIndex = 1:numTrials
    selectedCombination = allCombinations(trialOrder(trialIndex), :);
    trialHyperparameters = struct();
    trialHyperparameters.numFilters = selectedCombination(1);
    trialHyperparameters.numHiddenUnits = selectedCombination(2);
    trialHyperparameters.dropoutRate = selectedCombination(3);
    trialHyperparameters.initialLearnRate = selectedCombination(4);
    trialHyperparameters.miniBatchSize = selectedCombination(5);

    trialNet = trainCrnnClassifier( ...
        sequenceInnerTrain, ...
        labelInnerTrain, ...
        numClasses, ...
        sequenceLength, ...
        trialHyperparameters, ...
        searchConfig.tuningEpochs);

    validationPredictions = classify(trialNet, sequenceValidation, ...
        'MiniBatchSize', trialHyperparameters.miniBatchSize);
    validationAccuracy = mean(validationPredictions == labelValidation) * 100;

    fprintf('  Trial %d/%d -> Val Acc: %.2f%% | filters=%d hidden=%d drop=%.2f lr=%.4f batch=%d\n', ...
        trialIndex, numTrials, validationAccuracy, ...
        trialHyperparameters.numFilters, ...
        trialHyperparameters.numHiddenUnits, ...
        trialHyperparameters.dropoutRate, ...
        trialHyperparameters.initialLearnRate, ...
        trialHyperparameters.miniBatchSize);

    if validationAccuracy > bestValidationAccuracy
        bestValidationAccuracy = validationAccuracy;
        bestHyperparameters = trialHyperparameters;
    end
end
end

function trainedNet = trainCrnnClassifier(sequenceTrain, labelsTrain, numClasses, sequenceLength, hyperparameters, maxEpochs)
% Build and train a CRNN classifier with the provided hyperparameters.

filterSize = min(3, sequenceLength);

layers = [ ...
    sequenceInputLayer(1, 'Name', 'input')
    convolution1dLayer(filterSize, hyperparameters.numFilters, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'batchnorm1')
    reluLayer('Name', 'relu1')
    convolution1dLayer(filterSize, hyperparameters.numFilters, 'Padding', 'same', 'Name', 'conv2')
    reluLayer('Name', 'relu2')
    bilstmLayer(hyperparameters.numHiddenUnits, 'OutputMode', 'last', 'Name', 'bilstm')
    dropoutLayer(hyperparameters.dropoutRate, 'Name', 'dropout')
    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classification')];

trainingOpts = trainingOptions('adam', ...
    'InitialLearnRate', hyperparameters.initialLearnRate, ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', hyperparameters.miniBatchSize, ...
    'Shuffle', 'every-epoch', ...
    'GradientThreshold', 1, ...
    'Verbose', false, ...
    'Plots', 'none');

trainedNet = trainNetwork(sequenceTrain, labelsTrain, layers, trainingOpts);
end

function saveConfusionMatrixFigure(trueLabels, predictedLabels, variantName, accuracyPercent, outputPath)
% Save confusion matrix figure for one dataset variant.

figureHandle = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 900, 650]);
confusionChartHandle = confusionchart(trueLabels, predictedLabels);
confusionChartHandle.Title = sprintf('CRNN | %s | Accuracy: %.2f%%', variantName, accuracyPercent);
confusionChartHandle.RowSummary = 'row-normalized';
confusionChartHandle.ColumnSummary = 'column-normalized';

if isprop(confusionChartHandle, 'Colormap')
    confusionChartHandle.Colormap = localLightBlueColormap(256);
end

if isprop(confusionChartHandle, 'FontColor')
    confusionChartHandle.FontColor = [0, 0, 0];
end

axesHandles = findall(figureHandle, 'Type', 'Axes');
for axesIndex = 1:numel(axesHandles)
    set(axesHandles(axesIndex), ...
        'Color', [1, 1, 1], ...
        'XColor', [0, 0, 0], ...
        'YColor', [0, 0, 0]);
end

exportgraphics(figureHandle, outputPath, 'Resolution', 300);
close(figureHandle);
end

function colorMap = localLightBlueColormap(numColors)

if nargin < 1
    numColors = 256;
end

redChannel = linspace(1.0, 0.25, numColors)';
greenChannel = linspace(1.0, 0.55, numColors)';
blueChannel = ones(numColors, 1);
colorMap = [redChannel, greenChannel, blueChannel];
end
