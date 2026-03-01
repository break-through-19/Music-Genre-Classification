function normalizedFeatureTable = meanCenterAndNormalizeFeatureTable(featureTable, featureColumnNames)
% meanCenterAndNormalizeFeatureTable
% Apply mean-centering and z-score normalization to selected feature
% columns while preserving metadata columns unchanged.

featureMatrix = table2array(featureTable(:, featureColumnNames));

if ~isnumeric(featureMatrix)
    error('Selected feature columns must be numeric.');
end

featureMeans = mean(featureMatrix, 1);
featureStdDevs = std(featureMatrix, 0, 1);

% Avoid division by zero when a feature is constant.
featureStdDevs(featureStdDevs == 0) = 1;

normalizedMatrix = bsxfun(@minus, featureMatrix, featureMeans);
normalizedMatrix = bsxfun(@rdivide, normalizedMatrix, featureStdDevs);

normalizedFeatureTable = featureTable;
normalizedFeatureTable{:, featureColumnNames} = normalizedMatrix;
end
