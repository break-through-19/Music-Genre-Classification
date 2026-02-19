function rowTable = buildFeatureRowTable(className, wavFileName, segmentIndex, featureVector, featureColumnNames)
% buildFeatureRowTable
% Build one table row with metadata + selected audio feature columns.

numFeatures = numel(featureVector);
if nargin < 5 || isempty(featureColumnNames)
    error('featureColumnNames must be provided and must contain descriptive feature names.');
end

if numel(featureColumnNames) ~= numFeatures
    error('Feature vector length (%d) and feature-column count (%d) must match.', ...
        numFeatures, numel(featureColumnNames));
end

metadataTable = table( ...
    {className}, ...
    {wavFileName}, ...
    segmentIndex, ...
    'VariableNames', {'class_name', 'wav_file_name', 'segment_index'});

featureTable = array2table(featureVector, 'VariableNames', featureColumnNames);
rowTable = [metadataTable, featureTable];
end
