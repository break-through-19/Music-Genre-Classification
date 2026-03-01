function featureTable = readFeatureDataset(csvPath, requiredVariableNames)
% readFeatureDataset
% Read the feature CSV and verify that all expected columns are present.

validateattributes(csvPath, {'char'}, {'nonempty'}, mfilename, 'csvPath');

if exist(csvPath, 'file') ~= 2
    error('Feature CSV not found: %s', csvPath);
end

featureTable = readtable(csvPath);

requiredMetadata = {'class_name', 'wav_file_name', 'segment_index'};
allRequiredColumns = [requiredMetadata, requiredVariableNames];
availableColumns = featureTable.Properties.VariableNames;

for columnIndex = 1:numel(allRequiredColumns)
    columnName = allRequiredColumns{columnIndex};
    if ~ismember(columnName, availableColumns)
        error('Required column missing from feature CSV: %s', columnName);
    end
end
end
