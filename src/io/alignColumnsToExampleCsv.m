function alignedTable = alignColumnsToExampleCsv(featureTable, exampleCsvPath)
% alignColumnsToExampleCsv
% If an example CSV exists, reorder and align output columns to its schema.
% Missing columns are added as NaN/string missing values where appropriate.
%
% If no example path is supplied, input table is returned unchanged.

alignedTable = featureTable;

if nargin < 2 || isempty(exampleCsvPath)
    return;
end

if isa(exampleCsvPath, 'string')
    exampleCsvPath = char(exampleCsvPath);
end

if exist(exampleCsvPath, 'file') ~= 2
    warning('Example CSV not found. Using generated column order: %s', exampleCsvPath);
    return;
end

exampleTable = readtable(exampleCsvPath);
exampleColumns = exampleTable.Properties.VariableNames;
currentColumns = alignedTable.Properties.VariableNames;

for columnIndex = 1:numel(exampleColumns)
    columnName = exampleColumns{columnIndex};
    if ~ismember(columnName, currentColumns)
        exampleColumnData = exampleTable.(columnName);
        if isnumeric(exampleColumnData)
            alignedTable.(columnName) = nan(height(alignedTable), 1);
        elseif islogical(exampleColumnData)
            alignedTable.(columnName) = false(height(alignedTable), 1);
        else
            alignedTable.(columnName) = repmat({''}, height(alignedTable), 1);
        end
    end
end

currentColumns = alignedTable.Properties.VariableNames;
extraColumns = setdiff(currentColumns, exampleColumns, 'stable');
finalColumns = [exampleColumns, extraColumns];
alignedTable = alignedTable(:, finalColumns);
end
