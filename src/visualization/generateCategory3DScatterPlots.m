function generateCategory3DScatterPlots(featureTable, featureCategories, plotOutputRoot)
% generateCategory3DScatterPlots
% Generate one 3D scatter plot for every 3-feature combination within each
% category. With 4 features per category, this produces 4 plots/category.

validateattributes(plotOutputRoot, {'char'}, {'nonempty'}, mfilename, 'plotOutputRoot');

if ~isfolder(plotOutputRoot)
    mkdir(plotOutputRoot);
end

classLabels = localConvertClassLabels(featureTable.class_name);
uniqueClassLabels = unique(classLabels, 'stable');
plotColors = lines(numel(uniqueClassLabels));

for categoryIndex = 1:numel(featureCategories)
    categoryDefinition = featureCategories(categoryIndex);
    categoryOutputDir = fullfile(plotOutputRoot, categoryDefinition.directoryName);

    if ~isfolder(categoryOutputDir)
        mkdir(categoryOutputDir);
    end

    featureTriplets = nchoosek(1:numel(categoryDefinition.featureNames), 3);

    for tripletIndex = 1:size(featureTriplets, 1)
        selectedIndices = featureTriplets(tripletIndex, :);
        selectedFeatureNames = categoryDefinition.featureNames(selectedIndices);
        localPlotAndSaveTriplet( ...
            featureTable, ...
            classLabels, ...
            uniqueClassLabels, ...
            plotColors, ...
            categoryDefinition.displayName, ...
            selectedFeatureNames, ...
            categoryOutputDir, ...
            tripletIndex);
    end
end
end

function localPlotAndSaveTriplet(featureTable, classLabels, uniqueClassLabels, plotColors, categoryDisplayName, selectedFeatureNames, categoryOutputDir, tripletIndex)
% localPlotAndSaveTriplet
% Render and save one scatter plot for a selected feature triplet.

xValues = featureTable.(selectedFeatureNames{1});
yValues = featureTable.(selectedFeatureNames{2});
zValues = featureTable.(selectedFeatureNames{3});

figureHandle = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1200, 800]);
axesHandle = axes('Parent', figureHandle);
hold(axesHandle, 'on');

for classIndex = 1:numel(uniqueClassLabels)
    currentLabel = uniqueClassLabels{classIndex};
    classMask = strcmp(classLabels, currentLabel);

    scatter3( ...
        axesHandle, ...
        xValues(classMask), ...
        yValues(classMask), ...
        zValues(classMask), ...
        12, ...
        plotColors(classIndex, :), ...
        'filled', ...
        'DisplayName', currentLabel);
end

xlabel(axesHandle, strrep(selectedFeatureNames{1}, '_', ' '), 'Interpreter', 'none');
ylabel(axesHandle, strrep(selectedFeatureNames{2}, '_', ' '), 'Interpreter', 'none');
zlabel(axesHandle, strrep(selectedFeatureNames{3}, '_', ' '), 'Interpreter', 'none');

title( ...
    axesHandle, ...
    sprintf('%s | Combination %d', categoryDisplayName, tripletIndex), ...
    'Interpreter', 'none');

grid(axesHandle, 'on');
view(axesHandle, 45, 25);
legend(axesHandle, 'Location', 'eastoutside');
hold(axesHandle, 'off');

baseFileName = sprintf( ...
    '%02d_%s_vs_%s_vs_%s', ...
    tripletIndex, ...
    selectedFeatureNames{1}, ...
    selectedFeatureNames{2}, ...
    selectedFeatureNames{3});
outputImagePath = fullfile(categoryOutputDir, [localSanitizeFileName(baseFileName), '.png']);

saveas(figureHandle, outputImagePath);
close(figureHandle);
end

function classLabels = localConvertClassLabels(rawLabels)
% localConvertClassLabels
% Convert class labels to a cell array of character vectors.

if iscell(rawLabels)
    classLabels = rawLabels;
elseif isa(rawLabels, 'categorical')
    classLabels = cellstr(rawLabels);
elseif isa(rawLabels, 'string')
    classLabels = cellstr(rawLabels);
elseif ischar(rawLabels)
    classLabels = cellstr(rawLabels);
else
    error('Unsupported class label type for plotting.');
end
end

function sanitizedName = localSanitizeFileName(rawName)
% localSanitizeFileName
% Replace characters that are inconvenient for filenames.

sanitizedName = regexprep(rawName, '[^a-zA-Z0-9_]+', '_');
sanitizedName = regexprep(sanitizedName, '_+', '_');
sanitizedName = regexprep(sanitizedName, '^_|_$', '');
end
