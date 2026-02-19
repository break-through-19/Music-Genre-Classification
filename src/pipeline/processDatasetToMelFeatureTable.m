function featureTable = processDatasetToMelFeatureTable(datasetRoot, options)
% processDatasetToMelFeatureTable
% Traverse dataset folders, segment each WAV file, extract Mel features,
% and return one consolidated table.

validateattributes(datasetRoot, {'char'}, {'nonempty'}, mfilename, 'datasetRoot');
assert(isfolder(datasetRoot), 'Dataset folder not found: %s', datasetRoot);

audioFiles = listGenreAudioFiles(datasetRoot);
if isempty(audioFiles)
    error('No .wav files found under dataset root: %s', datasetRoot);
end

numFiles = numel(audioFiles);
rowTables = cell(numFiles, 1);
featureColumnNames = getImportantMelFeatureNames();
if numel(featureColumnNames) ~= 60
    error('Expected exactly 60 feature names, but got %d.', numel(featureColumnNames));
end

for fileIndex = 1:numFiles
    fileInfo = audioFiles(fileIndex);

    [audioSignal, sampleRate] = audioread(fileInfo.fullPath);

    % Convert to mono for consistent feature extraction.
    if size(audioSignal, 2) > 1
        audioSignal = mean(audioSignal, 2);
    end

    segmentSignals = splitAudioIntoFixedSegments( ...
        audioSignal, ...
        sampleRate, ...
        options.segmentDurationSeconds);

    segmentRows = cell(numel(segmentSignals), 1);
    for segmentIndex = 1:numel(segmentSignals)
        melFeatureVector = extractMelSpectrogramFeatureVector( ...
            segmentSignals{segmentIndex}, ...
            sampleRate, ...
            options);

        segmentRows{segmentIndex} = buildFeatureRowTable( ...
            fileInfo.className, ...
            fileInfo.fileName, ...
            segmentIndex, ...
            melFeatureVector, ...
            featureColumnNames);
    end

    rowTables{fileIndex} = vertcat(segmentRows{:});

    if mod(fileIndex, 25) == 0 || fileIndex == numFiles
        fprintf('Processed %d/%d files\n', fileIndex, numFiles);
    end
end

featureTable = vertcat(rowTables{:});
end
