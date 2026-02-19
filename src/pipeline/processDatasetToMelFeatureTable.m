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
isSuccessfulFile = false(numFiles, 1);
skippedFiles = cell(numFiles, 1);
numSkippedFiles = 0;
featureColumnNames = getImportantMelFeatureNames();
if numel(featureColumnNames) ~= 16
    error('Expected exactly 16 feature names, but got %d.', numel(featureColumnNames));
end

for fileIndex = 1:numFiles
    fileInfo = audioFiles(fileIndex);
    try
        [audioSignal, sampleRate] = audioread(fileInfo.fullPath);

        % Convert to mono for consistent feature extraction.
        if size(audioSignal, 2) > 1
            audioSignal = mean(audioSignal, 2);
        end

        segmentSignals = splitAudioIntoFixedSegments( ...
            audioSignal, ...
            sampleRate, ...
            options.segmentDurationSeconds);

        if isempty(segmentSignals)
            error('Audio file is shorter than the configured segment length.');
        end

        segmentRows = cell(numel(segmentSignals), 1);
        for segmentIndex = 1:numel(segmentSignals)
            featureVector = extractMelSpectrogramFeatureVector( ...
                segmentSignals{segmentIndex}, ...
                sampleRate, ...
                options);

            segmentRows{segmentIndex} = buildFeatureRowTable( ...
                fileInfo.className, ...
                fileInfo.fileName, ...
                segmentIndex, ...
                featureVector, ...
                featureColumnNames);
        end

        rowTables{fileIndex} = vertcat(segmentRows{:});
        isSuccessfulFile(fileIndex) = true;
    catch processingException
        numSkippedFiles = numSkippedFiles + 1;
        skippedFiles{numSkippedFiles} = fileInfo.fullPath;
        warning('Skipping file (%s): %s', fileInfo.fullPath, processingException.message);
    end

    if mod(fileIndex, 25) == 0 || fileIndex == numFiles
        fprintf('Processed %d/%d files\n', fileIndex, numFiles);
    end
end

if any(isSuccessfulFile)
    featureTable = vertcat(rowTables{isSuccessfulFile});
else
    error('No features were extracted. All files failed to process.');
end

if numSkippedFiles > 0
    fprintf('Skipped %d file(s) due to read/extraction errors.\n', numSkippedFiles);
    fprintf('First skipped file: %s\n', skippedFiles{1});
end
end
