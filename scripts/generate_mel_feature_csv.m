% generate_mel_feature_csv.m
% Build a Mel-spectrogram feature dataset CSV from genre-organized WAV files.
%
% Folder expectation:
%   dataset/<genre_name>/<audio_file>.wav
%
% Output:
%   data/features/mel_spectrogram_features.csv
%
% Notes:
% - Each 30-second audio is split into 3-second segments (10 segments/file).
% - Every segment becomes one row in the CSV.
% - Metadata columns are included: class_name, wav_file_name, segment_index.
% - Exactly 60 important features are written per segment:
%   45 Mel-based features + 15 harmonic features.
% - Optional example CSV path can be provided to align final column ordering.

clear;
clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(projectRoot, 'src')));

datasetRoot = resolveDatasetRoot(projectRoot);
outputDir = fullfile(projectRoot, 'data', 'features');
outputCsvPath = fullfile(outputDir, 'mel_spectrogram_features.csv');

% Optional: point this to your existing feature CSV to align column order.
exampleCsvPath = '';

options = struct();
options.segmentDurationSeconds = 3;
options.targetSampleRate = 22050;
options.windowDurationSeconds = 0.025;
options.hopDurationSeconds = 0.010;
options.numMelBands = 128;
options.minFrequencyHz = 20;
options.maxFrequencyHz = 8000;
options.useLogScale = true;

fprintf('Starting Mel feature extraction...\n');
fprintf('Dataset root: %s\n', datasetRoot);
fprintf('Output CSV:   %s\n', outputCsvPath);

featureTable = processDatasetToMelFeatureTable(datasetRoot, options);
featureTable = alignColumnsToExampleCsv(featureTable, exampleCsvPath);

if ~isfolder(outputDir)
    mkdir(outputDir);
end

writetable(featureTable, outputCsvPath);
fprintf('Done. Wrote %d rows and %d columns.\n', height(featureTable), width(featureTable));

function datasetRoot = resolveDatasetRoot(projectRoot)
% resolveDatasetRoot
% Prefer: <projectRoot>/dataset
% Fallback: <pwd>/Music-Genre-Classification/dataset

candidatePaths = {
    fullfile(projectRoot, 'dataset')
    fullfile(pwd, 'Music-Genre-Classification', 'dataset')
    };

for pathIndex = 1:numel(candidatePaths)
    if isfolder(candidatePaths{pathIndex})
        datasetRoot = candidatePaths{pathIndex};
        return;
    end
end

error('Dataset folder not found. Expected one of:\n1) %s\n2) %s', ...
    candidatePaths{1}, candidatePaths{2});
end
