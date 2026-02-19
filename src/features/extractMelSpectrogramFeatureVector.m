function melFeatureVector = extractMelSpectrogramFeatureVector(audioSignal, sampleRate, options)
% extractMelSpectrogramFeatureVector
% Compute 60 important features for one segment:
% - 45 Mel-based features (15 bands x 3 statistics)
% - 15 harmonic features
%
% Feature layout (60 total):
% 1:15   -> mean log-Mel energy (15 representative Mel bands)
% 16:30  -> std log-Mel energy  (same 15 Mel bands)
% 31:45  -> mean absolute temporal delta (same 15 Mel bands)
% 46:60  -> harmonic features (pitch/harmonicity descriptors)

targetSampleRate = options.targetSampleRate;
if sampleRate ~= targetSampleRate
    audioSignal = resample(audioSignal, targetSampleRate, sampleRate);
    sampleRate = targetSampleRate;
end

% Enforce fixed segment sample count so feature length is consistent.
targetSegmentSamples = round(options.segmentDurationSeconds * sampleRate);
if numel(audioSignal) < targetSegmentSamples
    audioSignal = [audioSignal; zeros(targetSegmentSamples - numel(audioSignal), 1)];
elseif numel(audioSignal) > targetSegmentSamples
    audioSignal = audioSignal(1:targetSegmentSamples);
end

windowLengthSamples = round(options.windowDurationSeconds * sampleRate);
hopLengthSamples = round(options.hopDurationSeconds * sampleRate);
fftLength = 2 ^ nextpow2(windowLengthSamples);
if exist('hamming', 'file') == 2
    analysisWindow = hamming(windowLengthSamples);
else
    analysisWindow = ones(windowLengthSamples, 1);
end

% This MATLAB version expects the actual window vector (Window), not WindowLength.
melMatrix = melSpectrogram( ...
    audioSignal, ...
    sampleRate, ...
    'Window', analysisWindow, ...
    'OverlapLength', max(windowLengthSamples - hopLengthSamples, 0), ...
    'FFTLength', fftLength, ...
    'NumBands', options.numMelBands, ...
    'FrequencyRange', [options.minFrequencyHz, options.maxFrequencyHz]);

if options.useLogScale
    % Stable log compression for improved numeric conditioning.
    melMatrix = log10(melMatrix + eps);
end

numImportantBands = 15;
if size(melMatrix, 1) < numImportantBands
    error('NumBands in melSpectrogram must be at least %d.', numImportantBands);
end

selectedBandIndices = round(linspace(1, size(melMatrix, 1), numImportantBands));
selectedMelMatrix = melMatrix(selectedBandIndices, :);

melBandMeans = mean(selectedMelMatrix, 2).';
melBandStds = std(selectedMelMatrix, 0, 2).';

if size(selectedMelMatrix, 2) > 1
    melTemporalDelta = diff(selectedMelMatrix, 1, 2);
    melBandDeltaMeanAbs = mean(abs(melTemporalDelta), 2).';
else
    melBandDeltaMeanAbs = zeros(1, numImportantBands);
end

harmonicFeatureVector = extractHarmonicFeatureVector( ...
    audioSignal, ...
    sampleRate, ...
    windowLengthSamples, ...
    hopLengthSamples, ...
    analysisWindow);

melFeatureVector = [melBandMeans, melBandStds, melBandDeltaMeanAbs, harmonicFeatureVector];

if numel(melFeatureVector) ~= 60
    error('Expected 60 features, but got %d.', numel(melFeatureVector));
end
end
