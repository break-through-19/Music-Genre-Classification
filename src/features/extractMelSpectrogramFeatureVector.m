function featureVector = extractMelSpectrogramFeatureVector(audioSignal, sampleRate, options)
% extractMelSpectrogramFeatureVector
% Compute 16 features for one segment:
% 1) spectral/timbral level features     -> 4
% 2) spectral variability features       -> 4
% 3) temporal dynamics features          -> 4
% 4) harmonic features                   -> 4

targetSampleRate = options.targetSampleRate;
if sampleRate ~= targetSampleRate
    audioSignal = resample(audioSignal, targetSampleRate, sampleRate);
    sampleRate = targetSampleRate;
end

audioSignal = audioSignal(:);

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

melMatrix = melSpectrogram( ...
    audioSignal, ...
    sampleRate, ...
    'Window', analysisWindow, ...
    'OverlapLength', max(windowLengthSamples - hopLengthSamples, 0), ...
    'FFTLength', fftLength, ...
    'NumBands', options.numMelBands, ...
    'FrequencyRange', [options.minFrequencyHz, options.maxFrequencyHz]);

if options.useLogScale
    melMatrix = log10(melMatrix + eps);
end

if isempty(melMatrix)
    error('Empty Mel spectrogram produced for segment.');
end

frameEnergy = mean(melMatrix, 1);
deltaFrameEnergy = diff(frameEnergy);

% 1) spectral/timbral level features (4)
timbralFeatures = [ ...
    mean(melMatrix(:)), ...
    median(melMatrix(:)), ...
    std(mean(melMatrix, 2)), ...
    mean(max(melMatrix, [], 1))];

% 2) spectral variability features (4)
bandRanges = max(melMatrix, [], 2) - min(melMatrix, [], 2);
variabilityFeatures = [ ...
    std(melMatrix(:)), ...
    mean(std(melMatrix, 0, 2)), ...
    std(frameEnergy), ...
    mean(bandRanges)];

% 3) temporal dynamics features (4)
if isempty(deltaFrameEnergy)
    deltaAbsMean = 0;
    deltaStd = 0;
else
    deltaAbsMean = mean(abs(deltaFrameEnergy));
    deltaStd = std(deltaFrameEnergy);
end

temporalFeatures = [ ...
    deltaAbsMean, ...
    deltaStd, ...
    localMeanZeroCrossingRate(audioSignal, windowLengthSamples, hopLengthSamples), ...
    localFrameRmsStd(audioSignal, windowLengthSamples, hopLengthSamples)];

% 4) harmonic features (4)
harmonicFeatures = extractHarmonicFeatureVector( ...
    audioSignal, ...
    sampleRate, ...
    windowLengthSamples, ...
    hopLengthSamples, ...
    analysisWindow);

featureVector = [timbralFeatures, variabilityFeatures, temporalFeatures, harmonicFeatures];

if numel(featureVector) ~= 16
    error('Expected 16 features, but got %d.', numel(featureVector));
end
end

function meanZcr = localMeanZeroCrossingRate(audioSignal, windowLengthSamples, hopLengthSamples)
numSamples = numel(audioSignal);
if numSamples < windowLengthSamples
    audioSignal = [audioSignal; zeros(windowLengthSamples - numSamples, 1)];
    numSamples = numel(audioSignal);
end

frameStarts = 1:hopLengthSamples:(numSamples - windowLengthSamples + 1);
numFrames = numel(frameStarts);
zcrValues = zeros(numFrames, 1);

for frameIndex = 1:numFrames
    frameStart = frameStarts(frameIndex);
    frameEnd = frameStart + windowLengthSamples - 1;
    frameSignal = audioSignal(frameStart:frameEnd);

    signChanges = abs(diff(sign(frameSignal)));
    zcrValues(frameIndex) = sum(signChanges > 0) / max(windowLengthSamples - 1, 1);
end

meanZcr = mean(zcrValues);
end

function rmsStd = localFrameRmsStd(audioSignal, windowLengthSamples, hopLengthSamples)
numSamples = numel(audioSignal);
if numSamples < windowLengthSamples
    audioSignal = [audioSignal; zeros(windowLengthSamples - numSamples, 1)];
    numSamples = numel(audioSignal);
end

frameStarts = 1:hopLengthSamples:(numSamples - windowLengthSamples + 1);
numFrames = numel(frameStarts);
rmsValues = zeros(numFrames, 1);

for frameIndex = 1:numFrames
    frameStart = frameStarts(frameIndex);
    frameEnd = frameStart + windowLengthSamples - 1;
    frameSignal = audioSignal(frameStart:frameEnd);
    rmsValues(frameIndex) = sqrt(mean(frameSignal .^ 2));
end

rmsStd = std(rmsValues);
end
