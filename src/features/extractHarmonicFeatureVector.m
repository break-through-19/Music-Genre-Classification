function harmonicFeatureVector = extractHarmonicFeatureVector(audioSignal, sampleRate, windowLengthSamples, hopLengthSamples, analysisWindow)
% extractHarmonicFeatureVector
% Compute 15 harmonic features from frame-wise autocorrelation and spectrum.
%
% Output order:
% 1  harmonic_voiced_frame_ratio
% 2  harmonic_f0_mean_hz
% 3  harmonic_f0_std_hz
% 4  harmonic_f0_median_hz
% 5  harmonic_f0_iqr_hz
% 6  harmonic_f0_min_hz
% 7  harmonic_f0_max_hz
% 8  harmonic_f0_range_hz
% 9  harmonic_f0_slope_hz_per_frame
% 10 harmonic_autocorr_peak_mean
% 11 harmonic_autocorr_peak_std
% 12 harmonic_energy_ratio_mean
% 13 harmonic_energy_ratio_std
% 14 harmonic_peak_prominence_mean
% 15 harmonic_peak_prominence_std

audioSignal = audioSignal(:);
signalLength = numel(audioSignal);

if signalLength < windowLengthSamples
    audioSignal = [audioSignal; zeros(windowLengthSamples - signalLength, 1)];
    signalLength = numel(audioSignal);
end

frameStarts = 1:hopLengthSamples:(signalLength - windowLengthSamples + 1);
numFrames = numel(frameStarts);

f0Values = nan(numFrames, 1);
autocorrPeakValues = zeros(numFrames, 1);
harmonicEnergyRatios = zeros(numFrames, 1);
peakProminenceValues = zeros(numFrames, 1);

minF0Hz = 80;
maxF0Hz = 800;
minLag = max(1, floor(sampleRate / maxF0Hz));
maxLag = max(minLag + 1, floor(sampleRate / minF0Hz));

for frameIndex = 1:numFrames
    frameStart = frameStarts(frameIndex);
    frameEnd = frameStart + windowLengthSamples - 1;
    frameSignal = audioSignal(frameStart:frameEnd) .* analysisWindow;
    frameSignal = frameSignal - mean(frameSignal);

    if sum(abs(frameSignal)) == 0
        continue;
    end

    maxLagForFrame = min(maxLag, windowLengthSamples - 1);
    acf = localNormalizedAutocorrelation(frameSignal, maxLagForFrame);
    pitchSearchLags = minLag:maxLagForFrame;

    if isempty(pitchSearchLags)
        continue;
    end

    pitchAcf = acf(pitchSearchLags + 1);
    [peakValue, peakIdx] = max(pitchAcf);
    bestLag = pitchSearchLags(peakIdx);
    autocorrPeakValues(frameIndex) = peakValue;

    if peakValue > 0.30
        f0Values(frameIndex) = sampleRate / bestLag;
    end

    [harmonicEnergyRatios(frameIndex), peakProminenceValues(frameIndex)] = ...
        localSpectrumHarmonicMeasures(frameSignal, sampleRate, f0Values(frameIndex));
end

voicedMask = ~isnan(f0Values);
voicedRatio = sum(voicedMask) / max(numFrames, 1);

f0Mean = localNanMean(f0Values);
f0Std = localNanStd(f0Values);
f0Median = localNanMedian(f0Values);
f0Iqr = localNanIqr(f0Values);
f0Min = localNanMin(f0Values);
f0Max = localNanMax(f0Values);
f0Range = f0Max - f0Min;

if sum(voicedMask) >= 2
    voicedPositions = find(voicedMask);
    slopeCoefficients = polyfit(voicedPositions(:), f0Values(voicedMask), 1);
    f0Slope = slopeCoefficients(1);
else
    f0Slope = 0;
end

harmonicFeatureVector = [ ...
    voicedRatio, ...
    f0Mean, ...
    f0Std, ...
    f0Median, ...
    f0Iqr, ...
    f0Min, ...
    f0Max, ...
    f0Range, ...
    f0Slope, ...
    mean(autocorrPeakValues), ...
    std(autocorrPeakValues), ...
    mean(harmonicEnergyRatios), ...
    std(harmonicEnergyRatios), ...
    mean(peakProminenceValues), ...
    std(peakProminenceValues)];
end

function acf = localNormalizedAutocorrelation(frameSignal, maxLag)
n = numel(frameSignal);
fftLength = 2 ^ nextpow2(2 * n - 1);
spectrum = fft(frameSignal, fftLength);
powerSpectrum = spectrum .* conj(spectrum);
acfFull = ifft(powerSpectrum);
acf = real(acfFull(1:(maxLag + 1)));

if acf(1) > 0
    acf = acf / acf(1);
else
    acf = zeros(size(acf));
end
end

function [harmonicEnergyRatio, peakProminence] = localSpectrumHarmonicMeasures(frameSignal, sampleRate, f0Hz)
fftLength = 2 ^ nextpow2(numel(frameSignal));
spectrumMagnitude = abs(fft(frameSignal, fftLength));
spectrumMagnitude = spectrumMagnitude(1:(floor(fftLength / 2) + 1));
totalEnergy = sum(spectrumMagnitude .^ 2) + eps;

peakProminence = max(spectrumMagnitude) / (mean(spectrumMagnitude) + eps);

if isnan(f0Hz) || f0Hz <= 0
    harmonicEnergyRatio = 0;
    return;
end

frequencyResolution = sampleRate / fftLength;
numHarmonics = 5;
harmonicEnergy = 0;

for harmonicIndex = 1:numHarmonics
    harmonicFrequency = harmonicIndex * f0Hz;
    if harmonicFrequency >= sampleRate / 2
        break;
    end

    centerBin = round(harmonicFrequency / frequencyResolution) + 1;
    binStart = max(1, centerBin - 1);
    binEnd = min(numel(spectrumMagnitude), centerBin + 1);
    harmonicEnergy = harmonicEnergy + sum(spectrumMagnitude(binStart:binEnd) .^ 2);
end

harmonicEnergyRatio = harmonicEnergy / totalEnergy;
end

function value = localNanMean(values)
validValues = values(~isnan(values));
if isempty(validValues)
    value = 0;
else
    value = mean(validValues);
end
end

function value = localNanStd(values)
validValues = values(~isnan(values));
if numel(validValues) <= 1
    value = 0;
else
    value = std(validValues);
end
end

function value = localNanMedian(values)
validValues = values(~isnan(values));
if isempty(validValues)
    value = 0;
else
    value = median(validValues);
end
end

function value = localNanIqr(values)
validValues = values(~isnan(values));
if numel(validValues) <= 1
    value = 0;
else
    validValues = sort(validValues);
    q1 = localPercentile(validValues, 25);
    q3 = localPercentile(validValues, 75);
    value = q3 - q1;
end
end

function value = localNanMin(values)
validValues = values(~isnan(values));
if isempty(validValues)
    value = 0;
else
    value = min(validValues);
end
end

function value = localNanMax(values)
validValues = values(~isnan(values));
if isempty(validValues)
    value = 0;
else
    value = max(validValues);
end
end

function value = localPercentile(sortedValues, percentileValue)
if isempty(sortedValues)
    value = 0;
    return;
end

position = (percentileValue / 100) * (numel(sortedValues) - 1) + 1;
lowerIndex = floor(position);
upperIndex = ceil(position);

if lowerIndex == upperIndex
    value = sortedValues(lowerIndex);
else
    weight = position - lowerIndex;
    value = sortedValues(lowerIndex) * (1 - weight) + sortedValues(upperIndex) * weight;
end
end
