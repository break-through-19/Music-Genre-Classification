function harmonicFeatureVector = extractHarmonicFeatureVector(audioSignal, sampleRate, windowLengthSamples, hopLengthSamples, analysisWindow)
% extractHarmonicFeatureVector
% Compute 4 harmonic features:
% 1 harmonic_voiced_frame_ratio
% 2 harmonic_f0_mean_hz
% 3 harmonic_f0_std_hz
% 4 harmonic_energy_ratio_mean

audioSignal = audioSignal(:);
signalLength = numel(audioSignal);

if signalLength < windowLengthSamples
    audioSignal = [audioSignal; zeros(windowLengthSamples - signalLength, 1)];
    signalLength = numel(audioSignal);
end

frameStarts = 1:hopLengthSamples:(signalLength - windowLengthSamples + 1);
numFrames = numel(frameStarts);

f0Values = nan(numFrames, 1);
harmonicEnergyRatios = zeros(numFrames, 1);

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

    if peakValue > 0.30
        f0Values(frameIndex) = sampleRate / bestLag;
    end

    harmonicEnergyRatios(frameIndex) = localSpectrumHarmonicRatio( ...
        frameSignal, ...
        sampleRate, ...
        f0Values(frameIndex));
end

voicedMask = ~isnan(f0Values);
voicedRatio = sum(voicedMask) / max(numFrames, 1);
f0Mean = localNanMean(f0Values);
f0Std = localNanStd(f0Values);

harmonicFeatureVector = [ ...
    voicedRatio, ...
    f0Mean, ...
    f0Std, ...
    mean(harmonicEnergyRatios)];
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

function harmonicEnergyRatio = localSpectrumHarmonicRatio(frameSignal, sampleRate, f0Hz)
fftLength = 2 ^ nextpow2(numel(frameSignal));
spectrumMagnitude = abs(fft(frameSignal, fftLength));
spectrumMagnitude = spectrumMagnitude(1:(floor(fftLength / 2) + 1));
totalEnergy = sum(spectrumMagnitude .^ 2) + eps;

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
