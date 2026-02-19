function segmentSignals = splitAudioIntoFixedSegments(audioSignal, sampleRate, segmentDurationSeconds)
% splitAudioIntoFixedSegments
% Split a waveform into fixed-length contiguous segments.
%
% For 30-second GTZAN files and 3-second segmentDurationSeconds,
% this returns 10 segments.
%
% If the tail is shorter than one full segment, it is discarded.

validateattributes(audioSignal, {'double', 'single'}, {'vector', 'nonempty'}, mfilename, 'audioSignal');
validateattributes(sampleRate, {'numeric'}, {'scalar', 'positive'}, mfilename, 'sampleRate');
validateattributes(segmentDurationSeconds, {'numeric'}, {'scalar', 'positive'}, mfilename, 'segmentDurationSeconds');

samplesPerSegment = round(segmentDurationSeconds * sampleRate);
totalSamples = numel(audioSignal);
numSegments = floor(totalSamples / samplesPerSegment);

segmentSignals = cell(numSegments, 1);
for segmentIndex = 1:numSegments
    startSample = (segmentIndex - 1) * samplesPerSegment + 1;
    endSample = segmentIndex * samplesPerSegment;
    segmentSignals{segmentIndex} = audioSignal(startSample:endSample);
end
end
