function audioFiles = listGenreAudioFiles(datasetRoot)
% listGenreAudioFiles
% Read WAV files from this exact layout:
%   dataset/<class_name>/<wav_file>.wav
%
% Return a struct array with:
%   - className: folder name (genre/class label)
%   - fileName: WAV file name
%   - fullPath: absolute path to WAV file

genreFolders = dir(datasetRoot);
genreFolders = genreFolders([genreFolders.isdir]);
genreFolders = genreFolders(~ismember({genreFolders.name}, {'.', '..'}));
genreNames = {genreFolders.name};
isHiddenGenre = cellfun(@(name) ~isempty(name) && name(1) == '.', genreNames);
genreFolders = genreFolders(~isHiddenGenre);

audioFiles = struct('className', {}, 'fileName', {}, 'fullPath', {});

for folderIndex = 1:numel(genreFolders)
    className = genreFolders(folderIndex).name;
    classFolder = fullfile(datasetRoot, className);
    wavFilesLower = dir(fullfile(classFolder, '*.wav'));
    wavFilesUpper = dir(fullfile(classFolder, '*.WAV'));
    wavFiles = [wavFilesLower; wavFilesUpper];

    % Keep only files directly under class folder and skip hidden files.
    wavFiles = wavFiles(~[wavFiles.isdir]);
    if ~isempty(wavFiles)
        wavNames = {wavFiles.name};
        isHiddenWav = cellfun(@(name) ~isempty(name) && name(1) == '.', wavNames);
        wavFiles = wavFiles(~isHiddenWav);
    end

    for wavIndex = 1:numel(wavFiles)
        audioFiles(end + 1) = struct( ... %#ok<AGROW>
            'className', className, ...
            'fileName', wavFiles(wavIndex).name, ...
            'fullPath', fullfile(wavFiles(wavIndex).folder, wavFiles(wavIndex).name));
    end
end

% Deterministic order: class name, then file name.
if ~isempty(audioFiles)
    sortKey = strcat({audioFiles.className}, '/', {audioFiles.fileName});
    [~, order] = sort(lower(sortKey));
    audioFiles = audioFiles(order);
end
end
