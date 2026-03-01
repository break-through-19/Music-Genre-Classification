%% Iteration 3: PCA via SVD and Dimensionality Reduction
clear; clc; close all;

% Set global theme for plots
set(groot, 'defaultTextColor', 'k', 'defaultAxesXColor', 'k', ...
    'defaultAxesYColor', 'k', 'defaultTextFontWeight', 'bold', ...
    'defaultAxesFontWeight', 'bold');

% 1. Load raw data
filename = 'mel_spectrogram_features.csv'; 
opts = detectImportOptions(filename);
opts.VariableNamingRule = 'preserve';
T = readtable(filename, opts);

% Define Class Labels
groups = categorical(T.class_name);
genres = categories(groups);
num_genres = length(genres);
custom_colors = lines(num_genres); 

% Extract strictly numeric feature data (Assuming columns 4 to end)
numeric_data = table2array(T(:, 4:end));
feature_names = T.Properties.VariableNames(4:end);
[n, p] = size(numeric_data);

% 3. Graph 3D Scatter Plots (Original Features - All Features Covered)

num_features = size(numeric_data, 2);
plot_count = 1;

% This loop cleans raw names (e.g., 'harmonic_f0_mean_hz') into 
% labels with units (e.g., 'Harmonic F0 Mean Hz (Hz)')
clean_names = cell(1, num_features);
for k = 1:num_features
    raw_name = feature_names{k};
    
    % 1. Replace underscores with spaces
    c_name = strrep(raw_name, '_', ' ');
    
    % 2. Capitalize the first letter of each word
    c_name = regexprep(c_name, '(^|\s)([a-z])', '${upper($2)}');
    
    % 3. Append units based on keywords in the feature name
    raw_lower = lower(raw_name);
    if contains(raw_lower, 'hz')
        c_name = [c_name, ' (Hz)'];
    elseif contains(raw_lower, 'ratio') || contains(raw_lower, 'rate')
        c_name = [c_name, ' (0-1)'];
    elseif contains(raw_lower, 'mel') || contains(raw_lower, 'energy') || contains(raw_lower, 'spread')
        c_name = [c_name, ' (dB)'];
    else
        c_name = [c_name, ' (Val)']; % Fallback unit
    end
    clean_names{k} = c_name;
end
% ------------------------------------------

% Loop through all features 3 at a time
for k = 1:3:num_features
    % Determine the three features to plot
    featA = k;
    
    % Handle the end of the list so we don't go out of bounds
    if k + 2 <= num_features
        featB = k + 1;
        featC = k + 2;
    else
        % Complete the final triad with the last 3 features
        featA = num_features - 2;
        featB = num_features - 1;
        featC = num_features;
    end
    
    figure('Name', sprintf('3D Original Features %d', plot_count), ...
           'Color', 'w', 'Position', [50+(plot_count*30), 50+(plot_count*30), 800, 600]);
    hold on;
    
    for i = 1:num_genres
        idx = (groups == genres{i});
        scatter3(numeric_data(idx, featA), numeric_data(idx, featB), numeric_data(idx, featC), ...
                 15, custom_colors(i,:), 'filled');
    end
    
    % Formatting using the newly cleaned names
    title(sprintf('Figure 3.%d: 3D Scatter Plot (%s, %s, %s)', ...
          plot_count, clean_names{featA}, clean_names{featB}, clean_names{featC}), 'Interpreter', 'none');
    
    % Apply cleaned labels with units to the axes
    xlabel(clean_names{featA}); 
    ylabel(clean_names{featB}); 
    zlabel(clean_names{featC});
    
    grid on; 
    view(45, 30); % Sets a clear 3D viewing angle
    
    legend(genres, 'Location', 'northeastoutside');
    hold off;
    
    % Stop the loop if we just plotted the end of the array
    if featC == num_features
        break;
    end
    plot_count = plot_count + 1;
end

% 4. Perform Singular Value Decomposition (SVD)
% Load the Normalized Data
filename = 'mel_spectrogram_features_normalized.csv'; 
opts = detectImportOptions(filename);
opts.VariableNamingRule = 'preserve';
T = readtable(filename, opts);

% Define Class Labels
groups = categorical(T.class_name);
genres = categories(groups);
num_genres = length(genres);
custom_colors = lines(num_genres); 

% Extract Feature Matrix
% We extract columns 4 to end to get strictly the acoustic features.
X = table2array(T(:, 4:end));
feature_names = T.Properties.VariableNames(4:end);
[n, p] = size(X);

% Format feature names for the Loading Vector plot
clean_names = strrep(feature_names, '_', ' ');

% Perform Singular Value Decomposition (SVD)
% The data is already mean-centered and normalized.
% We use 'econ' to compute only the required components efficiently.
[U, S, V] = svd(X, 'econ');

% V = Loading Vectors (Principal Directions)
% S = Singular Values (Diagonal Matrix)
% U = Left Singular Vectors

% Calculate Variance Explained
eigenvalues = diag(S).^2 / (n - 1);
variance_explained = (eigenvalues / sum(eigenvalues)) * 100;
cumulative_variance = cumsum(variance_explained);

% 5. Graph Scree Plot
figure('Name', 'Scree Plot', 'Color', 'w', 'Position', [100, 100, 800, 500]);
yyaxis left
bar(1:p, variance_explained, 'FaceColor', [0.2 0.6 0.8]);
ylabel('Variance Explained (%)');
ylim([0 max(variance_explained)*1.2]);

yyaxis right
plot(1:p, cumulative_variance, '-ro', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
ylabel('Cumulative Variance (%)');
ylim([0 105]);

title('Scree Plot: Variance Explained by Principal Components');
xlabel('Principal Component Number');
xticks(1:p); grid on;

% 6. Plot Loading Vectors (PC1, PC2, PC3, PC4, and Last PC)
% Define the specific Principal Components you want to plot
pcs_to_plot = [1, 2, 3, 4, p]; % 'p' is automatically the last PC (16)

% Generate a set of distinct colors for the different plots
bar_colors = lines(length(pcs_to_plot)); 

for i = 1:length(pcs_to_plot)
    pc_idx = pcs_to_plot(i);
    
    % Create a new figure for each PC
    figure('Name', sprintf('Loading Vector PC%d', pc_idx), ...
           'Color', 'w', 'Position', [150 + i*50, 150 + i*50, 900, 500]);
           
    % Plot the bar chart for the specific column in the V matrix
    b = bar(1:p, V(:, pc_idx), 'FaceColor', bar_colors(i, :));
    
    % Formatting and Labels
    title(sprintf('Loading Vector for Principal Component %d (Variance: %.2f%%)', ...
          pc_idx, variance_explained(pc_idx)));
    xlabel('Original Features');
    ylabel('Loading Weight (Coefficient)');
    
    % Apply the cleaned feature names to the X-axis
    xticks(1:p);
    xticklabels(clean_names);
    xtickangle(45); % Tilt labels for readability
    
    grid on;
end

% 7. Generate Score Vectors and 3D Score Plots
% The score vectors are the data projected into the new PCA space
Scores = U * S; % Mathematically equivalent to X * V

% Score Plot 1: PC1, PC2, PC3
figure('Name', '3D Score Plot (PC1, PC2, PC3)', 'Color', 'w', 'Position', [200, 200, 800, 600]);
hold on;
for i = 1:num_genres
    idx = (groups == genres{i});
    scatter3(Scores(idx, 1), Scores(idx, 2), Scores(idx, 3), ...
             15, custom_colors(i,:), 'filled');
end
title('3D Score Plot: Principal Components (1, 2, 3)');
xlabel(sprintf('PC1 (%.1f%% Variance)', variance_explained(1)));
ylabel(sprintf('PC2 (%.1f%% Variance)', variance_explained(2)));
zlabel(sprintf('PC3 (%.1f%% Variance)', variance_explained(3)));
grid on; view(45, 30);
legend(genres, 'Location', 'northeastoutside');
hold off;

% Score Plot 2: PC2, PC3, PC4
figure('Name', '3D Score Plot (PC2, PC3, PC4)', 'Color', 'w', 'Position', [250, 250, 800, 600]);
hold on;
for i = 1:num_genres
    idx = (groups == genres{i});
    scatter3(Scores(idx, 2), Scores(idx, 3), Scores(idx, 4), ...
             15, custom_colors(i,:), 'filled');
end
title('3D Score Plot: Principal Components (2, 3, 4)');
xlabel(sprintf('PC2 (%.1f%% Variance)', variance_explained(2)));
ylabel(sprintf('PC3 (%.1f%% Variance)', variance_explained(3)));
zlabel(sprintf('PC4 (%.1f%% Variance)', variance_explained(4)));
grid on; view(45, 30);
legend(genres, 'Location', 'northeastoutside');
hold off;