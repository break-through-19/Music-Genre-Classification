%% 1. Setup and Data Loading
clear; clc; close all;

filename = 'mel_spectrogram_features.csv'; 
opts = detectImportOptions(filename);
opts.VariableNamingRule = 'preserve';
T = readtable(filename, opts);

% Define Class Labels (Genres)
groups = categorical(T.class_name);
num_genres = length(categories(groups));
custom_colors = lines(num_genres); % Ensures unique colors for all genres
genres = categories(groups);

%% Plot 1. Select the 5 Most Important Feature
% Select 5 Features 
feature_subset = [T.timbral_log_mel_mean, ...
                  T.variability_log_mel_std_global, ...
                  T.temporal_zero_crossing_rate_mean, ...
                  T.harmonic_f0_mean_hz, ...
                  T.harmonic_voiced_frame_ratio];

% Add units for the scatterplots
feature_names = {'Mean Energy (dB)', 'Energy StdDev (dB)', 'Zero Crossing (rate)', 'Pitch (Hz)', 'Voiced Ratio (0-1)'};
num_features = size(feature_subset, 2);

% Create the Scatter Plot Grid
figure('Name', 'Pairwise Scatter Plots', 'Color', 'w', 'Position', [100, 100, 1200, 800]);
t = tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'compact');
title(t, 'Figure 1: Pairwise Scatter Plots of Key Audio Features', 'FontSize', 14, 'FontWeight', 'bold');

% Loop through every unique pair of features
for i = 1:num_features
    for j = (i+1):num_features % Start from i+1 to avoid self-plots (the histograms)
        
        % Move to the next tile in our grid
        nexttile;
        
        % Plot the individual scatter plot
        gscatter(feature_subset(:, i), feature_subset(:, j), groups, custom_colors, '.', 10, 'off');
        
        % Add Individual Labels with units
        xlabel(feature_names{i}, 'FontWeight', 'bold');
        ylabel(feature_names{j}, 'FontWeight', 'bold');
        grid on;
        
    end
end

% Create invisible dummy plots just to generate one clean legend for the whole figure
hold on;
dummy_lines = gobjects(num_genres, 1);
for k = 1:num_genres
    dummy_lines(k) = plot(nan, nan, '.', 'MarkerSize', 20, 'Color', custom_colors(k,:));
end
hold off;

% Place the master legend on the right side of the entire layout
lgd = legend(dummy_lines, genres, 'NumColumns', 1, 'FontSize', 10);
lgd.Layout.Tile = 'east'; % Snaps it to the right edge
title(lgd, 'Music Genres');


%% Plot 2: Dot Diagram & Superimposed Normal Curves (By Class)
% We will use one feature as an example for this plot:
feature_to_plot = T.timbral_log_mel_mean; 
feature_name = 'Timbral Log Mel Mean (dB)'; % Added Unit

figure('Name', 'Normality Check by Class', 'Color', 'w', 'Position', [150, 150, 900, 600]);
hold on;

% Loop through each music genre
for i = 1:num_genres
    % 1. Get the data observations for this specific genre
    genre_data = feature_to_plot(groups == genres{i});
    
    % 2. Calculate Mean and Standard Deviation for the normal curve
    mu = mean(genre_data);
    sigma = std(genre_data);
    
    % 3. Plot the Normal Curve
    x_range = linspace(min(genre_data), max(genre_data), 200);
    y_norm = normpdf(x_range, mu, sigma); % Calculate Bell Curve
    
    % Plot the curve
    plot(x_range, y_norm, 'LineWidth', 2, 'Color', custom_colors(i,:), 'DisplayName', [genres{i} ' (Normal Fit)']);
    
    % 4. Plot the "Dot Diagram" (Observations)
    y_offset = zeros(size(genre_data)) - (i * max(y_norm) * 0.05); 
    scatter(genre_data, y_offset, 15, custom_colors(i,:), 'filled', 'MarkerEdgeColor', 'none', 'HandleVisibility', 'off');
end

% Formatting with units
title(['Figure 2: Normality Check: Observations vs. Normal Distribution for ', feature_name]);
xlabel('Feature Value (dB)'); % Added Unit
ylabel('Probability Density (1/dB)'); % Added Unit
grid on;

% Add legend outside the plot
lgd = legend('Location', 'northeastoutside');
title(lgd, 'Normal Curves by Class');
hold off;


%% Plot 3: Covariance / Correlation Matrix
% Extract only the numeric features (Columns 4 to end)
feature_data = T(:, 4:end); 
feature_names_corr = feature_data.Properties.VariableNames;

% Calculate the Correlation Matrix
R = corr(table2array(feature_data));

figure('Name', 'Correlation Matrix', 'Color', 'w', 'Position', [100, 100, 900, 700]);
h = heatmap(feature_names_corr, feature_names_corr, R);

% Formatting
h.Title = 'Figure 3: Covariance & Correlation Matrix (Dimensionless [-1, 1])'; % Added note on scale
h.Colormap = parula;
h.ColorLimits = [-1 1]; 


%% Plot 4: Box and Whisker Plot
figure('Name', 'Box Plot', 'Color', 'w', 'Position', [100, 100, 800, 500]);
% Create boxplot grouped by genre
boxplot(feature_to_plot, groups, 'Colors', 'k', 'Symbol', 'ro');
title(['Figure 4: Box and Whisker Plot of ', feature_name]);
xlabel('Music Genre');
ylabel('Feature Value (dB)'); % Added Unit
grid on;


%% Plot 5: Normal Probability Plot (Q-Q Plot)
figure('Name', 'Probability Plot', 'Color', 'w', 'Position', [150, 150, 800, 500]);
hold on;

for i = 1:length(genres)
    genre_data = feature_to_plot(groups == genres{i});
    h = probplot('normal', genre_data);
    
    set(h(1), 'Color', custom_colors(i,:), 'Marker', '.'); 
    if length(h) > 1
        set(h(2), 'Color', custom_colors(i,:), 'LineStyle', '-'); 
    end
end
title(['Figure 5: Normal Probability Plot for ', feature_name]);
ylabel('Probability (Cumulative %)'); % Clarified unit
xlabel('Quantiles of Data (dB)'); % Added Unit
legend(genres, 'Location', 'southeast');
grid on;
hold off;


%% Plot 6: Empirical CDF (Cumulative Distribution Function)
figure('Name', 'CDF Plot', 'Color', 'w', 'Position', [200, 200, 800, 500]);
hold on;

for i = 1:length(genres)
    genre_data = feature_to_plot(groups == genres{i});
    [f, x] = ecdf(genre_data);
    plot(x, f, 'LineWidth', 2, 'Color', custom_colors(i,:));
end
title(['Figure 6: Cumulative Distribution Function of ', feature_name]);
xlabel('Feature Value (dB)'); % Added Unit
ylabel('Cumulative Probability (0 to 1)'); % Clarified unit
legend(genres, 'Location', 'southeast');
grid on;
hold off;