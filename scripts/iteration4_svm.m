%% Iteration 4: SVM Classification (Baseline, PCA Space, and Curated Space)
clear; clc; close all;

% Set global theme for plots
set(groot, 'defaultTextColor', 'k', 'defaultAxesXColor', 'k', ...
    'defaultAxesYColor', 'k', 'defaultTextFontWeight', 'bold', ...
    'defaultAxesFontWeight', 'bold');

% Ensure the output folder exists for your Overleaf plots
if ~exist('Iteration 4 plots', 'dir')
    mkdir('Iteration 4 plots');
end

%% 1. Load the Normalized Data
filename = 'mel_spectrogram_features_normalized.csv'; 
opts = detectImportOptions(filename);
opts.VariableNamingRule = 'preserve';
T = readtable(filename, opts);

% Define Class Labels
labels = categorical(T.class_name);
genres = categories(labels);

% Extract Feature Matrix (Columns 4 to end)
X_norm = table2array(T(:, 4:end));
feature_names = T.Properties.VariableNames(4:end);

% Re-calculate SVD (so we have U and S for Phase 2)
[U, S, V] = svd(X_norm, 'econ');

% 2. Setup Train/Test Split (80% Training, 20% Testing)
% We use a fixed random seed for reproducibility in your report
rng(42); 
cv = cvpartition(labels, 'HoldOut', 0.2);
idxTrain = training(cv);
idxTest = test(cv);

y_train = labels(idxTrain);
y_test = labels(idxTest);

% Define the SVM Template (Gaussian/RBF Kernel for overlapping boundaries)
svm_template = templateSVM('KernelFunction', 'gaussian', 'Standardize', true);

%% Phase 1: The Original Baseline (All 16 Features)
fprintf('Phase 1: Baseline SVM (All 16 Features)\n');

X_baseline_train = X_norm(idxTrain, :);
X_baseline_test  = X_norm(idxTest, :);

% Train and Predict using fitcecoc (Multi-class SVM)
% This tells MATLAB to spend a few minutes testing hundreds of combinations 
% to find the absolute maximum accuracy possible
mdl_baseline = fitcecoc(X_baseline_train, y_train, 'Learners', svm_template, ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', 'expected-improvement-plus', 'ShowPlots', false));
pred_baseline = predict(mdl_baseline, X_baseline_test);

% Calculate Accuracy
acc_baseline = sum(pred_baseline == y_test) / length(y_test) * 100;
fprintf('Baseline Accuracy: %.2f%%\n\n', acc_baseline);

% Plot and Save Confusion Matrix
fig1 = figure('Name', 'Phase 1: Baseline Confusion Matrix', 'Color', 'w', 'Position', [100, 100, 700, 500]);
confusionchart(y_test, pred_baseline, 'Title', sprintf('Phase 1: Baseline SVM (Acc: %.1f%%)', acc_baseline));
exportgraphics(fig1, 'Iteration 4 plots/SVM_Phase1_Baseline.png', 'Resolution', 300);


%% Phase 2: The Transformed PCA Space (U_r)
fprintf('Phase 2: PCA Transformed SVM (Top 4 Components)\n');

% Calculate the full Score Matrix (U * S)
Scores_full = U * S;

% Truncate to the top r components (Based on the Scree Plot)
r = 4; 
Scores_truncated = Scores_full(:, 1:r);

X_pca_train = Scores_truncated(idxTrain, :);
X_pca_test  = Scores_truncated(idxTest, :);

% Train and Predict
mdl_pca = fitcecoc(X_pca_train, y_train, 'Learners', svm_template);
pred_pca = predict(mdl_pca, X_pca_test);

% Calculate Accuracy
acc_pca = sum(pred_pca == y_test) / length(y_test) * 100;
fprintf('PCA Space Accuracy (r=%d): %.2f%%\n\n', r, acc_pca);

% Plot and Save Confusion Matrix
fig2 = figure('Name', 'Phase 2: PCA Confusion Matrix', 'Color', 'w', 'Position', [150, 150, 700, 500]);
confusionchart(y_test, pred_pca, 'Title', sprintf('Phase 2: PCA SVM r=%d (Acc: %.1f%%)', r, acc_pca));
exportgraphics(fig2, 'Iteration 4 plots/SVM_Phase2_PCA.png', 'Resolution', 300);


%% Phase 3: The Curated Original Space
fprintf('Phase 3: Curated Feature Space\n');

% Based on the Iteration 3 analysis, drop the redundant/noisy features.
% Column 2 = timbral_log_mel_median (Highly collinear with mean)
% Column 8 = variability_band_range_mean (Noisy, heavy-tailed)
cols_to_drop = [2, 8]; 

% Create curated dataset by deleting those columns
X_curated = X_norm;
X_curated(:, cols_to_drop) = []; 

X_curated_train = X_curated(idxTrain, :);
X_curated_test  = X_curated(idxTest, :);

% Train and Predict
mdl_curated = fitcecoc(X_curated_train, y_train, 'Learners', svm_template);
pred_curated = predict(mdl_curated, X_curated_test);

% Calculate Accuracy
acc_curated = sum(pred_curated == y_test) / length(y_test) * 100;
fprintf('Curated Space Accuracy: %.2f%%\n\n', acc_curated);

% Plot and Save Confusion Matrix
fig3 = figure('Name', 'Phase 3: Curated Confusion Matrix', 'Color', 'w', 'Position', [200, 200, 700, 500]);
confusionchart(y_test, pred_curated, 'Title', sprintf('Phase 3: Curated SVM (Acc: %.1f%%)', acc_curated));
exportgraphics(fig3, 'Iteration 4 plots/SVM_Phase3_Curated.png', 'Resolution', 300);

fprintf('Plots saved to "Iteration 4 plots" folder.');