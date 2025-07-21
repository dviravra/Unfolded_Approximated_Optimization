%% Testing code for LARPCA 
clear; clc;
approxU = [];  
approxV = [];
   
%% Load Data
data_path = "./data/n1000_r5_eta_matrix.mat";
load(data_path);  
[n, r] = size(U_star);  
X_star = sparse(double(U_star * V_star'));  % ground truth
Y = sparse(double(Y_star));  

%% Load Model
model_path = "./trained_models/LARPCA_r5_approxU_4,6,8,10,12,14_approxV_3,5,7,9,11,13,15.mat";
load(model_path);  
zeta = double(ths) * (1000/n) * (r/5);  % thresholds
etaU = double(stepU);  % matrix step sizes for U, shape: [maxIt x d1 x r]
etaV = double(stepV);  % matrix step sizes for V, shape: [maxIt x d2 x r]

%% Run the LearnedApproximationRPCA 
[X, L, R] = LearnedRPCA(Y, r, X_star, zeta, etaU, etaV, approxU, approxV);

function [X, L, R] = LearnedRPCA(Y, r, X_star, zeta, etaU, etaV, approxU, approxV)
    [~, T] = size(zeta);
    time_counter = 0;
    iterationLogs = struct('iteration', cell(1,T-1),'error', cell(1,T-1),'time', cell(1,T-1));
    % Initialization using SVD
    tStart = tic;
    [U0, Sigma0, V0] = svds(Y - Thre(Y, zeta(1)), r);
    L = U0 * sqrt(Sigma0);
    R = V0 * sqrt(Sigma0);
    time_counter = time_counter + toc(tStart);
    
    fprintf("===============LRPCA logs=============\n");
    % Main loop 
    for t = 1:(T-1)
        tStart = tic;
        X = L * R';
        S = Thre(Y - X, zeta(t+1));
        if ismember(t+1, approxU)
            L_plus = L;
        else
            update_L = (X + S - Y) * R / (R' * R + eps('double') * eye(r));
            L_plus = L - update_L .* squeeze(etaU(t+1,:,:));
        end
        if ismember(t+1, approxV)
            R_plus = R;
        else
            update_R = (X + S - Y)' * L / (L' * L + eps('double') * eye(r));
            R_plus = R - update_R .* squeeze(etaV(t+1,:,:));
        end
        L = L_plus;
        R = R_plus;
        time_counter = time_counter + toc(tStart);
        dist_X = norm(X - X_star, 'fro') / norm(X_star, 'fro');
        fprintf("k: %d Err: %e Time: %f\n", t, dist_X, time_counter);
        % save the k-iter
        iterationLogs(t).iteration = t;        
        iterationLogs(t).error     = dist_X;  % error value   
        iterationLogs(t).time      = time_counter; % run time   
    end
    fprintf("======================================\n");
end

%% Soft Thresholding Function
function S = Thre(S, theta)
    S = sign(S) .* max(abs(S) - theta, 0.0);
end

