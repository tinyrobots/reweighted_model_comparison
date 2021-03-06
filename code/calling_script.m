% Wrapper for fitting models with multiple components
% using bootstrapped cross-validated reweighting.
% katherine.storrs@gmail.com

% INPUT DATA FORMAT:
% Need to provide two sources of data, one containing target data to be
% fitted to (e.g. fMRI-derived RDMs) and one containing the set of models /
% model components which you wish to create a weighted combination of.
%       refRDMs -- brain data RDMs should be a single 3D matrix, with
%                  dimensions CONDITIONS x CONDITIONS x SUBJECTS
%       model_struct -- models should be in a 1 x NUM_MODELS struct, with
%                  the fields:
%                       -- "RDM": 2D CONDITIONS x CONDITIONS matrix
%                       -- "name": short string name for that model, to be
%                          used in plots.
% The two "data_demo" directories contain some example .mat data files in
% the above formats.

% OUTPUTS:
% As well as returning and saving a bar plot and a comparison matrix, 
% this script will return the following variables:
%       stats_p_r -- structure from
%                    FUNC_compareRefRDM2candRDMs_reweighting, containing
%                    similar statistical info usually returned by
%                    compareRefRDM2candRDMs in RSA Toolbox
%       model_corrs -- N_bootstraps x (N_models + 1) matrix listing the
%                    correlations obtained for each model on each bootstrap
%                    sample, including the final "model" which is the
%                    reweighted version of all components provided
%       low_ceiling_corrs -- 1 x N_bootstraps vector listing the lower
%                    bound obtained on each bootstrap sample
%       upp_ceiling_corrs -- 1 x N_bootstraps vector listing the upper
%                    bound obtained on each bootstrap sample
clear all

% This points to a fresh install (04.11.18) from github of the latest
% master branch of the RSA toolbox:
rsa_toolbox_dir = 'D:\Dropbox\coding\MATLAB\rsatoolbox-master';
addpath(genpath(rsa_toolbox_dir));

% where to save analysis results and figures
save_dir = '../results/';

%% EXAMPLE 1: 4-subject 92-image dataset 
% from Khaligh-Razavi & Kriegeskorte (2014)
% load data and set options...

load('../data_92demo/hIT_92imgs.mat') 
refRDMs = hIT_92imgs;

% precalculated AlexNet models
% load('../data_92demo/convNetRDMs.mat')
% load('../data_92demo/svmRDMs.mat')
% model_struct = [convNetRDMs(1:7), svmRDMs];

% or 27 shallow models (ignore final reweighted model)
load('../data_92demo/modelRDMs_28models.mat')
model_struct = modelRDMs(1:27);
highlevel_options.plotComparisonBars = false; % in line with original Fig 2, don't show comparisons

% restructure as a cell array for compatibility with compareRefRDM2candRDMs...
for i = 1:length(model_struct)
    model_RDMs{i} = model_struct(i);
end

% options used by FUNC_compareRefRDM2candRDMs_reweighting
highlevel_options.reweighting = true; % true = bootstrapped xval reweighting. Otherwise proceeds with standard RSA Toolbox analysis.
highlevel_options.resultsPath = save_dir;
highlevel_options.barsOrderedByRDMCorr = false;
highlevel_options.rootPath = pwd;

% options used by FUNC_bootstrap_wrapper
highlevel_options.boot_options.nboots = 100; % make small dummy value for test run
highlevel_options.boot_options.boot_conds = true;
highlevel_options.boot_options.boot_subjs = false; % insufficient subjects here

% options used by FUNC_reweighting_wrapper
highlevel_options.rw_options.nTestSubjects = 1; 
highlevel_options.rw_options.nTestImages = 23; % i.e. 1/4 of the 92 images
highlevel_options.rw_options.nImageLoops = 20; % number of crossvalidation loops within each bootstrap sample (stabilises estimate)
if highlevel_options.rw_options.nTestSubjects == 1
    highlevel_options.rw_options.nSubjectLoops = size(refRDMs,3); % will use exhaustive LOO xval
else
    highlevel_options.rw_options.nSubjectLoops = 20; % specify number of inner crossvalidation loops within each bootstrap sample (xvals reweighting over subjects)
end

%% EXAMPLE 2: 24-subject 62-image dataset 
% from Alex Walther & Niko Kriegeskorte
% load data and set options...

load('../data_62demo/hIT_62imgs.mat') 
refRDMs = hIT_62imgs;

load('../data_62demo/alexnetRDMs.mat')
model_struct = alexnetRDMs;

% restructure as a cell array for compatibility with compareRefRDM2candRDMs...
for i = 1:length(model_struct)
    model_RDMs{i} = model_struct(i);
end

% options used by FUNC_compareRefRDM2candRDMs_reweighting
highlevel_options.reweighting = true; % true = bootstrapped xval reweighting. Otherwise proceeds with standard RSA Toolbox analysis.
highlevel_options.resultsPath = save_dir;
highlevel_options.barsOrderedByRDMCorr = false;
highlevel_options.rootPath = pwd;

% options used by FUNC_bootstrap_wrapper
highlevel_options.boot_options.nboots = 100; % make small dummy value for test run
highlevel_options.boot_options.boot_conds = true;
highlevel_options.boot_options.boot_subjs = true;

% options used by FUNC_reweighting_wrapper
highlevel_options.rw_options.nTestSubjects = 10; % if 1, will do exhaustive LOO xval. Otherwise, select desired split of train/test subjects for inner xval loop.
highlevel_options.rw_options.nTestImages = 10; % of the 62 images
highlevel_options.rw_options.nImageLoops = 20; % number of outer crossvalidation loops within each bootstrap sample (stabilises estimate)
if highlevel_options.rw_options.nTestSubjects == 1
    highlevel_options.rw_options.nSubjectLoops = size(refRDMs,3); % will use exhaustive LOO xval
else
    highlevel_options.rw_options.nSubjectLoops = 20; % specify number of inner crossvalidation loops within each bootstrap sample (xvals reweighting over subjects)
end

%% analyse
tic
[stats_p_r, model_corrs, low_ceiling_corrs, upp_ceiling_corrs] = FUNC_compareRefRDM2candRDMs_reweighting(refRDMs, model_RDMs, highlevel_options)
toc
