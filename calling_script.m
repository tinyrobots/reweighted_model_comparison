% Wrapper for reanalysis of data from Khaligh-Razavi & Kriegeskorte (2014)
% using bootstrapped cross-validated reweighting.
% katherine.storrs@gmail.com

% OUTPUTS:
% As well as returning and saving plots and correlation matrices, 
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

% This was written to work with a fresh install from github of the
% latest master branch of the RSA toolbox (04.11.2018):
rsa_toolbox_dir = 'D:\Dropbox\coding\MATLAB\rsatoolbox-master';
addpath(genpath(rsa_toolbox_dir));

%% user variables 

% point to locations of models, data etc
refRDM_file = 'hIT_92imgs.mat';
model_dir = '.';

% where to save analysis results and figures
save_dir = './results/';

% load model and brain RDMs...
% BEWARE this is all hardcoded presently for the demo reanalyis of Seyed's data
% -- change file names in here to yours.

load(refRDM_file) 
refRDMs = hIT_92imgs;

load(strcat(model_dir, 'convNetRDMs.mat'))
load(strcat(model_dir, 'svmRDMs.mat'))

model_struct = [convNetRDMs(1:7), svmRDMs];

% restructure as a cell array for compatibility with compareRefRDM2candRDMs...
for i = 1:length(model_struct)
    model_RDMs{i} = model_struct(i);
end

%% set analysis parameters

% options used by FUNC_compareRefRDM2candRDMs_reweighting
highlevel_options.reweighting = true; % true = bootstrapped xval reweighting. Otherwise proceeds with standard RSA Toolbox analysis.
highlevel_options.resultsPath = save_dir;
highlevel_options.barsOrderedByRDMCorr = false;
highlevel_options.rootPath = pwd;

% options used by FUNC_bootstrap_wrapper
highlevel_options.boot_options.nboots = 1000; % make small dummy value for test run
highlevel_options.boot_options.boot_conds = true;
highlevel_options.boot_options.boot_subjs = false; % WARNING: will not currently work if true, I don't think.

% options used by FUNC_reweighting_wrapper
highlevel_options.rw_options.nTestSubjects = 1; % current version only does LOO crossval, suitable for 4 subject data
highlevel_options.rw_options.nTestImages = 23; % i.e. 1/4 of the 92 images
highlevel_options.rw_options.nLoops = 20; % number of crossvalidation loops within each bootstrap sample (stabilises estimate)

tic
[stats_p_r, model_corrs, low_ceiling_corrs, upp_ceiling_corrs] = FUNC_compareRefRDM2candRDMs_reweighting(refRDMs, model_RDMs, highlevel_options)
toc
