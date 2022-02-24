%% Example wrapper script, drawing data from ./data_62demo
% Estimates Representational Similarity between a model and some brain data,
% where the model has multiple components. The performance of each component
% separately is estimated, along with the performance of the best linearly-
% weighted combination of all components. Weighting is done by cross-
% -validating over both subjects and stimuli. A bootstrap distribution of 
% performance estimates (over subjects and/or stimuli) is returned.

% Procedure is as used in Storrs et al (2021) https://doi.org/10.1162/jocn_a_01755
% katherine.storrs@gmail.com

% INPUTS:
%       fmri_RDM_file -- .mat file containing target data (e.g. fMRI-derived RDMs)
%                   Expects .mat file to contain a single data matrix of
%                   dimensions N x N x S, where N is the number of stimuli
%                   and S is the number of participants/subjects. Each
%                   entry is the dissimilarity between two stimuli
%                   for a given subject.
%       model_RDM_file -- .mat file containing model data
%                   Expects .mat file to contain a cell array, with M cells,
%                   one for each of the M model components. Each cell contains 
%                   a structure with one field named ".rawRDM", within 
%                   is another structure with two fields:
%                       .RDM: NxN square Representational Dissimilarity 
%                          Matrix containing pairwise distances between 
%                          all N stimuli
%                       .name: string providing name of this model 
%                           component (e.g. "Layer 1")
%                   (nb. Nested structure was to facilitate extension to 
%                   more complicated analyses where models might have 
%                   multiple sets of nested components)
%       savedir -- where to save results
%       nboots -- number of bootstrap samples to perform
%       nCVs -- number of cross-validation folds to perform within each
%                   bootstrap sample. On each, a test set of subjects and
%                   stimuli is set aside, the model components are fitted
%                   on the training split, and evaluated on the remainder
%
% OUTPUTS:
% This script will output and save the following structures:
%       bootstrap_output_components -- cell array with M entries, one for each of 
%                   the M model components. Each cell contains a structure
%                   with one field named:
%                       .raw -- (nboots x 1) vector of bootstrap estimates 
%                           for this component alone
%       bootstrap_output_ceilings -- struct with 2 fields. Each field contain an 
%                   (nboots x 1) vector of bootstrap estimates of the 
%                   performances of the expected performance of the "true"
%                   model, given the inter-subject variability in the data.
%                   See Nili et al (2014, https://doi.org/10.1371/journal.pcbi.1003553) for background information on
%                   noise ceiling calculations, and see Storrs et al (2020, https://www.biorxiv.org/content/10.1101/2020.03.23.003046v1)
%                   for a note on correct ceiling calculations for
%                   reweighted models:
%                       .lower -- lower bound of noise ceiling
%                       .upper -- upper bound of noise ceiling
%       bootstrap_output_combined -- struct with 2 fields, each containing an 
%                   (nboots x 1) vector of bootstrap estimates of the 
%                   combined performance of all components in the model,
%                   when either:
%                       .raw_fitted -- combined according to the best
%                           linear reweighting, where weights are fitted in
%                           the cross-validation procedure
%                       .raw_unif -- combined with equal weights. Provides
%                           a helpful baseline to assess how much the model 
%                           benefits from reweighting

clear all

% point to human and model data files:
fmri_RDM_file = '../data_62demo/hIT_62imgs.mat';
model_RDM_file = '../data_62demo/alexnetRDMs.mat';

% set arguments for cross-validation process
nboots = 1000; % can make small for debugging. At least 1000 for full run.
nCV = 100; % can make small for debugging (minimum 2). At least 50 for full run.
nTestSubjects = 5; % approx 20% of the total number of participants
nTestImages = 12; % approx 20% of the total number of stimuli

% specify where to save results and create if it doesn't exist 
% (beware overwriting! e.g. use a unique savedir with an informative name 
% for each major run of the analysis)
savedir = '../results/'; 
try mkdir(savedir); end

%% 1. Load human target data
% Amend name to suit your dataset

load(fmri_RDM_file); % loads as whatever name the data RDM was saved under,
                     % in this demo it's called "hIT_62imgs"
refRDMs = hIT_62imgs; % point to correct name of your data RDM

% edit subject data to put NaNs in the diagonals, so that when
% bootstrap resampling causes some to end up in the off-diag spots they can
% be ignored (zeros would artificially inflate correlation)
for s = 1:size(refRDMs,3)
    this_subj = refRDMs(:,:,s);
    this_subj(logical(eye(size(this_subj,1)))) = NaN;
    refRDMs(:,:,s) = this_subj;
end

%% 2. Load model components
% Amend name to suit your dataset

load(model_RDM_file); % loads as whatever name the data RDM was saved under,
                      % in this demo it's called "alexnetRDMs"
model_RDMs = alexnetRDMs; % point to correct name of your data RDM

%% 3. Set options
% Mostly specified already at top of script, except for choice of what to
% bootstrap over. Default is both subjects and stimuli.

% options used by FUNC_bootstrap_wrapper
highlevel_options.boot_options.nboots = nboots; 
highlevel_options.boot_options.boot_conds = true; % true = will bootstrap over stimulus conditions
highlevel_options.boot_options.boot_subjs = true; % true = will bootstrap over subjects/participants

% options used by FUNC_reweighting_wrapper
highlevel_options.rw_options.nTestSubjects = nTestSubjects;
highlevel_options.rw_options.nTestImages = nTestImages;
highlevel_options.rw_options.nCVs = nCV; 

%% Run analysis and save results
[component_results, ceiling_results, combined_results] = FUNC_bootstrap_wrapper(refRDMs, model_RDMs, highlevel_options);

% save bootstrap distributions
save(strcat(savedir,'bootstrap_output_components'), 'component_results');
save(strcat(savedir,'bootstrap_output_ceilings'), 'ceiling_results');
save(strcat(savedir,'bootstrap_output_combined'), 'combined_results');
