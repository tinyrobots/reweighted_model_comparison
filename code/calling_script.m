%% Example arguments, using relative paths in 'JOCN_cluster_analyses' 
% directory to run and save all analyses for the trained Alexnet model:

% fmrifile = '../data/fMRI_RDMs_crossnobis_trans62.mat';
% pcaRDMdir = '../data/alexnet/rdm/pca/';
% randcaRDMdir = '../data/alexnet/rdm/randca/';
% rawRDMdir = '../data/alexnet/rdm/raw/';
% savedir = '../data/alexnet/results/cluster_test/'; 
% samplingfile = '../data/sampling_order_JOCN.mat';
% nCV = 100; % can make smaller for debugging. At least 100 for full run.
% nboots = 1000; % can make smaller for debugging. At least 1000 for full run.
% % Then just call: WRAPPER_FUNC_call_for_one_model_jocn(fmrifile, pcaRDMdir, randcaRDMdir, rawRDMdir, savedir, samplingfile, nboots, nCV)

%% Wrapper for fitting models with multiple components using bootstrapped 
% cross-validated reweighting.
% katherine.storrs@gmail.com

% This version customised for the Journal of Cognitive Neuroscience special
% issue submission "DNNs With Diverse Architectures All Predict human IT..."

% INPUT ARGUMENTS:
% Need to provide three sources of data:
%       fmrifile -- .mat file containing target data to be fitted to 
%                   (e.g. fMRI-derived RDMs)
%       pcaRDMdir -- relative path to directory containing PCA-derived RDMs,
%                   with one .mat file per network layer, each containing
%                   the 100 principal component RDMs for that layer
%       randcaRDMdir -- relative path to directory containing RDMs derived
%                   from random components analysis, with one .mat file per
%                   network layer, each containing the 100 random component
%                   RDMs for that layer
%       rawRDMdir -- relative path to directory containing "raw" RDMs
%                   (i.e. unfitted full feature-space RDMs) as .mat files,
%                   each containing the single raw RDM for that layer
%       savedir -- where to save results
%       samplingfile -- .mat file containing pre-computed structure
%                   containing the subject and stimulus bootstrap samples.
%                   These are pre-assigned so that all models use the same
%                   bootstrap sampling, and a distribution of differences
%                   between estimates on each bootstrap can be built up
%                   allowing statistical comparisons between models as if
%                   they had all been run within the same bootstrapping
%                   procedure.
%       nboots -- number of bootstrap samples to perform (must be equal to
%                   or less than the number of pre-assigned bootstrap
%                   samples in `samplingfile`
%       nCVs -- number of cross-validation folds to perform within each
%                   bootstrap sample. On each, a test set of subjects and
%                   stimuli is set aside, the model components are fitted
%                   on the training split, and evaluated on the remainder
%
% OUTPUT ARGUMENTS:
% This script will output and save the following structures:
%       layerwise_results -- (num_layers x 1) cell array, with each cell 
%                           containing 1 struct with 5 fields. Each field
%                           contain an (nboots x 1) vector of bootstrap 
%                           estimates of the performances of the following
%                           strategies. 
%           .fittedPCA -- fitted combination of 100 PCA RDMs
%           .fittedRANDCA -- fitted combination of 100 random component RDMs
%           .unifPCA -- unfitted equal weighting of 100 PCA RDMs
%           .unifRANDCA -- unfitted equal weighting of 100 random comp. RDMs
%           .raw -- unfitted unweighted single full raw feature RDM of layer
%
%       ceiling_results -- struct with 2 fields. Each field
%                          contain an (nboots x 1) vector of bootstrap 
%                          estimates of the performances of the ceilings:
%           .lower -- lower bound of noise ceiling
%           .upper -- upper bound of noise ceiling
%
%       wholenet_results -- struct with 6 fields. Each field
%                          contain an (nboots x 1) vector of bootstrap 
%                          estimates of the performances of the following
%                          strategies to estimate a whole-network
%                          performance:
%           .fittedPCA_fitted -- all-layer combinations of per-layer
%                               fitted PCA RDMs
%           .fittedRANDCA_fitted -- all-layer combinations of per-layer
%                               fitted random component RDMs
%           .unifPCA_fitted -- all-layer combinations of equally-weighted
%                               per-layer PCA RDMs
%           .unifRANDCA_fitted -- all-layer combinations of equally-weighted
%                               per-layer random component RDMs
%           .raw_fitted -- all-layer combinations of fixed raw per-layer RDMs
%           .raw_unif -- equally-weighted all-layer combinations of fixed raw per-layer RDMs

%% Set these to appropriate paths:

try mkdir(savedir); end

%% JOCN hIT architecture paper: 24-subject 62-image dataset 
% load data and set options...

load(fmrifile, 'fMRI_RDMs');
refRDMs = permute(fMRI_RDMs.IT,[2,3,1]); % or desired ROI
% amend subject data to put NaNs in the diagonals, so that when
% bootstrap resampling causes some to end up in the off-diag spots they can
% be ignored (zeros would artificially inflate correlation)
for s = 1:size(refRDMs,3)
    this_subj = refRDMs(:,:,s);
    this_subj(logical(eye(size(this_subj,1)))) = NaN;
    refRDMs(:,:,s) = this_subj;
end

layerlist = dir([pcaRDMdir,'*.mat']);

% specifically for this analysis -- load pre-generated sampling order
% -- this allows bootstrap simulations using the same random boot and cv
% samples to be run at separate times.
load(samplingfile, 'sampling_order');

% options used by FUNC_compareRefRDM2candRDMs_reweighting
highlevel_options.reweighting = true; % true = bootstrapped xval reweighting. Otherwise proceeds with standard RSA Toolbox analysis.
highlevel_options.resultsPath = savedir;
highlevel_options.barsOrderedByRDMCorr = false;
highlevel_options.rootPath = pwd;

% options used by FUNC_bootstrap_wrapper
highlevel_options.boot_options.nboots = nboots; 
highlevel_options.boot_options.boot_conds = true;
highlevel_options.boot_options.boot_subjs = true; 

% options used by FUNC_reweighting_wrapper
% -- WARNING: if using pre-computed sampling order, these must match what
% was used to generate those:
highlevel_options.rw_options.nTestSubjects = 5;
highlevel_options.rw_options.nTestImages = 12;
highlevel_options.rw_options.nCVs = nCV; % number of crossvalidation loops within each bootstrap sample (stabilises estimate)


%% Cycle through, loading up sets of component RDMs from each layer 
% and adding them to a whole-network cell array `model_RDMs`

for layer = 1:length(layerlist)
    
    % load both PCA and RandCA component RDM sets for this layer
    load(strcat(pcaRDMdir,layerlist(layer).name), 'layerpcs');
    pcaRDMs = layerpcs; % rename
    load(strcat(randcaRDMdir,layerlist(layer).name), 'layerpcs');
    randcaRDMs = layerpcs; % rename
    % also load full raw feature space RDM for this layer
    load(strcat(rawRDMdir,layerlist(layer).name), 'rdm_raw');
    
    % create uniformly-weighted RDM from each type of components
    unifPCA = zeros([62,62]); % hardcoded assuming trans62 stimulus set
    unifRandCA = zeros([62,62]); % hardcoded assuming trans62 stimulus set
    for i = 1:length(pcaRDMs)
        unifPCA = unifPCA + (1/length(pcaRDMs)).*pcaRDMs(i).RDM;
        unifRandCA = unifRandCA + (1/length(randcaRDMs)).*randcaRDMs(i).RDM;
    end
    
    % put these into a nested struct containing everything for this layer
    layer_struct.pcaRDMs = pcaRDMs; % inherits .RDM and .name fields
    layer_struct.randcaRDMs = randcaRDMs; % inherits .RDM and .name fields
    layer_struct.rawRDM.RDM = squareform(rdm_raw); % takes square, not triu, RDMs
    layer_struct.rawRDM.name = 'raw feature space';
    layer_struct.unifPCA.RDM = unifPCA;
    layer_struct.unifPCA.name = 'uniformly-weighted PCA';
    layer_struct.unifRandCA.RDM = unifRandCA;
    layer_struct.unifRandCA.name = 'uniformly-weighted random CA';
    
    model_RDMs{layer} = layer_struct; % add to whole-model cell array
end

%% analyse
[layerwise_results, ceiling_results, wholenet_results] = FUNC_bootstrap_wrapper(refRDMs, model_RDMs, highlevel_options, sampling_order);

% save bootstrap distributions
save(strcat(savedir,'bootstrap_output_layerwise'), 'layerwise_results');
save(strcat(savedir,'bootstrap_output_wholenet'), 'wholenet_results');
save(strcat(savedir,'bootstrap_output_ceilings'), 'ceiling_results');
