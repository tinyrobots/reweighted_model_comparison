function [layerwise_results, ceiling_results, wholenet_results] = FUNC_bootstrap_wrapper(refRDMs, model_RDMs, highlevel_options, sampling_order)
% Bootstrap resamples conditions and/or subjects and passes the resampled
% data to another function to perform crossvalidated reweighting of the
% component model RDMs.

% create temporary storages accessible within parfoor loop

tmp_layerwise_fittedPCA = zeros([highlevel_options.boot_options.nboots,length(model_RDMs)]);
tmp_layerwise_fittedRandCA = zeros([highlevel_options.boot_options.nboots,length(model_RDMs)]);
tmp_layerwise_unifPCA = zeros([highlevel_options.boot_options.nboots,length(model_RDMs)]);
tmp_layerwise_unifRandCA = zeros([highlevel_options.boot_options.nboots,length(model_RDMs)]);
tmp_layerwise_raw = zeros([highlevel_options.boot_options.nboots,length(model_RDMs)]);

tmp_wholenet_fittedPCA_fitted = zeros([highlevel_options.boot_options.nboots,1]);
tmp_wholenet_fittedRandCA_fitted = zeros([highlevel_options.boot_options.nboots,1]);
tmp_wholenet_unifPCA_fitted = zeros([highlevel_options.boot_options.nboots,1]);
tmp_wholenet_unifRandCA_fitted = zeros([highlevel_options.boot_options.nboots,1]);
tmp_wholenet_raw_fitted = zeros([highlevel_options.boot_options.nboots,1]);
tmp_wholenet_raw_unif = zeros([highlevel_options.boot_options.nboots,1]);

tmp_ceiling_lower = zeros([highlevel_options.boot_options.nboots,1]);
tmp_ceiling_upper = zeros([highlevel_options.boot_options.nboots,1]);

parfor boot = 1:highlevel_options.boot_options.nboots
    fprintf(' %d ... ',boot)
    
    if highlevel_options.boot_options.boot_conds == true
        cond_ids = datasample(1:size(refRDMs,1),size(refRDMs,1),'Replace',true);
    else
        cond_ids = 1:size(refRDMs,1);
    end
    
    if highlevel_options.boot_options.boot_subjs == true
        subj_ids = datasample(1:size(refRDMs,3),size(refRDMs,3),'Replace',true);
    else
        subj_ids = 1:size(refRDMs,3);
    end
    
    % Calculates results for a single bootstrap sample:
    [layerwise_oneboot, ceiling_oneboot, wholenet_oneboot] = FUNC_reweighting_wrapper(refRDMs, model_RDMs, highlevel_options.rw_options, cond_ids, subj_ids); % only needs sampling order info for this boot
    
    tmp_layerwise_fittedPCA(boot,:) = layerwise_oneboot.fittedPCA;
    tmp_layerwise_fittedRandCA(boot,:) = layerwise_oneboot.fittedRandCA;
    tmp_layerwise_unifPCA(boot,:) = layerwise_oneboot.unifPCA;
    tmp_layerwise_unifRandCA(boot,:) = layerwise_oneboot.unifRandCA;
    tmp_layerwise_raw(boot,:) = layerwise_oneboot.raw;

    tmp_wholenet_fittedPCA_fitted(boot) = wholenet_oneboot.fittedPCA_fitted;
    tmp_wholenet_fittedRandCA_fitted(boot) = wholenet_oneboot.fittedRandCA_fitted;
    tmp_wholenet_unifPCA_fitted(boot) = wholenet_oneboot.unifPCA_fitted;
    tmp_wholenet_unifRandCA_fitted(boot) = wholenet_oneboot.unifRandCA_fitted;
    tmp_wholenet_raw_fitted(boot) = wholenet_oneboot.raw_fitted;
    tmp_wholenet_raw_unif(boot) = wholenet_oneboot.raw_unif;

    tmp_ceiling_lower(boot) = ceiling_oneboot.lower;
    tmp_ceiling_upper(boot) = ceiling_oneboot.upper;

end

% assign temporary values to structures
layerwise_results.fittedPCA = tmp_layerwise_fittedPCA;
layerwise_results.fittedRandCA = tmp_layerwise_fittedRandCA;
layerwise_results.unifPCA = tmp_layerwise_unifPCA;
layerwise_results.unifRandCA = tmp_layerwise_unifRandCA;
layerwise_results.raw = tmp_layerwise_raw;

wholenet_results.fittedPCA_fitted = tmp_wholenet_fittedPCA_fitted;
wholenet_results.fittedRandCA_fitted = tmp_wholenet_fittedRandCA_fitted;
wholenet_results.unifPCA_fitted = tmp_wholenet_unifPCA_fitted;
wholenet_results.unifRandCA_fitted = tmp_wholenet_unifRandCA_fitted;
wholenet_results.raw_fitted = tmp_wholenet_raw_fitted;
wholenet_results.raw_unif = tmp_wholenet_raw_unif;

ceiling_results.lower = tmp_ceiling_lower;
ceiling_results.upper = tmp_ceiling_upper;
end