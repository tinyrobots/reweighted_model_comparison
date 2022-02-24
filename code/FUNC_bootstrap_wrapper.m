function [component_results, ceiling_results, combined_results] = FUNC_bootstrap_wrapper(refRDMs, model_RDMs, highlevel_options)
% Bootstrap resamples conditions and/or subjects and passes the resampled
% data to another function to perform crossvalidated reweighting of the
% component model RDMs.

% nb. naming and comments assume that the multiple model components are
% different layers within a neural network - but they could be different
% feature maps within a layer, or anything else.

% create temporary storages accessible within parfoor loop
tmp_layerwise_raw = zeros([highlevel_options.boot_options.nboots,length(model_RDMs)]);

tmp_wholenet_raw_fitted = zeros([highlevel_options.boot_options.nboots,1]);
tmp_wholenet_raw_unif = zeros([highlevel_options.boot_options.nboots,1]);

tmp_ceiling_lower = zeros([highlevel_options.boot_options.nboots,1]);
tmp_ceiling_upper = zeros([highlevel_options.boot_options.nboots,1]);

parfor boot = 1:highlevel_options.boot_options.nboots % can change to `for` during debugging etc
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
    
    tmp_layerwise_raw(boot,:) = layerwise_oneboot.raw;
    
    tmp_wholenet_raw_fitted(boot) = wholenet_oneboot.raw_fitted;
    tmp_wholenet_raw_unif(boot) = wholenet_oneboot.raw_unif;

    tmp_ceiling_lower(boot) = ceiling_oneboot.lower;
    tmp_ceiling_upper(boot) = ceiling_oneboot.upper;

end

% assign temporary values to structures
component_results.raw = tmp_layerwise_raw;

combined_results.raw_fitted = tmp_wholenet_raw_fitted;
combined_results.raw_unif = tmp_wholenet_raw_unif;

ceiling_results.lower = tmp_ceiling_lower;
ceiling_results.upper = tmp_ceiling_upper;
end