function [layerwise_oneboot, ceiling_oneboot, wholenet_oneboot] = FUNC_reweighting_wrapper(refRDMs, model_RDMs, rw_options, cond_ids, subj_ids)

% Performs a single bootstrap sample in which multiple crossvalidation
% folds are performed. On each crossval fold, data are split into training 
% and test portions, model components are linearly combined to best predict
% training portion, and then this same weighted combination is tested on 
% the test portion. 

% Intended to be embedded within a bootstrap loop, which supplies the 
% indices of the selected subjects and conditions on this bootstrap 
% (i.e. expecting to be called from `FUNC_bootstrap_wrapper.m`)

% nb. naming and comments assume that the multiple model components are
% different layers within a neural network - but they could be different
% feature maps within a layer, or anything else.

%% extract info from data

nConds = size(refRDMs,1);
nSubjs = size(refRDMs,3);

%% Cross-validation procedure

% create temporary storages for per-crossval fold results for each estimate
loop_store_layerwise.raw = zeros([rw_options.nCVs,length(model_RDMs)]); % one column per layer x nCV rows

loop_store_wholenet.raw_fitted = zeros([rw_options.nCVs,1]); % single column with nCV rows
loop_store_wholenet.raw_unif = zeros([rw_options.nCVs,1]); % single column with nCV rows

loop_store_ceilings.lower =  zeros([rw_options.nCVs,1]);
loop_store_ceilings.upper =  zeros([rw_options.nCVs,1]);

% cycle through crossvalidation procedure, which splits data into both separate
% stimuli and subject groups. This xval loop has the purpose of stabilising 
% the estimates obtained within each bootstrap sample
for loop = 1:rw_options.nCVs
    
    %% 1. Preparation: split human data into training and test partitions #####################
    
    % STIMULI: We select exactly nTestImages that *are present* in this
    % bootstrap sample (and then sample multiply according to how many
    % times they are present in the sample)
    cond_ids_test = datasample(unique(cond_ids), rw_options.nTestImages, 'Replace', false);
    cond_ids_train = setdiff(unique(cond_ids),cond_ids_test); % use the others for training
   
    % find locations of where these are present in the bootstrapped sample,
    % and append to two lists of cond_id entries we're going to use for
    % training and testing. Note that these change size from boot to boot:
    cond_locs_test = [];
    for i = 1:length(cond_ids_test)
        cond_locs_test = [cond_locs_test, find(cond_ids==cond_ids_test(i))];
    end
    cond_locs_train = [];
    for i = 1:length(cond_ids_train)
        cond_locs_train = [cond_locs_train, find(cond_ids==cond_ids_train(i))];
    end

    % SUBJECTS: apply same logic here, only selecting *available* subjects
    subj_ids_test = datasample(unique(subj_ids), rw_options.nTestSubjects, 'Replace', false);
    subj_ids_train = setdiff(unique(subj_ids),subj_ids_test); % use the others for training

    % find locations of any of these present in the (possibly) bootstrapped sample,
    % and append to two lists of subj_id entries we're going to use for
    % training and testing:
    subj_locs_test = [];
    for i = 1:length(subj_ids_test)
        subj_locs_test = [subj_locs_test, find(subj_ids==subj_ids_test(i))];
    end
    subj_locs_train = [];
    for i = 1:length(subj_ids_train)
        subj_locs_train = [subj_locs_train, find(subj_ids==subj_ids_train(i))];
    end
    
    % training data
    c_sel_train = cond_ids(cond_locs_train);
    s_sel_train = subj_ids(subj_locs_train);
    dataRDM_train = refRDMs(c_sel_train,c_sel_train,s_sel_train);
    dataRDM_train = mean(dataRDM_train,3);
    % need to replace diagonals with zeros so we can use squareform
    dataRDM_train(logical(eye(size(dataRDM_train,1)))) = 0;
    dataRDM_train_ltv = squareform(dataRDM_train);

    % test data
    c_sel_test = cond_ids(cond_locs_test);
    s_sel_test = subj_ids(subj_locs_test);
    dataRDMs_test = refRDMs(c_sel_test,c_sel_test,s_sel_test);
    clear dataRDMs_test_ltv % to avoid problems with size changes from boot to boot
    for s = 1:size(dataRDMs_test,3)
        this_subj = dataRDMs_test(:,:,s);
        this_subj(logical(eye(size(this_subj,1)))) = 0;
        dataRDMs_test_ltv(s,:) = squareform(this_subj);
    end

    % also create an RDM of ALL subjects' data for test images,
    % for calculating the UPPER bound of the noise ceiling
    dataRDMs_test_all_subjs = refRDMs(c_sel_test,c_sel_test,subj_ids); % nb. if bootstrapping Ss, this can contain duplicates
    % rank-transform each subject's RDM before we average them 
    % (assuming Spearman correlation is used to assess model performance; 
    % see RSA Toolbox (Nili et al., 2014). If using a different correlation
    % metric see RSA Toolbox for advice on how to transform data before
    % averaging to create the "best fit" model for the upper noise ceiling.
    for s = 1:size(dataRDMs_test_all_subjs,3)
        dataRDMs_test_all_subjs(:,:,s) = tiedrank(dataRDMs_test_all_subjs(:,:,s));
    end
    % Take the average, since we only use this RDM for noise ceiling
    % calculations:
    dataRDMs_test_all_subjs = mean(dataRDMs_test_all_subjs,3); 
    dataRDMs_test_all_subjs(logical(eye(size(dataRDMs_test_all_subjs,1)))) = 0;
    dataRDMs_test_all_subjs_ltv = squareform(dataRDMs_test_all_subjs);

    % ...plus an RDM of TRAINING subjects' data for TEST images,
    % for calculating the LOWER bound of the noise ceiling
    dataRDMs_test_train_subjs = refRDMs(c_sel_test,c_sel_test,s_sel_train); % nb. if bootstrapping Ss, this can contain duplicates - but cannot overlap w training data
    dataRDMs_test_train_subjs = mean(dataRDMs_test_train_subjs,3); % we only ever need the mean
    dataRDMs_test_train_subjs(logical(eye(size(dataRDMs_test_train_subjs,1)))) = 0;
    dataRDMs_test_train_subjs_ltv = squareform(dataRDMs_test_train_subjs);

    % remove NaN columns from human test data because lsqnonneg function
    % doesn't handle NaNs, and we don't want to inflate our correlation 
    % by setting the values to zeros. Provided we do the same to human and
    % model data we can simply remove  those pairwise comparisons -
    % i.e. this is removing same-image-comparison entries in the RDM
    dataRDM_train_ltv = dataRDM_train_ltv(~isnan(dataRDM_train_ltv)); 
    if size(dataRDMs_test_ltv,1) > 1 % multiple test subjects - eliminate by columns
        dataRDMs_test_ltv = dataRDMs_test_ltv(:,all(~isnan(dataRDMs_test_ltv)));
    else % just one subject - eliminate individual entries
        dataRDMs_test_ltv = dataRDMs_test_ltv(~isnan(dataRDMs_test_ltv)); 
    end
    dataRDMs_test_all_subjs_ltv = dataRDMs_test_all_subjs_ltv(~isnan(dataRDMs_test_all_subjs_ltv));
    dataRDMs_test_train_subjs_ltv = dataRDMs_test_train_subjs_ltv(~isnan(dataRDMs_test_train_subjs_ltv));

    %% 2. calculate performace of each model component individually #####################
        
    for layer = 1:length(model_RDMs)
        
        rawRDM_test = model_RDMs{layer}.rawRDM.RDM;
        rawRDM_test(logical(eye(size(rawRDM_test,1)))) = NaN; % put NaNs in diagonal
        rawRDM_test = rawRDM_test(c_sel_test,c_sel_test); % resample rows and columns
        rawRDM_test(logical(eye(size(rawRDM_test,1)))) = 0; % put zeros in diagonal
        rawRDM_test_ltv = squareform(rawRDM_test);
        rawRDM_test_ltv = rawRDM_test_ltv(~isnan(rawRDM_test_ltv)); % drop NaNs
        
        clear tm % very temp storage
        for test_subj = 1:size(dataRDMs_test_ltv,1)
            tm(test_subj) = corr(rawRDM_test_ltv', dataRDMs_test_ltv(test_subj,:)', 'Type', 'Spearman');
        end
        % get means of the correlations over held-out subjects
        loop_store_layerwise.raw(loop, layer) = mean(tm); 

    end
    
    %% 3. perform reweighting to estimate performance of all components combined #####################
    
    % A. First construct a "baseline" combined model, by combining all
    % components using equal uniform weights:
    raw_unif = zeros([62,62]); % hardcoded assuming trans62 stimulus set
    for layer = 1:length(model_RDMs)
        raw_unif = raw_unif + (1/length(model_RDMs)).*model_RDMs{layer}.rawRDM.RDM;
    end
    raw_unif_test = raw_unif;
    raw_unif_test(logical(eye(size(raw_unif_test,1)))) = NaN; % put NaNs in diagonal
    raw_unif_test = raw_unif_test(c_sel_test,c_sel_test); % resample rows and columns
    raw_unif_test(logical(eye(size(raw_unif_test,1)))) = 0; % put zeros in diagonal
    raw_unif_test_ltv = squareform(raw_unif_test);
    raw_unif_test_ltv = raw_unif_test_ltv(~isnan(raw_unif_test_ltv)); % drop NaNs
    clear tm % very temp storage
    % calculate performance on the test subjects and images
    for test_subj = 1:size(dataRDMs_test_ltv,1)
        tm(test_subj) = corr(raw_unif_test_ltv', dataRDMs_test_ltv(test_subj,:)', 'Type', 'Spearman');
    end
    % get means of the correlations over held-out subjects
    loop_store_wholenet.raw_unif(loop) = mean(tm);
    
    % B. Reweighting. Gather the component RDMs, fit weights using the train
    % portion, and use them to calculate a reweighted predicted RDM 
    % that we will evaluate on the test portion
    clear modelRDMs_train_ltv modelRDMs_test_ltv % reuse generic names
    for layer = 1:length(model_RDMs)
        cModelRDM_train = model_RDMs{layer}.rawRDM.RDM;
        cModelRDM_train(logical(eye(size(cModelRDM_train,1)))) = NaN; % put NaNs in diagonal
        cModelRDM_train = cModelRDM_train(c_sel_train,c_sel_train); % resample rows and columns
        cModelRDM_train(logical(eye(size(cModelRDM_train,1)))) = 0; % put zeros in diagonal
        modelRDMs_train_ltv(:,layer) = squareform(cModelRDM_train);

        cModelRDM_test = model_RDMs{layer}.rawRDM.RDM;
        cModelRDM_test(logical(eye(size(cModelRDM_test,1)))) = NaN; % put NaNs in diagonal
        cModelRDM_test = cModelRDM_test(c_sel_test,c_sel_test); % resample rows and columns
        cModelRDM_test(logical(eye(size(cModelRDM_test,1)))) = 0; % put zeros in diagonal
        modelRDMs_test_ltv(:,layer) = squareform(cModelRDM_test);
    end
    % do regression to estimate layer weights
    % dropping same-image-pair entries, as we have done for the human data
    modelRDMs_train_ltv = modelRDMs_train_ltv(all(~isnan(modelRDMs_train_ltv),2),:); % as before, but with rows
    modelRDMs_test_ltv = modelRDMs_test_ltv(all(~isnan(modelRDMs_test_ltv),2),:);
    % main call to fitting library - this could be replaced with
    % glmnet to allow for ridge regression, etc., but GLMnet is not compiled
    % to work in Matlab post ~2014ish.
    weights = lsqnonneg(double(modelRDMs_train_ltv), double(dataRDM_train_ltv'));
    % combine each layer in proportion to the estimated weights
    this_model_test_ltv_weighted = modelRDMs_test_ltv*weights;
    % calculate performance on the held out subjects and images
    clear tm % very temp storage
    for test_subj = 1:size(dataRDMs_test_ltv,1)
        tm(test_subj) = corr(this_model_test_ltv_weighted, dataRDMs_test_ltv(test_subj,:)', 'Type', 'Spearman');
    end
    % get means of the correlations over held-out subjects
    loop_store_wholenet.raw_fitted(loop) = mean(tm);
    
    %% 4. Estimate noise ceilings #####################
    clear tcl tcu % very temp storages
    for test_subj = 1:size(dataRDMs_test_ltv,1)
        % Model for lower bound = correlation between each subject and mean  
        % of test data from TRAINING subjects (this captures a "perfectly fitted" 
        % model, which has not been allowed to peek at any of the training Ss' data)
        tcl(test_subj) = corr(dataRDMs_test_train_subjs_ltv', dataRDMs_test_ltv(test_subj,:)', 'Type', 'Spearman');
        % Model for upper noise ceiling = correlation between each subject 
        % and mean of ALL train and test subjects' data, including themselves (overfitted)
        tcu(test_subj) = corr(dataRDMs_test_all_subjs_ltv', dataRDMs_test_ltv(test_subj,:)', 'Type', 'Spearman');
    end
    loop_store_ceilings.lower(loop) = mean(tcl);
    loop_store_ceilings.upper(loop) = mean(tcu);

end % end of crossvalidation loop

% average over crossvalidation loops at the end of this bootstrap sample
layerwise_oneboot.raw = mean(loop_store_layerwise.raw);

wholenet_oneboot.raw_fitted = mean(loop_store_wholenet.raw_fitted);
wholenet_oneboot.raw_unif = mean(loop_store_wholenet.raw_unif);

ceiling_oneboot.lower = mean(loop_store_ceilings.lower); % will be split into multiple substructures
ceiling_oneboot.upper = mean(loop_store_ceilings.upper); % will be split into multiple substructures

end
