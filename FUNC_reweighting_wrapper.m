function [model_corr_list, low_ceiling_corr, upp_ceiling_corr] = FUNC_reweighting_wrapper(refRDMs, model_RDMs, rw_options, cond_ids, subj_ids)

% Does a single crossvalidation fold in which data RDMs are split into
% training and test portions, models are fitted to training portion, and
% tested on test portion. Intended to be embedded within a bootstrap loop,
% which supplies the indices of the selected subjects and conditions on
% this bootstrap.

% nb. naming and comments assume that the multiple model components are
% different layers within a neural network - but they could be different
% feature maps within a layer, or anything else.

import rsa.util.*

if rw_options.nTestSubjects ~= 1
    fprintf('nTestSubjects must be 1 in order to use exhaustive LOO method for subjects, as currently implemented in this function')
    return
end

%% extract info from data

nConds = size(refRDMs,1);
nSubjs = size(refRDMs,3);
nModels = 1; % Currently only works for nModels == 1. 
             % This is the number of independent models e.g. networks 
             % NOT number of components to be reweighted e.g. layers or feature maps 

%% Cross-validation procedure

% reset temporary storage for crossval loop results for each model
loop_store_models = zeros([rw_options.nLoops,size(model_RDMs,3)+1]); 
loop_store_low_ceil =  zeros([rw_options.nLoops,1]);
loop_store_upp_ceil = zeros([rw_options.nLoops,1]);

% cycle through crossvalidation procedure
for loop = 1:rw_options.nLoops
    
    % 1. split data into training and test partitions #####################
    
    % select images to leave out for testing
    cond_set_ok = 0;
    while cond_set_ok == 0
        % try choosing a random sample of test condition IDs
        cond_ids_test = datasample(1:nConds, rw_options.nTestImages, 'Replace', false);
        cond_ids_train = setdiff(1:nConds,cond_ids_test); % use the others for training
        % check that a reasonable number of them is present in this bootstrap sample
        if (numel(intersect(cond_ids,cond_ids_test)) >= 3)
            cond_set_ok = 1;
        else
            'fewer than 3 test or train conditions selected; repicking'
        end 
    end
    
    % find locations of any of these present in the bootstrapped sample,
    % and append to two lists of cond_id entries we're going to use for
    % training and testing:
    cond_locs_test = [];
    for i = 1:length(cond_ids_test)
        cond_locs_test = [cond_locs_test, find(cond_ids==cond_ids_test(i))];
    end
    cond_locs_train = [];
    for i = 1:length(cond_ids_train)
        cond_locs_train = [cond_locs_train, find(cond_ids==cond_ids_train(i))];
    end
    
    % restructure subject data with NaNs in the diagonals so that they end up in the off-diag spots
    for s = 1:size(refRDMs,3)
        this_subj = refRDMs(:,:,s);
        this_subj(logical(eye(size(this_subj,1)))) = NaN;
        refRDMs(:,:,s) = this_subj;
    end

    % create temporary storage for the values from each LOO fold
    temp_models = zeros([nSubjs,size(model_RDMs,3)+1]); % nb assumes ONE model, with multiple reweightable component RDMs - uses columns to store all components + reweighted performances
    temp_ceil_upp = zeros([nSubjs,1]);
    temp_ceil_low = zeros([nSubjs,1]);

    % systematically step through, leaving out each one of the subjects in turn
    for subj_count = 1:nSubjs
%         fprintf('subj:%d...', subj_count)
        % use this subject as the test subject
        subj_locs_test = subj_count;
        subj_locs_train = setdiff(1:nSubjs,subj_locs_test); % use the others for training

        % 2. construct training and test RDMs #################################

        % training data
        c_sel_train = cond_ids(cond_locs_train);
        s_sel_train = subj_ids(subj_locs_train);
        dataRDM_train = refRDMs(c_sel_train,c_sel_train,s_sel_train);
        dataRDM_train = mean(dataRDM_train,3);
        % rank transform it - doesn't affect (Spearman) correlations with
        % models, but does ensure our noise ceiling calculations are correct
        % (see Appendix of Nili et al. 2015)
        % need to replace diagonals with zeros so we can use squareform
        dataRDM_train(logical(eye(size(dataRDM_train,1)))) = 0;
        dataRDM_train_ltv = rankTransform_equalsStayEqual(squareform(dataRDM_train)); % take mean, and put in ltv format for glmnet

        % test data
        c_sel_test = cond_ids(cond_locs_test);
        s_sel_test = subj_ids(subj_locs_test);
        dataRDMs_test = refRDMs(c_sel_test,c_sel_test,s_sel_test);
        % rank transform it - doesn't affect (Spearman) correlations with
        % models, but does ensure our noise ceiling calculations are correct
        % (see Appendix of Nili et al. 2015)
        % first need to replace diagonals with zeros so we can use squareform
        dataRDMs_test(logical(eye(size(dataRDMs_test,1)))) = 0;
        dataRDMs_test_ltv = rankTransform_equalsStayEqual(squareform(dataRDMs_test));
        
        % also create an RDM of ALL subjects' data for test images, for
        % calculating the upper bound of the noise ceiling
        dataRDMs_test_all_subjs = refRDMs(c_sel_test,c_sel_test,:);
        clear dataRDMs_test_all_subjs_ltv
        for s = 1:nSubjs
            this_subj = dataRDMs_test_all_subjs(:,:,s);
            this_subj(logical(eye(size(this_subj,1)))) = 0;
            dataRDMs_test_all_subjs_ltv(s,:) = rankTransform_equalsStayEqual(squareform(this_subj));
        end

        % remove NaN columns from human test data because Spearman
        % correlation function doesn't handle NaNs, and we don't want to
        % inflate our correlation by setting the values to zeros
        dataRDM_train_ltv(isnan(dataRDM_train_ltv)) = []; 
        dataRDMs_test_ltv(isnan(dataRDMs_test_ltv)) = []; 
        dataRDMs_test_all_subjs_ltv(:,isnan(mean(dataRDMs_test_all_subjs_ltv,1))) = []; 

        % for each model, gather its component layers, fit weights, and store a
        % reweighted predicted RDM for the test conditions for that model
        clear modelRDMs_test_ltv_weighted
        for model_num = 1:nModels
            clear modelRDMs_train_ltv modelRDMs_test_ltv
            for layer = 1:size(model_RDMs,3)
                cModelRDM_train = model_RDMs(:,:,layer);
                cModelRDM_train(logical(eye(size(cModelRDM_train,1)))) = NaN; % put NaNs in diagonal
                cModelRDM_train = cModelRDM_train(c_sel_train,c_sel_train); % resample rows and columns
                cModelRDM_train(logical(eye(size(cModelRDM_train,1)))) = 0; % put zeros in diagonal
                modelRDMs_train_ltv(:,layer) = rankTransform_equalsStayEqual(squareform(cModelRDM_train));

                cModelRDM_test = model_RDMs(:,:,layer);
                cModelRDM_test(logical(eye(size(cModelRDM_test,1)))) = NaN; % put NaNs in diagonal
                cModelRDM_test = cModelRDM_test(c_sel_test,c_sel_test); % resample rows and columns
                cModelRDM_test(logical(eye(size(cModelRDM_test,1)))) = 0; % put zeros in diagonal
                modelRDMs_test_ltv(:,layer) = rankTransform_equalsStayEqual(squareform(cModelRDM_test));
            end

            % 3. do regression to estimate layer weights ########################## 

            % dropping same-image-pair entries - check whether this is
            % necessary, was originally to work around glmnet's data format
            % requirements
            modelRDMs_train_ltv(isnan(mean(modelRDMs_train_ltv,2)),:) = []; 
            modelRDMs_test_ltv(isnan(mean(modelRDMs_test_ltv,2)),:) = []; 

            % main call to fitting library
            weights = lsqnonneg(double(modelRDMs_train_ltv), double(dataRDM_train_ltv(:)));

            % 4. calculate performance on the held out subjects and images ########

            % combine each layer in proportion to the estimated weights
            this_model_test_ltv_weighted = (modelRDMs_test_ltv*weights)';

            % add to our list of model predicted RDMs
            modelRDMs_test_ltv_weighted(model_num,:) = this_model_test_ltv_weighted;
        end

        % now we have all our models, evaluate each one, along with the noise ceiling
        
        % for each of the model components
        for comp_num = 1:size(model_RDMs,3)
            temp_models(subj_count,comp_num) = 1-pdist([modelRDMs_test_ltv(:, comp_num)'; dataRDMs_test_ltv(1,:)],'spearman');
        end
        % ...and finally for the weighted combination of these components
        temp_models(subj_count,size(model_RDMs,3)+1) = 1-pdist([modelRDMs_test_ltv_weighted(1,:); dataRDMs_test_ltv(1,:)],'spearman');

        % Upper noise ceiling = correlation between each subject and mean of all
        % test subjects' data, including themselves (overfitted)
        avg_dataRDMs_test = mean(dataRDMs_test_all_subjs_ltv,1);
        temp_ceil_upp(subj_count) = 1-pdist([avg_dataRDMs_test; dataRDMs_test_ltv(1,:)],'spearman');

        % Lower bound = correlation between each subject and mean of other 
        % data from other test subjects
        rem_dataRDMs_test = dataRDMs_test_all_subjs_ltv;
        rem_dataRDMs_test(subj_count,:) = [];
        rem_dataRDMs_test = mean(rem_dataRDMs_test,1);
        temp_ceil_low(subj_count) = 1-pdist([rem_dataRDMs_test; dataRDMs_test_ltv(1,:)],'spearman');

    end
    
    % get means of the correlations within this cv fold, and store in loop results
    loop_store_models(loop, :) = mean(temp_models,1); 
    loop_store_low_ceil(loop) = mean(temp_ceil_low);
    loop_store_upp_ceil(loop) = mean(temp_ceil_upp);
end

% average over crossval loops at the end of this bootstrap sample
model_corr_list = mean(loop_store_models, 1);
low_ceiling_corr = mean(loop_store_low_ceil);
upp_ceiling_corr = mean(loop_store_upp_ceil);
end
