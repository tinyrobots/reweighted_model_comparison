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

%% extract info from data

nConds = size(refRDMs,1);
nSubjs = size(refRDMs,3);
nModels = 1; % Currently only works for nModels == 1. 
             % This is the number of independent models e.g. networks 
             % NOT number of components to be reweighted e.g. layers or feature maps 

%% Cross-validation procedure

% reset temporary storage for crossval loop results for each model
loop_store_models = zeros([rw_options.nImageLoops,size(model_RDMs,3)+1]); 
loop_store_low_ceil =  zeros([rw_options.nImageLoops,1]);
loop_store_upp_ceil = zeros([rw_options.nImageLoops,1]);

% cycle through crossvalidation procedure
% nb. This consists of two nested crossvalidation loops, one splitting 
% stimuli on every "loop", and the other splitting subjects multiple
% times within each "loop".
% This outer xval loop has the purpose of stabilising the estimates
% obtained within each bootstrap sample
for loop = 1:rw_options.nImageLoops
    
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
    
    % restructure subject data to put NaNs in the diagonals, so that when
    % crossvalidation causes some to end up in the off-diag spots they can
    % be ignored (zeros would artificially inflate correlation)
    for s = 1:size(refRDMs,3)
        this_subj = refRDMs(:,:,s);
        this_subj(logical(eye(size(this_subj,1)))) = NaN;
        refRDMs(:,:,s) = this_subj;
    end

    % create temporary storage for the values from each subject xval fold
    temp_models = zeros([rw_options.nSubjectLoops,size(model_RDMs,3)+1]); % nb assumes ONE model, with multiple reweightable component RDMs - uses columns to store all components + reweighted performances
    temp_ceil_upp = zeros([rw_options.nSubjectLoops,1]);
    temp_ceil_low = zeros([rw_options.nSubjectLoops,1]);

    % inner loop of subject xval
    for subj_count = 1:rw_options.nSubjectLoops
%         fprintf('subj:%d...', subj_count)

        % do exhaustive LOO if nTestSubjects is 1 (suitable for small sample sizes)
        if rw_options.nTestSubjects == 1
            % use this subject as the test subject
            subj_locs_test = subj_count;
            subj_locs_train = setdiff(1:nSubjs,subj_locs_test); % use the others for training
        
        else % otherwise randomly choose a train/test split of subjects
            
            subj_set_ok = 0;
            while subj_set_ok == 0
                % try choosing a random sample of test condition IDs
                subj_ids_test = datasample(1:nSubjs, rw_options.nTestSubjects, 'Replace', false);
                subj_ids_train = setdiff(1:nSubjs,subj_ids_test); % use the others for training
                % check that a reasonable number of them is present in this bootstrap sample
                if (numel(intersect(subj_ids,subj_ids_test)) >= 3)
                    subj_set_ok = 1;
                else
                    'fewer than 3 test or train subjects selected; repicking'
                end 
            end
            % find locations of any of these present in the bootstrapped sample,
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
        end
        
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
        clear dataRDMs_test_ltv
        for s = 1:size(dataRDMs_test,3)
            this_subj = dataRDMs_test(:,:,s);
            this_subj(logical(eye(size(this_subj,1)))) = 0;
            dataRDMs_test_ltv(s,:) = rankTransform_equalsStayEqual(squareform(this_subj));
        end
        
        % also create an RDM of ALL subjects' data for test images,
        % for calculating the upper bound of the noise ceiling
        dataRDMs_test_all_subjs = refRDMs(c_sel_test,c_sel_test,:); % TODO: need to change this if bootstrapping Ss?
        dataRDMs_test_all_subjs = mean(dataRDMs_test_all_subjs,3); % we only ever need the mean
        % rank transform it
        dataRDMs_test_all_subjs(logical(eye(size(dataRDMs_test_all_subjs,1)))) = 0;
        dataRDMs_test_all_subjs_ltv = rankTransform_equalsStayEqual(squareform(dataRDMs_test_all_subjs));
        
        % ...plus an RDM of TRAIN subjects' data for TEST images,
        % for calculating the LOWER bound of the noise ceiling
        dataRDMs_test_train_subjs = refRDMs(c_sel_test,c_sel_test,s_sel_train); % TODO: need to change this if bootstrapping Ss?
        dataRDMs_test_train_subjs = mean(dataRDMs_test_train_subjs,3); % we only ever need the mean
        % rank transform it
        dataRDMs_test_train_subjs(logical(eye(size(dataRDMs_test_train_subjs,1)))) = 0;
        dataRDMs_test_train_subjs_ltv = rankTransform_equalsStayEqual(squareform(dataRDMs_test_train_subjs));

        % remove NaN columns from human test data because Spearman
        % correlation function doesn't handle NaNs, and we don't want to
        % inflate our correlation by setting the values to zeros
        % i.e. this is removing same-image-comparison entries in the RDM
        dataRDM_train_ltv = dataRDM_train_ltv(~isnan(dataRDM_train_ltv)); 
        if size(dataRDMs_test_ltv,1) > 1 % multiple test subjects - eliminate by columns
            dataRDMs_test_ltv = dataRDMs_test_ltv(:,all(~isnan(dataRDMs_test_ltv)));
        else % just one subject - eliminate individual entries
            dataRDMs_test_ltv = dataRDMs_test_ltv(~isnan(dataRDMs_test_ltv)); 
        end
        dataRDMs_test_all_subjs_ltv = dataRDMs_test_all_subjs_ltv(~isnan(dataRDMs_test_all_subjs_ltv));
        dataRDMs_test_train_subjs_ltv = dataRDMs_test_train_subjs_ltv(~isnan(dataRDMs_test_train_subjs_ltv));

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

            % dropping same-image-pair entries, as we have done for the
            % human data.
            modelRDMs_train_ltv = modelRDMs_train_ltv(all(~isnan(modelRDMs_train_ltv),2),:); % as before, but with rows
            modelRDMs_test_ltv = modelRDMs_test_ltv(all(~isnan(modelRDMs_test_ltv),2),:);

            % main call to fitting library - this could be replaced with
            % glmnet for ridge regression, etc.
            weights = lsqnonneg(double(modelRDMs_train_ltv), double(dataRDM_train_ltv(:)));

            % 4. calculate performance on the held out subjects and images ########

            % combine each layer in proportion to the estimated weights
            this_model_test_ltv_weighted = (modelRDMs_test_ltv*weights)';

            % add to our list of model predicted RDMs
            modelRDMs_test_ltv_weighted(model_num,:) = this_model_test_ltv_weighted;
        end

        % Now we have added the reweighted version to our list of models, 
        % evaluate each one, along with the noise ceiling
        % - need to do this individually against each of the test Ss
        clear tm tcl tcu % very temp storages
        for test_subj = 1:size(dataRDMs_test_ltv,1)
            % for each of the component models
            for comp_num = 1:size(model_RDMs,3)
                tm(test_subj,comp_num) = 1-pdist([modelRDMs_test_ltv(:, comp_num)'; dataRDMs_test_ltv(test_subj,:)],'spearman');
            end
            % ...and finally for the weighted combination of these components
            tm(test_subj,size(model_RDMs,3)+1) = 1-pdist([modelRDMs_test_ltv_weighted(1,:); dataRDMs_test_ltv(test_subj,:)],'spearman');
            
            % Model for lower bound = correlation between each subject and mean  
            % of test data from TRAINING subjects (this captures a "perfectly fitted" 
            % model, which has not been allowed to peek at any of the training Ss' data)
            tcl(test_subj) = 1-pdist([dataRDMs_test_train_subjs_ltv; dataRDMs_test_ltv(test_subj,:)],'spearman');
            
            % Model for upper noise ceiling = correlation between each subject 
            % and mean of ALL train and test subjects' data, including themselves (overfitted)
            tcu(test_subj) = 1-pdist([dataRDMs_test_all_subjs_ltv; dataRDMs_test_ltv(test_subj,:)],'spearman');
        end
        
        % average over the individual test subjects
        temp_ceil_low(subj_count) = mean(tcl);
        temp_ceil_upp(subj_count) = mean(tcu);
        temp_models(subj_count,:) = mean(tm,1);

    end % end of inner subject xval procedure
    
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
