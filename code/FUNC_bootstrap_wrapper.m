function [model_corrs, low_ceiling_corrs, upp_ceiling_corrs] = FUNC_bootstrap_wrapper(refRDMs, model_RDMs, save_dir, boot_options, rw_options)

% Bootstrap resamples conditions and/or subjects and passes the resampled
% data to another function to perform crossvalidated reweighting of the
% component model RDMs.

if rw_options.nTestSubjects == 1
    fprintf('Procedure will use exhaustive Leave-One-Subject-Out method in inner reweighting crossvalidation loop \n')
end

fprintf('Bootstrap sample number: \n')

parfor boot = 1:boot_options.nboots
    fprintf(' %d ... ',boot)
    
    if boot_options.boot_conds == true
%         cond_ids = datasample(1:size(refRDMs,1),size(refRDMs,1),'Replace',true);
        cond_ids = sampling_order(boot).cond_ids;
    else
        cond_ids = 1:size(refRDMs,1);
    end
    
    if boot_options.boot_subjs == true
%         subj_ids = datasample(1:size(refRDMs,3),size(refRDMs,3),'Replace',true);
        subj_ids = sampling_order(boot).subj_ids;
    else
        subj_ids = 1:size(refRDMs,3);
    end
    
    [mc, lcc, ucc] = FUNC_reweighting_wrapper(refRDMs, model_RDMs, rw_options, cond_ids, subj_ids);
    
    model_corrs(boot,:) = mc;
    low_ceiling_corrs(boot) = lcc;
    upp_ceiling_corrs(boot) = ucc;
end

%% save stuff

try mkdir(save_dir); end
save(strcat(save_dir,'bootstrap_output.mat'),'model_corrs', 'low_ceiling_corrs', 'upp_ceiling_corrs', 'model_RDMs');

end
