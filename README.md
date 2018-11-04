# reweighted_model_comparison

_WARNING: These scripts are currently adapted for the special case of reanalysing the 4-subject 92-image dataset. I will extend them asap to the more general case where there is a large enough number of both subjects and conditions to bootstrap and cross-validate over both._

Start at `calling_script.m`
- This loads data, sets options, and calls `FUNC_compareRefRDM2candRDMs_reweighting.m`...
- ...which is a version of the standard `comparedRefRDM2candRDMs` function from the RSA Toolbox, modified to include a `userOptions.reweighting` option. If this is chosen, the usual analyses will be bypassed and the bootstrapped crossvalidated reweighting procedure will be done instead. The usual plotting functionality is used, with minor modifications. For the reweighting analysis, this function passes the data first to `FUNC_bootstrap_wrapper.m`...
- ...which is a small wrapper function which just does the sampling for each bootstrap. Each bootstrap sample is then passed to `FUNC_reweighting_wrapper.m`...
- ...which performs the bulk of the analysis, doing a nested cross-validation procedure (looping once over stimuli, and once over subjects).

_This repository contains example brain and AlexNet RDMs for the 4-subject 92-image dataset, so should run out-of-the-box as a demo._
