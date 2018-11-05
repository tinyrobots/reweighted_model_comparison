# reweighted_model_comparison

_Use with caution - still in development - and let me know if/when you find bugs!_

_P.S. I have not yet made any attempts to optimise for running time. Can hopefully be made a fair bit faster!_

Start at `calling_script.m` in `/code`
- This loads data, sets options, and calls `FUNC_compareRefRDM2candRDMs_reweighting.m`...
- ...which is a version of the standard `comparedRefRDM2candRDMs` function from the RSA Toolbox, modified to include a `userOptions.reweighting` option. If this is chosen, the usual analyses will be bypassed and the bootstrapped crossvalidated reweighting procedure will be done instead. The usual plotting functionality is used, with minor modifications. For the reweighting analysis, this function passes the data first to `FUNC_bootstrap_wrapper.m`...
- ...which is a small wrapper function which just does the sampling for each bootstrap. Each bootstrap sample is then passed to `FUNC_reweighting_wrapper.m`...
- ...which performs the bulk of the analysis, doing a nested cross-validation procedure (looping once over stimuli, and once over subjects).

_The `calling_script.m` contains two examples which draw from data and models provided in the two `/demo` folders -- these should run out-of-the-box as tests._
