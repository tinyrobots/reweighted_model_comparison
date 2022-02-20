# reweighted_model_comparison

_**Demos:** The `calling_script.m` contains two examples which draw from data and models provided in the two `/demo` folders -- these should run out-of-the-box as tests._

Start at `calling_script.m` in `/code`
- This loads data, sets options, and calls `FUNC_bootstrap_wrapper.m`...
- ...which is a small wrapper function which just does the sampling for each bootstrap. Each bootstrap sample is then passed to `FUNC_reweighting_wrapper.m`...
- ...which performs the bulk of the analysis, using a cross-validation procedure (over stimuli and/or subjects, as chosen).
- Once analysis has run and results saved, can use `plot_and_analyse.m` to visualise raw and reweighted model performance and conduct various statistical tests.

Timing note - uses parallel for-loop in `FUNC_bootstrap_wrapper.m` to speed things up:
- Takes approx XX mins to compute 1,000 bootstrap samples of 20 cross-validation folds, on a dataset with 24 subjects and 62 stimulus conditions.
- If parallel computing toolbox not available or desired, simply change `parfor` to `for` at line 24

Caveat:
_Use with caution and let me know if you find bugs!_


