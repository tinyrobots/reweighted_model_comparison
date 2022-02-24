%% Example barchart and significance tests using output from cross-validated
% reweighting procedure. Assumes a results folder containing:
% -- bootstrap_output_ceilings.mat
% -- bootstrap_output_combined.mat
% -- bootstrap_output_components.mat
% These are saved at the end of `calling_script.m` 

clear all

resultsdir = strcat('../results/');
    
load(strcat(resultsdir,'bootstrap_output_ceilings.mat'),'ceiling_results');
load(strcat(resultsdir,'bootstrap_output_combined.mat'),'combined_results');
load(strcat(resultsdir,'bootstrap_output_components.mat'),'component_results');

% average bootstrap estimates of lower and upper noise ceiling 
lowceil = mean(ceiling_results.lower);
uppceil = mean(ceiling_results.upper);

% can specify names for each component here to label plot
component_names = {'Layer 1','Layer 2','Layer 3','Layer 4','Layer 5','Layer 6','Layer 7'};

%% PLOT 1: performance of each component separately

figure(1)

% noise ceiling first in drawing order
patch([0 0 size(component_results.raw,2)+1 size(component_results.raw,2)+1],[lowceil, uppceil, uppceil, lowceil],[0.5, 0.5, 0.5],'edgecolor','none','FaceAlpha',0.5)
hold on 

% barchart of mean and standard deviation across bootstraps for each model component
bar(mean(component_results.raw,1),'FaceColor','flat', 'BarWidth',1, 'EdgeColor', 'w')
errorbar([1:size(component_results.raw,2)], mean(component_results.raw,1), std(component_results.raw,1), 'color','k','LineWidth',2,'LineStyle','none','CapSize',0);

% aesthetics
box off
xlim([0.5,size(component_results.raw,2)+0.5])
ylim([0, uppceil+0.1])
xticks(1:length(component_names))
xticklabels(component_names);
xtickangle(45);
set(gcf,'color','w');
set(gcf,'Position',[100 100 900 650])
set(gca,'FontSize',18);
title('Performance of each model component');
ylabel({'\fontsize{16}RDM correlation with hIT';'\fontsize{12}(Spearman)'});

% PLOT 1: Statistical test indicators

thresh = 0.05; % uncorrected for now; could define Bonferroni or other correction

% Test 1: each component vs zero
% (does this model explain significant variance in the data?)
diffs = component_results.raw; % diff from zero is just the performance itself
for i = 1:size(diffs,2)
    % since we have a bootstrap distribution, the 95% confidence interval
    % is simply the centre 95% of the distribution:
    ci = quantile(diffs(:,i), [thresh/2, 1-thresh/2])
    % draw asterisk at base of bar if significantly better than zero
    % (i.e. if confidence interval is entirely above zero)
    if ci(1) > 0
        text(i, 0.03, '*', 'FontSize',14);
    else
        text(i, 0.03, 'ns', 'FontSize',14);
    end
end

% Test 2: each component vs lower bound of noise ceiling
% (does this model do significantly worse than can be achieved by
% predicting each subjects' data from that of other subjects?)
% nb Can't draw any strong conclusion from a non-significant result - 
% may be non-significant either because it is indeed very close to the 
% noise ceiling, or else because the data are noisy / test under-powered. 
diffs = repmat(ceiling_results.lower,[1, size(component_results.raw,2)])-component_results.raw;
for i = 1:size(diffs,2)
    % since we have a bootstrap distribution, the 95% confidence interval
    % is simply the centre 95% of the difference distribution:
    ci = quantile(diffs(:,i), [thresh/2, 1-thresh/2])
    % draw asterisk at bottom of noise ceiling if significantly worse than
    % lower bound of noise ceiling
    if ci(1) > 0
        text(i, lowceil+0.03, '*', 'FontSize',14);
    else
        text(i, lowceil+0.03, 'ns', 'FontSize',14);
    end
end

%% PLOT 2: performance of uniformly and optimally reweighted components

figure(2)

% noise ceiling first in drawing order
patch([0 0 3 3],[lowceil, uppceil, uppceil, lowceil],[0.5, 0.5, 0.5],'edgecolor','none','FaceAlpha',0.5)
hold on 

% barchart of mean and standard deviation across bootstraps for each of the
% two versions (uniformly and optimally reweighted model components)
bar([mean(combined_results.raw_unif,1), mean(combined_results.raw_fitted,1)],'FaceColor','flat', 'BarWidth',1, 'EdgeColor', 'w')
errorbar([1:2], [mean(combined_results.raw_unif,1), mean(combined_results.raw_fitted,1)], [std(combined_results.raw_unif,1), std(combined_results.raw_fitted,1)], 'color','k','LineWidth',2,'LineStyle','none','CapSize',0);

% aesthetics
box off
xlim([0.5,2+0.5])
ylim([0, uppceil+0.1])
xticks(1:2)
xticklabels({'uniformly weighted','optimally reweighted'});
xtickangle(45);
set(gcf,'color','w');
set(gcf,'Position',[100 100 650 650])
set(gca,'FontSize',18);
title('Performance of combined model components');
ylabel({'\fontsize{16}RDM correlation with hIT';'\fontsize{12}(Spearman)'});

% PLOT 2: Statistical test indicators

thresh = 0.05; % uncorrected for now; could define Bonferroni or other correction

% Test 1: each version of the combined model vs zero
% (does this model explain significant variance in the data?)
diffs = combined_results.raw_unif; % diff from zero is just the performance itself
ci = quantile(diffs, [thresh/2, 1-thresh/2])
% draw result for first bar (uniform weighting)
if ci(1) > 0
    text(1, 0.03, '*', 'FontSize',14);
else
    text(1, 0.03, 'ns', 'FontSize',14);
end

% now do same for second bar (optimal reweighting)
diffs = combined_results.raw_fitted; % diff from zero is just the performance itself
ci = quantile(diffs, [thresh/2, 1-thresh/2])
if ci(1) > 0
    text(2, 0.03, '*', 'FontSize',14);
else
    text(2, 0.03, 'ns', 'FontSize',14);
end

% Test 2: each version of the combined model vs lower bound of noise ceiling
% (does this model do significantly worse than can be achieved by
% predicting each subjects' data from that of other subjects?)
% nb Can't draw any strong conclusion from a non-significant result - 
% may be non-significant either because it is indeed very close to the 
% noise ceiling, or else because the data are noisy / test under-powered. 
diffs = ceiling_results.lower-combined_results.raw_unif;
ci = quantile(diffs, [thresh/2, 1-thresh/2]);
% draw result for first bar (uniform weighting)
if ci(1) > 0
    text(1, lowceil+0.03, '*', 'FontSize',14);
else
    text(1, lowceil+0.03, 'ns', 'FontSize',14);
end
    
% now do same for second bar (optimal reweighting)
diffs = ceiling_results.lower-combined_results.raw_fitted;
ci = quantile(diffs, [thresh/2, 1-thresh/2]);
% draw result for first bar (uniform weighting)
if ci(1) > 0
    text(2, lowceil+0.03, '*', 'FontSize',14);
else
    text(2, lowceil+0.03, 'ns', 'FontSize',14);
end
