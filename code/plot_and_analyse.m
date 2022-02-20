% 23.09.2019
% Barchart and scatterplot of output of whole-network-reweighting analyses.

trained_names = {'alexnet','vgg16','googlenet','resnet18','resnet50','squeezenet','densenet201','inceptionresnetv2','mobilenetv2'};
random_names = {'alexnet_rand01','vgg16_rand01','googlenet_rand01','resnet18_rand01','resnet50_rand01','squeezenet_rand01','densenet201_rand01','inceptionresnetv2_rand01','mobilenetv2_rand01'};
pretty_names = {'Alexnet', 'VGG-16', 'Googlenet', 'Resnet-18','Resnet-50','Squeezenet','Densenet-201','InceptionResnet','Mobilenet'}; % used for plotting

% load trained net results (fitted and unfitted)
trainednetperfs = [];
trainednetperfs_fitted = [];
for model = 1:length(trained_names)
    
%     resultsdir = strcat('../../JOCN_cluster_analyses/v1_summary_xmas/',trained_names{model},'/results/V1/');
    resultsdir = strcat('../../JOCN_cluster_analyses/kate18/data/',trained_names{model},'/results/cluster0428/');
    
    load(strcat(resultsdir,'bootstrap_output_wholenet.mat'),'wholenet_results');
    trainednetperfs = [trainednetperfs, wholenet_results.raw_fitted];
    trainednetperfs_fitted = [trainednetperfs_fitted, wholenet_results.fittedPCA_fitted];

    if model == 1 % assuming same from all models
        load(strcat(resultsdir,'bootstrap_output_ceilings.mat'),'ceiling_results');
        lowceil = mean(ceiling_results.lower);
        uppceil = mean(ceiling_results.upper);
    end
end

% load random net results (fitted and unfitted)
randomnetperfs = [];
randomnetperfs_fitted = [];
for model = 1:length(random_names)
    
%     resultsdir = strcat('../../JOCN_cluster_analyses/v1_summary_xmas/',random_names{model},'/results/V1/');
    resultsdir = strcat('../../JOCN_cluster_analyses/kate18/data/',random_names{model},'/results/cluster0428/');

    load(strcat(resultsdir,'bootstrap_output_wholenet.mat'),'wholenet_results');
    randomnetperfs = [randomnetperfs, wholenet_results.raw_fitted];
    randomnetperfs_fitted = [randomnetperfs_fitted, wholenet_results.fittedPCA_fitted];
    
end

% collate so we can plot as grouped bars
groupedperfs = [nanmean(randomnetperfs); nanmean(randomnetperfs_fitted); nanmean(trainednetperfs); nanmean(trainednetperfs_fitted)]';
groupedstds = [nanstd(randomnetperfs); nanstd(randomnetperfs_fitted); nanstd(trainednetperfs); nanstd(trainednetperfs_fitted)]';

%% PLOT

bars = bar(groupedperfs,'FaceColor','flat', 'BarWidth',1, 'EdgeColor', 'w')
hold on

% assigning colours to bars (to match layerwise plots)
cm = bone(5);
untrained_unfit_col = cm(4,:);
bars(1).CData = untrained_unfit_col;
untrained_fit_col = cm(3,:);
bars(2).CData = untrained_fit_col;
cm = parula(5);
trained_unfit_col = cm(2,:);
bars(3).CData = trained_unfit_col;
trained_fit_col = cm(1,:);
bars(4).CData = trained_fit_col;

% noise ceiling first in drawing order
patch([0 0 length(trained_names)+1 length(trained_names)+1],[lowceil, uppceil, uppceil, lowceil],[0.5, 0.5, 0.5],'edgecolor','none','FaceAlpha',0.5)
hold on

% Calculating the width for each bar group for proper error bar placement
ngroups = size(groupedperfs, 1);
nbars = size(groupedperfs, 2);
groupwidth = min(0.8, nbars/(nbars + 1.5));
for i = 1:nbars
    barx = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    errorbar(barx, groupedperfs(:,i), groupedstds(:,i), 'color','k','LineWidth',2,'LineStyle','none','CapSize',0);
end

% aesthetics
box off
xlim([0.5,length(trained_names)+0.5])
ylim([0, uppceil+0.08])
xticks(1:length(trained_names))
xticklabels(pretty_names);
xtickangle(45);
legend('untrained + unfitted', 'untrained + fitted', 'trained + unfitted', 'trained + fitted', 'Box', 'on', 'EdgeColor', 'w', 'Position', [0.17 0.77 0.2 0.1]);
set(gcf,'color','w');
set(gcf,'Position',[100 100 900 650])
set(gca,'FontSize',18);
% title('Combined performance of all layers for each network');
ylabel({'\fontsize{16}RDM correlation with V1';'\fontsize{12}(Spearman)'});

%% statistical test indicators

thresh = 0.05; % uncorrected for now

% 1. each model vs lower bound of noise ceiling for trained and fitted nets
% (only ones close to noise ceiling)
diffs = repmat(ceiling_results.lower,[1, size(trainednetperfs_fitted,2)])-trainednetperfs_fitted;
for i = 1:size(diffs,2)
    ci = quantile(diffs(:,i), [thresh, 1-thresh])
    if ci(1) > 0
        % since nearly all models are significantly below noise ceiling,
        % it's often neater to only indicate `ns` results, hence this line
        % commented out by default
%         text(barx(i)-0.1, lowceil+0.03, '*', 'FontSize',14);
    else
        text(barx(i)-0.1, lowceil+0.03, 'ns', 'FontSize',14);
    end
end

%% model comparison matrix of all trained and fitted versions

% each trained and fitted model vs every other
pairwise_sigs = zeros(length(trained_names),length(trained_names));
count=4;
for m = 1:length(trained_names)
    for n = 1:length(trained_names)
        if m > n
            diffs = trainednetperfs_fitted(:,m) - trainednetperfs_fitted(:,n);
            ci = quantile(diffs, [thresh, 1-thresh]);
            % check whether CI contains 0 to either side
            if m ~= n
                if ~((ci(1) < 0) && (0 < ci(2)))
                    pairwise_sigs(m,n) = 1;
                end
            end
        end
    end
end

colormap('copper')
imagesc(pairwise_sigs)
axis('square')

%% within-architecture comparisons:

% eg each trained unfitted model vs its untrained unfitted version
thresh = 0.05;%/length(trained_names);
sigs = zeros(1,length(trained_names));
count = 1;
for m = 1:length(trained_names)
    diffs = trainednetperfs(:,m) - trainednetperfs_fitted(:,m);
    ci = quantile(diffs, [thresh, 1-thresh]);
    % check whether CI contains 0 to either side
    if ~((ci(1) < 0) && (0 < ci(2)))
        sigs(m) = 1;
    end
    figure(count)
    hist(diffs,50)
    hold on
    plot([ci(1),ci(1)],[0,60])
    plot([ci(2),ci(2)],[0,60])
    title(trained_names{m})
    drawnow
    count = count+1;
end

sigs

%% scatterplot
figure(4)
cm = parula(5);
scatter(nanmean(randomnetperfs),nanmean(trainednetperfs),200, 'MarkerFaceColor', cm(2,:), 'MarkerFaceAlpha',0.7, 'MarkerEdgeColor', cm(2,:), 'LineWidth', 1.5);

hold on
patch([0 0 uppceil+0.1 uppceil+0.1],[lowceil, uppceil, uppceil, lowceil],[0.5, 0.5, 0.5],'edgecolor','none','FaceAlpha',0.5,'HandleVisibility','off')
patch([lowceil, uppceil, uppceil, lowceil],[0 0 uppceil+0.1 uppceil+0.1],[0.5, 0.5, 0.5],'edgecolor','none','FaceAlpha',0.5,'HandleVisibility','off')
% xlim([0, uppceil+0.02])
% ylim([0, uppceil+0.02])
% xticks([0:0.1:0.5])
% yticks([0:0.1:0.5])
% axis square; box off
% set(gcf,'color','w');
% set(gcf,'Position',[200 200 500 500])
% set(gca,'FontSize',14);
% xlabel('hIT correlation of untrained network')
% ylabel('hIT correlation of trained network')

% if we want to know the correlation:
[r,p] = corr([nanmean(randomnetperfs);nanmean(trainednetperfs)]')
% %% scatterplot
% figure(5)
% hold on 
cm = parula(5);
scatter(nanmean(randomnetperfs_fitted),nanmean(trainednetperfs_fitted),200, 'MarkerFaceColor', cm(1,:), 'MarkerFaceAlpha',0.7, 'MarkerEdgeColor', cm(1,:), 'LineWidth', 1.5);

plot([0,1],[0,1],'k--')

xlim([0, uppceil+0.02])
ylim([0, uppceil+0.02])
xticks([0:0.1:0.5])
yticks([0:0.1:0.5])
axis square; box off
set(gcf,'color','w');
set(gcf,'Position',[200 200 500 500])
set(gca,'FontSize',16);
% legend({'unfitted','fitted'},'Box', 'on', 'EdgeColor', 'w', 'Position', [0.17 0.72 0.2 0.1])
xlabel('correlation of untrained network')
ylabel('correlation of trained network')

% if we want to know the correlation:
[r,p] = corr([nanmean(randomnetperfs_fitted);nanmean(trainednetperfs_fitted)]')


%% Permutation test of variability among fitted vs unfitted models

pairdists_trainednetperfs = pdist(mean(trainednetperfs)');
pairdists_trainednetperfs_fitted = pdist(mean(trainednetperfs_fitted)');
[p, observeddifference, effectsize] = permutationTest(pairdists_trainednetperfs, pairdists_trainednetperfs_fitted, 1000, 'plotresult', 1)

%% Equivalence test of differences between trained and fitted models
% relative to spread of lower noise ceiling

centred_lowceil = ceiling_results.lower - mean(ceiling_results.lower);
d = quantile(centred_lowceil, [0.05/2, 1-0.05/2]);

for m = 1:size(trainednetperfs_fitted,2)
    for n = 1:size(trainednetperfs_fitted,2)
        if m > n % i.e. only both testing unique model pairs
            %The null hypothesis is rejected if max([p1, p2]) > alpha, or if the
            %confidence interval falls outside of the equivalence interval
            [p1, p2, CI] = TOST(trainednetperfs_fitted(:,m), trainednetperfs_fitted(:,n), d(1), d(2), 0.05/36)
%             [p1, p2, CI] = TOST(trainednetperfs(:,m), trainednetperfs(:,n), d(1), d(2), 0.05/36)
        end
    end
end

