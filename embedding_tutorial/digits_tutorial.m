% MDS and tSNE tutorial using digit data
% Digit data is from Kaggle competition
% (https://www.kaggle.com/c/digit-recognizer/data)
%
% Gabrielle M. Schroeder
% CNNP Journal Club
% 24 April 2019

close all; clearvars; clc

%% Load digit data.
% Each row = a digit

load('digits.mat')

%% example digit
px = 28; % dimensions of each digit are px x px
x = 10;  % index of the digit to plot

figure(1)
imagesc(reshape(digits(x,:),px,px)') % reshape digit pixel info to a square
title(num2str(labels(x)))
clrs = gray(200);
colormap(flipud(clrs))

clearvars x px

%% use subset of matrix
n = length(labels); % total number of observations
n_sample = 500;     % number of samples to select
rng(100)            % set seed for reproducability

% sample digits and digit labels
[digits_small,idx] = datasample(digits,n_sample,1);
labels_small = labels(idx);

clearvars idx

%% visualise distance matrix
D = squareform(pdist(digits_small));        % make distance matrix

[sorted_labels,idx] = sort(labels_small);   % sort by digit label

figure(2)
imagesc(D(idx,idx))
title('distance matrix (sorted)','FontSize',16)
colormap parula
colorbar
label_diff = diff(sorted_labels);
ticks = [find(label_diff>0); length(labels_small)]; % digit labels
set(gca,'XTick',ticks,'XTickLabels',sorted_labels(ticks),...
    'YTick',ticks,'YTickLabels',sorted_labels(ticks))
xlabel('digit label')
ylabel('digit label')

clearvars sorted_labels idx labels_diff ticks
%% MDS

%%% choose stress function
%criterion = 'metricstress';     % normalised stress (type of metric MDS)
criterion = 'stress';           % Kruskal's stress, type 1 (type of non-metric MDS)
%criterion = 'strain';           % 'classical' MDS (Torgerson's), equivalent to PCA

% number of dimensions
n_dim = 2;

% MDS; disparities = mapped distances
% Note that disparities will be equal to the original distances if metric
% MDS is used (because distances are not mapped)
[Y,stress,disparities] = mdscale(D,n_dim,'Criterion',criterion);

%%% If get error "points in the configuration have co-located," try the below
%%% code with different initializations
%rng(10);
%[Y,stress,disparities] = mdscale(D,n_dim,'Criterion',criterion,'start','random');


% plot MDS results 

% unlabelled points
f=figure();
set(f,'Position',[20 20 1300 500])
subplot(1,3,1)
scatter(Y(:,1),Y(:,2),20,[0 0 0],'fill')

% points coloured by digit label
subplot(1,3,2)
scatter(Y(:,1),Y(:,2),20,labels_small,'fill')
colormap jet

% points coloured by digit label and labelled using text
subplot(1,3,3)
scatter(Y(:,1),Y(:,2),20,labels_small,'fill')
colormap jet
text(Y(:,1),Y(:,2),num2cell(labels_small),'FontSize',10)
xlabel('dimension 1')
ylabel('dimension 2')
axis equal
title([criterion ', stress = ' num2str(stress)])

%% tSNE

% While Matlab, by default, initialises iterative MDS algorithms using the
% "classical" (strain) MDS results, tSNE begins with a random initialisation;
% therefore, set a random seed.
rng(5) 

% Perplexity balances the local vs global aspects of the topology.
% It controls local neighbours of each point.
% Generally, a perplexity between 5 and 50 is okay; will need larger perplexity for
% larger datasets.
% Matlab default = 30
perplex = 10; % try other perplexity values (e.g., 5, 30, 50)
Y_tsne = tsne(digits_small,'Perplexity',perplex);

% plot tSNE results 

% unlabelled points
f=figure();
set(f,'Position',[20 20 1300 500])
subplot(1,3,1)
scatter(Y_tsne(:,1),Y_tsne(:,2),20,[0 0 0],'fill')

% points coloured by digit label
subplot(1,3,2)
scatter(Y_tsne(:,1),Y_tsne(:,2),20,labels_small,'fill')
colormap jet

% points coloured by digit label and labelled using text
subplot(1,3,3)
scatter(Y_tsne(:,1),Y_tsne(:,2),20,labels_small,'fill')
colormap jet
text(Y_tsne(:,1),Y_tsne(:,2),num2cell(labels_small),'FontSize',12,'FontWeight','bold')
xlabel('dimension 1')
ylabel('dimension 2')
axis equal
title(['tSNE, perplexity = ' num2str(perplex)])
