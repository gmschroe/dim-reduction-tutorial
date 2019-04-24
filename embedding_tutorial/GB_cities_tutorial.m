% Demonstrating MDS using driving distances between GB cities.
% Distance data is from https://www.mileage-charts.com
%
% Gabrielle M. Schroeder
% CNNP Journal Club
% 24 April 2019

clc; close all; clearvars

%% load data
% note that distance is the *driving distance* in kilometers

load('GB_cities.mat');
%% ensure distance matrix is symmetric
km = (km + km')./2;

%% visualize distance matrix
n = length(cities);

figure(1)
imagesc(km)
set(gca,'XTick',1:n,'YTick',1:n,...
    'XTickLabels',cities,'YTickLabels',cities,...
    'FontSize',8)
xtickangle(45)
axis square
title('driving distances (km) between cities','FontSize',16)

%% Torgerson's MDS (referred to as classical MDS by Matlab)

[Y_metric,eigenvalues] = cmdscale(km);

% plot eigenvalues
figure(2);
plot(eigenvalues,'-o')
title('eigenvalues')

% plot points (first two dimensions)
figure(3)
scatter(Y_metric(:,2),Y_metric(:,1))
text(Y_metric(:,2),Y_metric(:,1),cities,'FontSize',10)
xlabel('dimension 2')
ylabel('dimension 1')
axis equal

%% Shepard's plot (distances vs dissimilarities)

figure(4)
D = pdist(Y_metric(:,1:2)); % distances in embedding (using first two dimensions)
scatter(squareform(km),D)
xlabel('dissimilarities (original distances)')
ylabel('distances (in MDS embedding)')
hold on
plot([0 max(D)],[0 max(D)],'LineWidth',2)
hold off

%% iterative forms of MDS
close all

% choose stress function
criterion = 'metricstress';     % normalised stress (type of metric MDS)
criterion = 'stress';           % Kruskal's stress, type 1 (type of non-metric MDS)

% number of dimensions
n_dim = 2;

% MDS; disparities = mapped distances
% note that disparities will be equal to the original distances if metric
% MDS is used
[Y,stress,disparities] = mdscale(km,n_dim,'Criterion',criterion);

% plot MDS results 
figure(5)
scatter(Y(:,2),Y(:,1))
text(Y(:,2),Y(:,1),cities,'FontSize',10)
xlabel('dimension 2')
ylabel('dimension 1')
axis equal
title([criterion ', stress = ' num2str(stress)])


switch criterion
    case 'metricstress'
        figure(6)
        D = pdist(Y);
        scatter(squareform(km),D)
        xlabel('dissimilarities (original distances)')
        ylabel('distances (in MDS embedding)')
        hold on
        plot([0 max(D)],[0 max(D)],'LineWidth',2)
        hold off
    case 'stress'
        D = pdist(Y);
        [sorted_dispar,idx] = sort(squareform(disparities));
        km_sorted = squareform(km);
        km_sorted=km_sorted(idx);
        
        figure(6)
        plot(km_sorted,sorted_dispar,'o-')
        xlabel('dissimilarities (original distances)')
        ylabel('disparities (mapped original distances)')
        
        figure(7)
        plot(sorted_dispar,D(idx),'o-');
        ylabel('distances (in MDS embedding)')
        xlabel('disparities (mapped original distances)')  
end

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
perplex = 5;
Y_tsne = tsne(km,'Perplexity',perplex);

% plot 
figure(8)
scatter(Y_tsne(:,1),Y_tsne(:,2))
text(Y_tsne(:,1),Y_tsne(:,2),cities,'FontSize',10)
xlabel('dimension 1')
ylabel('dimension 2')
axis equal
title('tSNE','FontSize',16)


