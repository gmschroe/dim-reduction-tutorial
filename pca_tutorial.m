% PCA Tutorial
% Computational Neuroscience, Neurology, and Psychiatry Journal Club
% 29 January 2018
%
% Answers to the exercises are at the end of the script.
%
% Gabrielle M. Schroeder
% Newcastle University School of Computing
% Contact: g.m.schroeder1@hcl.ac.uk

close all; clearvars; clc

%% Divergent colormap, to use for plots
% red = positive, blue = negative, when centered at zero
m=51;    
m1 = floor(m*0.5);
r = (0:m1-1)'/max(m1,1);
g = r;
r = [r; ones(m1+1,1)];
g = [g; 1; flipud(g)];
b = flipud(r);
my_colormap=[r g b];

%% Load data
% We will use MATLAB data on breakfast cereals
load cereal.mat

data=[Calories Carbo Fat Fiber Potass Protein Shelf Sodium Sugars Vitamins Weight];
var_names={'Calories','Carbohydrates','Fat','Fiber',...
    'Potassium','Protein','Display Shelf','Sodium','Sugars','Vitamins','Weight'};
clearvars -except var_names data Variables Name Mfg Type my_colormap

% Excluded Cups because many cereals don't have this data, and PCA doesn't
% work well with missing values

%% Remove observations with missing values
% missing values are -1 in this dataset
imagesc(data<0) 
title('Missing values (yellow)')
xlabel('variables')
ylabel('observations')
[i,j]=find(data<0);
keep_obs=ones(1,size(data,1));
keep_obs(i)=0;
data=data(keep_obs==1,:);

% other info about observations (name, manufacturer, type)
Name=Name(keep_obs==1);
Mfg=Mfg(keep_obs==1);
Type=Type(keep_obs==1);

%% Info about data 
close all

n = size(data,1); % number of observations 
p = size(data,2); % number of variables

disp([num2str(n) ' observations']);
disp([num2str(p) ' variables']);

%% Look at distributions of values for each variable
f=figure(1);
set(f,'Position',[50 50 1200 500]);

% boxplots for each variable
subplot(1,2,1)
boxplot(data)
set(gca,'XTick',1:p,'XTickLabel',var_names);
xtickangle(45)
title('Distributions of the observed values for each variable')
ylabel('variable-specific units')
xlabel('variables')

% standard deviation for each variable
subplot(1,2,2)
bar(std(data))
set(gca,'XTick',1:p,'XTickLabel',var_names);
xtickangle(45)
title('Standard deviation of the observed values for each variable')
xlabel('variables')
ylabel('standard deviation, in variable-specific units')

%% Standardize

% For this dataset, we need to z-score each column (i.e., the values for
% each variable) because the data have different units and scales -
% otherwise, the PCs will be dominated by the variables that have the
% largest variance (here, sodium and potassium)
%
% To convert the values to z-scores, for each column (i.e., each variable),
% we
% 1) Subtract the mean value, then
% 2) Divide by the standard deviation

[data_proc,data_mu,data_std]=zscore(data); % z score each column

% To see what happens if you don't standardize this dataset, you could
% instead run the below code, which just centers the data (i.e., subtracts
% the mean value):

%%%data_proc=detrend(data,'constant'); % center each column 

% Here are the new distributions:
f=figure(1);
set(f,'Position',[50 50 1200 500]);

% boxplots for each variable
subplot(1,2,1)
boxplot(data_proc)
set(gca,'XTick',1:p,'XTickLabel',var_names);
xtickangle(45)
title({'Distributions of the observed values for each variable,','after being z-scored'})
ylabel('z-score')
xlabel('variables')

% standard deviation for each variable
subplot(1,2,2)
bar(std(data_proc))
set(gca,'XTick',1:p,'XTickLabel',var_names);
xtickangle(45)
title({'Standard deviation of the observed values for each variable,', 'after being z-scored'})
xlabel('variables')
ylabel('standard deviation')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Part 1: Perform PCA using built in MATLAB function

%  pca Principal Component Analysis (pca) on raw data.
%     COEFF = pca(X) returns the principal component coefficients for the N
%     by P data matrix X. Rows of X correspond to observations and columns to
%     variables. Each column of COEFF contains coefficients for one principal
%     component. The columns are in descending order in terms of component
%     variance (LATENT). pca, by default, centers the data and uses the
%     singular value decomposition algorithm. For the non-default options,
%     use the name/value pair arguments.

% Our data is the in correct format: the values for each observation 
% (= each cereal) each make up one row

[coeff, score, ~, ~, explained, ~] = pca(data_proc,'centered',true);

% By default, the MATLAB pca function centers (but does NOT standardize -
% i.e., z-score) each column of the data. This step is redundant since we
% already centered the data before performing PCA.
% Be aware that PCA functions in other programs may have different
% defaults!

%% Loadings/coefficients of principal components
% Each column corresponds to a PC

figure(2)
imagesc(coeff)
set(gca,'YTick',1:p,'YTickLabels',var_names);
xlabel('PCs')
ylabel('original variables')
colormap(my_colormap)
caxis([-1*max(abs(coeff(:))),max(abs(coeff(:)))]) % center colormap at zero
colorbar

%% Percentage of variance explained by each PC

figure(3)

% Percentage explained by each PC - called a "scree plot"
subplot(1,2,1)
plot(explained,'ko:','LineWidth',2);
xlim([1 p])
title('Percentage explained (scree plot)')
xlabel('PC #')
ylabel('% explained')

% Percentage explained by the sum of r PCs
subplot(1,2,2)
plot(cumsum(explained),'ko:','LineWidth',2)
ylim([0 100])
xlim([1 p])
title('Cumulative percentage explained')
xlabel('number of PCs')
ylabel('% explained')


%% Reoconstructing the data using the PC coefficients and scores 
% We can reconstruct the original dataset by multiplying the matrix of 
% PC scores (= each observation in terms of a combination of the PCs, size
% n x p)
% and the matrix of 
% coefficients (= the linear combinations of variables that make up each
% PC, size p x p)
%
% To do so, we multiply score * coeff'
% (We take the transpose the coefficients matrix because we want each
% row of score to correspond to the PC scores of one observation, and each 
% column of coeff to correspond to one variable (and its loading in each PC).

f=figure(4);
set(f,'Position',[50 50 1000 700]);

% Here's the "score" matrix - the data in terms of the PCs, rather than the
% original variables
subplot(2,2,1)
imagesc(score)
caxis([-1*max(abs(score(:))),max(abs(score(:)))]) % center colormap at zero
colormap(my_colormap)
title('Data in terms of PCs (PC scores)')
ylabel('observations')
xlabel('PC #')
set(gca,'XTick',1:p)
colorbar

% We've already plotted this, but as a reminder, here are the coefficients
% of the PCs - this time with each row corresponding to a PC
subplot(2,2,2)
imagesc(coeff')
set(gca,'XTick',1:p,'XTickLabels',var_names);
xtickangle(45)
title('Coefficients of each PC')
ylabel('PCs')
xlabel('original variables')
colormap(my_colormap)
caxis([-1*max(abs(coeff(:))),max(abs(coeff(:)))]) % center colormap at zero
colorbar

% Here's the original data, after pre-processing (i.e., centered or
% z-scored)
subplot(2,2,3)
imagesc(data_proc)
caxis([-1*max(abs(data_proc(:))),max(abs(data_proc(:)))]) % center colormap at zero
colormap(my_colormap)
title('Original data matrix, preprocessed')
ylabel('observations')
xlabel('variables')
set(gca,'XTick',1:p,'XTickLabels',var_names)
xtickangle(45)
colorbar

% And now here's the reconstructed data, using the PC coefficients and
% scores
subplot(2,2,4)
data_recon=score*coeff';
imagesc(data_recon)
caxis([-1*max(abs(data_recon(:))),max(abs(data_recon(:)))]) % center colormap at zero
colormap(my_colormap)
title('Data reconstructed from the PC coefficients and scores')
ylabel('observations')
xlabel('variables')
set(gca,'XTick',1:p,'XTickLabels',var_names)
xtickangle(45)
colorbar

%% Reconstructing the original data, before pre-processing
% Note that if you wanted to reconstruct the original data (before it was
% z-scored) you could do so by multipying each column by the
% original standard deviation of the corresponding variable, and then
% adding the original mean value of the variable.
%
% EXERCISE 1: Write the code for computing the original data matrix.
% When we computed the z-scores, we stored the column means as data_mu and
% the column standard deviations as data_std. Also remember that we've
% stored our reconstruction of the pre-processed ata as data_recon.
%
% Hint: the MATLAB functions bsxfun or repmat (you will only need one) will
% be useful for adding/multiplying each column of the data by a vector. If
% you aren't familiar with MATLAB code, try starting with "pseudocode" -
% write down what you want each step to do, in the order they should be
% executed.
%
% Put your answer in the below code to compare your reconstruction to the
% original data matrix.


%my_recon = <your reconstruction>

% plot
f=figure(5);
set(f,'Position',[50 50 1000 500]);

% Your reconstruction
subplot(1,2,2)
imagesc(my_recon)
title('My reconstruction of the original data matrix')
ylabel('observations')
xlabel('variables')
set(gca,'XTick',1:p,'XTickLabels',var_names)
xtickangle(45)
colorbar
colormap parula

% The original data, before preprocessing
subplot(1,2,1)
imagesc(data)
title('Original data matrix')
ylabel('observations')
xlabel('variables')
set(gca,'XTick',1:p,'XTickLabels',var_names)
xtickangle(45)
colorbar
colormap parula

%% Use first r PCs to reconstruct data

% Often, however, we want to use the computed PCs to construct an
% approximation of the original data. We do so using the same method -
% score * coeff'
% but this time, only use the first r components:
% score(:,1:r) * coeff(:,1:r)'
% This is a (n x r) matrix times a (r x p) matrix, which results in the
% desired (n x p) matrix
%
% We'll make a few reconstructions using successively more PCs

n_pc=1:6; % how many PCs to use for each reconstruction 
data_part_recon=zeros(n,p,length(n_pc)); % array for storing reconstructions

try close(6); catch ; end
f=figure(6);
set(f,'Position',[50 50 1200 700]);

count=1;
for i=n_pc
    ax=subplot(2,ceil(length(n_pc)/2),count);
    recon=(score(:,1:i)*coeff(:,1:i)'); % (n x r) * (r x p) --> n x p
    imagesc(recon);
    
    % We'll use the colorbar limits of our preprocessed data matrix so it's
    % easier to compare the reconstruction with the data (hopefully they
    % should be the same!)
    caxis([-1*max(abs(data_proc(:))),max(abs(data_proc(:)))]) % center colormap at zero
    title(['r=' num2str(i)])
    ylabel('observations')
    set(gca,'XTick',1:p,'XTickLabels',var_names,'YTick',[])
    xtickangle(45)
    colormap(my_colormap)
    
    % This code keeps the plot the same size when you add the colorbar
    % (normally it shrinks the plot)
    if i==n_pc(end)
        pos=get(ax,'Position');
        colorbar
        set(gca,'Position',pos);
    end
    
    data_part_recon(:,:,count)=recon; % save reconstruction in array
    count=count+1;
end
clearvars i recon pos ax count
suptitle('Reconstruction of data using first r PCs')

%% How does each PC contribute to the data reconstruction?
% Our reconstruction with r PCs is the sum of r rank 1 matrices (each one
% corresponding to a PC)
%
% "Rank 1" just means that each row of the matrix is a multiple of the same
% vector (here, the PC coefficients).
% The score of each observation tells you what to multiply the PC by for 
% each row.

n_pc=1:8; 

try close(7); catch ; end
f=figure(7);
set(f,'Position',[50 50 1200 700]);

count=1;
for i=n_pc
    ax=subplot(2,ceil(length(n_pc)/2),count); 
    rankone_mat=(score(:,i)*coeff(:,i)'); % (n x 1) * (1 x p) --> n x p
    imagesc(rankone_mat);
    caxis([-1*max(abs(data_proc(:))),max(abs(data_proc(:)))]) % center colormap at zero
    title(['r=' num2str(i)])
    set(gca,'XTick',1:p,'XTickLabels',var_names,'YTick',[])
    xtickangle(45)
    colormap(my_colormap)
    if i==n_pc(end)
        pos=get(ax,'Position');
        colorbar
        set(gca,'Position',pos);
    end
    
    count=count+1;
end
clearvars i rankone_mat pos ax count
suptitle('Contribution of each PC to the data reconstruction using first r PCs')

%% Using PCA for visualization

% Our data in the original variable space - each pair of variables plotted
% vs each other
corrplot(data_proc)

%% Now, plot the data in PC space 

% Choose two PCs 
pc1=1;
pc2=2;

% Plot
figure(9);
subplot(1,2,1)

scatter(score(:,pc1),score(:,pc2),'fill')
hold on
text(score(:,pc1),score(:,pc2),Name,'FontSize',6) % plot name of each cereal
hold off
xlabel(['PC ' num2str(pc1) ' scores']);
ylabel(['PC ' num2str(pc2) ' scores']);
title('Cereals')

subplot(1,2,2)
gscatter(score(:,pc1),score(:,pc2),Mfg)
xlabel(['PC ' num2str(pc1) ' scores']);
ylabel(['PC ' num2str(pc2) ' scores']);
title('Cereals, coloured by manufacturer')

%% Project new data into PC space
% When projecting new data into the PC space, we want to multiply the PC
% coefficients (with each PC = one row) by the new observations (each
% column = one observation).
% If you ever want to check your math - try using this method to project
% the original data into PC space, and then check that you get the same
% answer as MATLAB!

% Data for Special K Red Berries
data_new=[110 27-9 0 3 70 2 1 190 9 25 1.1];

% First, normalize by mean and standard deviation of the original data set
data_new_proc=bsxfun(@minus,data_new,data_mu);
data_new_proc=bsxfun(@rdivide,data_new_proc,data_std);

% Then, multiply by PC coefficients
score_new=coeff'*data_new_proc';
score_new=score_new'; % take the transpose so each row = one observation

% PC scores to plot
pc1=1;
pc2=2;

% Plot new point in PC space
figure(10);
scatter(score(:,pc1),score(:,pc2),'b','fill')
hold on
scatter(score_new(pc1),score(pc2),'r','fill') % new data
% Note that if we had multiple new observations, the code would instead be
% scatter(score_new(:,pc1),score(:,pc2),'r','fill')
hold off
xlabel(['PC ' num2str(pc1) ' scores']);
ylabel(['PC ' num2str(pc2) ' scores']);
title('Cereals')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Part 2: Finding PCs using the covariance matrix
%  We can also find the principal components ourselves by finding the 
% eigenvectors of the covariance or correlation matrix

% Covariance matrix (if z-scored first, will be correlation matrix)
C=cov(data_proc); % each column corresponds to a variable
figure(11)
imagesc(C)
axis square; axis off
xlabel('variables')
ylabel('variables')
title('Covariance matrix (correlation matrix if data was standardized)')
colorbar

%% Compute PCs
% We will use MATLAB's function eig to find the eigenvalues
%
% PC will contain each principal component (each eigenvector) as a column.
% V will be a diagonal matrix with the corresponding eigenvalues on the
% diagonal, in ascending order (lowest to highest).

[PC,V]=eig(C);

% Plot
f=figure(12);
set(f,'Position',[50 50 900 500]);

subplot(1,2,1)
imagesc(PC)
set(gca,'YTick',1:p,'YTickLabels',var_names);
title('PCs computed from covariance matrix (not in order of importance!)')
xlabel('PCs')
ylabel('original variables')
colormap(my_colormap)
caxis([-1*max(abs(coeff(:))),max(abs(coeff(:)))]) % center colormap at zero
colorbar

subplot(1,2,2)
imagesc(V)
colorbar
title('Diagonal matrix with eigenvalues')
axis square; axis off

%% Re-order eigenvectors and eigenvalues, and take diagonal of eigenvalue matrix
% However, the PC with the LARGEST eigenvalue is the one that explains the
% most variation, so we want that one to be first.
% We also just need the diagonal V, which contains the eigenvalues

% Diagonal of V = lambda, the eigenvalues
lambda=diag(V);

% Put eigenvalues and eigenvectors in descending order
[lambda,idx]=sort(lambda,'descend'); % reorder eigenvalues
PC=PC(:,idx);                        % reorder eigenvectors

% Plot

figure(12)
subplot(1,2,1)
imagesc(PC)
set(gca,'YTick',1:p,'YTickLabels',var_names);
title('PCs computed from covariance matrix')
xlabel('PCs')
ylabel('original variables')
colormap(my_colormap)
caxis([-1*max(abs(coeff(:))),max(abs(coeff(:)))]) % center colormap at zero
colorbar

subplot(1,2,2)
bar(lambda)
title('Spectrum of eigenvalues')

%% Amount of variance explained
% The eigenvalues are proportional to the total amount of variance, which
% is the trace (= sum of the diagonal elements) of the covariance matrix:

var_explained=lambda./trace(C);

% Plot
figure(13)
plot(var_explained,'bo:','LineWidth',2);
xlim([1 p])
title('Percentage explained (scree plot), computed from eigenvalues')
xlabel('PC #')
ylabel('% explained')

%% All that is left to do is to project the data onto the PCs
% EXERCISE 2: Project the data (data_proc) onto the PCs that you found.
% You will use the same method as before when we projected new data onto our
% MATLAB PCs.
% Make each column in your project correspond to the scores of one PC, and
% each row corespond to an observation.

%pc_score=<your projections here>

% Plot
pc1=1;
pc2=2;

figure(14);
scatter(pc_score(:,pc1),pc_score(:,pc2),'fill')
xlabel(['PC ' num2str(pc1) ' scores']);
ylabel(['PC ' num2str(pc2) ' scores']);
title('Cereals in PC space')




%% ANSWERS TO EXERCISES

% EXERCISE 1
% Reconstructing the original data matrix
my_recon = bsxfun(@times,data_recon,data_std);
my_recon = bsxfun(@plus,my_recon,data_mu); 

% EXERCISE 2
% Projecting the data onto the principal components
pc_score=PC'*data_proc';
pc_score=pc_score';



