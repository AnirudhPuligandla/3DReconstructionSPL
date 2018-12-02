clear variables
close all
clc

addpath ../Data_Lidar/
addpath ../Data_Lidar/Multispectral_data
addpath src/
addpath ../Data_Lidar/new_data_40m
addpath ../Data_Lidar/synth_data

% Load calibration matrix
load new_data_40m/F_real_40m_F100.mat
F=F(1001:3500,1001:3500);

% Load synthetic data
load synth_data/synth_F100_cube_2000ppp_sb.mat;

% Save ground truth
a_gt = a; b_gt = b; d_gt = d;
clear a b d x;

% Remove first and last time stamps of the measurements
Y = Y(:,:,2:end-1);

F = F/(60/dt); % original capture time for F was 60 seconds

% Estimate sigma_f
sigma_f = sum(F(:,1000));

% Change data to get appropriate mean
factr = 0.0008;
sigma_f = factr*sigma_f;
Y_DATA = binornd(Y,factr);
%a_gt = a_gt.*factr;
b_gt = b_gt.*factr;

% Split the data into two parts for baseline and response intensity
Tb = 1200;
% for baseline
Yb = sum(Y_DATA(:,:,1:Tb),3);
Ta = 2500;
% for response intensity
Ya=Y_DATA(:,:,1201:1200+Ta); % Discard early and late measurements
Ya = sum(Ya,3);

% Functions to reshape the data
dims = size(Yb);
sq = @(x) reshape(x,dims); % recover shape
vc = @(x) x(:); % vectorize
Nrm_se = @(x,y) sqrt(mean((x-y).^2)/mean(x.^2));%/(max(y)-min(y));

% Build a mask for sub-sampling
N = prod(dims);                 % total no. of pixels
sub_sample = 8;                 % Sub-sampling ratio (1,2,4,8 etc.)
                                % set to 1 for full sampling
num = N/sub_sample;             % no. of samples
ind = randperm(N,num);          % Random sampling locations
M = sparse(1:num,ind,ones(1,num),num,N);    % measurement operator

% Functions for measurement operator
if(sub_sample == 1)
    K = @(x) x;
    Kt = @(x) x;
    
    % select measurements
    Ya_ = Ya(:);
    Yb_ = Yb(:);
else
    K = @(x) M*x;
    Kt = @(x) M'*x;
    
    % select measurements
    Ya_ = M*Ya(:);
    Yb_ = M*Yb(:);
end

% Calculate inv(K'*K + 2I)
Kmat = K(sparse(eye(N)));
Kmat=sparse(Kmat);
ktk = Kmat'*Kmat;
ktk = sparse(ktk);
ktk2Iinv = inv(ktk + 2*sparse(eye(size(ktk,1))));
ktk2Iinv = sparse(ktk2Iinv);

% Set algorithmic parameters
clear options;
options.verbose = 1;
options.vc = vc;
options.sq = sq;
options.sigma_f = sigma_f;
options.rel_tol = 1e-2;
options.max_iter = 1000;
options.min_iter = 2;
options.tau = 2e3/dt;              % regularization parameter for tv norm
%options.tau2 = 2e2; %2e1       % regularization parameter for nuclear norm
options.mu = 1e1*options.tau;   % kind of 1/step size
options.sub=1;                  % Set sub_sampling true or false
options.a=0;                    % '0'-estimate b | '1'-estimate a
options.ktk = ktk2Iinv;         % inv(K'*K + 2I)



%-----------Estimation---------------
tstart_a = tic;
fprintf('Running a_pidal_tvnn\n');

%-------baseline-------------------

[b,options.d] = pidal_tv(Yb_,K,Kt,options,Tb);

%options.max_iter = 5000;
%-------intensity-----------------
options.a=1;
options.rel_tol = 1e-5;
options.tau = 5.314e-2;
options.mu = 5e-2;
%options.rel_tol = 5e-5;
%options.tau = 5.314e-2;
%options.mu = 5e-2;

[a,~] = pidal_tv(Ya_,K,Kt,options,Ta,K(b(:)));
a = min(a,1);
a = max(a,0);

%final_a = K(a).*sub_rsf;

tend = toc(tstart_a);
fprintf('a_pidal_tvnn runtime: %.3fs\n\n', tend);