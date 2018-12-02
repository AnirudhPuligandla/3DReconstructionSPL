% Code to test econstruction from hyperspectral data

clear variables
close all
clc

addpath ../Data_Lidar/Multispectral_data
addpath src/
addpath ../Data_Lidar/synth_multi_data

% Load data
load Multispectral_data/init1_f8_blocks.mat     % Impulse response
load Multispectral_data/init4_f8_blocks.mat     % wavelenght vignetting effect
%load synth_multi_data/synth_f8_blocks_cube.mat  % measurements
%load Multispectral_data/init3_f8_blocks_300us.mat; dt = 3e-4;   % Real Data
load Multispectral_data/init3_f8_blocks.mat;    % real data with acquisition time of 10ms
dt = 1e-2;

% reshape data into 190x190 images
l = sqrt(size(sY,1)); % assumed to be square
nwl = size(sY,2); % number of wavelengths
sY = reshape(sY,[l l nwl]);
sY = sY(6:end-5,6:end-5,:); l = l - 10; % discard edges
sY = reshape(sY,[l*l nwl]);
Y = sY(:,1:32);

% Save ground truth
load gt_01_001.mat;
a_gt = a; %b_gt = b;
clear a;

%sF0 = sF0/(1e-3/dt); % original capture time for F was 100 seconds with a=0.1
% sF0 = sF0.*(10/100).*dt;

sigma_f = sF0(round(end/2),1:32);
%clear sF0;

[N, nwl] = size(Y); % no. of pixels N and wavelengths nwl
l = sqrt(N);
dims = [l,l];       % images dimension

% Cells to hold the results
load('test_multi_real_js_dwt.mat')
% final_a = cell(1,2);
% final_a_33 = cell(1,2);
% final_a_1 = cell(1,2);
% final_a_15 = cell(1,2);
% nrmse_a = cell(1,2);
%var_a = cell(8,5);
%mean_nrmse_a = cell(8,5);
%mean_var_a = cell(8,5);

% Comparison metric function handler
%Nrm_se = @(x,y) sqrt(mean((x-y).^2)/mean(x.^2));
%Nrm_se_h = @(x,y) sqrt(mean(mean((x-y).^2))/mean(mean(x.^2)));
%Nmse_h = @(x,y) mean(mean((x-y).^2))/mean(mean(x.^2));
mse_h = @(x,y) mean(mean((x-y).^2));
var_mse_h = @(x,y) var(mean((x-y).^2));

% mean and variance over 32 wavelengths // remove gt from denominator

factr_count = 4;
sub_ratio = [4 1 2 4 8 16];
required_means = [2 1 2 4 8];%[0.1 0.2 0.4 0.8 1.6];
mean_val_DATA = mean(Y);

Y_DATA = zeros(size(Y));

for mean_val = 1
    
    tau_count = 1;
    
    for i = 1:size(sub_ratio,2)
        % Change data to get appropriate mean
        factr = required_means(i)./mean_val_DATA;
        %sigma_f = factr.*sigma_f;
        for wl = 1:size(Y,2)
            Y_DATA(:,wl) = binornd(Y(:,wl),factr(wl));
            %b_gt = b_gt.*factr;
            % rsf representing R0*a*sigma_f from model equation
            rsf = repmat(factr,N,1).*repmat(10*sigma_f,N,1).*R0(:,1:32);
        end
        
        % Build a mask for sub-sampling
        sub_sample = sub_ratio(i);                 % Sub-sampling ratio (1,2,4,8 etc.)
                                                   % set to 1 for full sampling
        num = round(N/sub_sample);             % no. of samples
        ind = randperm(N,num);          % Random sampling locations
        M = sparse(1:num,ind,ones(1,num),num,N);    % measurement operator
        
        % Functions for measurement operator
        if(sub_sample == 1)
            K = @(x) x;
            Kt = @(x) x;
            
            % select measurements
            Y_ = Y_DATA;
            rsf_ = rsf;
        else
            K = @(x) M*x;
            Kt = @(x) M'*x;
            
            % select measurements
            Y_ = M*Y_DATA;
            rsf_ = M*rsf;
        end
        
        sq = @(x) reshape(x,dims); % recover as an image
        vc = @(x) x(:); % vectorize
        
        % Calculate inv(K'*K + 2I)
        Kmat = K(sparse(eye(N)));
        Kmat=sparse(Kmat);
        ktk = Kmat'*Kmat;
        ktk = sparse(ktk);
        ktk2Iinv = diag(1./(diag(ktk)+2));
        %ktk2Iinv = inv(ktk + 2*sparse(eye(size(ktk,1))));
        ktk2Iinv = sparse(ktk2Iinv);
        
        % Set algorithmic parameters
        clear options;
        options.verbose = 1;
        options.vc = vc;
        options.sq = sq;
        options.sigma_f = rsf_;
        options.rel_tol = 1e-5;
        options.max_iter = 1000;
        options.min_iter = 2;
        options.tau = 8;%20;%1.5              % regularization parameter for tv norm
        %options.tau2 = 1.5; %2e1       % regularization parameter for nuclear norm
        options.mu = 5e-2;   % kind of 1/step size
        %options.sub=0;                  % Set sub_sampling true or false
        options.a=1;                    % '0'-estimate b | '1'-estimate a
        options.ktk = ktk2Iinv;         % inv(K'*K + 2I)
        options.js = 1;                 % set to 1 if using joint sparsity model
        % diag(1./diag(KtK+2I))
        
        % %  Daubechies Wavelet parameters
        L = zeros([8,1]);
        nel = zeros([8,1]);
        s = cell([8,1]);
        for i = 1:8
            wname = ['db' num2str(i)];
            L(i) = wmaxlev(dims,wname);
            %     L(i) = 2;
            [temp,s{i}] = wavedec2(randn(dims),L(i),wname);
            nel(i) = numel(temp);
        end
        lastel = cumsum(nel);
        
        % DCT/DWT transform for sparsity basis
        if(options.js == 1)
            %P = @(x) vc(dct2(sq(x)));
            %Pt = @(x) vc(idct2(sq(x)));
            P = @(x) [wavedec2(sq(x),L(1),'db1')...
                wavedec2(sq(x),L(2),'db2')...
                wavedec2(sq(x),L(3),'db3')...
                wavedec2(sq(x),L(4),'db4')...
                wavedec2(sq(x),L(5),'db5')...
                wavedec2(sq(x),L(6),'db6')...
                wavedec2(sq(x),L(7),'db7')...
                wavedec2(sq(x),L(8),'db8')]'/sqrt(8);
            
            Pt = @(x) vc(waverec2(x(1:lastel(1)),s{1},'db1')+...
                waverec2(x(lastel(1)+1:lastel(2)),s{2},'db2')+...
                waverec2(x(lastel(2)+1:lastel(3)),s{3},'db3')+...
                waverec2(x(lastel(3)+1:lastel(4)),s{4},'db4')+...
                waverec2(x(lastel(4)+1:lastel(5)),s{5},'db5')+...
                waverec2(x(lastel(5)+1:lastel(6)),s{6},'db6')+...
                waverec2(x(lastel(6)+1:lastel(7)),s{7},'db7')+...
                waverec2(x(lastel(7)+1:lastel(8)),s{8},'db8'))/sqrt(8);
        else
            P = @(x) x;
            Pt = @(x) x;
        end
        
        options.P = P;
        options.Pt = Pt;
        
        %-----------Estimation---------------
        tstart_a = tic;
        fprintf('Running a_pidal_tvnn\n');
        
        T = 1;
        if(options.js==1)
            [a,~] = pidal_fa(Y_,K,Kt,options,T,0);
        else
            [a,~] = pidal_tv(Y_,K,Kt,options,T,0);
        end
        %a = min(a,1);
        
        tend = toc(tstart_a);
        fprintf('a_pidal_tvnn runtime: %.3fs\n\n', tend);
        
        % Save results
        final_a{factr_count,tau_count} = a;
        final_a_32{factr_count,tau_count} = a(:,32);
        final_a_1{factr_count,tau_count} = a(:,1);
        final_a_15{factr_count,tau_count} = a(:,15);
        nrmse_a{factr_count,tau_count} = mse_h(a_gt,a);
        var_a{factr_count,tau_count} = var_mse_h(a_gt,a);
%         
         save(['test_multi_real_js_dwt.mat'],...
         'final_a',...
        'final_a_32',...
        'final_a_1',...
        'final_a_15',...
        'nrmse_a',...
        'var_a');
        
        % Increment counter
        tau_count = tau_count + 1;
        
    end
    
    %factr_count = factr_count + 1;
    
end
%a = max(a,0);
% save(['test_multi.mat'],...
%     'final_a',...
%     'nrmse_a');