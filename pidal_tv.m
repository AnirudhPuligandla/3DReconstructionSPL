function [xsol,d] = pidal_tv(y,K,Kt,options,T,b)
% Implementation of the adapted PIDAL-TV ADMM algorithm for restoration of
% Poissonian images described in:
% 
% Figueiredo et al., "Restoration of Poissonian Images Using
% Alternating Direction Optimization".
% 
% And adapted for intensity estimation.
% 
% y - measurements
% rsf - For wavelength vignetting effect r
% options - structure with options and parameters
% 
% Rodrigo Daudt
% 01/2017

% Optional input arguments.
if ~isfield(options, 'verbose'), options.verbose = 1; end
if ~isfield(options, 'sigma_f'), options.sigma_f = 1; end
if ~isfield(options, 'sq'), error('No sq function'); end
if ~isfield(options, 'vc'), error('No vc function'); end
if ~isfield(options, 'rel_tol'), options.rel_tol = 1e-4; end
if ~isfield(options, 'rel_tol2'), options.rel_tol2 = 1e-4; end
if ~isfield(options, 'max_iter'), options.max_iter = 200; end
if ~isfield(options, 'min_iter'), options.min_iter = 1; end
% lagrange multiplier
if ~isfield(options, 'mu'), options.mu = 1e2; end
% regularization parameters
if ~isfield(options, 'tau'), options.tau= 1; end
if ~isfield(options, 'tau2'), options.tau2 = 0; end
% prameters for varying mu
if ~isfield(options, 'tau_incr'), options.tau_incr = 2; end
if ~isfield(options, 'tau_decr'), options.tau_decr = 2; end
if ~isfield(options, 'rho'), options.rho = 10; end

if options.verbose > 0
    fprintf('   a_pidal_tv2: Starting...\n');
end

% Function value to check convergence
f2 = @(xt,xtm1) norm(xt - xtm1)/(min(norm(xt),norm(xtm1))+eps);

% Initialization

% Initialize r*sigma_f
rsf = options.sigma_f;

% variables for tau optimization
alph_tau = 1;
beta_tau = 1;
k_tau = 1;      % =1 for majority of norms (l1, tv, nuclear etc.)

if ~isfield(options, 'init')
    % u1 and d1
    % Check whether to estimate a or b
    if(options.a == 1)
        s1 = size(y);
        u1 = y./rsf;%zeros(s1);
        d1 = zeros(s1);
    else
        s1 = size(y);
        u1 = zeros(s1);
        d1 = zeros(s1);
    end

    % u2 and d2
    %if(options.sub == 1)
        u2 = Kt(u1);
        %u2(u2 == 0) = mean(u2(u2 ~= 0));
    %else
    %    u2 = u1;
    %end
    s2 = size(u2);
    d2 = zeros(s2);

    % u3 and d3
    u3 = u2;
    d3 = zeros(s2);
    
    % u4 and d4
    % u4 will not exist for single spectral case
    if(options.tau2~=0)
        u4 = u2;
        d4 = zeros(s2);
    else
        u4=0;
        d4=0;
    end
    
    % initialize z
    z=zeros(s2);
    
    % Initialize residuals
    % for varying penalty parameter mu
    %r = zeros(s2);
    %s = zeros(s2);
else
    % u1 and d1
    u1 = (options.init);
    d1 = options.d.d1/options.mu;

    % u2 and d2
    u2 = (options.init);
    d2 = options.d.d2/options.mu;

    % u3 and d3
    u3 = options.init;
    d3 = options.d.d3/options.mu;
    
    % u4 and d4
    if(options.tau2 ~= 0)
        u4 = options.init;
        d4 = options.d.d4/options.mu;
    end
end

% Initialize total number of functions for ADMM
if(options.tau2 ~= 0)
    J = 3;
else
    J = 2;
end

% To avoid the same matrix inversion every time
% (K'K + 2I)^-1
if ~isfield(options, 'ktk')
    Kmat = K(sparse(eye(size(u3,1))));
    Kmat=sparse(Kmat);
    ktk = Kmat'*Kmat;
    ktk = sparse(ktk);
    ktk2Iinv = inv(ktk + J*sparse(eye(size(ktk,1))));
    ktk2Iinv = sparse(ktk2Iinv);
else
    ktk2Iinv = options.ktk;
end

% Main loop
o.verbose = 0;

%--------Plot f to check convergence------------------
% h1=figure();
% plot(1:options.max_iter,repmat(options.rel_tol,1,options.max_iter),'b-');
% hold on;
f_plot=[];
%-----------------------------------------------------

for t = 1:options.max_iter
    % ADMM - PIDAL-TV
    if options.verbose > 0
        fprintf('   a_pidal_tv2: Iteration %i/%i\n',t,options.max_iter);
    end
    
    % For convergence condition
    z_prev = z;
    u1_prev = u1;
    u2_prev = u2;
    u3_prev = u3;
    u4_prev = u4;
    
    zeta1 = u1 + d1;
    zeta2 = u2 + d2;
    zeta3 = u3 + d3;
    zeta4 = u4 + d4;
    
    gamma = Kt(zeta1) + zeta2 + zeta3 + zeta4;
    
    z = ktk2Iinv*gamma;
    
    % Likelihood function changes with estimation of a or b
    if(options.a == 1)
        nu1 = K(z) - d1;
        b_bhas = (T.*b)./rsf + rsf./options.mu - nu1;
        c_bhas = (T.*b - y)./options.mu - (T.*nu1.*b)./rsf;
        u1 = (0.5).*(-b_bhas + sqrt((b_bhas).^2 - 4.*c_bhas));
        clear b_bhas c_bhas;
    else
        nu1 = K(z) - d1;
        b_bhas = T./options.mu - nu1;
        c_bhas = -y./options.mu;
        u1 = (0.5).*(-b_bhas + sqrt((b_bhas).^2 - 4*c_bhas));
        clear b_bhas c_bhas;
    end
    
    nu2 = z - d2;
    %tvnorm=0;
    for l = 1:size(y,2)
        [sol, ~] = prox_tv(options.sq(nu2(:,l)),...
                             2*options.tau/options.mu,o);
        u2(:,l) = options.vc(sol);
    end
    
    % update tau
    %options.tau = (size(y,1)/k_tau + alph_tau - 1)/(tvnorm + beta_tau);

    nu3 = z - d3;
    u3 = max(0,nu3);
    
    
    if options.tau2 ~= 0
        nu4 = z - d4;
        [U,S,V] = svd(nu4);
        u4 = U*wthresh(S,'s',2*options.tau2/options.mu)*V';
        clear U S V;
    end
    
    d1 = d1 - K(z) + u1;
    d2 = d2 - z + u2;
    d3 = d3 - z + u3;
    if(options.tau2 ~= 0)
        d4 = d4 - z + u4;
    end
    
    % Update residuals
    r = z + Kt(u1) + u2 + u3 + u4;
    s = (options.mu).*(Kt(u1-u1_prev) + (u2-u2_prev) + (u3-u3_prev) + (u4-u4_prev));
    norm_s_temp = zeros(1,size(y,2));
    norm_r_temp = zeros(1,size(y,2));
    for l2=1:size(y,2)
        norm_s_temp(l2) = norm(s(:,l2));
        norm_r_temp(l2) = norm(r(:,l2));
    end
    norm_r = mean(norm_r_temp);
    norm_s = mean(norm_s_temp);
    
    % update mu
    if(mod(t,2)==0)
        if (norm_r>options.rho*norm_s)
            options.mu = options.tau_incr*options.mu;
            d1 = d1./options.tau_incr;
            d2 = d2./options.tau_incr;
            d3 = d3./options.tau_incr;
            d4 = d4./options.tau_incr;
        elseif (norm_s>options.rho*norm_r)
            options.mu = options.mu/options.tau_decr;
            d1 = d1.*options.tau_decr;
            d2 = d2.*options.tau_decr;
            d3 = d3.*options.tau_decr;
            d4 = d4.*options.tau_decr;
        else
            options.mu = options.mu;
        end
    end
    
    % Check for convergence
    f_temp = zeros(1,size(y,2));
    for l1 = 1:size(y,2)
        f_temp(l1) = f2(z(:,l1),z_prev(:,l1));
    end
    f = mean(f_temp);
    f_plot = [f_plot,f];

    if f < options.rel_tol && t >= options.min_iter
        fprintf('   a_pidal_tv2: Break condition satisfied\n');
        break;
    end
    
    %---------plot f---------------
%     plot(t,f,'go-');
%     hold on;
%     if(mod(t,1000)==0)
%         figure(h1);
%     end
    %------------------------------
end

xsol = z;

figure,plot(2:size(f_plot,2),f_plot(2:end),'-');

d.d1 = d1*options.mu;
d.d2 = d2*options.mu;
d.d3 = d3*options.mu;
%d.d4 = d4*options.mu;
end