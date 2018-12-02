function [xsol,d] = a_pidal_tvnn(y,rsf,options)
% Implementation of the adapted PIDAL-TV ADMM algorithm for restoration of
% Poissonian images described in:
% 
% Figueiredo et al., "Restoration of Poissonian Images Using
% Alternating Direction Optimization".
% 
% And adapted for intensity estimation.
% 
% y - measurements
% K and Kt - function handles for measurement operator and its adjoint
% Kmat - measurement matrix
% P and Pt - function handles for sparsity basis and its adjoint
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
% if ~isfield(options, 'rel_tol2'), options.rel_tol2 = 1e-4; end
if ~isfield(options, 'max_iter'), options.max_iter = 200; end
if ~isfield(options, 'min_iter'), options.min_iter = 1; end
if ~isfield(options, 'mu'), options.mu = 1e2; end
if ~isfield(options, 'tau'), options.tau = 2e1; end
if ~isfield(options, 'tau2'), options.tau2 = 0; end

if options.verbose > 0
    fprintf('   a_pidal_tvnn: Starting...\n');
end

% %Useful functions.
% sc = @(z) z*min(epsilon/norm(z(:)), 1); % scaling
% % xtp1 = @(xt,st,nt,vt) Psi(wthresh(Psit(xt - options.delta*real(Phi(st + nt - vt))),'s',options.delta/options.rho));
% xtp1 = @(xt,st,nt,vt) max(0,Psi(wthresh(Psit(xt - options.delta*real(Phi(st + nt - vt))),'s',options.delta/options.rho)));
% stp1 = @(xt) Phit(xt) - y;
% ntp1 = @(vt,st) sc(vt-st);
% vtp1 = @(st,nt,vt) vt - (st + nt);
% TV = @(xt) sum(sum(abs(xt(2:end,:)-xt(1:end-1,:)))) + ...
%     sum(sum(abs(xt(:,2:end)-xt(:,1:end-1))));
% f = @(xt) sum(sum(xt - log(T*b+xt+eps).*y)) + TV(xt);
f2 = @(xt,xtm1) norm(xt(:) - xtm1(:))/(norm(xt(:))+eps);

%%Initializations.

if ~isfield(options, 'init')
    % u1 and d1
    s1 = size(y);
    u1 = y./rsf;
    d1 = zeros(s1);

    % u2 and d2
    u2 = u1;
%     u2(u2 == 0) = mean(u2(u2 ~= 0));
    s2 = size(u2);
    d2 = zeros(s2);

    % u3 and d3
    u3 = u2;
    d3 = zeros(s2);
    
    u4 = u1;
    d4 = zeros(s2);
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
    u4 = options.init;
    d4 = options.d.d4/options.mu;
end

% To avoid the same matrix inversion every time
% if ~isfield(options, 'ktk')
%     Kmat = K(eye(size(u3,1)));
%     ktk = Kmat'*Kmat;
%     ktk2Iinv = inv(ktk + 2*eye(size(ktk,1)));
% else
%     ktk2Iinv = options.ktk;
% end

%% Main loop
o.verbose = 0;

% ft = Inf;
% sc_ratio = 10;
% sc = 2;
for t = 1:options.max_iter
    % ADMM - PIDAL-FA
    
    if options.verbose > 0
        fprintf('   a_pidal_tvnn: Iteration %i/%i\n',t,options.max_iter);
    end
    
%     u1_prev = u1;
%     u2_prev = u2;
    u3_prev = u3;
    
    zeta1 = u1 + d1;
    zeta2 = u2 + d2;
    zeta3 = u3 + d3;
    zeta4 = u4 + d4;
    
%     gamma = zeta1 + zeta2 + zeta3 + zeta4;
    
%     z = ktk2Iinv*gamma;
%     z = gamma/4; % fully sampled
    z = (zeta1 + zeta2 + zeta3 + zeta4)/4; % fully sampled
    
    nu1 = z - d1;
    b_bhas = rsf/options.mu - nu1;
    c_bhas = -y/options.mu;
    u1 = 0.5*(-b_bhas + sqrt((b_bhas).^2 - 4*c_bhas));
    clear b_bhas c_bhas;
    
    nu2 = z - d2;
    for l = 1:size(y,2)
        u2(:,l) = options.vc(prox_tv(options.sq(nu2(:,l)),...
                             2*options.tau/options.mu,o));
    end

    nu3 = z - d3;
    u3 = max(0,nu3);
    
    if options.tau2 ~= 0
        nu4 = z - d4;
        [U,S,V] = svd(nu4);
        u4 = U*wthresh(S,'s',2*options.tau2/options.mu)*V';
        clear U S V;
    else
        u4 = z - d4;
    end
    
    
    
    d1 = d1 - z + u1;
    d2 = d2 - z + u2;
    d3 = d3 - z + u3;
    d4 = d4 - z + u4;
    
    
%     if mod(t,10) == 0
% %         r = sqrt(norm(u1 - K(u3),'fro')^2 + norm(u2 - P(u3),'fro')^2);
%         r = sqrt(norm(u1 - K(z),'fro')^2 + norm(u2 - (z),'fro')^2 + norm(u3 - z,'fro')^2);
% %         rv = [rv r];
% 
% %         s = options.mu*sqrt(norm(K(u3-u3_prev),'fro')^2 + norm(P(u3-u3_prev),'fro')^2);
%         s = options.mu*sqrt(norm(u1-u1_prev,'fro')^2 + norm(u2-u2_prev,'fro')^2 + norm(u3-u3_prev,'fro')^2);
% %         sv = [sv s];
% 
%         if r > sc_ratio*s
%             options.mu = options.mu*sc;
%             d1 = d1*sc;
%             d2 = d2*sc;
%             d3 = d3*sc;
% %             d4 = d4*sc;
%         elseif s > sc_ratio*r
%             options.mu = options.mu/sc;
%             d1 = d1/sc;
%             d2 = d2/sc;
%             d3 = d3/sc;
% %             d4 = d4/sc;
%         end
% 
%     end
    
    % Break conditions
%     fprev = ft;
%     ft = f(u3)
    
%     val = abs(ft - fprev)/ft
%     b1 = (abs(ft - fprev)/ft) < options.rel_tol;
%     b2 = norm(s(:)) <= epsilon; %% CHANGE S
%     b3 = (norm(z(:) - xprev(:))/norm(z(:))) < options.rel_tol2;
    
%     if options.verbose == 2
%         disp(['iter = ', num2str(t)])
%         disp(['f = ', num2str(ft)])
%         disp(['norm(s) = ', num2str(norm(s(:)))])
%     end
    
%     if (b1 && b2) || b3
    f = f2(u3,u3_prev);
%     if abs((fprev-f)/f) < options.rel_tol && t >= options.min_iter
    if f < options.rel_tol && t >= options.min_iter
        fprintf('   a_pidal_tvnn: Break condition satisfied\n');
        break;
    end

end

% fval = ft;
% fval = 0;


xsol = u3;

d.d1 = d1*options.mu;
d.d2 = d2*options.mu;
d.d3 = d3*options.mu;
d.d4 = d4*options.mu;

end























