function model = RFFGPCR_train(X,Y,options)
% X: (NxD) matrix of features for the train set. Each row is an instance
% Y: (NxR) matrix of annotations (0/1). Negative numbers are used for missing values.
% options: struct to indicate the algorithm parameters. It contains the fields:
%    - maxiter: Maximum number of loops for the mean-field process.
%    - maxiter_par: Maximum number of iterations for the gradient descent
%           in the (W,gamma_var) update
%    - DF: Number of Fourier frequencies to use
%    - W: manual Fourier frequencies (if not provided, they are sampled from N(0,I))
%    - thr: threshold for the hyperparameters convergence
%    - iterOut: number of iterations in a row that we require the
%           parameters not to satisfy the stopping criterion before it
%           becomes valid (otherwise, specially in big problems, the
%           stopping criterion is satisfied in the first step because
%           hyperparameters do not move much at the beginning)
%    - alpha_a0,alpha_b0,beta_a0,beta_b0: initial values for alpha and beta parameters
%% Defining variables and initializing parameters
[N, D] = size(X);
R = size(Y,2);
Y1mask = double(Y==1);
Y0mask = double(Y==0);

alpha_a0 = options.alpha_a0;
alpha_b0 = options.alpha_b0;
beta_a0 = options.beta_a0;
beta_b0 = options.beta_b0;

if isequal(alpha_a0,1)
    alpha_a0 = ones(R,1);
end
if isequal(alpha_b0,1)
    alpha_b0 = ones(R,1);
end
if isequal(beta_a0,1)
    beta_a0 = ones(R,1);
end
if isequal(beta_b0,1)
    beta_b0 = ones(R,1);
end
alphaKL_cte = betaln(alpha_a0,alpha_b0);
betaKL_cte = betaln(beta_a0,beta_b0);
% Initializing xi (not necessary)
xi = ones(N,1);

% Initializing (sigma_var,gamma_var) and the corresponding Phi(necessary)
gamma_var = 10;
sigma_var = estimateSigma(X);
if(isfield(options,'W'))
    W = options.W;
    DF = size(W,1);
else
    DF = options.DF;
    W = randn(DF,D);
end
XW = X*W';
Phi = zeros(N,2*DF);
Phi(:,1:2:(2*DF)) = cos(XW/sigma_var)./realsqrt(DF);
Phi(:,2:2:(2*DF)) = sin(XW/sigma_var)./realsqrt(DF);

% Initializing rho and the corresponding LiPhi (L^{-1}*Phi') and PhimuRHO (necessary)
%SigmaRHOi = eye(2*DF)./gamma_var;
muRHO = zeros(2*DF,1);
LiPhi = realsqrt(gamma_var)*Phi';
PhimuRHO = zeros(N,1);

% Initializing alpha and beta (necessary)
alpha_a = alpha_a0;
alpha_b = alpha_b0;
beta_a = beta_a0;
beta_b = beta_b0;
mean_alpha = alpha_a./(alpha_a+alpha_b);
mean_beta = beta_a./(beta_a+beta_b);

% Initializing z (probability of z=1) (necessary)
z = sum(Y1mask,2)./sum(Y0mask+Y1mask,2);
v = z-0.5;

stuck = true;
counter = 0;
ELBO_LB = [];
fprintf('Iter |ELBO_LB| \t xi \t\t gamma \t\t sigma \t\t muRHO \t\t mean_alpha \t mean_beta \t\t z| \t gamma \t\t sigma \t mean(muRHO) \t mean(z)\n');
fprintf('%i |%.7f| \t %.7f \t %.7f \t %.7f \t %.7f \t %.7f \t %.7f \t %.7f | \t %.7f \t %.7f \t %.7f \t %.7f \n',...
    [0,double(0),double(0),double(0),double(0),double(0),double(0),double(0),double(0)...
    gamma_var,sigma_var,mean(muRHO),mean(z)])
for iter = 1:options.maxiter
    % Keeping old values
    xi_old = xi;
    z_old = z;
    gamma_var_old = gamma_var;
    sigma_var_old = sigma_var;
    muRHO_old = muRHO;
    mean_alpha_old = mean_alpha;
    mean_beta_old = mean_beta;
    
    % Updating xi
    xi = realsqrt(sum(LiPhi.^2,1)'+(PhimuRHO).^2); % LiPhi (L^{-1}*Phi')
    g_xi = (1./(2*xi)) .* ((1./(1+exp(-xi))) - 0.5); 
    
    % Updating (sigma_var,gamma_var) and the corresponding Phi
    logGammaSigma = minimize([log(gamma_var);log(sigma_var)],@minus_posterior,-options.maxiter_par,XW,v,g_xi);
    gamma_var = exp(logGammaSigma(1));
    sigma_var = exp(logGammaSigma(2));
    Phi(:,1:2:(2*DF)) = cos(XW/sigma_var)./realsqrt(DF);
    Phi(:,2:2:(2*DF)) = sin(XW/sigma_var)./realsqrt(DF);
    
    % Updating rho
    SigmaRHOi = eye(2*DF)./gamma_var + Phi'*bsxfun(@times,2*g_xi,Phi);
    L = chol(SigmaRHOi,'lower');
    LiPhi = L\Phi'; % L^{-1}*Phi' (used later in xi update)
    muRHO = (L')\(LiPhi*v);
    PhimuRHO = Phi*muRHO;
    
    % Updating alpha beta
    alpha_a = alpha_a0 + sum(repmat(z,1,R).*Y1mask)';
    alpha_b = alpha_b0 + sum(repmat(z,1,R).*Y0mask)';
    beta_a = beta_a0 + sum(repmat(1-z,1,R).*Y0mask)';
    beta_b = beta_b0 + sum(repmat(1-z,1,R).*Y1mask)';
    mean_alpha = alpha_a./(alpha_a+alpha_b);
    mean_beta = beta_a./(beta_a+beta_b);
    
    % Updating z
    log_beta = psi(beta_a) - psi(beta_a+beta_b);
    log_1mbeta = psi(beta_b) - psi(beta_a+beta_b);
    log_alpha = psi(alpha_a) - psi(alpha_a+alpha_b);
    log_1malpha = psi(alpha_b) - psi(alpha_a+alpha_b);
    
    unnor_log_z0 = sum(repmat(log_1mbeta',N,1).*Y1mask + repmat(log_beta',N,1).*Y0mask,2);
    unnor_log_z1 = PhimuRHO + sum(repmat(log_alpha',N,1).*Y1mask + repmat(log_1malpha',N,1).*Y0mask,2);
    z = exp(unnor_log_z1)./(exp(unnor_log_z1)+exp(unnor_log_z0));
    v = z-0.5;
    
    ELBO_LB_KLalpha = sum(-betaln(alpha_a,alpha_b)+(alpha_a-alpha_a0).*log_alpha+(alpha_b-alpha_b0).*log_1malpha+alphaKL_cte);
    ELBO_LB_KLbeta = sum(-betaln(beta_a,beta_b)+(beta_a-beta_a0).*log_beta+(beta_b-beta_b0).*log_1mbeta+betaKL_cte);
    ELBO_LB_KLnormal = 0.5*(sum(sum((L\eye(2*DF)).^2))+muRHO'*muRHO)./gamma_var-DF+DF*log(gamma_var)+sum(log(diag(L)));
    ELBO_LB_nKL1 = z'*Y1mask*log_alpha+z'*Y0mask*log_1malpha+(1-z)'*Y0mask*log_beta+(1-z)'*Y1mask*log_1mbeta;
    ELBO_LB_nkL2 = (z-0.5)'*Phi*muRHO-sum(sum(bsxfun(@times,LiPhi,realsqrt(g_xi)').^2))-PhimuRHO'*(g_xi.*PhimuRHO)+sum(g_xi.*(xi.^2)+0.5*xi-log(1+exp(xi)));
    auxi = z.*log(z)+(1-z).*log(1-z);
    auxi(isnan(auxi))=0;
    ELBO_z_mentropy = sum(auxi);
    ELBO_LB_i = -ELBO_LB_KLalpha-ELBO_LB_KLbeta-ELBO_LB_KLnormal+ELBO_LB_nKL1+ELBO_LB_nkL2-ELBO_z_mentropy;
    ELBO_LB = [ELBO_LB; ELBO_LB_i];
    % Check convergence
    xi_c = norm(xi-xi_old)/norm(xi_old);
    z_c = norm(z-z_old)/norm(z_old);
    gamma_var_c = abs(gamma_var-gamma_var_old)/abs(gamma_var_old);
    sigma_var_c = abs(sigma_var-sigma_var_old)/abs(sigma_var_old);
    muRHO_c = norm(muRHO-muRHO_old)/norm(muRHO_old);
    mean_alpha_c = norm(mean_alpha-mean_alpha_old)/norm(mean_alpha_old);
    mean_beta_c = norm(mean_beta-mean_beta_old)/norm(mean_beta_old);
    
    fprintf('%i | %.7f | \t %.7f \t %.7f \t %.7f \t %.7f \t %.7f \t %.7f \t %.7f | \t %.7f \t %.7f \t %.7f \t %.7f \n',...
        [iter,ELBO_LB_i,xi_c,gamma_var_c,sigma_var_c,muRHO_c,mean_alpha_c,mean_beta_c,z_c...
        gamma_var,sigma_var,mean(muRHO),mean(z)])
    
    if isequal(options.stopCriteria,'ELBO')
        if iter == 1
            satisfiedStop = false;
        else
            satisfiedStop = ((ELBO_LB_i-ELBO_LB(iter-1))<options.thr_ELBO);
        end
    elseif isequal(options.stopCriteria,'parameters')
        satisfiedStop = (xi_c<options.thr && gamma_var_c<options.thr && sigma_var_c<options.thr &&  ...
            muRHO_c<options.thr_muRHO && z_c<options.thr && mean_alpha_c<options.thr && mean_beta_c<options.thr);
    end
    
    if satisfiedStop
        if(~stuck)
            fprintf('Iter |ELBO_LB| \t xi \t\t gamma \t\t sigma \t\t muRHO \t\t mean_alpha \t mean_beta \t\t z| \t gamma \t\t sigma \t mean(muRHO) \t mean(z)\n\n');
            fprintf('Convergence reached at iteration %i.\n',iter)            
            break;
        else
            counter = 0;
        end
    else
        counter = counter + 1;
        if(counter==options.iterOut)
            stuck = false;
        end
    end
end
model.xi = xi;
model.gamma = gamma_var;
model.sigma = sigma_var;
model.sigma_sq = sigma_var^2;
model.W = W;
model.muRHO = muRHO;
model.L = L;
model.alpha_a = alpha_a;
model.alpha_b = alpha_b;
model.beta_a = beta_a;
model.beta_b = beta_b;
model.z = z;
model.ELBO_LB = ELBO_LB;
end

function [f, df] = minus_posterior(logGammaSigma,XW,v,g_xi)
N = size(XW,1);
DF = size(XW,2);
df = zeros(size(logGammaSigma));
gamma_var = exp(logGammaSigma(1));
sigma_var = exp(logGammaSigma(2));

Phi = zeros(N,2*DF);
Phi(:,1:2:(2*DF)) = cos(XW/sigma_var)./realsqrt(DF);
Phi(:,2:2:(2*DF)) = sin(XW/sigma_var)./realsqrt(DF);
dPhidsigma = zeros(N,2*DF);
dPhidsigma(:,1:2:(2*DF)) = XW.*sin(XW/sigma_var)./(realsqrt(DF)*sigma_var^2);
dPhidsigma(:,2:2:(2*DF)) = -XW.*cos(XW/sigma_var)./(realsqrt(DF)*sigma_var^2);

% Some needed common expressions
aux0 = bsxfun(@times,2*g_xi,Phi); % (2A)Phi
L = chol(eye(2*DF)./gamma_var + Phi'*aux0,'lower');
aux1 = L\(Phi'*v);  % L^{-1}Phi'v
aux2 = (L')\aux1;   % (K+I)^{-1}Phi'v
aux3 = L'\(L\(aux0'));   % (K+I)^{-1}Phi'(2A)
aux4 = (aux2'*Phi').*(2*g_xi)';  % v'Phi(K+I)^{-1}Phi'(2A)
% Calculating f
f = 2*DF*log(gamma_var)+2*sum(log(abs(diag(L))))-aux1'*aux1;
% gradient for gamma
dfdgamma = sum(sum(aux3'.*Phi))/gamma_var -(aux2'*aux2)./(gamma_var^2);
df(1) = gamma_var*dfdgamma; %We need the derivative with respect to log(\gamma) 
% gradient for sigma
dfdsigma = 2*sum(sum(aux3'.*dPhidsigma))-...
    2*v'*dPhidsigma*aux2+...
    2*aux4*dPhidsigma*aux2;
df(2) = sigma_var*dfdsigma; %We need the derivative with respect to log(\sigma)
end

function sigma = estimateSigma(X)
% Subsampling
n = size(X,1);
if n>1000
    idx = randperm(n);
    X = X(idx(1:1000),:);
end
d = pdist(X);
sigma = mean(d(d>0));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MINIMIZE, UNWRAP, WRAP (from GPML) %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [X, fX, i] = minimize(X, f, length, varargin)

INT = 0.1;    % don't reevaluate within 0.1 of the limit of the current bracket
EXT = 3.0;                  % extrapolate maximum 3 times the current step-size
MAX = 20;                         % max 20 function evaluations per line search
RATIO = 10;                                       % maximum allowed slope ratio
SIG = 0.1; RHO = SIG/2; 
if max(size(length)) == 2, red=length(2); length=length(1); else red=1; end
if length>0, S='Linesearch'; else S='Function evaluation'; end 

i = 0;                                            % zero the run length counter
ls_failed = 0;                             % no previous line search has failed
[f0 df0] = feval(f, X, varargin{:});          % get function value and gradient
Z = X; X = unwrap(X); df0 = unwrap(df0);
%fprintf('%s %6i;  Value %4.6e\r', S, i, f0);
if exist('fflush','builtin') fflush(stdout); end
fX = f0;
i = i + (length<0);                                            % count epochs?!
s = -df0; d0 = -s'*s;           % initial search direction (steepest) and slope
x3 = red/(1-d0);                                  % initial step is red/(|s|+1)

while i < abs(length)                                      % while not finished
  i = i + (length>0);                                      % count iterations?!

  X0 = X; F0 = f0; dF0 = df0;                   % make a copy of current values
  if length>0, M = MAX; else M = min(MAX, -length-i); end

  while 1                             % keep extrapolating as long as necessary
    x2 = 0; f2 = f0; d2 = d0; f3 = f0; df3 = df0;
    success = 0;
    while ~success && M > 0
      try
        M = M - 1; i = i + (length<0);                         % count epochs?!
        
        [f3 df3] = feval(f, rewrap(Z,X+x3*s), varargin{:});
        df3 = unwrap(df3);
        if isnan(f3) || isinf(f3) || any(isnan(df3)+isinf(df3)), error(' '),end
        success = 1;
      catch                                % catch any error which occured in f
        x3 = (x2+x3)/2;                                  % bisect and try again
      end
    end
    if f3 < F0, X0 = X+x3*s; F0 = f3; dF0 = df3; end         % keep best values
    d3 = df3'*s;                                                    % new slope
    if d3 > SIG*d0 || f3 > f0+x3*RHO*d0 || M == 0  % are we done extrapolating?
      break
    end
    x1 = x2; f1 = f2; d1 = d2;                        % move point 2 to point 1
    x2 = x3; f2 = f3; d2 = d3;                        % move point 3 to point 2
    A = 6*(f1-f2)+3*(d2+d1)*(x2-x1);                 % make cubic extrapolation
    B = 3*(f2-f1)-(2*d1+d2)*(x2-x1);
    x3 = x1-d1*(x2-x1)^2/(B+sqrt(B*B-A*d1*(x2-x1))); % num. error possible, ok!
    if ~isreal(x3) || isnan(x3) || isinf(x3) || x3 < 0 % num prob | wrong sign?
      x3 = x2*EXT;                                 % extrapolate maximum amount
    elseif x3 > x2*EXT                  % new point beyond extrapolation limit?
      x3 = x2*EXT;                                 % extrapolate maximum amount
    elseif x3 < x2+INT*(x2-x1)         % new point too close to previous point?
      x3 = x2+INT*(x2-x1);
    end
  end                                                       % end extrapolation

  while (abs(d3) > -SIG*d0 || f3 > f0+x3*RHO*d0) && M > 0  % keep interpolating
    if d3 > 0 || f3 > f0+x3*RHO*d0                         % choose subinterval
      x4 = x3; f4 = f3; d4 = d3;                      % move point 3 to point 4
    else
      x2 = x3; f2 = f3; d2 = d3;                      % move point 3 to point 2
    end
    if f4 > f0           
      x3 = x2-(0.5*d2*(x4-x2)^2)/(f4-f2-d2*(x4-x2));  % quadratic interpolation
    else
      A = 6*(f2-f4)/(x4-x2)+3*(d4+d2);                    % cubic interpolation
      B = 3*(f4-f2)-(2*d2+d4)*(x4-x2);
      x3 = x2+(sqrt(B*B-A*d2*(x4-x2)^2)-B)/A;        % num. error possible, ok!
    end
    if isnan(x3) || isinf(x3)
      x3 = (x2+x4)/2;               % if we had a numerical problem then bisect
    end
    x3 = max(min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2));  % don't accept too close
    [f3 df3] = feval(f, rewrap(Z,X+x3*s), varargin{:});
    df3 = unwrap(df3);
    if f3 < F0, X0 = X+x3*s; F0 = f3; dF0 = df3; end         % keep best values
    M = M - 1; i = i + (length<0);                             % count epochs?!
    d3 = df3'*s;                                                    % new slope
  end                                                       % end interpolation

  if abs(d3) < -SIG*d0 && f3 < f0+x3*RHO*d0          % if line search succeeded
    X = X+x3*s; f0 = f3; fX = [fX' f0]';                     % update variables
    %fprintf('%s %6i;  Value %4.6e\r', S, i, f0);
    if exist('fflush','builtin') fflush(stdout); end
    s = (df3'*df3-df0'*df3)/(df0'*df0)*s - df3;   % Polack-Ribiere CG direction
    df0 = df3;                                               % swap derivatives
    d3 = d0; d0 = df0'*s;
    if d0 > 0                                      % new slope must be negative
      s = -df0; d0 = -s'*s;                  % otherwise use steepest direction
    end
    x3 = x3 * min(RATIO, d3/(d0-realmin));          % slope ratio but max RATIO
    ls_failed = 0;                              % this line search did not fail
  else
    X = X0; f0 = F0; df0 = dF0;                     % restore best point so far
    if ls_failed || i > abs(length)         % line search failed twice in a row
      break;                             % or we ran out of time, so we give up
    end
    s = -df0; d0 = -s'*s;                                        % try steepest
    x3 = 1/(1-d0);                     
    ls_failed = 1;                                    % this line search failed
  end
end
X = rewrap(Z,X); 
%fprintf('\n'); 
if exist('fflush','builtin') fflush(stdout); end
end

function v = unwrap(s)
% Extract the numerical values from "s" into the column vector "v". The
% variable "s" can be of any type, including struct and cell array.
% Non-numerical elements are ignored. See also the reverse rewrap.m. 
v = [];   
if isnumeric(s)
  v = s(:);                        % numeric values are recast to column vector
elseif isstruct(s)
  v = unwrap(struct2cell(orderfields(s))); % alphabetize, conv to cell, recurse
elseif iscell(s)
  for i = 1:numel(s)             % cell array elements are handled sequentially
    v = [v; unwrap(s{i})];
  end
end                                                   % other types are ignored
end

function [s v] = rewrap(s, v)
% Map the numerical elements in the vector "v" onto the variables "s" which can
% be of any type. The number of numerical elements must match; on exit "v"
% should be empty. Non-numerical entries are just copied. See also unwrap.m.
if isnumeric(s)
  if numel(v) < numel(s)
    error('The vector for conversion contains too few elements')
  end
  s = reshape(v(1:numel(s)), size(s));            % numeric values are reshaped
  v = v(numel(s)+1:end);                        % remaining arguments passed on
elseif isstruct(s) 
  [s p] = orderfields(s); p(p) = 1:numel(p);      % alphabetize, store ordering
  [t v] = rewrap(struct2cell(s), v);                 % convert to cell, recurse
  s = orderfields(cell2struct(t,fieldnames(s),1),p);  % conv to struct, reorder
elseif iscell(s)
  for i = 1:numel(s)             % cell array elements are handled sequentially 
    [s{i} v] = rewrap(s{i}, v);
  end
end
end