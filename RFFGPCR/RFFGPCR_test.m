function output = RFFGPCR_test(model,Xts,Yts,zts)
% - model: posterior distributions and hyperparameter estimations from
%       train (see the object returned by RFFGPCR_train)
% - Xts: (Nts x D) matrix of features for the test set
% - Yts: (optional) 0/1 annotations for the test instances (negative numbers for missing values)
% - zts: True labels for test set (in order to compute the classification metrics)
mask = double(Yts>=0);
annot = sum(mask(:))>0;

muRHO = model.muRHO;
sigma_var = realsqrt(model.sigma_sq);
W = model.W;
L = model.L;

DF = length(muRHO)/2;
Nts = size(Xts,1);

% Fourier features for test points
Phi_ts = zeros(Nts,2*DF);
XtsW = Xts*W';
Phi_ts(:,1:2:(2*DF)) = cos(XtsW/sigma_var)./realsqrt(DF);
Phi_ts(:,2:2:(2*DF)) = sin(XtsW/sigma_var)./realsqrt(DF);

% means and variances for f*'s
mean_test = Phi_ts*muRHO; 
var_test = sum((L\(Phi_ts')).^2,1)';  

% probability of z=1 if there are no test annotations
sigmoid = 1./(1+exp(-mean_test./realsqrt(1+0.125*pi*var_test)));

% Products coming from the test annotations
if annot
    alpha_mean = model.alpha_a./(model.alpha_a+model.alpha_b);
    beta_mean = model.beta_a./(model.beta_a+model.beta_b);
    prod1 = exp(sum(Yts.*mask.*log(repmat(alpha_mean',Nts,1))+...
        (1-Yts).*mask.*log(repmat(1-alpha_mean',Nts,1)),2));
    prod0 = exp(sum(Yts.*mask.*log(repmat(1-beta_mean',Nts,1))+...
        (1-Yts).*mask.*log(repmat(beta_mean',Nts,1)),2));
else
    prod1 = ones(Nts,1);
    prod0 = ones(Nts,1);
end

% Probabilities of z=1 and predictions
probabilities = (sigmoid.*prod1)./(sigmoid.*prod1+(1-sigmoid).*prod0);
z_predic = probabilities >= 0.5;

% Measuring results
res = measures(zts,z_predic);
[res.roc.X,res.roc.Y,res.roc.T,res.AUC] = perfcurve(zts,probabilities,'1');
res.ML = (sum(probabilities(zts==1))+sum(1-probabilities(zts==0)))/length(zts);

% Preparing output
output.res = res;
output.prob = probabilities;
output.z_predic = z_predic;
end

function results = measures(y_test,y_predic)
%% Only binary case
CM = zeros(2,2);
ind = (y_test==1);

%% Confussion Matrix

    % True Positives
        CM(1,1) = sum(y_predic(ind)==1);
    % True Negatives
        CM(2,2) = sum(y_predic(not(ind))==0);
    % False Positives
        CM(1,2) = sum(y_predic(not(ind))==1);
    % False Negatives
        CM(2,1) = sum(y_predic(ind)==0);

results.CM = CM;

%% Overall Accuracy
    results.OA = 100*(sum(diag(CM)) / sum(sum(CM)));

%% Precision and Recall
    PR(1) = CM(1,1)/(CM(1,1) + CM(1,2));
    PR(2) = CM(1,1)/(CM(1,1) + CM(2,1));
    
    results.Pre_Rec = PR;
%% F-score
    results.Fscore = 2*PR(1)*PR(2)/sum(PR);
    
%% TPR and FPR

    T_F(1) = CM(1,1)/(CM(1,1) + CM(2,1));
    T_F(2) = CM(1,2)/(CM(1,2) + CM(2,2));

    results.TF_ratio = T_F;
end