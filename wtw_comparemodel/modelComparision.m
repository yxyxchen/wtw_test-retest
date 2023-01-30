


fileDir = sprintf('waic_compare_sess1.csv');
waic = csvread(fileDir);

% transfer into the loss domain 
L = -waic';

L = L(1:2,:);

% exclude participants with disconvergent model fitting results 
% L = L(1:5, all(L(7:11,:)));

% run the group-level Bayesian model selection, Stephan et al., 2009 
[posterior,out] = VBA_groupBMC(L);
f  = out.Ef;
EP = out.ep;

csvwrite('../../genData/EF5.csv', EF5_, 3, 5);
csvwrite('../../genData/PXP5.csv', PXP5_, 3, 5);
