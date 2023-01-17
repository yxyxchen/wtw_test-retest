
nExp = 3;
% 
% % compare the four variants of the RL model
% EF4_ = zeros([3, 4]); % expeted model frequencies 
% PXP4_ = zeros([3, 4]); % protected prob of exceedance 
% for i = 1 : nExp
% 	fileDir = sprintf('../../genData/wtw_exp%d/waic.csv', i);
%     % fileDir = sprintf('waic_exp%d.csv', i);
% 	waic = csvread(fileDir);
% 	
% 	% transfer into the loss domain 
% 	L = -waic';
% 
% 	% exclude participants with disconvergent model fitting results 
% 	L = L(1:4, all(L(7:10,:)));
% 
% 	% run the group-level Bayesian model selection, Stephan et al., 2009 
% 	[posterior,out] = VBA_groupBMC(L);
% 	f  = out.Ef;
% 	EP = out.ep;
% 	EF4_(i, :)  = out.Ef;
% 	PXP4_(i, :) = out.ep;
% end 
% csvwrite('../../genData/EF4.csv', EF4_, 3, 4);
% csvwrite('../../genData/PXP4.csv', PXP4_, 3, 4);

% add the non-learning benchmark model
EF5_ = zeros([3, 5]); % expeted model frequencies 
PXP5_ = zeros([3, 5]); % protected prob of exceedance 
for i = 1 : nExp
	fileDir = sprintf('../../genData/wtw_exp%d/waic.csv', i);
	waic = csvread(fileDir);
	
	% transfer into the loss domain 
	L = -waic';

	% exclude participants with disconvergent model fitting results 
	L = L(1:5, all(L(7:11,:)));

	% run the group-level Bayesian model selection, Stephan et al., 2009 
	[posterior,out] = VBA_groupBMC(L);
	f  = out.Ef;
	EP = out.ep;
	EF5_(i, :)  = out.Ef;
	PXP5_(i, :) = out.ep;
end 
csvwrite('../../genData/EF5.csv', EF5_, 3, 5);
csvwrite('../../genData/PXP5.csv', PXP5_, 3, 5);