for fileDir = ['active_waic_compare_sess1.csv', 'active_waic_compare_sess2.csv', 'passive_waic_compare_sess1.csv', 'passive_waic_compare_sess2.csv']
	fileDir = sprintf('waic_compare_sess1.csv');
	waic = csvread(fileDir);

	% transfer into the loss domain 
	L = -waic';

	L = L(1:2,:);

	% run the group-level Bayesian model selection, Stephan et al., 2009 
	[posterior,out] = VBA_groupBMC(L);
	f  = out.Ef;
	EP = out.ep;
end

