
########################### import modules ############################
import pandas as pd
import numpy as np
import os
import glob
import importlib
import re
import matplotlib.pyplot as plt
import itertools
import copy # pay attention to copy 
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.interpolate import interp1d
import code
# my customized modules
import subFxs
from subFxs import analysisFxs
from subFxs import expParas
from subFxs import modelFxs
import seaborn as sns
import scipy.stats as stats
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from subFxs import simFxs 
from subFxs import normFxs
from subFxs import loadFxs
from subFxs import figFxs
from subFxs import analysisFxs
from datetime import datetime as dt
import scipy 
import statsmodels.formula.api as smf
from plotnine import ggplot, aes, facet_grid, labs, geom_point, geom_errorbar, geom_text, position_dodge, scale_fill_manual, labs, theme_classic, ggsave, geom_bar, scale_x_discrete
from scipy.stats import mannwhitneyu
from sklearn.linear_model import Lasso
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
import sklearn

# plot styles
plt.style.use('classic')
sns.set(font_scale = 1)
sns.set_style("white")
condition_palette = ["#762a83", "#1b7837"]
UPPS_subscales = ["NU", "PU", "PM", "PS", "SS"]
BIS_l1_subscales = ["Attentional", "Motor", "Nonplanning"]
BIS_l2_subscales = ["attention", "cogstable", "motor", "perseverance", "selfcontrol", "cogcomplex"]

# passive version
statsdf_ = []
selfdf_ = [] 
for expname in ["active", "passive"]:
	hdrdata_sess1, trialdata_sess1_ = loadFxs.group_quality_check(expname, 1, plot_quality_check = True)
	hdrdata_sess2, trialdata_sess2_ = loadFxs.group_quality_check(expname, 2, plot_quality_check = True)
	########### let me only include participants complete both sessions
	hdrdata_sess1 = hdrdata_sess1[np.isin(hdrdata_sess1["id"], hdrdata_sess2["id"])]
	trialdata_sess1_ = {x: y for x,y in trialdata_sess1_.items() if x[0] in hdrdata_sess2["id"].values}
	############ conduct behavioral analysis ######
	s1_stats, s1_Psurv_b1_, s1_Psurv_b2_, s1_WTW_ = analysisFxs.group_MF(trialdata_sess1_, plot_each = False)   
	s2_stats, s2_Psurv_b1_, s2_Psurv_b2_, s2_WTW_ = analysisFxs.group_MF(trialdata_sess2_, plot_each = False)   
	s1_df = analysisFxs.pivot_by_condition(s1_stats)
	s2_df = analysisFxs.pivot_by_condition(s2_stats)
	statsdf = analysisFxs.agg_across_sessions(s1_df, s2_df)
	statsdf_.append(statsdf)
	# self
	####################### analyze selfreport data ####################
	if expname == "passive":
		s1_selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
		s2_selfdf = loadFxs.parse_group_selfreport(expname, 2, isplot = False)
		s1_selfdf = s1_selfdf[np.isin(s1_selfdf["id"], s2_selfdf["id"])]
		selfdf = analysisFxs.agg_across_sessions(s1_selfdf, s2_selfdf)
	else:
		selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
		s1_selfdf = selfdf
	selfdf["exp"] = expname
	selfdf_.append(selfdf)

selfdf = pd.concat(selfdf_)
statsdf = pd.concat(statsdf_)
df = statsdf.merge(selfdf, on = "id")

# normalize the data 
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(df.select_dtypes("number"))
df.loc[:, df.dtypes == "float64"]= scaler.transform(df.select_dtypes("number"))

df["exp"] = pd.Categorical(df["exp"], categories = ["passive", "active"], ordered = True)

################ # calculate simple correlations 
self_vars = ["discount_logk"] + BIS_l2_subscales + UPPS_subscales 

# self_vars = ["BIS", "UPPS", "discount_logk"] 
task_vars = ["auc", "auc_delta", "std_wtw"]
nselfvar = len(self_vars)
ntaskvar = len(task_vars)

cb_p_df = pd.DataFrame(np.zeros((ntaskvar, nselfvar)), index = task_vars, columns = self_vars)
cb_r_df = pd.DataFrame(np.zeros((ntaskvar, nselfvar)), index = task_vars, columns = self_vars)
interaction_pvals = pd.DataFrame(np.zeros((ntaskvar, nselfvar)), index = task_vars, columns = self_vars)
for self_var, task_var in itertools.product(self_vars, task_vars):
	#fit = smf.ols(self_var +  " ~ %s * exp" % task_var, data = df).fit()
	#interaction_pvals.loc[task_var, self_var] = fit.pvalues[3]
	cb_r_df.loc[task_var, self_var],cb_p_df.loc[task_var, self_var]  = spearmanr(df[self_var], df[task_var], nan_policy = "omit")


import statsmodels.stats.multitest
fdr_sigs, fdr_pvals = statsmodels.stats.multitest.fdrcorrection(cb_p_df.values.flatten(), alpha=0.05, method='indep', is_sorted=False)


########## plot the histgram of all correlations 
# plt.style.use('classic')
# sns.set(font_scale = 2)
# sns.set_style("white")
# fig, ax  = plt.subplots()
# ax.hist(cb_r_df.values.reshape(-1), color = "grey", edgecolor = "black")
# ax.set_xlim([-0.5, 0.5])
# ax.set_xticks([ -0.4, -0.2, 0, 0.2, 0.4])
# ax.set_xlabel("Spearman's correlation")
# fig.tight_layout()
# fig.savefig(os.path.join("../figures/combined/rho_dist_task.pdf"))


# ######## plot the only significant one ############
# df = statsdf.merge(selfdf, on = "id")
# fig, ax  = plt.subplots()
# figFxs.my_regplot(df["PU"], df["std_wtw"], ax = ax, equal_aspect = False)
# ax.set_ylabel(r"$\sigma_{wtw}$")
# fig.tight_layout()
# fig.set_size_inches(w = 6, h = 6)
# fig.savefig(os.path.join("../figures/combined/rho_pu_sigma-wtw.pdf"))


################# find promising (not necessarily significant) correlations. arrange ito a long list #############
# cb_p_df[interaction_pvals < 0.05] = np.nan
cb_p_df[cb_p_df > 0.01] = np.nan
cb_sig_r_df = []
for self_var, task_var in itertools.product(self_vars, task_vars):
	if  ~np.isnan(cb_p_df.loc[task_var, self_var]):
		tmp = pd.DataFrame({
			"selfvar": self_var,
			"task_var": task_var,
			"r": cb_r_df.loc[task_var, self_var],
			"p": cb_p_df.loc[task_var, self_var]
			}, index = [0])
		cb_sig_r_df.append(tmp)
cb_sig_r_df = pd.concat(cb_sig_r_df)

################ permutation tests, combined for both experiments ##################
n_perm = 5000
cb_max_abs_r_dist = [] # to record the max abs correlation for each iteration
coef_dist_ = dict(zip(['%s-%s'%(x,y) for x, y in itertools.product(self_vars, task_vars)], [[]]*len(self_vars)*len(task_vars)))
coef_dist_list = [] # to record all abs correlations 
for i in np.arange(n_perm):
	df[["rd_" + x for x in task_vars]] = np.random.permutation(df[task_vars])
	r_, _ = analysisFxs.calc_prod_correlations(df, ["rd_" + x for x in task_vars], self_vars)
	cb_max_abs_r_dist.append(np.max(np.abs(r_.values)))
	for self_var in self_vars:
		for task_var in task_vars:
			coef_dist_['%s-%s'%(self_var, task_var)] = coef_dist_['%s-%s'%(self_var, task_var)] + [abs(r_.loc['rd_' + task_var, self_var])]
			coef_dist_list.append(abs(r_.loc['rd_' + task_var, self_var]))


########### compare H0 and H1 distributions 
plt.hist(coef_dist_list)
plt.style.use('classic')
sns.set(font_scale = 2)
sns.set_style("white")
fig, ax = plt.subplots()
h0_dist = ax.hist(coef_dist_list, density = True, color = "#bdbdbd", edgecolor = "black", alpha = 0.8, label='Null dist')
h1_dist = ax.hist(np.abs(cb_r_df.values.reshape(-1)), density = True, color = "#dd1c77", edgecolor = "black", alpha = 0.6, label='Observed dist')
ax.legend(['Null dist', 'Observed dist'])
ax.set_xlabel("Spearman's correlation")
ax.set_ylabel("Density")
fig.tight_layout()
fig.savefig(os.path.join("../figures", "combined", "h0_h1_correlations.pdf"))

# maybe adding a KL test. KL divergence is not symmetric?
# h0_ecdf = statsmodels.distributions.empirical_distribution.ECDF(coef_dist_list)
# h1_ecdf = statsmodels.distributions.empirical_distribution.ECDF(np.abs(cb_r_df.values.reshape(-1)))
# h0_pmf = h0_ecdf(np.linspace(0, 0.25, 20))
# h1_pmf = h1_ecdf(np.linspace(0, 0.25, 20))


##################
mc_pvals = [] # permutation p values, based on max values  
mc_pvals2 = [] # permutation p values, and then fwdr correction
for _, row in cb_sig_r_df.iterrows():
	mc_pvals.append(np.mean(np.array(cb_max_abs_r_dist) > abs(row['r'])))
	mc_pvals2.append(np.mean(np.array(coef_dist_['%s-%s'%(row['selfvar'], row['task_var'])])> abs(row['r']))* 3 * 12)
cb_sig_r_df['corrected_two-sided_p'] = mc_pvals
cb_sig_r_df['simple_corrected_two-sided_p'] = mc_pvals2

######## concatenate the list ###########



############## let me try this ########
best_params_ = []
for var in task_vars:
	# X = df[['discount_logk', 'UPPS', 'BIS']][~np.isnan(df['discount_logk'])]
	# X = df[BIS_l2_subscales][~np.isnan(df['discount_logk'])]
	X = df[UPPS_subscales][np.logical_and(df['exp'] == "passive", ~np.isnan(df['discount_logk']))]
	y = df[var][np.logical_and(df['exp'] == "passive", ~np.isnan(df['discount_logk']))]
	alpha_vals = 10**np.linspace(np.log10(0.01), np.log10(1), 10)
	gv = GridSearchCV(Lasso(max_iter = 1000000), {"alpha": alpha_vals}, cv = RepeatedKFold(2, 10), scoring = 'neg_mean_squared_error')
	res = gv.fit(X, y)
	#alpha_vals = [0.001, 0.01, 0.1, 1, 10, 100]
	plotdf = pd.DataFrame({
		"alpha": [x['alpha'] for x in res.cv_results_['params'] ],
		"score": res.cv_results_['mean_test_score']
		})
	print(res.best_params_)
	best_params_.append(res.best_params_['alpha'])
	fig, ax = plt.subplots()
	ax.set_xlabel("log10(alpha)")
	ax.set_ylabel("neg mean squared error")
	ax.plot(np.log10(plotdf['alpha']), plotdf['score'])
	fig.savefig(os.path.join("../figures", "combined", "BIS_%s_alpha_score.pdf"%var))


for var in task_vars:
	for predictors in predictors_:
		X = df[predictors][~np.isnan(df['discount_logk'])]
		y = df[var][~np.isnan(df['discount_logk'])]
		reg = Lasso(alpha = 0.1).fit(X, y)
		# reg = LinearRegression().fit(X, y)
		# print(pd.DataFrame(zip(['discount_logk', 'UPPS', 'BIS'] , reg.coef_)))
		# print(pd.DataFrame(zip(BIS_l2_subscales , reg.coef_)))
		print(pd.DataFrame(zip(predictors, reg.coef_)))

# permutation tests 
predictors_ = [['discount_logk', 'UPPS', 'BIS'], BIS_l2_subscales, UPPS_subscales]
# predictors_ = [['discount_logk', 'UPPS', 'BIS']]
max_coef_ = []
coef_dist_ = []
for p in np.arange(500):
	df[["rd_" + x for x in task_vars]] = np.random.permutation(df[task_vars])
	tmp = []
	for predictors in predictors_:
		for var in task_vars:
			# X = df[['discount_logk', 'UPPS', 'BIS']][~np.isnan(df['discount_logk'])]
			# X = df[BIS_l2_subscales][~np.isnan(df['discount_logk'])]
			X = df[predictors][~np.isnan(df['discount_logk'])]
			y = df['rd_' + var][~np.isnan(df['discount_logk'])]
			res = Lasso(alpha = 0.1).fit(X, y)
			tmp.append(np.max(np.abs(res.coef_)))
	max_coef_.append(np.max(tmp))

np.mean(np.array(max_coef_) > 0.04698)


################### change the directions 
self_vars = ["discount_logk"] + BIS_l2_subscales + UPPS_subscales
best_params_ = []
fig, axes = plt.subplots(len(self_vars))
for i, var in enumerate(self_vars):
	# X = df[['discount_logk', 'UPPS', 'BIS']][~np.isnan(df['discount_logk'])]
	# X = df[BIS_l2_subscales][~np.isnan(df['discount_logk'])]
	X = df[task_vars][np.logical_and(np['exp'] == 'passive', ~np.isnan(df['discount_logk']))]
	y = df[var][np.logical_and(np['exp'] == 'passive', ~np.isnan(df['discount_logk']))]
	alpha_vals = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.5]
	# alpha_vals = 10**np.linspace(np.log10(0.01), np.log10(1), 10)
	gv = GridSearchCV(Lasso(max_iter = 1000000), {"alpha": alpha_vals}, cv = RepeatedKFold(2, 10), scoring = 'neg_mean_squared_error')
	res = gv.fit(X, y)
	plotdf = pd.DataFrame({
		"alpha": [x['alpha'] for x in res.cv_results_['params'] ],
		"score": res.cv_results_['mean_test_score']
		})
	print(res.best_params_)
	best_params_.append(res.best_params_['alpha'])
	axes.flatten()[i].plot(plotdf['alpha'], plotdf['score'])
	# ax.set_xlabel("log10(alpha)")
	# ax.set_ylabel("neg mean squared error")

res_ = []
for i, var in enumerate(self_vars):
	print(var)
	# X = df[task_vars][np.logical_and(df['exp'] == "active", ~np.isnan(df['discount_logk']))]
	# y = df[var][np.logical_and(df['exp'] == "active", ~np.isnan(df['discount_logk']))]
	X = df[task_vars][~np.isnan(df['discount_logk'])]
	y = df[var][~np.isnan(df['discount_logk'])]
	# reg = Lasso(alpha = best_params_[i]).fit(X, y)
	reg = Lasso(alpha = 0.1).fit(X, y)
	# reg = LinearRegression().fit(X, y)
	# print(pd.DataFrame(zip(['discount_logk', 'UPPS', 'BIS'] , reg.coef_)))
	# print(pd.DataFrame(zip(BIS_l2_subscales , reg.coef_)))
	res_.append(reg.coef_)

res = pd.DataFrame(np.vstack(res_), index = self_vars, columns = task_vars)

coef_dist_ = dict(zip(['%s-%s'%(x,y) for x, y in itertools.product(self_vars, task_vars)], [[]]*len(self_vars)*len(task_vars)))
max_coef_ = []
for p in np.arange(1000):
	df[["rd_" + x for x in task_vars]] = np.random.permutation(df[task_vars])
	X = df[task_vars][~np.isnan(df['discount_logk'])]
	# X = df[['rd_'+x for x in task_vars]][np.logical_and(df['exp'] == "active", ~np.isnan(df['discount_logk']))]
	if p % 100 == 0:
		print(p)
	tmp = []
	for i, var in enumerate(self_vars):
		y = df[var][~np.isnan(df['discount_logk'])]
		# y = df[var][np.logical_and(df['exp'] == "active", ~np.isnan(df['discount_logk']))]
		# reg = Lasso(alpha = best_params_[i]).fit(X, y)
		reg = Lasso(alpha = 0.1).fit(X, y)
		tmp.append(np.max(np.abs(reg.coef_)))
		for i, task_var in enumerate(task_vars):
			coef_dist_['%s-%s'%(var, task_var)] = coef_dist_['%s-%s'%(var, task_var)] + [abs(reg.coef_[i])]
	max_coef_.append(np.max(tmp))

np.mean(np.array(max_coef_) > 0.0489)	## hmmmm wierd

# fig, axes = plt.subplots(len(self_vars))
# count = 0
# for y in coef_dist_.values():
# 	axes.flatten[count].hist(y)
# 	count = count + 1

# a = list(coef_dist_.keys())
# plt.hist(coef_dist_[a[-1]])

# this works. 
# Let me debug tomorrow 
p_df = pd.DataFrame(columns = task_vars, index = self_vars)
for self_var in self_vars:
	for task_var in task_vars:
		p_df.loc[self_var, task_var] = np.mean(np.array(coef_dist_['%s-%s'%(self_var, task_var)]) >= abs(res.loc[self_var, task_var]))


##########
cb_sig_r_df['simple_corrected_two-sided_p'] = mc_pvals
smf.ols("std_wtw ~ discount_logk + NU + PU + PM + PS + SS + selfcontrol + cogstable + attention + motor + cogcomplex + perseverance", data = df).fit().summary()


smf.ols("auc ~ discount_logk + NU + PU + PM + PS + SS + selfcontrol + cogstable + attention + motor + cogcomplex + perseverance", data = df[df["exp"] == "passive"]).fit().summary()


smf.ols("std_wtw ~ NU + PU + PM + PS + SS ", data = df[df["exp"] == "passive"]).fit().summary()


smf.ols("auc ~ selfcontrol + cogstable + attention + motor + cogcomplex + perseverance", data = df[df["exp"] == "passive"]).fit().summary()


smf.ols("auc ~ NU + PU + PM + PS + SS ", data = df[df["exp"] == "passive"]).fit().summary()
smf.ols("discount_logk ~ auc_delta + std_wtw + auc", data = df).fit().summary() # yes .... I am not sure
smf.ols("PU ~ auc_delta + std_wtw + auc", data = df).fit().summary() 
smf.ols("motor ~ auc_delta + std_wtw + auc", data = df).fit().summary() # No 
smf.ols("selfcontrol ~ auc_delta + std_wtw + auc", data = df[df['exp'] == 'passive']).fit().summary() # No
smf.ols("cogstable ~ auc + auc_delta + std_wtw", data = df[df['exp'] == 'passive']).fit().summary() # No
smf.ols("Nonplanning ~ auc_delta + std_wtw + auc", data = df[df['exp'] == 'passive']).fit().summary()


smf.ols("discount_logk ~ std_wtw * exp", data = df).fit().summary() # the best 
smf.ols("discount_logk ~ std_wtw + exp", data = df).fit().summary() 
smf.ols("discount_logk ~ std_wtw", data = df).fit().summary() 
smf.ols("discount_logk ~ std_wtw", data = df[df["exp"] == "passive"]).fit().summary()
smf.ols("discount_logk ~ std_wtw", data = df[df["exp"] == "active"]).fit().summary()


# 
smf.ols("auc ~ selfcontrol * exp", data = df).fit().summary()
spearmanr(df.loc[df["exp"]=="passive"]["auc"], df.loc[df["exp"]=="passive"]["selfcontrol"], nan_policy = "omit")


spearmanr(df["auc"], df["PS"], nan_policy = "omit")



smf.ols("auc ~ selfcontrol", data = df[df["exp"] == "passive"]).fit().summary()
smf.ols("auc ~ selfcontrol", data = df[df["exp"] == "active"]).fit().summary()


# 
smf.ols("auc ~ Nonplanning * exp", data = df).fit().summary()
################
smf.ols("UPPS ~ std_wtw * exp", data = df).fit().summary() # the best; BIS is not ..
smf.ols("UPPS ~ std_wtw + exp", data = df).fit().summary() 
smf.ols("UPPS ~ std_wtw", data = df).fit().summary() 
smf.ols("UPPS ~ std_wtw", data = df[df["exp"] == "passive"]).fit().summary() ....
smf.ols("UPPS ~ std_wtw", data = df[df["exp"] == "active"]).fit().summary()


smf.ols("std_wtw ~ UPPS * exp", data = df).fit().summary() # the best; upps is not significant
smf.ols("std_wtw ~ UPPS + exp", data = df).fit().summary() 
smf.ols("UPPS ~ std_wtw", data = df).fit().summary() # significant ....
smf.ols("UPPS ~ std_wtw", data = df[df["exp"] == "passive"]).fit().summary() ....
smf.ols("UPPS ~ std_wtw", data = df[df["exp"] == "active"]).fit().summary()

########
smf.ols("Motor ~ std_wtw", data = df).fit().summary() 
smf.ols("motor ~ std_wtw", data = df).fit().summary() 


# why adding the interaction term makes the effect smaller 
smf.ols("discount_logk ~ auc_delta", data = df[df["exp"] == "passive"]).fit().summary()
smf.ols("UPPS ~ exp + std_wtw", data = df).fit().summary()

############
r_, p_ = analysisFxs.calc_prod_correlations(df, ['Motor', "Nonplanning", "Attentional"], ['auc', "std_wtw", "auc_delta"])
r_, p_ = analysisFxs.calc_prod_correlations(df, ['motor', "perseverance", "cogstable", "selfcontrol", "attention", "cogcomplex"], ['auc', "std_wtw", "auc_delta"])

r_, p_ = analysisFxs.calc_prod_correlations(df, ['BIS', "UPPS", "log_GMK"], ['auc', "std_wtw", "auc_delta"])
r_, p_ = analysisFxs.calc_prod_correlations(df, ["NU", "PU", "PM", "PS", "SS"], ["NU", "PU", "PM", "PS", "SS"])

r_, p_ = analysisFxs.calc_prod_correlations(df, ['BIS', "UPPS", "log_GMK"], ["log_alpha", "log_alphaU", "tau", "gamma", "log_eta"])
r_, p_ = analysisFxs.calc_prod_correlations(df, ["log_alpha", "log_alphaU", "tau", "gamma", "log_eta"], ["log_alpha", "log_alphaU", "tau", "gamma", "log_eta"])



r_, p_ = analysisFxs.calc_prod_correlations(df, ['auc', "std_wtw", "auc_delta"], ['auc', "std_wtw", "auc_delta"])
r_, p_ = analysisFxs.calc_prod_correlations(df, ['motor', "selfcontrol", "cogstable"], ['auc', "std_wtw", "auc_delta"])

r_, p_ = analysisFxs.calc_prod_correlations(df, ["Motor", "Nonplanning", "Attentional"], ['auc', "std_wtw", "auc_delta"])

r_, p_ = analysisFxs.calc_prod_correlations(df, ["BIS", "UPPS", "log_GMK"], ['auc', "std_wtw", "auc_delta"])

r_, p_ = analysisFxs.calc_prod_correlations(df, ['motor', "perseverance",  "cogcomplex", "selfcontrol", "attention", "cogstable"], ['auc', "std_wtw", "auc_delta"])

sns.pairplot(df[["Motor", "Nonplanning", "Attentional"]])
plt.savefig(os.path.join("..", "figures", expname, "BIS_corr.pdf"))

sns.pairplot(df[["log_alpha", "log_nu", "log_tau", "gamma", "log_eta"]]) # strong auto-correlations among parameters; 
plt.savefig(os.path.join("..", "figures", expname, "para_corr_%s_wtw_%s_stepsize%.2f.pdf"%(modelname, fitMethod, stepsize)))


################### simple correlations among selfreports and model measures ###############
source_ = []
sp_var_ = []
bh_var_ = []
coef_ = [] 
pval_ = []
impulse_vars = ["BIS", "UPPS", "discount_logk"]
if expname == "passive":
	sources = ["sess1", "sess2", "combined"]
elif expname == "active":
	sources = ["sess1", "combined"]

# impulse_vars = ["NU", "PU", "PM", "PS", "SS"]
for source in sources:
	if source == "sess1":
		df = s1_df.merge(s1_selfdf, on = "id").merge(hdrdata_sess1[["id", "age", "gender"]], on = "id")
	elif source == "sess2":
		df = s2_df.merge(s2_selfdf, on = "id").merge(hdrdata_sess1[["id", "age", "gender"]], on = "id")
	elif source == "combined":
		df = statsdf.merge(selfdf, on = "id").merge(hdrdata_sess1[["id", "age", "gender"]], on = "id")
	r_, p_ = analysisFxs.calc_prod_correlations(df, impulse_vars, ['auc', "std_wtw", "auc_delta"])
	coef_ = coef_ + list(r_.reset_index().melt(id_vars = ["index"])["value"])
	sp_var_ = sp_var_ + list(r_.reset_index().melt(id_vars = ["index"])["index"])
	bh_var_ = bh_var_ + list(r_.reset_index().melt(id_vars = ["index"])["variable"])
	pval_ = pval_ + list(p_.reset_index().melt(id_vars = ["index"])["value"])
	source_ = source_ + [source] * len(impulse_vars) * 3

plotdf = pd.DataFrame({
	"source": source_,
	"bh_var": bh_var_,
	"sp_var": sp_var_,
	"coef": coef_,
	"pval": [figFxs.tosig(x) for x in pval_],
	"label_y": [x + 0.03 * np.sign(x) for x in coef_]
	})
plotdf["source"] =  pd.Categorical(plotdf["source"], categories = ["sess1", "sess2", "combined"])
plotdf["bh_var"] =  pd.Categorical(plotdf["bh_var"], categories = ["auc", "std_wtw", "auc_delta"])

# only combined sessions
p = (ggplot(plotdf[plotdf["source"] == "combined"]) \
	+ aes(x="bh_var", y="coef") \
	+ geom_bar(stat = "identity", position="dodge", width=0.75) \
	# + geom_errorbar(aes(ymin = "ymin", ymax = "ymax"), position = "dodge", width = 0.9)\
	+ facet_grid(facets="~sp_var") 
	+ geom_text(aes(y="label_y", label = "pval"), position=position_dodge(width=0.75)) 
	+ scale_fill_manual(values = ["#ca0020", "#0571b0", "#bababa"]) 
	+ labs(x = "", y = r"$Spearman's \rho$") + theme_classic() +
	scale_x_discrete(labels= ["AUC", r"$\sigma_{wtw}$", r"$\Delta AUC$"]))
ggsave(plot = p, filename= "../figures/%s/simple_coef_combined_BIS.pdf"%(expname))


# all sessions 
p = (ggplot(plotdf) \
	+ aes(x="bh_var", y="coef", fill = "source") \
	+ geom_bar(stat = "identity", position="dodge", width=0.75) \
	# + geom_errorbar(aes(ymin = "ymin", ymax = "ymax"), position = "dodge", width = 0.9)\
	+ facet_grid(facets="~sp_var") 
	+ geom_text(aes(y="label_y", label = "pval"), position=position_dodge(width=0.75)) 
	+ scale_fill_manual(values = ["#ca0020", "#0571b0", "#bababa"]) 
	+ labs(x = "", y = r"$Spearman's \rho$") +
	scale_x_discrete(labels= ["AUC", r"$\sigma_{wtw}$", r"$\Delta AUC$"]))
ggsave(plot = p, filename= "../figures/%s/simple_coef_all-sessions_BIS.pdf"%(expname))


################### regressions
source_ = []
sp_var_ = []
bh_var_ = []
coef_ = [] 
se_ = []
pval_ = []
impulse_vars = ["BIS", "UPPS",  "discount_logk"]
bh_vars = ["auc", "std_wtw", "auc_delta"]
if expname == "passive":
	sources = ["sess1", "sess2", "combined"]
elif expname == "active":
	sources = ["sess1", "combined"]

# impulse_vars = ["NU", "PU", "PM", "PS", "SS"]
for impulse_var in impulse_vars:
	for source in sources:
		if source == "sess1":
			df = s1_df.merge(s1_selfdf, on = "id").merge(hdrdata_sess1[["id", "age", "gender", "seq"]], on = "id")
		elif source == "sess2":
			df = s2_df.merge(s2_selfdf, on = "id").merge(hdrdata_sess1[["id", "age", "gender", "seq"]], on = "id")
		elif source == "combined":
			df = statsdf.merge(selfdf, on = "id").merge(hdrdata_sess1[["id", "age", "gender", "seq"]], on = "id")
		df = df[df["gender"] != "Neither/Do not wish to disclose"]
		# df = df[df["age"] < 50]
		# df = df[1:140]
		for bh_var in bh_vars:
			df[df.select_dtypes('number').columns] = df.select_dtypes('number').apply(lambda x:scipy.stats.zscore(x, nan_policy = "omit")) 
			results = smf.ols(bh_var + ' ~ ' + impulse_var , data=df).fit()
			source_ = source_ + [source] * len(results.params[1:])
			sp_var_ = sp_var_ + list(results.params[1:].index.values)
			bh_var_ = bh_var_ + [bh_var] * len(results.params[1:])
			coef_ = coef_ + list(results.params[1:].values)
			se_ = se_ + list(results.bse[1:].values)
			pval_ = pval_ + list(results.pvalues[1:].values)

plotdf = pd.DataFrame({
	"source": source_,
	"bh_var": bh_var_,
	"sp_var": sp_var_,
	"coef": coef_,
	"se": se_,
	"ymax": [x + y for x, y in zip(coef_, se_)],
	"ymin": [x - y for x, y in zip(coef_, se_)],
	"pval": [figFxs.tosig(x, marginal = True) for x in pval_],
	"label_y": [x + 0.03 * np.sign(x) for x in coef_]
	})
plotdf["source"] =  pd.Categorical(plotdf["source"], categories = ["sess1", "sess2", "combined"])

plotdf = plotdf[np.logical_and(plotdf["source"] == "combined", np.isin(plotdf["sp_var"], ["BIS", "UPPS",  "discount_logk",  "Motor", "Nonplanning", "Attentional", "motor"]))]
plotdf["bh_var"] =  pd.Categorical(plotdf["bh_var"], categories = ["auc", "std_wtw", "auc_delta"])

# only combined sessions
p = (ggplot(plotdf) \
	+ aes(x="bh_var", y="coef") \
	+ geom_bar(stat = "identity", position="dodge", width=0.75) \
	# + geom_errorbar(aes(ymin = "ymin", ymax = "ymax"), position = "dodge", width = 0.9)\
	+ facet_grid(facets="~sp_var") 
	+ geom_text(aes(y="label_y", label = "pval"), position=position_dodge(width=0.75)) 
	+ scale_fill_manual(values = ["#ca0020", "#0571b0", "#bababa"]) 
	+ labs(x = "", y = "Standardized coef") + theme_classic() + scale_x_discrete(labels= [r"AUC", r"$\sigma_{WTW}$", r"$\Delta AUC$"]))
ggsave(plot = p, filename= "../figures/%s/coef_combined_BIS.pdf"%(expname), width = 10, height = 5)



############################# look at subscores ###########
source_ = []
sp_var_ = []
bh_var_ = []
coef_ = [] 
se_ = []
pval_ = []
impulse_vars = ["Motor", "Nonplanning", "Attentional", "attention", "cogstable", "motor", "perseverance", "selfcontrol", "cogcomplex"]
bh_vars = ["auc", "std_wtw", "auc_delta"]
if expname == "passive":
	sources = ["sess1", "sess2", "combined"]
elif expname == "active":
	sources = ["sess1", "combined"]

# impulse_vars = ["NU", "PU", "PM", "PS", "SS"]
for impulse_var in impulse_vars:
	for source in sources:
		if source == "sess1":
			df = s1_df.merge(s1_selfdf, on = "id").merge(hdrdata_sess1[["id", "age", "gender", "seq"]], on = "id")
		elif source == "sess2":
			df = s2_df.merge(s2_selfdf, on = "id").merge(hdrdata_sess1[["id", "age", "gender", "seq"]], on = "id")
		elif source == "combined":
			df = statsdf.merge(selfdf, on = "id").merge(hdrdata_sess1[["id", "age", "gender", "seq"]], on = "id")
		df = df[df["gender"] != "Neither/Do not wish to disclose"]
		# df = df[df["age"] < 50]
		# df = df[1:140]
		for bh_var in bh_vars:
			df[df.select_dtypes('number').columns] = df.select_dtypes('number').apply(lambda x:scipy.stats.zscore(x, nan_policy = "omit")) 
			results = smf.ols(bh_var + ' ~ ' + impulse_var, data=df).fit()
			source_ = source_ + [source] * len(results.params[1:])
			sp_var_ = sp_var_ + list(results.params[1:].index.values)
			bh_var_ = bh_var_ + [bh_var] * len(results.params[1:])
			coef_ = coef_ + list(results.params[1:].values)
			se_ = se_ + list(results.bse[1:].values)
			pval_ = pval_ + list(results.pvalues[1:].values)

plotdf = pd.DataFrame({
	"source": source_,
	"bh_var": bh_var_,
	"sp_var": sp_var_,
	"coef": coef_,
	"se": se_,
	"ymax": [x + y for x, y in zip(coef_, se_)],
	"ymin": [x - y for x, y in zip(coef_, se_)],
	"pval": [figFxs.tosig(x, marginal = True) for x in pval_],
	"label_y": [x + 0.03 * np.sign(x) for x in coef_]
	})
plotdf["source"] =  pd.Categorical(plotdf["source"], categories = ["sess1", "sess2", "combined"])
plotdf = plotdf[np.logical_and(plotdf["source"] == "combined", np.isin(plotdf["sp_var"], ["Motor", "Nonplanning", "Attentional", "attention", "cogstable", "motor", "perseverance", "selfcontrol", "cogcomplex"]))]
plotdf["bh_var"] =  pd.Categorical(plotdf["bh_var"], categories = ["auc", "std_wtw", "auc_delta"])

p = (ggplot(plotdf)\
	+ aes(x="bh_var", y="coef") \
	+ geom_bar(stat = "identity", position="dodge", width=0.75) \
	# + geom_errorbar(aes(ymin = "ymin", ymax = "ymax"), position = "dodge", width = 0.9)\
	+ facet_grid(facets="~sp_var") 
	+ geom_text(aes(y="label_y", label = "pval"), position=position_dodge(width=0.75)) 
	+ scale_fill_manual(values = ["#ca0020", "#0571b0", "#bababa"]) 
	+ labs(x = "", y = "Standardized coef") + theme_classic() + scale_x_discrete(labels= [r"AUC", r"$\sigma_{WTW}$", r"$\Delta AUC$"]))
ggsave(plot = p, filename= "../figures/%s/coef_combined_subscores.pdf"%(expname), width = 20, height = 5)


#  all variale regressions
source_ = []
sp_var_ = []
bh_var_ = []
coef_ = [] 
se_ = []
pval_ = []
impulse_vars = ["BIS", "UPPS",  "discount_logk"]
bh_vars = ["auc", "std_wtw", "auc_delta"]
if expname == "passive":
	sources = ["sess1", "sess2", "combined"]
elif expname == "active":
	sources = ["sess1", "combined"]

# impulse_vars = ["NU", "PU", "PM", "PS", "SS"]
for impulse_var in impulse_vars:
	for source in sources:
		if source == "sess1":
			df = s1_df.merge(s1_selfdf, on = "id").merge(hdrdata_sess1[["id", "age", "gender", "seq"]], on = "id")
		elif source == "sess2":
			df = s2_df.merge(s2_selfdf, on = "id").merge(hdrdata_sess1[["id", "age", "gender", "seq"]], on = "id")
		elif source == "combined":
			df = statsdf.merge(selfdf, on = "id").merge(hdrdata_sess1[["id", "age", "gender", "seq"]], on = "id")
		df = df[df["gender"] != "Neither/Do not wish to disclose"]
		# df = df[df["age"] < 50]
		# df = df[1:140]
		df[df.select_dtypes('number').columns] = df.select_dtypes('number').apply(lambda x:scipy.stats.zscore(x, nan_policy = "omit")) 
		results = smf.ols(impulse_var + '~ auc + std_wtw + auc_delta', data=df).fit()
		source_ = source_ + [source] * len(results.params[1:])
		bh_var_ = bh_var_ + list(results.params[1:].index.values)
		sp_var_ = sp_var_ + [impulse_var] * len(results.params[1:])
		coef_ = coef_ + list(results.params[1:].values)
		se_ = se_ + list(results.bse[1:].values)
		pval_ = pval_ + list(results.pvalues[1:].values)

plotdf = pd.DataFrame({
	"source": source_,
	"bh_var": bh_var_,
	"sp_var": sp_var_,
	"coef": coef_,
	"se": se_,
	"ymax": [x + y for x, y in zip(coef_, se_)],
	"ymin": [x - y for x, y in zip(coef_, se_)],
	"pval": [figFxs.tosig(x, marginal = True) for x in pval_],
	"label_y": [x + 0.03 * np.sign(x) for x in coef_]
	})
plotdf["source"] =  pd.Categorical(plotdf["source"], categories = ["sess1", "sess2", "combined"])
plotdf = plotdf[np.logical_and(plotdf["source"] == "combined", np.isin(plotdf["bh_var"], ["auc", "std_wtw", "auc_delta"]))]



