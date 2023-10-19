
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


