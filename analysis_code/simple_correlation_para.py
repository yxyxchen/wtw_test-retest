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


# plot styles
plt.style.use('classic')
sns.set(font_scale = 1)
sns.set_style("white")
condition_palette = ["#762a83", "#1b7837"]
UPPS_subscales = ["NU", "PU", "PM", "PS", "SS"]
BIS_l1_subscales = ["Attentional", "Motor", "Nonplanning"]
BIS_l2_subscales = ["attention", "cogstable", "motor", "perseverance", "selfcontrol", "cogcomplex"]

modelname = "QL2reset_slope_two_simple"
fitMethod = "whole"
stepsize = 0.5
# passive version
paradf_ = [] 
selfdf_ = []
for expname in ["active", "passive"]:
	hdrdata_sess1, trialdata_sess1_ = loadFxs.group_quality_check(expname, 1, plot_quality_check = True)
	hdrdata_sess2, trialdata_sess2_ = loadFxs.group_quality_check(expname, 2, plot_quality_check = True)
	########### let me only include participants complete both sessions
	hdrdata_sess1 = hdrdata_sess1[np.isin(hdrdata_sess1["id"], hdrdata_sess2["id"])]
	trialdata_sess1_ = {x: y for x,y in trialdata_sess1_.items() if x[0] in hdrdata_sess2["id"].values}
	############ conduct behavioral analysis ######
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
	s1_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, fitMethod, stepsize)
	s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname, fitMethod, stepsize)
	paradf = analysisFxs.agg_across_sessions(s1_paradf, s2_paradf)
	paradf["exp"] = expname
	paradf_.append(paradf)

paradf = pd.concat(paradf_)
selfdf = pd.concat(selfdf_)
df = paradf.merge(selfdf, on = ["id", "exp"])
df.loc[:, df.dtypes == "float64"] = df.select_dtypes("number") - df.select_dtypes("number").apply(np.mean, axis = 0)
df["exp"] = pd.Categorical(df["exp"], categories = ["passive", "active"], ordered = True)


paranames = modelFxs.getModelParas(modelname)
self_vars = ["BIS", "UPPS", "discount_logk"] + BIS_l2_subscales + UPPS_subscales
npara = len(paranames)
nselfvar = len(self_vars)
# record whether there is an interaction of exp 
cb_p_df = pd.DataFrame(np.zeros((nselfvar, npara)), index = self_vars, columns = paranames)
ps_p_df = pd.DataFrame(np.zeros((nselfvar, npara)), index = self_vars, columns = paranames)
ac_p_df = pd.DataFrame(np.zeros((nselfvar, npara)), index = self_vars, columns = paranames)
cb_r_df = pd.DataFrame(np.zeros((nselfvar, npara)), index = self_vars, columns = paranames)
ps_r_df = pd.DataFrame(np.zeros((nselfvar, npara)), index = self_vars, columns = paranames)
ac_r_df = pd.DataFrame(np.zeros((nselfvar, npara)), index = self_vars, columns = paranames)
interaction_pvals = pd.DataFrame(np.zeros((nselfvar, npara)), index = self_vars, columns = paranames)
for para, self_var in itertools.product(paranames, self_vars):
	fit = smf.ols(para +  " ~ %s * exp" % self_var, data = df).fit()
	interaction_pvals.loc[self_var, para] = fit.pvalues[3]
	cb_r_df.loc[self_var, para],cb_p_df.loc[self_var, para]  = spearmanr(df[para], df[self_var], nan_policy = "omit")
	ps_r_df.loc[self_var, para],ps_p_df.loc[self_var, para] = spearmanr(df.loc[df["exp"]=="passive", para], df.loc[df["exp"]=="passive", self_var], nan_policy = "omit")
	ac_r_df.loc[self_var, para],ac_p_df.loc[self_var, para]  = spearmanr(df.loc[df["exp"]=="active", para], df.loc[df["exp"]=="active", self_var], nan_policy = "omit")

########## visualize 
# plt.style.use('classic')
# sns.set(font_scale = 2)
# sns.set_style("white")
# fig, ax  = plt.subplots()
# ax.hist(cb_r_df.values.reshape(-1), color = "grey", edgecolor = "black")
# ax.set_xlim([-0.5, 0.5])
# ax.set_xticks([ -0.4, -0.2, 0, 0.2, 0.4])
# ax.set_xlabel("Spearman's correlation")
# fig.tight_layout()
# fig.savefig(os.path.join("../figures/combined/rho_dist_para.pdf"))


##### permutation tests #######
# shuffle within each condition # 
self_vars = ["discount_logk"] + BIS_l2_subscales + UPPS_subscales
n_perm = 5000
cb_max_abs_r_dist = []
coef_dist_ = dict(zip(['%s-%s'%(x,y) for x, y in itertools.product(self_vars, paranames)], [[]]*len(self_vars)*len(paranames)))
coef_dist_list = [] # to record all abs correlations 
for i in np.arange(n_perm):
	for para in paranames:
		df["rd_" + para] = np.random.permutation(df[para])
	r_, _ = analysisFxs.calc_prod_correlations(df, ["rd_" + x for x in paranames], self_vars)
	cb_max_abs_r_dist.append(np.max(np.abs(r_.values)))
	for self_var in self_vars:
		for para in paranames:
			coef_dist_['%s-%s'%(self_var, para)] = coef_dist_['%s-%s'%(self_var, para)] + [abs(r_.loc['rd_' + para, self_var])]
			coef_dist_list.append(abs(r_.loc['rd_' + para, self_var]))
 

################ compare h0 and h1 correlation dists 
plt.style.use('classic')
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
fig.savefig(os.path.join("../figures", "combined", "h0_h1_para_correlations.pdf"))


