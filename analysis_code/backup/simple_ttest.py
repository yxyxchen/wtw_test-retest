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
import scipy as sp
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
	####################### analyze only selfreport data ####################
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
df.loc[:, df.dtypes == "float64"] = df.select_dtypes("number") - df.select_dtypes("number").apply(np.mean, axis = 0)
df["exp"] = pd.Categorical(df["exp"], categories = ["passive", "active"], ordered = True)

cb_p_df = pd.DataFrame(np.zeros((ntaskvar, nselfvar)), index = task_vars, columns = self_vars)
for self_var in self_vars:
	q1, q3 = scipy.stats.mstats.mquantiles(df[self_var], [0.2, 0.8])
	df["top"] = [1 if x > q3 else 0 for x in df[self_var]]
	df["bottom"] = [1 if x < q1 else 0 for x in df[self_var]]
	for task_var in task_vars:
		cb_p_df.loc[task_var, self_var] = mannwhitneyu(df.loc[df["top"] == 1, task_var], df.loc[df["bottom"] == 1, task_var])[1]


######

self_vars = ["BIS", "UPPS", "discount_logk"]
cb_min_p_dist = []
for i in np.arange(n_perm):
	p_ = []
	for task_var in task_vars:
		df.loc[df["exp"] == "passive", "rd_" + task_var] = np.random.permutation(df.loc[df["exp"] == "passive", task_var])
		df.loc[df["exp"] == "active", "rd_" + task_var] = np.random.permutation(df.loc[df["exp"] == "active", task_var])
	for self_var in self_vars:
		q1, q3 = scipy.stats.mstats.mquantiles(df[self_var], [0.20, 0.80])
		df["top"] = [1 if x > q3 else 0 for x in df[self_var]]
		df["bottom"] = [1 if x < q1 else 0 for x in df[self_var]]
		for task_var in task_vars:
			tmp = mannwhitneyu(df.loc[df["top"] == 1, 'rd_' + task_var], df.loc[df["bottom"] == 1, 'rd_' + task_var])[1]
			p_.append(tmp)
	# print(p_)
	cb_min_p_dist.append(np.min(p_))

np.mean(np.array(cb_min_p_dist) < 0.05)


