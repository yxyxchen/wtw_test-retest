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
statsdf_ = []
for expname in ["active"]:
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
	statsdf["exp"] = expname
	statsdf_.append(statsdf)
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
statsdf = pd.concat(statsdf_)
df = paradf.merge(selfdf, on = ["id", "exp"]).merge(statsdf, on = ["id", "exp"])
#df.loc[:, df.dtypes == "float64"] = df.select_dtypes("number") - df.select_dtypes("number").apply(np.mean, axis = 0)
df["exp"] = pd.Categorical(df["exp"], categories = ["passive", "active"], ordered = True)



UPPS_subscales = ["NU", "PU", "PM", "PS", "SS"]
BIS_l1_subscales = ["Attentional", "Motor", "Nonplanning"]
BIS_l2_subscales = ["attention", "cogstable", "motor", "perseverance", "selfcontrol", "cogcomplex"]
self_vars = UPPS_subscales + BIS_l1_subscales + ["discount_logk"]
task_vars = ["auc", "std_wtw", "auc_delta"]
paranames = ["alpha", "alphaU", "tau","eta"]
vars = self_vars + task_vars + paranames


df.loc[:,vars].to_csv(os.path.join("../figures", "combined", "all_measures.csv"))

################### get reliability 
s1_taskdf_ = []
s2_taskdf_ = []
s1_paradf_ = []
s2_paradf_ = []

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
	s1_taskdf_.append(s1_df)
	s2_taskdf_.append(s2_df)
	s1_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, fitMethod, stepsize)
	s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname, fitMethod, stepsize)
	s1_paradf_.append(s1_paradf)
	s2_paradf_.append(s2_paradf)
	if expname == "passive":
		s1_selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
		s2_selfdf = loadFxs.parse_group_selfreport(expname, 2, isplot = False)


s1_taskdf = pd.concat(s1_taskdf_)
s2_taskdf = pd.concat(s2_taskdf_)
s1_paradf = pd.concat(s1_paradf_)
s2_paradf = pd.concat(s2_paradf_)
# find common ids 
from functools import reduce
common_ids = reduce(np.intersect1d, [s1_taskdf["id"], s2_taskdf["id"], s1_paradf["id"], s2_paradf["id"]])
common_self_ids = np.intersect1d(s1_selfdf["id"], s2_selfdf["id"])

# so far I am not sure about that ...
task_rhos = []
for var in task_vars:
	task_rhos.append(spearmanr(s1_taskdf.loc[np.isin(s1_taskdf["id"], common_ids), var], s2_taskdf.loc[np.isin(s2_taskdf["id"], common_ids), var])[0])


para_rhos = []
for var in paranames:
	para_rhos.append(spearmanr(s1_paradf.loc[np.isin(s1_paradf["id"], common_ids), var], s2_paradf.loc[np.isin(s2_paradf["id"], common_ids), var])[0])


# I still haven't include that participant who didn't fill the questionaire !!!!! 
self_rhos = []
self_ns = []
for var in self_vars:
	self_rhos.append(spearmanr(s1_selfdf.loc[np.isin(s1_selfdf["id"], common_self_ids), var], s2_selfdf.loc[np.isin(s2_selfdf["id"], common_self_ids), var], nan_policy = "omit")[0])
	self_ns.append(len(np.intersect1d(s1_selfdf.loc[~np.isnan(s1_selfdf[var]), "id"], s2_selfdf.loc[~np.isnan(s2_selfdf[var]), "id"])))


# ok so I am
reliability_df = pd.DataFrame({
	"var": self_vars + task_vars + paranames,
	"rho": self_rhos + task_rhos + para_rhos
	})

reliability_df.to_csv(os.path.join("../figures", "combined", "all_reliability.csv"))


# figure that quitted participant
# figure which n to use 







