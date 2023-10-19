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
paradf = paradf[['id', 'exp'] + ['alpha', 'alphaU', 'tau', 'eta']]
selfdf = pd.concat(selfdf_)
selfdf = selfdf[['id', 'exp'] + UPPS_subscales + BIS_l2_subscales + 'discount_logk']
statsdf = pd.concat(statsdf_)

statsdf = statsdf[['id', 'exp'] + ['auc', 'std_wtw', 'auc_delta']]
df = paradf.merge(selfdf, on = ["id", "exp"]).merge(statsdf, on = ["id", "exp"])
df.loc[:, df.dtypes == "float64"] = df.select_dtypes("number") - df.select_dtypes("number").apply(np.mean, axis = 0)
df["exp"] = pd.Categorical(df["exp"], categories = ["passive", "active"], ordered = True)
df[UPPS_subscales + BIS_l2_subscales + ['auc', 'std_wtw', 'auc_delta'] + ['alpha', 'alphaU', 'tau', 'eta']].to_csv('EFA/all_measures.csv')









