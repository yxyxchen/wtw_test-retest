
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
	statsdf = analysisFxs.sub_across_sessions(s1_df, s2_df)
	statsdf_.append(statsdf)
	# self
	####################### analyze only selfreport data ####################
	s1_selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
	s2_selfdf = loadFxs.parse_group_selfreport(expname, 2, isplot = False)
	s1_selfdf = s1_selfdf[np.isin(s1_selfdf["id"], s2_selfdf["id"])].reset_index()
	if expname == "passive":
		selfdf = analysisFxs.sub_across_sessions(s1_selfdf, s2_selfdf, vars = ['discount_logk', 'UPPS', 'BIS', 'PAS', 'NAS'] + UPPS_subscales + BIS_l2_subscales + BIS_l1_subscales)
	else:
		selfdf = analysisFxs.sub_across_sessions(s1_selfdf, s2_selfdf, vars = ['PAS', 'NAS'])
		selfdf = selfdf[~np.isnan(selfdf['PAS_sess1'])]		
	selfdf["exp"] = expname
	selfdf_.append(selfdf)

selfdf = pd.concat(selfdf_)
statsdf = pd.concat(statsdf_)
df = statsdf.merge(selfdf, on = "id")
# scaler = sklearn.preprocessing.StandardScaler
# a = preprocessing.StandardScaler()
# a.fit(df.select_dtypes("number"))
# df.loc[:, df.dtypes == "float64"]= a.transform(df.select_dtypes("number"))
df["exp"] = pd.Categorical(df["exp"], categories = ["passive", "active"], ordered = True)


################ 
# record whether there is an interaction of exp  
# calculate simple correlations 

mood_vars = [x + '_diff' for x in ['PAS', 'NAS']]
vars = [x + '_diff' for x in ["auc", "auc_delta", "std_wtw"]]
nmoodvar = len(mood_vars)
nvar = len(vars)

cb_p_df = pd.DataFrame(np.zeros((nmoodvar, nvar)), index = mood_vars, columns = vars)
ps_p_df = pd.DataFrame(np.zeros((nmoodvar, nvar)), index = mood_vars, columns = vars)
ac_p_df = pd.DataFrame(np.zeros((nmoodvar, nvar)), index = mood_vars, columns = vars)
cb_r_df = pd.DataFrame(np.zeros((nmoodvar, nvar)), index = mood_vars, columns = vars)
ps_r_df = pd.DataFrame(np.zeros((nmoodvar, nvar)), index = mood_vars, columns = vars)
ac_r_df = pd.DataFrame(np.zeros((nmoodvar, nvar)), index = mood_vars, columns = vars)
interaction_pvals = pd.DataFrame(np.zeros((nmoodvar, nvar)), index = mood_vars, columns = vars)
for var, mood_var in itertools.product(vars, mood_vars):
	fit = smf.ols(var +  " ~ %s * exp" % mood_var, data = df).fit()
	interaction_pvals.loc[mood_var, var] = fit.pvalues[3]
	cb_r_df.loc[mood_var, var],cb_p_df.loc[mood_var, var]  = spearmanr(df[var], df[mood_var], nan_policy = "omit")
	ps_r_df.loc[mood_var, var],ps_p_df.loc[mood_var, var] = spearmanr(df.loc[df["exp"]=="passive", var], df.loc[df["exp"]=="passive", mood_var], nan_policy = "omit")
	ac_r_df.loc[mood_var, var],ac_p_df.loc[mood_var, var]  = spearmanr(df.loc[df["exp"]=="active", var], df.loc[df["exp"]=="active", mood_var], nan_policy = "omit")

###########################
ood_vars = [x + '_diff' for x in ['PAS', 'NAS']]
vars = [x + '_diff' for x in ["discount_logk", "UPPS", "BIS"] + BIS_l2_subscales + BIS_l1_subscales + UPPS_subscales]
nmoodvar = len(mood_vars)
nvar = len(vars)

cb_p_df = pd.DataFrame(np.zeros((nmoodvar, nvar)), index = mood_vars, columns = vars)
cb_r_df = pd.DataFrame(np.zeros((nmoodvar, nvar)), index = mood_vars, columns = vars)
interaction_pvals = pd.DataFrame(np.zeros((nmoodvar, nvar)), index = mood_vars, columns = vars)
for var, mood_var in itertools.product(vars, mood_vars):
	fit = smf.ols(var +  " ~ %s * exp" % mood_var, data = df).fit()
	interaction_pvals.loc[mood_var, var] = fit.pvalues[3]
	cb_r_df.loc[mood_var, var],cb_p_df.loc[mood_var, var]  = spearmanr(df[var], df[mood_var], nan_policy = "omit")





