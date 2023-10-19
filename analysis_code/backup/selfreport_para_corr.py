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

# plot styles
plt.style.use('classic')
sns.set(font_scale = 1)
sns.set_style("white")
condition_palette = ["#762a83", "#1b7837"]


expname = 'passive'

############### model parameter 
modelname = 'QL2reset'
fitMethod = "whole"
stepsize = 0.5
paranames = modelFxs.getModelParas('QL2')
s1_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, fitMethod, stepsize)
s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname, fitMethod, stepsize)

################## behavioral data ##################
hdrdata_sess1, trialdata_sess1_ = loadFxs.group_quality_check(expname, 1, plot_quality_check = False)
hdrdata_sess2, trialdata_sess2_ = loadFxs.group_quality_check(expname, 2, plot_quality_check = False)
s1_stats, s1_Psurv_b1_, s1_Psurv_b2_, s1_WTW_emp = analysisFxs.group_MF(trialdata_sess1_, plot_each = False)   
s2_stats, s2_Psurv_b1_, s2_Psurv_b2_, s2_WTW_emp = analysisFxs.group_MF(trialdata_sess2_, plot_each = False)   

s1_df = analysisFxs.pivot_by_condition(s1_stats)
s2_df = analysisFxs.pivot_by_condition(s2_stats)

############ correlations among behavioral measures ############
df = analysisFxs.agg_across_sessions(s1_df, s2_df)
paradf = analysisFxs.agg_across_sessions(s1_paradf, s2_paradf)

df[['auc', 'std_wtw', 'auc_delta']].corr()
paradf[paranames].corr()

tmp = df.merge(paradf, on = "id")
row_vars = ['auc', 'std_wtw', "auc_delta", "auc_HP", "auc_LP"]
col_vars = modelFxs.getModelParas('QL2')
r_table, p_table, perm_r_, perm_p_table = figFxs.corr_analysis(tmp[row_vars], tmp[col_vars], 10)


########## correlations among paradf #################

s1_paradf[paranames].corr()
s2_paradf[paranames].corr() # maybe not a good story to tell




########### correlations between model paramters and task measures #######
########## maybe it is better to average them first #############
tmp = s1_df.merge(s1_paradf, on = "id")
row_vars = ['auc_ave', 'std_wtw_ave', "auc_delta", "auc_HP", "auc_LP"]
col_vars = modelFxs.getModelParas('QL2')
r_table, p_table, perm_r_, perm_p_table = figFxs.corr_analysis(tmp[row_vars], tmp[col_vars], 10)


############# 
d = selfdf.merge(df, on = "id")
r_table, p_table, perm_r_, perm_p_table = figFxs.corr_analysis(d[['UPPS','BIS', 'GMK']], d[['auc_ave', 'std_wtw_ave', 'auc_delta']], 10)





#####
row_vars = modelFxs.getModelParas('QL2')
col_vars = ['NU', 'PU', 'PM', 'PS', 'SS', 'attention', 'cogstable', 'motor', 'perseverance', 'selfcontrol', 'cogcomplex']
col_vars = ['UPPS','BIS', 'GMK']
r_table, p_table, perm_r_, perm_p_table = figFxs.corr_analysis(s1_paradf[row_vars], s1_selfdf.loc[np.isin(s1_selfdf.id, s1_paradf.id), col_vars], 1000)

# I lost a data here?
d = s1_paradf.merge(s2_selfdf, on = "id")
r_table, p_table, perm_r_, perm_p_table = figFxs.corr_analysis(d[row_vars], d[col_vars], 10)

# hmm let me avarage these numbers 
selfdf = s1_selfdf.merge(s2_selfdf, on = "id", suffixes = ["_sess1", "_sess2"])
selfdf['BIS'] = (selfdf['BIS_sess1'] + selfdf['BIS_sess2']) / 2
selfdf['GMK'] = (selfdf['GMK_sess1'] + selfdf['GMK_sess2']) / 2

paradf = s1_paradf.merge(s2_paradf, on = "id", suffixes = ["_sess1", "_sess2"])
paradf['tau'] = (paradf['tau_sess1'] * paradf['tau_sess2']) ** 0.5
paradf['nu'] = (paradf['nu_sess1'] * paradf['nu_sess2']) ** 0.5

d = selfdf.merge(paradf, on = "id")
spearmanr(d['tau'], d['BIS'], nan_policy = 'omit')
spearmanr(d['tau'], d['GMK'], nan_policy = 'omit')
spearmanr(d['nu'], d['GMK'], nan_policy = 'omit')

# 
spearmanr(d['tau_sess1'], d['GMK_sess1'], nan_policy = 'omit')

########### AUC and self-report
########## there must be an average measure somewhere.
row_vars = ['auc', 'std_wtw']
col_vars = ['UPPS','BIS', 'GMK']
d = s1_stats[s1_stats['condition'] == "HP"].merge(s1_selfdf, on = "id")
r_table, p_table, perm_r_, perm_p_table = figFxs.corr_analysis(d[row_vars], d[col_vars], 10)










