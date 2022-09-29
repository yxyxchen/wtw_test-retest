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
sns.set(font_scale = 2)
sns.set_style("white")
condition_palette = ["#762a83", "#1b7837"]



expname = "passive"
s1_selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
hdrdata_sess1, trialdata_sess1_ = loadFxs.group_quality_check(expname, 1, plot_quality_check = True)
hdrdata_sess2, trialdata_sess2_ = loadFxs.group_quality_check(expname, 2, plot_quality_check = True)
s1_stats, s1_Psurv_b1_, s1_Psurv_b2_, s1_WTW_ = analysisFxs.group_MF(trialdata_sess1_, plot_each = False)   
s2_stats, s2_Psurv_b1_, s2_Psurv_b2_, s2_WTW_ = analysisFxs.group_MF(trialdata_sess2_, plot_each = False)   


n_sub = 4
# calc within-block adaptation using the non-parametric method
s1_stats['wb_adapt_np'] = s1_stats['end_wtw'] - s1_stats['init_wtw']
s2_stats['wb_adapt_np'] = s2_stats['end_wtw'] - s2_stats['init_wtw']

# calc within-block adaptation using AUC values
s1_stats['wb_adapt'] = s1_stats['auc'+str(n_sub)] - s1_stats['auc1']
s2_stats['wb_adapt'] = s2_stats['auc'+str(n_sub)] - s2_stats['auc1']

# calc std_wtw using the moving window method
s1_stats['std_wtw_mw'] = np.mean(s1_stats[['std_wtw' + str(i+1) for i in np.arange(n_sub)]]**2,axis = 1)**0.5
s2_stats['std_wtw_mw'] = np.mean(s2_stats[['std_wtw' + str(i+1) for i in np.arange(n_sub)]]**2,axis = 1)**0.5

# colvars = ['auc_end_start', 'auc', 'auc1', 'auc2', "auc_rh", 'std_wtw', 'std_wtw1', 'std_wtw2', "std_wtw_rh"]
# colvars = ['auc', "std_wtw", "std_wtw_mw", "init_wtw", "wb_adapt_np", 'wb_adapt']
colvars = ['auc', "std_wtw", "std_wtw_mw", "wb_adapt_np", "wb_adapt", "init_wtw"]
s1_HP = s1_stats.loc[s1_stats['condition'] == 'HP', colvars + ['id']]
s1_LP = s1_stats.loc[s1_stats['condition'] == 'LP', colvars + ['id']]
s1_df = s1_HP.merge(s1_LP, left_on = 'id', right_on = 'id', suffixes = ['_HP', "_LP"])


s2_HP = s2_stats.loc[s2_stats['condition'] == 'HP', colvars + ['id']]
s2_LP = s2_stats.loc[s2_stats['condition'] == 'LP', colvars + ['id']]
s2_df = s2_HP.merge(s2_LP, left_on = 'id', right_on = 'id', suffixes = ['_HP', "_LP"])

# add auc_delta and auc_ave
auc_vars = ['auc']
for var in ['auc']:
    s1_df[var + '_delta'] = s1_df.apply(func = lambda x: x[var + '_HP'] - x[var + '_LP'], axis = 1)
    s2_df[var + '_delta'] = s2_df.apply(func = lambda x: x[var + '_HP'] - x[var + '_LP'], axis = 1)
    s1_df[var + '_ave'] = (s1_df.apply(func = lambda x: x[var + '_HP'] + x[var + '_LP'], axis = 1)) / 2
    s2_df[var + '_ave'] = (s2_df.apply(func = lambda x: x[var + '_HP'] + x[var + '_LP'], axis = 1)) / 2


######################
## plot reliability ##
######################
# AUC reliability #
fig, ax = plt.subplots()
figFxs.AUC_reliability(s1_stats, s2_stats) # maybe I don't need it yet 
plt.gcf().set_size_inches(8, 6)
plt.savefig(os.path.join("..", "figures", expname, "AUC_reliability.pdf"))


###





#####
df = s1_df.merge(s2_df, on = 'id', suffixes = ['_sess1', '_sess2']) 
# delta AUC 
fig, ax = plt.subplots()
figFxs.my_regplot(df.loc[:, 'auc_delta_sess1'], df.loc[:, 'auc_delta_sess2'], color = "grey")
fig.gca().set_ylabel(r'$\Delta$' + 'AUC SESS2 (s)')
fig.gca().set_xlabel(r'$\Delta$' + 'AUC SESS1 (s)')
plt.gcf().set_size_inches(4, 4)
plt.tight_layout()
plt.savefig(os.path.join("..", "figures", expname, "delta_auc_reliability.pdf"))
