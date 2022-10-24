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



########## additional measures ######
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
## plot reliability for variables I am interested in ##
######################
s1_df = analysisFxs.pivot_by_condition(s1_stats)
s2_df = analysisFxs.pivot_by_condition(s2_stats)
df = analysisFxs.hstack_sessions(s1_df, s2_df)
fig, ax = plt.subplots()
ax.set_xlabel("Session 1")

ax.set_ylabel("Session 2")
figFxs.my_regplot(df['auc_sess1'], df['auc_sess2'], ax = ax)
ax.set_title("AUC (s)")


fig, ax = plt.subplots()
figFxs.my_regplot(df['std_wtw_sess1'], df['std_wtw_sess2'], ax = ax)
ax.set_xlabel("Session 1")
ax.set_ylabel("Session 2")
ax.set_title(r"$\sigma_{wtw}$ (s)")


fig, ax = plt.subplots()
figFxs.my_regplot(df['auc_delta_sess1'], df['auc_delta_sess2'], ax = ax)
ax.set_xlabel("Session 1")
ax.set_ylabel("Session 2")
ax.set_title(r'$\Delta$AUC (s)')


#########################
# plot practice effects 
########################
df = analysisFxs.vstack_sessions(s1_df, s2_df)
ax = sns.swarmplot(data = df, x = "sess", y = "auc", color = "grey", edgecolor = "black", alpha = 0.4, linewidth=1)
sns.boxplot(x="sess", y="auc", data=df, boxprops={'facecolor':'None'}, medianprops={"linestyle":"--", "color": "red"}, ax=ax)
ax.set_xlabel("")
ax.set_ylabel("AUC (s)")


##### 
# plot the reliability distribution #

##### also let me generate the table #####



######################## spilt half reliability #############
for sess in [1, 2]:
if sess == 1:
	trialdata_ = trialdata_sess1_
else:
	trialdata_ = trialdata_sess2_
odd_trialdata_, even_trialdata_ = analysisFxs.split_odd_even(trialdata_)
stats_odd, _, _, _ = analysisFxs.group_MF(odd_trialdata_, plot_each = False)  
stats_even, _, _, _ = analysisFxs.group_MF(even_trialdata_, plot_each = False) 

odd_df = analysisFxs.pivot_by_condition(stats_odd)
even_df = analysisFxs.pivot_by_condition(stats_even)
df = analysisFxs.hstack_sessions(odd_df, even_df, suffixes = ["_odd", "_even"])
fig, ax = plt.subplots()
figFxs.my_regplot(df['auc_odd'], df['auc_even'], ax = ax)
ax.set_xlabel("Odd")
ax.set_ylabel("Even")
ax.set_title('AUC (s)')


odd_df = analysisFxs.pivot_by_condition(stats_odd)
even_df = analysisFxs.pivot_by_condition(stats_even)
df = analysisFxs.hstack_sessions(odd_df, even_df, suffixes = ["_odd", "_even"])
fig, ax = plt.subplots()
figFxs.my_regplot(df['std_wtw_odd'], df['std_wtw_even'], ax = ax)
ax.set_xlabel("Odd")
ax.set_ylabel("Even")
ax.set_title(r"$\sigma_{wtw}$ (s)")



