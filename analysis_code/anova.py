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
from scipy.stats import mannwhitneyu

# plot styles
plt.style.use('classic')
sns.set(font_scale = 1)
sns.set_style("white")
condition_palette = ["#762a83", "#1b7837"]

#############   
def var2label(var):
    if var == "auc":
        label = "AUC (s)"
    elif var == "std_wtw":
        label = r"$\sigma_\mathrm{wtw}$ (s)"
    elif var == "auc_delta":
        label = r'$\Delta$ AUC (s)'
    else:
        return var
    return label

vars = ['auc', 'std_wtw', 'auc_delta']
labels = ["AUC (s)", r"$\sigma_\mathrm{wtw}$ (s)", r'$\Delta$ AUC (s)']

stats_df_ = []
################
for expname in ['active', 'passive']:
    s1_selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
    hdrdata_sess1, trialdata_sess1_ = loadFxs.group_quality_check(expname, 1, plot_quality_check = True)
    hdrdata_sess2, trialdata_sess2_ = loadFxs.group_quality_check(expname, 2, plot_quality_check = True)
    # only include participants who completed two sessions
    hdrdata_sess1 = hdrdata_sess1[np.isin(hdrdata_sess1["id"], hdrdata_sess2["id"])]
    trialdata_sess1_ = {x: y for x,y in trialdata_sess1_.items() if x[0] in hdrdata_sess2["id"].values}
    s1_stats, s1_Psurv_b1_, s1_Psurv_b2_, s1_WTW_ = analysisFxs.group_MF(trialdata_sess1_, plot_each = False)   
    s2_stats, s2_Psurv_b1_, s2_Psurv_b2_, s2_WTW_ = analysisFxs.group_MF(trialdata_sess2_, plot_each = False)   
    stats_df_.append(s1_stats)
    stats_df_.append(s2_stats)

stats_df_[0]['exp'] = 'active'
stats_df_[1]['exp'] = 'active'
stats_df_[2]['exp'] = 'passive'
stats_df_[3]['exp'] = 'passive'
stats_df = pd.concat(stats_df_)
stats_df['block'] = pd.Categorical(stats_df['block'])
stats_df['sess'] = pd.Categorical(stats_df['sess'])
smf.ols('auc ~ block * sess * exp', data = stats_df).fit().summary()
