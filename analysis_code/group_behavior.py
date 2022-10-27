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




expname = "active"
s1_selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
hdrdata_sess1, trialdata_sess1_ = loadFxs.group_quality_check(expname, 1, plot_quality_check = True)
hdrdata_sess2, trialdata_sess2_ = loadFxs.group_quality_check(expname, 2, plot_quality_check = True)






########
s1_stats, s1_Psurv_b1_, s1_Psurv_b2_, s1_WTW_ = analysisFxs.group_MF(trialdata_sess1_, plot_each = False)  


fig, ax = plt.subplots()
figFxs.plot_group_KMSC_both(s1_Psurv_b1_, s1_Psurv_b2_, s2_Psurv_b1_, s2_Psurv_b2_, hdrdata_sess1, hdrdata_sess2, ax)
plt.savefig(os.path.join('..', 'figures', expname, 'KMSC.pdf'))


fig, ax = plt.subplots()
plt.style.use('classic')
sns.set(font_scale = 4)
plt.tight_layout()
sns.set_style("white")
figFxs.plot_group_WTW_both(s1_WTW_, s2_WTW_, hdrdata_sess1, hdrdata_sess2, ax)
ax.set_ylim([4, 10])
ax.set_ylabel("Willingness-to-wait")
fig.savefig(os.path.join('..', 'figures', expname, 'WTW.pdf'))


############# make sure there is learning 
for sess in [1, 2]:
    if sess == 1:
        tmp = s1_stats.melt(id_vars = ["id", "condition"], value_vars = ['auc1', 'auc2', 'auc3', 'auc4'])
    else:
        tmp = s2_stats.melt(id_vars = ["id", "condition"], value_vars = ['auc1', 'auc2', 'auc3', 'auc4'])
    plotdf = tmp.groupby(["condition", "variable"]).agg(mean = ("value", np.mean), se = ("value", analysisFxs.calc_se)).reset_index()
    g = sns.FacetGrid(plotdf, col = "condition")
    g.map(sns.barplot, "variable", "mean")
    g.map(plt.errorbar, "variable", "mean", "se", color = "red")
    plt.tight_layout()
    g.savefig(os.path.join('..', 'figures', expname, 'sub_auc_sess%d.pdf'%sess))

s1_stats.groupby('condition').apply(lambda x: (x['auc4'] - x['auc1']).mean()).round(2)
s1_stats.groupby('condition').apply(lambda x: (stats.ttest_rel(x['auc4'], x['auc1'])))
s2_stats.groupby('condition').apply(lambda x: (x['auc4'] - x['auc1']).mean()).round(2)
s2_stats.groupby('condition').apply(lambda x: (stats.ttest_rel(x['auc4'], x['auc1'])))

for sub1 in range(1, 4):
    print(sub1)
    # s1_stats.groupby('condition').apply(lambda x: (x['auc'+str(sub1+1)] - x['auc'+str(sub1)]).mean()).round(2)
    # s1_stats.groupby('condition').apply(lambda x: (stats.ttest_rel(x['auc'+str(sub1+1)], x['auc'+str(sub1)])))
    s2_stats.groupby('condition').apply(lambda x: (x['auc'+str(sub1+1)] - x['auc'+str(sub1)]).mean()).round(2)
    s2_stats.groupby('condition').apply(lambda x: (stats.ttest_rel(x['auc'+str(sub1+1)], x['auc'+str(sub1)])))

###########

