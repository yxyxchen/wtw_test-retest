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


################## 
expname = "passive"
s1_selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
hdrdata_sess1, trialdata_sess1_ = loadFxs.group_quality_check(expname, 1, plot_quality_check = True)
hdrdata_sess2, trialdata_sess2_ = loadFxs.group_quality_check(expname, 2, plot_quality_check = True)


########### only include participants complete both sessions
hdrdata_sess1 = hdrdata_sess1[np.isin(hdrdata_sess1["id"], hdrdata_sess2["id"])]
trialdata_sess1_ = {x: y for x,y in trialdata_sess1_.items() if x[0] in hdrdata_sess2["id"].values}


#################
_, _, _, s1_WTW_ = analysisFxs.group_MF(trialdata_sess1_, plot_each = False, isTrct = False)  
_,_,_, s2_WTW_ = analysisFxs.group_MF(trialdata_sess2_, plot_each = False, isTrct = False)


s1_stats, s1_Psurv_b1_, s1_Psurv_b2_, _ = analysisFxs.group_MF(trialdata_sess2_, plot_each = False)    
s2_stats, s2_Psurv_b1_, s2_Psurv_b2_, _ = analysisFxs.group_MF(trialdata_sess2_, plot_each = False)  

plt.style.use('classic')
sns.set(font_scale = 2)
plt.tight_layout()
sns.set_style("white")
fig, ax = plt.subplots()
figFxs.plot_group_KMSC_both(s1_Psurv_b1_, s1_Psurv_b2_, s2_Psurv_b1_, s2_Psurv_b2_, hdrdata_sess1, hdrdata_sess2, ax)
fig.tight_layout()
plt.savefig(os.path.join('..', 'figures', expname, 'KMSC.pdf'))

fig, ax = plt.subplots()
plt.style.use('classic')
sns.set(font_scale = 4)
plt.tight_layout()
sns.set_style("white")
figFxs.plot_group_WTW_both(s1_WTW_, s2_WTW_, hdrdata_sess1, hdrdata_sess2, ax)
ax.set_ylim([4, 12])
ax.set_ylabel("Willingness-to-wait (s)")
fig.tight_layout()
fig.savefig(os.path.join('..', 'figures', expname, 'WTW.pdf'))

############# make sure there is learning 
plt.style.use('classic')
sns.set(font_scale = 2)
sns.set_style("white")
for i in np.arange(3):
    s1_stats["auc%d-auc%d"%(i+2, i+1)] = s1_stats["auc%d"%(i+2)] - s1_stats["auc%d"%(i+1)] 
    s2_stats["auc%d-auc%d"%(i+2, i+1)] = s2_stats["auc%d"%(i+2)] - s2_stats["auc%d"%(i+1)] 

statsdf = pd.concat([s1_stats, s2_stats], axis = 0)
plotdf = statsdf.melt(id_vars = ["id", "condition", "sess"], value_vars =["auc2-auc1", "auc3-auc2", "auc4-auc3"])

fig, axes = plt.subplots(1, 2)
for i, condition in enumerate(["LP", "HP"]):
    sns.boxplot(data = plotdf[plotdf["condition"] == condition], x="variable", y="value", hue = "sess", ax = axes[i], palette = ["#b2182b", "#2166ac"])
    axes[i].plot([-1,3], [0,0], linestyle = "dotted", color = "orange", linewidth = 3)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
    if i == 0:
        axes[i].legend().set_visible(False)
    else:
        axes[i].legend(loc = "upper left")
fig.savefig(os.path.join('..', 'figures', expname, 'within-block_auc_adaptation.pdf'))

#########
plotdf = statsdf.melt(id_vars = ["id", "condition", "sess"], value_vars =["auc%d"%(x + 1) for x in np.arange(4)])
fig, axes = plt.subplots(1, 2)
for i, condition in enumerate(["LP", "HP"]):
    sns.pointplot(data = plotdf[plotdf["condition"] == condition], x="variable", y="value", hue = "sess", ax = axes[i], palette = ["#b2182b", "#2166ac"])
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
    if i == 0:
        axes[i].legend().set_visible(False)
    else:
        axes[i].legend(loc = "upper left")
fig.savefig(os.path.join('..', 'figures', expname, "sub_auc.pdf"))

#########


