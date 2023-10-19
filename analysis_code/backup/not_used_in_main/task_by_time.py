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
import scipy as sp
import scipy

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

################
s1_df_ = []
s2_df_ = []
s1_stats_ = []
s2_stats_ = []
trialdata_sess1_list = []
trialdata_sess2_list = []
for expname in ['active', 'passive', "combined"]:
    if expname == "combined":
        #s1_df, s2_df, s1_stats, s2_stats = pd.concat(s1_df_), pd.concat(s2_df_), pd.concat(s1_stats_), pd.concat(s2_stats_)
        trialdata_sess1_, trialdata_sess2_ = copy.copy(trialdata_sess1_list[0]), copy.copy(trialdata_sess2_list[0])
        trialdata_sess1_.update(trialdata_sess1_list[1])
        trialdata_sess2_.update(trialdata_sess2_list[1])
    else:
        hdrdata_sess1, trialdata_sess1_ = loadFxs.group_quality_check(expname, 1, plot_quality_check = True)
        hdrdata_sess2, trialdata_sess2_ = loadFxs.group_quality_check(expname, 2, plot_quality_check = True)
        # only include participants who completed two sessions
        hdrdata_sess1 = hdrdata_sess1[np.isin(hdrdata_sess1["id"], hdrdata_sess2["id"])]
        trialdata_sess1_ = {x: y for x,y in trialdata_sess1_.items() if x[0] in hdrdata_sess2["id"].values}
        trialdata_sess1_list.append(trialdata_sess1_)
        trialdata_sess2_list.append(trialdata_sess2_)



# loop over cutoffs 
rl_list = [] 
cr_list = []
for cutoff in [(x+1)* 60 for x in np.arange(10)]:
    s1_stats, s1_Psurv_b1_, s1_Psurv_b2_, s1_WTW_ = analysisFxs.group_MF(trialdata_sess1_, plot_each = False, cutoff = cutoff)   
    s2_stats, s2_Psurv_b1_, s2_Psurv_b2_, s2_WTW_ = analysisFxs.group_MF(trialdata_sess2_, plot_each = False, cutoff = cutoff)   
    s1_df = analysisFxs.pivot_by_condition(s1_stats)
    s2_df = analysisFxs.pivot_by_condition(s2_stats)
    s1_df_.append(s1_df)
    s2_df_.append(s2_df)
    s1_stats_.append(s1_stats)
    s2_stats_.append(s2_stats)
    trialdata_sess1_list.append(trialdata_sess1_)
    trialdata_sess2_list.append(trialdata_sess2_)
    # reliability table
    df = analysisFxs.hstack_sessions(s1_df, s2_df)
    _, _, _, _, _, report = analysisFxs.calc_zip_reliability(df, [(x + '_sess1', x + '_sess2') for x in ["auc", "std_wtw", "auc_delta"]])
    rl_list.append(report.loc[:,'spearman_rho'].values)
    print(report.round(3))
    # hmmm correlations 
    ###### correlations among measures #########
    statsdf = analysisFxs.agg_across_sessions(s1_df, s2_df)
    tmp = statsdf[vars].corr(method = 'spearman')
    cr_list.append([tmp.iloc[1,0], tmp.iloc[2,0], tmp.iloc[2,1]])


# I need to plot the figure here 
rldf = pd.DataFrame(dict({"var": np.repeat(labels, 10),
    "rl": np.vstack(rl_list).flatten(order = "F"),\
    "time": np.tile([(x+1) for x in np.arange(10)], 3)}))

# 
g = sns.FacetGrid(data = rldf, col = "var")
g.map(sns.lineplot, "time", "rl", color = "grey")
g.map(sns.scatterplot, "time", "rl", color = "grey")
g.set(xlabel ="Time spent in each environment (min)", ylabel = "Spearman's rho")
g.set_titles("{col_name}")


pd.DataFrame(cr_list)

crdf = pd.DataFrame(dict({"name": np.tile(["AUC~$\\sigma_\\mathrm{wtw}$", "AUC~"+'$\\Delta$ AUC', '$\\sigma_\\mathrm{wtw}$~'+'$\\Delta$ AUC'], 10),
    "cr": np.concatenate(cr_list),\
    "time": np.repeat([(x+1) for x in np.arange(10)], 3)}))
g = sns.FacetGrid(data = crdf, col = "name")
g.map(sns.lineplot, "time", "cr", color = "grey")
g.map(sns.scatterplot, "time", "cr", color = "grey")
g.set(xlabel ="Time spent in each environment (min)", ylabel = "Spearman's rho")
g.set_titles("{col_name}")

####################### let me try the PCA analysis 
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(s1_df[['auc', 'std_wtw', 'auc_delta']])
y = StandardScaler().fit_transform(s2_df[['auc', 'std_wtw', 'auc_delta']])
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pcafit = pca.fit(x)
x_pca = pcafit.transform(x)
y_pca = pcafit.transform(y)


spearmanr(x_pca[:,0], y_pca[:,0])[0]
spearmanr(x_pca[:,1], y_pca[:,1])[0]
spearmanr(x_pca[:,2], y_pca[:,2])[0]

# PCA explained 
pca.explained_variance_ratio_

pd.DataFrame(pca.components_, columns = vars, index = ['feature_' + str(x + 1) for x in np.arange(3)])

########### for self-report 
expname = "passive"
s1_selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
s2_selfdf = loadFxs.parse_group_selfreport(expname, 2, isplot = False)
s1_selfdf = s1_selfdf.dropna()
s2_selfdf = s2_selfdf.dropna()

s1_selfdf = s1_selfdf[np.isin(s1_selfdf["id"], s2_selfdf["id"])]
s2_selfdf = s2_selfdf[np.isin(s2_selfdf["id"], s1_selfdf["id"])]
# I need to exclude data ... 
# 
self_s1 = StandardScaler().fit_transform(s1_selfdf[['UPPS', 'BIS', 'discount_logk']])
self_s2 = StandardScaler().fit_transform(s2_selfdf[['UPPS', 'BIS', 'discount_logk']])
pca = PCA(n_components=3)
pcafit = pca.fit(self_s1)
x_pca = pcafit.transform(self_s1)
y_pca = pcafit.transform(self_s2)

spearmanr(x_pca[:,0], y_pca[:,0])[0]
spearmanr(x_pca[:,1], y_pca[:,1])[0]
spearmanr(x_pca[:,2], y_pca[:,2])[0]



