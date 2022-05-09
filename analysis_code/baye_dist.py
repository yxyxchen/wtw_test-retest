
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
from sksurv.nonparametric import kaplan_meier_estimator as km
from scipy.interpolate import interp1d
import code



def corrdot(*args, **kwargs):
    corr_r = args[0].corr(args[1], 'pearson')
    corr_text = f"{corr_r:2.2f}".replace("0.", ".")
    ax = plt.gca()
    ax.set_axis_off()
    marker_size = abs(corr_r) * 10000
    ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
               vmin=-1, vmax=1, transform=ax.transAxes)
    font_size = abs(corr_r) * 40 + 5
    ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                ha='center', va='center', fontsize=font_size)


1225


fig, axs = plt.subplots(5)
for i, id in enumerate(ids):
    
##################
id = "s1008"
fig, ax = plt.subplots()
analysisFxs.plot_ind_both_wtw(trialdata_sess1_[(id, 1)], trialdata_sess2_[(id, 2)], ax)

sampledf = pd.read_csv("../analysis_results/passive/modelfit/QL1/%s_sess1_sample.txt"%id, index_col = None, header = None)
summarydf = pd.read_csv("../analysis_results/passive/modelfit/QL1/%s_sess1_summary.txt"%id, index_col = None, header = None)

sampledf.columns = ["alpha", "tau", "gamma", "eta", "totalLL"]

sampledf.apply(min)
sampledf.apply(max)

sns.set(style='white', font_scale=1.6)
g = sns.PairGrid(sampledf.iloc[:,:4], aspect=1.4, diag_sharey=False)
g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={'color': 'black'})
g.map_diag(sns.distplot, kde_kws={'color': 'black'})
g.map_upper(corrdot)

