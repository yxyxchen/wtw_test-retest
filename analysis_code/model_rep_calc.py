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
import pickle
import random

random.seed(10)

# plot styles
plt.style.use('classic')
sns.set(font_scale = 1)
sns.set_style("white")
condition_palette = ["#762a83", "#1b7837"]


for expname in ["active", "passive"]:
    # load data 
    hdrdata_sess1, trialdata_sess1_ = loadFxs.group_quality_check(expname, 1, plot_quality_check = False)
    hdrdata_sess2, trialdata_sess2_ = loadFxs.group_quality_check(expname, 2, plot_quality_check = False)

    ## only exclude valid participants 
    hdrdata_sess1 = hdrdata_sess1[np.isin(hdrdata_sess1["id"], hdrdata_sess2["id"])]
    trialdata_sess1_ = {x: y for x,y in trialdata_sess1_.items() if x[0] in hdrdata_sess2["id"].values}

    ###
    s1_stats, s1_Psurv_b1_, s1_Psurv_b2_, s1_WTW_emp = analysisFxs.group_MF(trialdata_sess1_, plot_each = False)   
    s2_stats, s2_Psurv_b1_, s2_Psurv_b2_, s2_WTW_emp = analysisFxs.group_MF(trialdata_sess2_, plot_each = False)   

    for modelname in ['QL2reset', 'QL2reset_slope', 'QL2reset_slope_two', 'QL2reset_slope_two_simple']:
        fitMethod = "whole"
        stepsize = 0.5
        paranames = modelFxs.getModelParas(modelname)
        npara = len(paranames)
        # replicate task data 
        s1_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, fitMethod, stepsize)
        s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname, fitMethod, stepsize)
        # if it is the first time 
        s1_stats_rep, s1_WTW_rep, s1_dist_vals_ = modelFxs.group_model_rep(trialdata_sess1_, s1_paradf, modelname, 'whole', stepsize, isTrct = True, plot_each = False)
        s2_stats_rep, s2_WTW_rep, s2_dist_vals_ = modelFxs.group_model_rep(trialdata_sess2_, s2_paradf, modelname, 'whole', stepsize, isTrct = True, plot_each = False)
        s1_stats_rep.to_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_sess1_%s_stepsize%.2f.csv'%(modelname, fitMethod, stepsize)), index = None)
        s2_stats_rep.to_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_sess2_%s_stepsize%.2f.csv'%(modelname, fitMethod, stepsize)), index = None)
        dbfile = open(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_%s_stepsize%.2f'%(modelname, fitMethod, stepsize)), "wb")
        dbobj = {
            "s1_WTW_rep": s1_WTW_rep,
            "s2_WTW_rep": s2_WTW_rep
        }
        pickle.dump(dbobj, dbfile)
