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


eta_vals = np.linspace(0.5, 5, 10)
tau_vals = np.linspace(1, 10, 10)

ts = np.linspace(1, 20, 20)
auc_ = np.zeros(shape = (len(eta_vals), len(tau_vals)))
for i, j in itertools.product(np.arange(len(tau_vals)), np.arange(len(eta_vals))):
    tau = tau_vals[i]
    eta = eta_vals[j]
    pwaits = 1 / (1 + np.exp((0.1*ts - eta) * tau))
    psurvivals = np.cumprod(pwaits)
    auc_[i,j] = np.sum(np.concatenate([[1], psurvivals[:-1]]) + psurvivals) / 2


pd.DataFrame(auc_, columns = ["eta = %.1f"%x for x in eta_vals], index = ["tau = %.1f"%x for x in tau_vals])



########### simulation based on empirical data 
########### maybe later let me choose one sequence, and then 
########### model changes I can make, I probably prefer this method ...., and also change gamma to customized gamma 
########### 
expname = 'passive'
hdrdata_sess1, trialdata_sess1_ = loadFxs.group_quality_check(expname, 1, plot_quality_check = False)
hdrdata_sess2, trialdata_sess2_ = loadFxs.group_quality_check(expname, 2, plot_quality_check = False)

# load 
modelname = 'QL2reset'
fitMethod = "whole"
stepsize = 0.5
subtitles = [r'$\mathbf{log(\alpha)}$', r'$\mathbf{log(\nu)}$', r'$\mathbf{\tau}$', r'$\mathbf{\gamma}$', r'$\mathbf{log(\eta)}$']
paranames = modelFxs.getModelParas(modelname)
npara = len(paranames)
s1_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, fitMethod, stepsize)
s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname, fitMethod, stepsize)

trialdata_ = trialdata_sess1_
paradf = s1_paradf
key = ('s1005', 1)
trialdata = trialdata_[key]

paravals = []
paravals = paradf.loc[paradf['id'] == key[0], paranames].values[0]
trialdata = trialdata[trialdata.sellTime <= expParas.blocksec - np.max(expParas.tMaxs)]
scheduledDelays = trialdata['scheduledDelay'].values
scheduledRewards = np.full(scheduledDelays.shape, expParas.tokenValue)
conditions = trialdata['condition'].values
blockIdxs = trialdata['blockIdx'].values
trialEarnings_ = trialdata['trialEarnings'].values
timeWaited_ = trialdata['timeWaited'].values
paras = dict(zip(paranames, paravals))
##########################
for chosen_para in ["eta", "alpha", "tau"]:
    if chosen_para == "eta":
        if modelname = "QLreset":
            chosen_para_vals = [0.2, 0.4, 0.6, 0.8, 1]
        if modelname = "QL2reset_ind":
            chosen_para_vals = [1, 2, 3, 4, 5]
    elif chosen_para == "tau":
        if modelname = "QLreset":
            chosen_para_vals = [round(x,1) for x in np.exp(np.linspace(np.log(0.5), np.log(32), 5))]
        if modelname = "QL2reset_ind":
            chosen_para_vals = [round(x, 1) for x in np.exp(np.linspace(np.log(5), np.log(40), 5))]
    elif chosen_para == "alpha":
        chosen_para_vals = [0.01, 0.02, 0.03, 0.04, 0.05]  
    stats_ = []
    WTW_ = []
    value_df_ = []
    for i, val in enumerate(chosen_para_vals):
        paras = dict(zip(paranames, paravals))
        paras[chosen_para] = val
        stats, Psurv_block1, Psurv_block2, WTW, dist_vals, value_df = modelFxs.ind_model_rep(modelname, paras, trialdata, key, 10, stepsize, False)
        stats[chosen_para] = val
        stats_.append(stats)
        WTW_.append(WTW)
        value_df_.append(value_df)
    statsdf = pd.concat(stats_).reset_index()
    g = sns.FacetGrid(data = statsdf, col = "condition")
    g.map(sns.barplot, chosen_para, "auc")
    g.savefig(os.path.join("..", "figures", "combined", "sim_%s_%s_auc.pdf"%(modelname, chosen_para)))
    statsdf = pd.concat(stats_).reset_index()
    g = sns.FacetGrid(data = statsdf, col = "condition")
    g.map(sns.barplot, chosen_para, "std_wtw")
    g.savefig(os.path.join("..", "figures", "combined", "sim_%s_%s_std_wtw.pdf"%(modelname, chosen_para)))
    diffdf = pd.DataFrame({
        chosen_para: chosen_para_vals,
        "auc_delta": statsdf.loc[statsdf["condition"] == "HP", "auc"].values - statsdf.loc[statsdf["condition"] == "LP", "auc"].values
        })
    fig, ax = plt.subplots()
    sns.barplot(data = diffdf, x = chosen_para, y = "auc_delta", ax = ax)
    fig.savefig(os.path.join("..", "figures", "combined", "sim_%s_%s_auc_delta.pdf"%(modelname, chosen_para)))
    wtwdf = pd.DataFrame({
        "wtw": np.concatenate(WTW_),
        "time": np.tile(expParas.TaskTime, len(chosen_para_vals)),
        chosen_para: np.repeat(chosen_para_vals, len(expParas.TaskTime))
        })
    fig, ax = plt.subplots()
    sns.lineplot(data = wtwdf, x = "time", y = "wtw", hue = chosen_para, ax = ax)
    fig.savefig(os.path.join("..", "figures", "combined", "sim_%s_%s_wtw.pdf"%(modelname, chosen_para)))


