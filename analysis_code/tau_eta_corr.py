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
modelname = 'QL2reset_ind'
fitMethod = "whole"
stepsize = 0.5
subtitles = [r'$\mathbf{log(\alpha)}$', r'$\mathbf{log(\nu)}$', r'$\mathbf{\tau}$', r'$\mathbf{\gamma}$', r'$\mathbf{log(\eta)}$']
paranames = modelFxs.getModelParas(modelname)
npara = len(paranames)
s1_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, fitMethod, stepsize)
s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname, fitMethod, stepsize)


ind_fit_sim(modelname, paras, condition_, blockIdx_, scheduledDelay_, scheduledReward_, observed_trialEarnings_, observed_timeWaited_, stepsize, empirical_iti = expParas.iti)


random.seed(10)
for key, trialdata in trialdata_.items():
    print(key)
    if key[0] in paradf['id'].values:
        paravals = paradf.loc[paradf['id'] == key[0], paranames].values[0]
    else:
        continue
trialdata = trialdata[trialdata.sellTime <= expParas.blocksec - np.max(expParas.tMaxs)]
scheduledDelays = trialdata['scheduledDelay'].values
scheduledRewards = np.full(scheduledDelays.shape, expParas.tokenValue)
conditions = trialdata['condition'].values
blockIdxs = trialdata['blockIdx'].values
trialEarnings_ = trialdata['trialEarnings'].values
timeWaited_ = trialdata['timeWaited'].values
paras = dict(zip(paranames, paravals))
simdata, Qwaits_, Qquit_ =  simFxs.ind_fit_sim(modelname, paras, conditions, blockIdxs, scheduledDelays, scheduledRewards,trialEarnings_, timeWaited_, stepsize)
# visualize action values 

ts = np.arange(0, max(expParas.tMaxs), stepsize) 
Qwaits_ = Qwaits_ - np.tile(Qquit_,len(ts)).reshape(len(ts),10)
plotdf = pd.DataFrame({
	"time": np.tile(ts, 10),
	"record_time": np.tile(np.repeat((np.arange(5)+1) * 2, len(ts)), 2),
	"Qwait": Qwaits_.transpose().reshape(-1) * paras["tau"],
	"condition" : np.repeat(np.repeat(("LP", "HP"), 5), len(ts))
	})
g = sns.FacetGrid(data = plotdf, hue = "record_time", col = "condition")
g.map(sns.lineplot, "time", "Qwait")
sns.lineplot(data = plotdf[plotdf["condition"] == "HP"], x = "time", y = "Qwait", hue = "record_time")
