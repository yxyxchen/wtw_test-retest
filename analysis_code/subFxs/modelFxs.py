# solve the credict assignment issue in the last timestep 
# nMadeAction

# Ts should be continous though. even when we generate it it is impossible to generate exactly the same. 
# it can always quit ealier though. I don't think it will make a huge difference though lol

import random
import numpy as np
import pandas as pd
import subFxs
from subFxs import analysisFxs
from subFxs import expParas
from subFxs import stancode
from subFxs import simFxs
import code
import stan
import os
import arviz as az
from sksurv.nonparametric import kaplan_meier_estimator as km
import matplotlib.pyplot as plt
import seaborn as sns
# plot styles
plt.style.use('classic')
sns.set_style("white")
sns.set_context("poster")

import concurrent.futures

def getModelParas(modelname):
    """Function to return parameter names of a given model
    """
    if modelname[:3] == 'QL1':
        return ['alpha', 'tau', 'gamma', 'eta']
    elif modelname[:3] == 'QL2':
        return ['alpha', 'nu', 'tau', 'gamma', 'eta']
    elif modelname[:3] == 'RL1':
        return ['alpha', 'tau', 'eta', 'beta']
    elif modelname[:3] == 'RL2':
        return ['alpha', 'nu', 'tau', 'eta', 'beta']

def getModelGroupParas(modelname):
    """Function to return parameter names of a given model
    """
    if modelname == "QL2reset_HM_simple":
        return ['alpha', 'tau', 'eta']
    elif modelname == "QL2reset_slope_simple":
        return ['alpha', "nu", 'tau', 'eta']
    else:
        return ['alpha', 'nu', 'tau', "gamma", 'eta']

def 
def check_stan_diagnosis(fit_summary):
    if any(fit_summary['n_divergent'] > 0) or any(fit_summary['Rhat'] > 1.05) or any(fit_summary['n_eff'] < 400):
        return False
    else:
        return True

##################### 
def ind_model_fit_rep(modelname, paras, trialdata, key, stepsize, plot_each):
    # make sure to use values so that it is not affected by indices
    scheduledDelays = trialdata['scheduledDelay'].values
    scheduledRewards = np.full(scheduledDelays.shape, expParas.tokenValue)
    conditions = trialdata['condition'].values
    blockIdxs = trialdata['blockIdx'].values
    trialEarnings_ = trialdata['trialEarnings'].values
    timeWaited_ = trialdata['timeWaited'].values


    simdata, Qwaits_, Qquit_ =  simFxs.ind_fit_sim(modelname, paras, conditions, blockIdxs, scheduledDelays, scheduledRewards, stepsize)

    # I can plot comparison here honestly
    if plot_each:
        emp_stats, emp_objs = analysisFxs.ind_MF(trialdata, key, isTrct = True, plot_RT = False, plot_trial = False, plot_KMSC = False, plot_WTW = False)
        fig, ax = plt.subplots()
        ax.plot(expParas.TaskTime, emp_objs['WTW'], label = 'Observed')
        ax.plot(expParas.TaskTime, WTW, label = 'Simulated')
    return stats, Psurv_block1, Psurv_block2, WTW, dist_vals 
#################### functions to replicate choice data using individually fitted parameters and analyze them##########
def ind_model_rep(modelname, paras, trialdata, key, nsim, stepsize, plot_each):
    # make sure to use values so that it is not affected by indices
    scheduledDelays = trialdata['scheduledDelay'].values
    scheduledRewards = np.full(scheduledDelays.shape, expParas.tokenValue)
    conditions = trialdata['condition'].values
    blockIdxs = trialdata['blockIdx'].values
    trialEarnings_ = trialdata['trialEarnings'].values
    timeWaited_ = trialdata['timeWaited'].values

    # initialize analysis results of simulated data to return
    # I think I probably only need mean and se?

    # loop over simulations 
    simdata_ = []
    value_df_ = []
    for i in range(nsim):
        simdata, _, _, value_df =  simFxs.ind_sim(modelname, paras, conditions, blockIdxs, scheduledDelays, scheduledRewards, stepsize)
        simdata_.append(simdata)
        value_df_.append(value_df)
    # code.interact(local = dict(globals(), **locals()))
    tmp = pd.concat(value_df_)
    value_df = tmp.groupby(["time", "record_time", "condition"]).agg({"decision_value":np.mean, "relative_value":np.mean}).reset_index()

    # analyze simulated datasets
    stats_, Psurv_block1_, Psurv_block2_, WTW_ = analysisFxs.group_sim_MF(simdata_, trialdata)

    # return summary statistics
    stats = stats_.groupby(['block', 'condition']).agg('mean')
    Psurv_block1 = np.nanmean(Psurv_block1_, axis = 0)
    Psurv_block2 = np.nanmean(Psurv_block2_, axis = 0)
    WTW = np.nanmean(WTW_, axis = 0)

    # return distance 
    dist_vals = analysisFxs.group_sim_dist(simdata_, trialdata)

    # I can plot comparison here honestly
    if plot_each:
        emp_stats, emp_objs = analysisFxs.ind_MF(trialdata, key, isTrct = True, plot_RT = False, plot_trial = False, plot_KMSC = False, plot_WTW = False)
        fig, ax = plt.subplots()
        ax.plot(expParas.TaskTime, emp_objs['WTW'], label = 'Observed')
        ax.plot(expParas.TaskTime, WTW, label = 'Simulated')
    return stats, Psurv_block1, Psurv_block2, WTW, dist_vals, value_df 



#############
# hmm I kinda want to rewrite it later, ... but whatever
def group_model_rep(trialdata_, paradf, modelname, fitMethod, stepsize, isTrct = True, plot_each = False):
    # set random seed
    random.seed(10)

    # get parameter names for this model
    paranames = getModelParas(modelname)

    # initialize outputs
    stats_ = []
    WTW_ = []
    dist_vals_ = []
    # loop over participants
    for key, trialdata in trialdata_.items():
        print(key)
        # code.interact(local = dict(locals(), **globals()))
        # try:
        #     fit_summary = pd.read_csv(os.path.join('..', 'analysis_results', expname, 'modelfit', modelname, '%s_sess%s_summary.txt'%key), header = None)
        # except:
        #     print("the file for %s, sess%d not found"%key)
        #     continue
        # fit_summary.index = paranames + ['totalLL']
        # fit_summary.columns = ['mean', 'se_mean', 'sd', '2.5%', '25%', '50%', '75%', '97.5%', 'n_effe', 'Rhat', 'n_divergent']
        # # yeah for this R version I need to add several more lines

        # if not check_stan_diagnosis(fit_summary):
        #     print(key, "not valid")
        #     continue
        # code.interact(local = dict(locals(), **globals()))
        # # extract mean parameter estimates
        # paravals = fit_summary['mean'].iloc[:-1]
        if key[0] in paradf['id'].values:
            # code.interact(local = dict(locals(), **globals()))
            paravals = paradf.loc[paradf['id'] == key[0], paranames].values[0]
        else:
            continue
        # prepare inputs
        if isTrct:        
            trialdata = trialdata[trialdata.sellTime <= expParas.blocksec - np.max(expParas.tMaxs)]

        if fitMethod == 'trct':
            trialdata = trialdata[trialdata.trialStartTime > 30]
        # replicate original behaviors
        paras = dict(zip(paranames, paravals))
        if plot_each:
            stats, _, _, WTW, dist_vals, _ = ind_model_rep(modelname, paras, trialdata, key, 10, stepsize, plot_each = True)
            plt.show()
            input("Press Enter to continue or ESC to exit...")
            plt.clf()
        else:
            stats, _, _, WTW, dist_vals, _ = ind_model_rep(modelname, paras, trialdata, key, 10, stepsize, plot_each = False)
        stats['id'] = np.full(2, key[0])
        stats['sess'] = np.full(2, key[1])
        stats['key'] = np.full(2, str(key))
        stats_.append(stats)
        dist_vals_.append(dist_vals)
        WTW_.append(WTW)
        # add more code tomorrow.
        # emp_stats, emp_obj = analysisFxs.ind_MF(trialdata, key, plot_trial = True, plot_KMSC = True, plot_WTW = True)
    stats_ = pd.concat(stats_)
    stats_ = stats_.reset_index()
    WTW_ = np.array([e for e in WTW_])
    dist_vals_ = np.array(dist_vals_)
    # code.interact(local = dict(locals(), **globals()))
    # temporarily let's save it to save time
    # stats_.to_csv(os.path.join('..', 'analysis_results', 'taskstats', 'rep_%s_sess%d.csv'%(modelname, key[1])), index = None)
    return stats_, WTW_, dist_vals_
       






