import pandas as pd
import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
# plot styles
plt.style.use('classic')
sns.set(font_scale = 1.5)
sns.set_style("white")
import itertools
import copy # pay attention to copy 
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sksurv.nonparametric import kaplan_meier_estimator as km
from scipy.interpolate import interp1d
from subFxs import expParas
import code


#############  some basic helper functions 
def calc_se(x):
    """calculate standard error after removing na values 
    """
    # if not isinstance(x, pd.Series):
    #     x = pd.Series(x)

    if not isinstance(x, np.ndarray):
        x = np.array(x)
    size = len(x)
    x = x[~np.isnan(x)]
    ndrop = size - len(x) 
    if ndrop > 0:
        print("Remove NaN values in calculating standard error"%ndrop)
    return np.nanstd(x) / np.sqrt(len(x))

####################################################
def score_PANAS(choices, isplot = True):
    """score PANAS questionaire answers

    Inputs:
        choices: choice data, from 1 to 4
        isplot: whether to plot figures or not
    """
    # questionaire inputs: items and reversed items for each component 
    PAS_items = [1, 3, 5, 9, 10, 12, 14, 16, 17, 19] # positive affect items. Scores can range from 10-50, higher scores -> higher lvels of positive affect
    NAS_items  = [2, 4, 6, 7, 8, 11, 13, 15, 18, 20] # negative affect items. higher scores -> higher lvels of positive affect

    # calculate PAS and NAS scores
    PAS = choices.loc[['PA-' + str(x) for x in PAS_items]].sum()
    NAS = choices.loc[['PA-' + str(x) for x in NAS_items]].sum()

    # plot
    if isplot:
        fig, ax = plt.subplots()
        ax.acorr(choices, maxlags = 10)
        ax.set_ylabel("PANAS Autocorrelation")
        ax.set_xlabel("Lags")

    return PAS, NAS

def score_BIS(choices, isplot = True):
    """score BIS questionaire answers

    Inputs:
        choices: choice data, from 1 to 4
        isplot: whether to plot figures or not

    Outputs:

    """
    # questionaire inputs: items and reversed items for each component
    attention_items = [[5, 11, 28], [9, 20]]
    cogstable_items = [[6, 24, 26], []]
    motor_items = [[2, 3, 4, 17, 19, 22, 25], []]
    perseverance_items = [[16, 21, 23], [30]]
    selfcontrol_items = [[], [1, 7, 8, 12, 13,  14]]
    cogcomplex_items = [[18, 27], [10, 15, 29]]

    def items2score(items):
        score = np.sum([choices.loc['BIS-'+str(item)] for item in items[0]]) +\
        np.sum([4 - choices.loc['BIS-'+str(item)] for item in items[1]])
        return score

    # calculate scores 
    attention = items2score(attention_items)
    cogstable = items2score(cogstable_items)
    motor = items2score(motor_items)
    perseverance = items2score(perseverance_items)
    selfcontrol = items2score(selfcontrol_items)
    cogcomplex = items2score(cogcomplex_items)
    
    Attentional = attention + cogstable
    Motor = motor + perseverance
    Nonplanning  = selfcontrol + cogcomplex
    BIS = Attentional + Motor + Nonplanning

    # plot 
    if isplot:
        fig, ax = plt.subplots()
        ax.acorr(choices, maxlags = 10)
        ax.set_ylabel("BIS Autocorrelation")
        ax.set_xlabel("Lags")
    return (attention, cogstable, motor, perseverance, selfcontrol, cogcomplex), Attentional, Motor, Nonplanning, BIS

def score_upps(choices, isplot = True):
    """score upps questionaire answers 
    Inputs:
        choices: choice data, from 1 to 4
        isplot: whether to plot figures or not 
    
    Outputs:
        NU: negative urgency, 
        PU: postitive urgency 
        PM: lack of premeditation
        PS: lack of perseverance
        SS: sensation seeking
        UPPS: overall score. Higher values mean more impulsive 
        
    """
    # questionaire inputs: items and reversed items for each component
    NU_items =  [[53], [2, 7, 12, 17, 22, 29, 34, 39, 44, 50, 58]] # negative urgency 
    PU_items = [[], [5, 10,  15, 20, 25, 30, 35, 40, 45, 49, 52, 54, 57, 59]] # positive urgency
    PM_items = [[1, 6, 11, 16, 21, 28, 33, 38, 43, 48, 55], []] # lack of premeditation
    PS_items = [[4, 14, 19, 24, 27, 32, 37,  42], [9, 47]] # lack of perseverance
    SS_items = [[], [3, 8, 13, 18, 23, 26, 31, 36, 41, 46, 51, 56]] # sensation seeking
    
    def items2score(items):
        score = np.sum([choices.loc['UP-'+str(item)] for item in items[0]]) +\
        np.sum([4 - choices.loc['UP-'+str(item)] for item in items[1]])
        return score

    # calculate scores 
    NU = items2score(NU_items)
    PU = items2score(PU_items)
    PM = items2score(PM_items)
    PS = items2score(PS_items)
    SS = items2score(SS_items)
    UPPS = NU + PU + PM + PS + SS
    
    # determine data quality 
    if isplot:
        fig, ax = plt.subplots()
        ax.acorr(choices, maxlags = 10)
        ax.set_ylabel("UPPS Autocorrelation")
        ax.set_xlabel("Lags")
    # c = collections.Counter(choices.values)
    return NU, PU, PM, PS, SS, UPPS
        
    
def calc_k(ddchoices, isplot = True):
    """Estimate k, log(k) and the standard error of log(k) for the questionaire
    
    Inputs:
        ddchoices: a pandas series of choice data. 1 -> immediate reward, 2-> delayed reward
        
    Outputs:
        k: discounting parameter k
        logk: log(k)
        se_logk: standard error of log(k)
        
    Comments:
        Here we use the logistic regression method described in Wileyto et al. (2004).
        It enables us to gauge the uncertianty in parameter estimation.
        The alternative method, one that is based on indifference points,is ideal for determining the shape of the function. 
    """
    #print(ddchoices)
    # questionaire inputs
    Vi = np.array([54, 55, 19, 31, 14, 47, 15, 25, 78, 40, 11, 67, 34, 27, 69, 49, 80, 24 ,33, 28, 34, 25, 41, 54, 54, 22, 20]) # immediate reward
    Vd = np.array([55, 75, 25, 85, 25, 50, 35, 60, 80, 55, 30, 75, 35, 50, 85, 60, 85, 35, 80, 30, 50, 30, 75, 60, 80, 25, 55]) # delayed reward
    T = np.array([117, 61, 53, 7, 19, 160, 13, 14, 162, 62, 7, 119, 186, 21, 91, 89, 157, 29, 14, 179, 30, 80, 20, 111, 30, 136, 7]) # delay in days
    
    # transformed variables 
    R = Vi / Vd #reward ratio
    TR = 1 - 1/R # transformed reward ratio
    pD = ddchoices.values.astype("float") - 1 # prob of choosing delayed rewards
    percentD = pD.sum() / len(pD)
    
    # logistic regression
    np.column_stack((pD,TR, T))
    regdf = pd.DataFrame({
        "Vi" : Vi,
        "Vd" : Vd, 
        'R': R,
        "T": T,
        "TR": TR,
        "pD": pD,
        
    })
    if all(pD == 1) or all(pD == 0) or sum(abs(np.diff(regdf.sort_values(by = 'TR').pD))) <= 1 \
    or sum(abs(np.diff(regdf.sort_values(by = 'T').pD))) <= 1: 
        print("All the same intertemporal choices.")
        k = np.nan
        logk = np.nan
        var_logk = np.nan
        se_logk = np.nan
    else:
        try:
            results = smf.glm("pD ~ -1 + TR + T",data = regdf, family=sm.families.Binomial()).fit()
        except:
            print("Use Bayesian Methods here")
            code.interact(local = dict(locals(), **globals()))

        # calculate k and related stats
        k = results.params[1]/results.params[0]
        logk = np.log(k)
        g = np.array([-1 / results.params[0], 1 / results.params[1]]) # first order derivative of logk on betas 
        var_logk = g.dot(results.cov_params()).dot(g.T)
        se_logk = np.sqrt(var_logk)
    
        # calculate SV
        regdf['SV'] = regdf.eval("Vd / (1 + @k * T) / Vi")
        
        # bin the SV variable
        regdf['SV_bin'] = pd.qcut(regdf['SV'], 5)

        # if plot figures 
        if isplot:
            fig, ax = plt.subplots(1,2)
            # check data quality
            ax[0].plot(pD)
            
            # check model fit 
            plotdf = regdf.groupby("SV_bin").mean()
            plotdf.plot("SV", "pD", ax = ax[1])
            ax[1].vlines(1,0,1, color = "r", linestyles ="dotted")
            ax[1].hlines(0.5,0,3, color = "r", linestyles ="dotted")
            ax[1].set_xlabel("SV (fraction of the immediate reward)")
            ax[1].set_ylabel("P(delayed)")
            ax[1].set_title("k = %.3f, logk_se = %.3f"%(k, se_logk))
            ax[1].set_ylim([-0.1, 1.1])
        
    return k, logk, se_logk, percentD


def resample(ys, xs, Xs):
    """ Resample pair-wise sequences, to the closet right point 
    
    Inputs:
        ys: y in the original sequence
        xs: x in the original sequence
        Xs: x in the new sequence
    
    Outputs: 
    Ys : y in the new sequence 
    """
    Ys = [ys[xs >= X][0] if X <= max(xs) else ys[-1] for X in Xs]
    return Ys

def kmsc(data, tMax, Time, plot_KMSC = False):
    """Survival analysis of willingness to wait 
    Inputs:
        data: task data
        tMax: duration of the analysis time window
        plotKMSC: whether to plot 
        Time: upsampled time data
        
    Outputs:
        time:time data
        psurv: prob of survival data
        Time: upsampled time data
        Psurv: upsampled prob of survival data
        auc: area under the survival curve
        std_wtw: std of willingness to wait across trials. 
    """
    durations = data.timeWaited
    event_observed = np.equal(data.trialEarnings, 0) # 1 if the participant quits and 0 otherwise 
    time, psurv = km(event_observed, durations)
    
    # add the first and the last datapoints
    psurv = psurv[time < tMax]
    time = time[time < tMax]
    time = np.insert(time, 0, 0);  time = np.append(time, tMax)
    psurv = np.insert(psurv, 0, 1);  psurv = np.append(psurv, np.max(psurv[-1],0))
    
    # upsample to a high resolution 
    Psurv = resample(psurv, time, Time) 
    
    # plot 
    if plot_KMSC:
        plt.plot(time, psurv)
        plt.legend()
    
    # calculate AUC
    auc = np.sum(np.diff(time) * psurv[:-1])
    
    # calculate std_wtw
    cdf_wtw = 1 - psurv
    cdf_wtw[-1] = 1  # assume everyone quits at tMax
    pdf_wtw = np.diff(cdf_wtw)
    
    # np.sum(pdf_wtw * time[1:]) #check, right-aligned rule
    var_wtw = np.sum(pdf_wtw * (time[1:] - auc)**2)
    std_wtw = np.sqrt(var_wtw)
    
    return time, psurv, Time, Psurv, auc, std_wtw


def rtplot_multiblock(trialdata):
    """ Plot figures to visually check RT, for multiple blocks
    """ 
    # calc ready_RT
    trialdata.eval("ready_RT = trialStartTime - trialReadyTime", inplace = True)
    blockbounds = [max(trialdata.totalTrialIdx[trialdata.blockIdx == i]) for i in np.unique(trialdata.blockIdx)]

    # plot
    fig, ax = plt.subplots(1,6)
    # ready RT timecourse
    trialdata.plot("totalTrialIdx",  "ready_RT", ax = ax[0])
    ax[0].set_ylabel("Ready RT (s)")
    ax[0].set_xlabel("Trial")
    ax[0].get_legend().remove()
    ax[0].vlines(blockbounds, 0, max(trialdata.ready_RT), color = "grey", linestyles = "dotted", linewidth=2)
    #  ready RT histogram
    trialdata['ready_RT'].plot.hist(ax = ax[1])
    ax[1].set_xlabel("Ready RT (s)")
    # sell RT timecourse
    trialdata.loc[trialdata.trialEarnings!=0,:].plot("totalTrialIdx", "RT", ax = ax[2])
    ax[2].set_ylabel("Sell RT(s)")
    ax[2].set_xlabel("Trial")
    ax[2].get_legend().remove()
    ax[2].vlines(blockbounds, 0, max(trialdata.RT), color = "grey", linestyles = "dotted", linewidth=2)
    # sell RT histogram
    trialdata.loc[trialdata.trialEarnings!=0,'RT'].plot.hist(ax = ax[3])
    ax[3].set_xlabel("Sell RT (s)")
    # 
    # code.interact(local = dict(globals(), **locals()))
    trialdata[np.logical_and(trialdata.trialEarnings != 0, trialdata.blockIdx == 1)].plot.scatter("timeWaited",  "RT", color = expParas.conditionColors[0], ax = ax[4])
    trialdata[np.logical_and(trialdata.trialEarnings != 0, trialdata.blockIdx == 2)].plot.scatter("timeWaited",  "RT", color = expParas.conditionColors[1], ax = ax[5])

def trialplot_multiblock(trialdata):
    """Plot figures to visually check trial-by-trial behavior, for multiple blocks

    """
    fig, ax = plt.subplots()
    trialdata[trialdata.trialEarnings != 0].plot('totalTrialIdx', 'timeWaited', ax = ax, color = "blue", label = 'rwd')
    trialdata[trialdata.trialEarnings != 0].plot.scatter('totalTrialIdx', 'timeWaited', ax = ax, color = "blue", label='_nolegend_', s = 100)

    trialdata[trialdata.trialEarnings == 0].plot('totalTrialIdx', 'timeWaited', ax = ax, color = "red", label = 'unrwd')
    trialdata[trialdata.trialEarnings == 0].plot.scatter('totalTrialIdx', 'timeWaited', ax = ax, color = "red", label='_nolegend_', s = 100)

    trialdata[trialdata.trialEarnings == 0].plot.scatter('totalTrialIdx', 'scheduledDelay', ax = ax, color = "black", label = "scheduled", s = 100)
    blockbounds = [max(trialdata.totalTrialIdx[trialdata.blockIdx == i]) for i in np.unique(trialdata.blockIdx)]
    ax.vlines(blockbounds, 0, max(expParas.tMaxs), color = "grey", linestyles = "dotted")
    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Trial")
    ax.set_ylim([-2, max(expParas.tMaxs) + 2])
    ax.set_xlim([-2, trialdata.shape[0] + 2])
    ax.legend(loc='upper right', frameon=False)

def wtwTS(trialEarnings_, timeWaited_, sellTime_, blockIdx_, tMax, TaskTime, plot_WTW = False):
    """
    sellTime_ here is a continous time.
    I uppack data here since the required inputs are different sometimes
    """
    # check whether they are values 

    # For trials before the first quit trial, wtw = the largest timeWaited value 
    if any(trialEarnings_ == 0):
        first_quit_idx = np.where(trialEarnings_ == 0)[0][0] # in case there is no quitting 
    else:
        first_quit_idx = len(trialEarnings_) - 1
    wtw_now = max(timeWaited_[:first_quit_idx+1])
    wtw = [wtw_now for i in range(first_quit_idx+1)]

    # For trials after the first quit trial, quitting indicates a new wtw value 
    # Otherwise, only update wtw if the current timeWaited is larger 
    for i in range(first_quit_idx+1, len(trialEarnings_)):
        if trialEarnings_[i] == 0:
            wtw_now = timeWaited_[i]
        else:
            wtw_now = max(timeWaited_[i], wtw_now)
        wtw.append(wtw_now)

    # code.interact(local = dict(locals(), **globals()))
    # cut off
    wtw = np.array([min(x, tMax) for x in wtw])

    # upsample 
    WTW = resample(wtw, sellTime_, TaskTime)

    # plot 
    if plot_WTW:
        fig, ax = plt.subplots()
        trialIdx_ = np.arange(len(trialEarnings_))
        ax.plot(trialIdx_, wtw)
        blockbounds = [max(trialIdx_[blockIdx_ == i]) for i in np.unique(blockIdx_)]
        ax.vlines(blockbounds, 0, tMax, color = "grey", linestyles = "dotted", linewidth = 3)
        ax.set_ylabel("WTW (s)")
        ax.set_xlabel("Trial")
        ax.set_ylim([-0.5, tMax + 0.5]) 

    return wtw, WTW, TaskTime

def desc_RT(trialdata):
    """Return descriptive stats of sell_RT and ready_RT, pooling all data together
    """
    # calc 
    trialdata.eval("ready_RT = trialStartTime - trialReadyTime", inplace = True)
    # calc summary stats
    out = trialdata.agg({
            "ready_RT": ["median", "mean", calc_se]
        })
    ready_RT_median, ready_RT_mean, ready_RT_se = out.ready_RT
    # code.interact(local = dict(globals(), **locals()))
    out = trialdata.loc[trialdata.trialEarnings != 0, :].agg({
            "RT": ["median", "mean"]
        })

    sell_RT_median,  sell_RT_mean = out.RT
    sell_RT_se  = calc_se(trialdata.loc[trialdata.trialEarnings != 0, :].RT)
    return ready_RT_median, ready_RT_mean, ready_RT_se, sell_RT_median, sell_RT_mean, sell_RT_se

############################ individual level analysis functions ###############
def ind_MF(trialdata, key, isTrct = True, plot_RT = False, plot_trial = False, plot_KMSC = False, plot_WTW = False):
    """Conduct model-free (MF) analysis for a single participant 
    Inputs:
        trialdata: a pd dataframe that contains task data
        key: in the format of (id, sessIdx). For example, ("s0001", 1)
        plot_RT: whether to plot 
        plot_trial:
    """
    # initialize the output
    stats = [] # for scalar outputs
    objs = {} # for others

    # RT visual check
    if plot_RT:
        rtplot_multiblock(trialdata)

    # trial-by-trial behavior visual check
    if plot_trial:
        trialplot_multiblock(trialdata)

    # WTW timecourse
    wtw, WTW, TaskTime = wtwTS(trialdata['trialEarnings'].values, trialdata['timeWaited'].values, trialdata['accumSellTime'].values, trialdata['blockIdx'].values, expParas.tMax, expParas.TaskTime, plot_WTW)
    objs['WTW'] = WTW

    ################## calculate summary stats for each block ###############
    if isTrct:
        trialdata = trialdata[trialdata.sellTime <= expParas.blocksec - np.max(expParas.tMaxs)]

    ################## this part of code  can be modified for different experiments ##########
    # initialize the figure 
    if plot_KMSC:
        fig, ax = plt.subplots()
        ax.set_xlim([0, expParas.tMax])
        ax.set_ylim([-0.1, 1.1])
        ax.set_xlabel("Elapsed time (s)")
        ax.set_ylabel("Survival rate")
        
    nBlock = len(np.unique(trialdata.blockIdx))
    for i in range(nBlock):
        blockdata = trialdata[trialdata.blockIdx == i + 1]
        condition = blockdata.condition.values[0]
        conditionColor = expParas.conditionColors[condition]
        # Survival analysis
        time, psurv, Time, Psurv, block_auc, block_std_wtw = kmsc(blockdata, expParas.tMax, expParas.Time, False)
        if plot_KMSC:
            ax.plot(time, psurv, color = conditionColor, label = condition)

        # Survival analysis into subblocks 
        sub_aucs = []
        sub_std_wtws = []
        for k in range(4):
            # code.interact(local = dict(locals(), **globals()))
            filter = np.logical_and(blockdata['sellTime'] >= k * expParas.blocksec / 4, blockdata['sellTime'] < (k + 1) * expParas.blocksec / 4)
            try:
                time, psurv, Time, Psurv, auc, std_wtw = kmsc(blockdata.loc[filter, :], expParas.tMax, Time, False)
            except:
                code.interact(local = dict(locals(), **globals()))
            sub_aucs.append(auc) 
            sub_std_wtws.append(std_wtw)

        # RT stats 
        ready_RT_median, ready_RT_mean, ready_RT_se, sell_RT_median, sell_RT_mean, sell_RT_se = desc_RT(blockdata)
        
        # organize the output
        tmp = pd.DataFrame({"id": key[0], "sess": key[1], "key": str(key), "block": i + 1, "auc": block_auc, "std_wtw": block_std_wtw,\
            "auc1": sub_aucs[0], "auc2": sub_aucs[1], "auc3": sub_aucs[2], "auc4": sub_aucs[3], \
            "std_wtw1": sub_std_wtws[0], "std_wtw2": sub_aucs[1], "std_wtw3": sub_std_wtws[2], "std_wtw4": sub_std_wtws[3], \
            "ready_RT_mean": ready_RT_mean,"ready_RT_se": ready_RT_se,"sell_RT_median": sell_RT_median,\
            "sell_RT_mean": sell_RT_mean, "sell_RT_se": sell_RT_se,\
            "condition": condition}, index = [i])
        stats.append(tmp) 
        objs['Psurv_block'+str(i+1)] = Psurv

    stats = pd.concat(stats, ignore_index = True)
        
    ############ return  ############# y
    return stats, objs


def ind_sim_MF(simdata, key, plot_trial = False, plot_KMSC = False, plot_WTW = False):
    """ 
        # this is for replication simulation though
        using wtw analysis here will be risky because balabala time longer will be truncated
        AUC for a block will not change 
        AUC analysis for subblocks will be affected
    """
    # initialize the output
    stats = [] # for scalar outputs
    objs = {} # for others
    wtw = [] # wtw for each trial
    WTW = [] # wtw resampled, trials beyond 600s are included

    # code.interact(local = dict(locals(), **globals()))
    # trial-by-trial behavior visual check
    if plot_trial:
        trialplot_multiblock(simdata)

    if plot_KMSC:
        fig, ax = plt.subplots()

    # calc AUC values and WTW for each block
    nBlock = len(np.unique(simdata.blockIdx))
    for i in range(nBlock):
        blockdata = simdata[simdata.blockIdx == i + 1]
        condition = blockdata.condition.values[0]
        conditionColor = expParas.conditionColors[condition]
        # code.interact(local = dict(locals(), **globals()))

        # Survival analysis
        time, psurv, Time, Psurv, block_auc, block_std_wtw = kmsc(blockdata, expParas.tMax, expParas.Time, False)
        if plot_KMSC:
            ax.plot(time, psurv, color = conditionColor, label = condition)

        # save results
        tmp = pd.DataFrame({"key": str(key), "block": i + 1, "auc": block_auc, "std_wtw": block_std_wtw,\
            "condition": condition}, index = [i])
        stats.append(tmp) 
        objs['Psurv_block'+str(i+1)] = Psurv

        # WTW analysis
        block_wtw, block_WTW, block_TaskTime = \
        wtwTS(
            blockdata['trialEarnings'].values,
            blockdata['timeWaited'].values,
            blockdata['sellTime'].values,
            blockdata['blockIdx'].values,
            expParas.tMax, 
            np.linspace(0, expParas.blocksec, 600),
            False
        )
        wtw.append(block_wtw)
        WTW.append(block_WTW)

    wtw = np.concatenate(wtw)
    if plot_WTW:
        fig, ax = plt.subplots()
        ax.plot(simdata.totalTrialIdx, wtw)

    # combine results from blocks
    WTW = np.concatenate(WTW)
    objs['WTW'] = WTW
    stats = pd.concat(stats, ignore_index = True)
        
    ############ return  ############# 
    return stats, objs

########################## group-level analysis functions ##############
def group_MF(trialdata_, plot_each = False):
    # check sample sizes 
    nsub = len(trialdata_)
    print("Analyze %d valid participants"%nsub)
    print("\n")
    # analysis constants
    Time = expParas.Time
    TaskTime = expParas.TaskTime
    stats_ = []
    # initialize outputs 
    Psurv_block1_ = np.empty([nsub, len(Time)])
    Psurv_block2_ = np.empty([nsub, len(Time)])
    WTW_ = np.empty([nsub, len(TaskTime)])

    # run MF for each participant 
    idx = 0
    for key, trialdata in trialdata_.items():
        if plot_each:
            stats, objs  = ind_MF(trialdata, key, plot_RT = True, plot_trial = True, plot_KMSC = False, plot_WTW = True)
            plt.show()
            input("Press Enter to continue...")
            plt.clf()
        else:
            stats, objs  = ind_MF(trialdata, key)
            stats_.append(stats)
        Psurv_block1_[idx, :] = objs['Psurv_block1']
        Psurv_block2_[idx, :] = objs['Psurv_block2']
        WTW_[idx, :] = objs['WTW']
        idx += 1

    # plot the group-level results
    # if plot_group:
    #   fig1, ax1 = plot_group_WTW(WTW_, TaskTime)
    #   fig2, ax2 = plot_group_KMSC(Psurv_block1_, Psurv_block2_, Time)

    stats_ = pd.concat(stats_)
    stats_.to_csv(os.path.join('..', 'analysis_results', 'taskstats', 'emp_sess%d.csv'%key[1]), index = None)
    # save some data 
    # code.interact(local = dict(globals(), **locals()))
    # stats_.to_csv(os.path.join(logdir, "stats_sess%d.csv"%sess), index = False)

    return stats_, Psurv_block1_, Psurv_block2_, WTW_

def group_sim_MF(simdata_, plot_each = False):
    """
        conduct MF analysis for multiple participants/simulated datasets
    """
    # tGrid constants
    Time = expParas.Time
    TaskTime = expParas.TaskTime

    # initialize outputs
    # code.interact(local = dict(locals(), **globals()))
    stats_ = []
    Psurv_block1_ = np.empty((len(simdata_), len(Time)))
    Psurv_block2_ = np.empty((len(simdata_), len(Time)))
    WTW_ = np.empty((len(simdata_), len(TaskTime)))


    # loop over participants 
    for i, simdata in enumerate(simdata_):
        if plot_each:
            stats, objs  = stats, objs  = ind_sim_MF(simdata, 'sim_%d'%i, plot_trial = True, plot_KMSC = False, plot_WTW = True)
            plt.show()
            input("Press Enter to continue...")
            plt.clf()
        else:
            stats, objs  = stats, objs  = ind_sim_MF(simdata, 'sim_%d'%i)
        
        # append data
        stats_.append(stats)
        Psurv_block1_[i, :] = objs['Psurv_block1']
        Psurv_block2_[i, :] = objs['Psurv_block2']
        WTW_[i, :] = objs['WTW']

    stats_ = pd.concat(stats_)
    # plot for the group level 
    # if plot_group:
    #   fig1, ax1 = plot_group_WTW(WTW_, TaskTime)
    #   fig2, ax2 = plot_group_KMSC(Psurv_block1_, Psurv_block2_, Time)

    return stats_, Psurv_block1_, Psurv_block2_, WTW_


def plot_group_WTW(WTW_, TaskTime, ax):
    """Plot group-level WTW timecourse 
    """
    # fig, ax = plt.subplots()
    df = pd.DataFrame({
            "mu": np.apply_along_axis(np.mean, 0, WTW_),
            "se": np.apply_along_axis(calc_se, 0, WTW_),
            "TaskTime": TaskTime
        })
    # code.interact(local = dict(globals(), **locals()))
    df = df.assign(ymin = df.mu - df.se, ymax = df.mu + df.se)
    df.plot("TaskTime", "mu", color = "black", ax = ax, label = '_nolegend_')
    ax.fill_between(df.TaskTime, df.ymin, df.ymax, facecolor='grey', edgecolor = "none",alpha = 0.4, interpolate=True)
    ax.set_xlabel("")
    ax.set_ylabel("WTW (s)")
    ax.set_xlabel("Task time (s)")
    ax.vlines(expParas.blocksec, 0, expParas.tMax, color = "red", linestyles = "dotted") # I might want to change it later
    ax.get_legend().remove()
    # plt.savefig(savepath)

def plot_group_AUC(stats, ax):
    ax.scatter(stats.loc[stats['condition'] == 'LP', 'auc'], stats.loc[stats['condition'] == 'HP', 'auc'], color = 'grey')
    ax.plot([-0.5, expParas.tMax + 0.5], [-0.5, expParas.tMax + 0.5], color = 'red', ls = "--")
    ax.set_ylim([-0.5, expParas.tMax + 0.5])
    ax.set_xlim([-0.5, expParas.tMax + 0.5])
    ax.set_xlabel("LP AUC (s)")
    ax.set_ylabel("HP AUC (s)")

def plot_group_KMSC(Psurv_block1_, Psurv_block2_, Time, ax):
    """ Plot group-level survival curves 
    """
    # fig, ax = plt.subplots()
    df1 = pd.DataFrame({
            "mu": np.apply_along_axis(np.mean, 0, Psurv_block1_),
            "se": np.apply_along_axis(calc_se, 0, Psurv_block1_),
            "Time": Time
        })
    df1 = df1.assign(ymin = lambda df: df.mu - df.se, ymax = lambda df: df.mu + df.se)
    df2 = pd.DataFrame({
            "mu": np.apply_along_axis(np.mean, 0, Psurv_block2_),
            "se": np.apply_along_axis(calc_se, 0, Psurv_block2_),
            "Time": Time
        })
    df2 = df2.assign(ymin = lambda df: df.mu - df.se, ymax = lambda df: df.mu + df.se)

    df1.plot("Time", "mu", color = expParas.conditionColors['LP'], ax = ax)
    ax.fill_between(df1.Time, df1.ymin, df1.ymax, facecolor= expParas.conditionColors['LP'], edgecolor = "none",alpha = 0.4, interpolate=True)
    df2.plot("Time", "mu", color = expParas.conditionColors['HP'], ax = ax)
    ax.fill_between(df2.Time, df2.ymin, df2.ymax, facecolor= expParas.conditionColors['HP'], edgecolor = "none",alpha = 0.4, interpolate=True)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Survival rate (%)")
    ax.set_ylim((0, 1))
    ax.set_xlim((0, expParas.tMax))
    ax.get_legend().remove()
    # plt.savefig(savepath)


