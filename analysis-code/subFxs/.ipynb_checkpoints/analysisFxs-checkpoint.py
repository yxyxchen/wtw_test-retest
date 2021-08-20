import pandas as pd
import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
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
    if x.empty:
        print("Calculate standard error of an empty object")
        return None 
    else: 
        size = len(x)
        x = x.dropna()
        ndrop = size - len(x) 
        if ndrop > 0:
            print("Remove NaN values in calculating standard error"%ndrop)
        return np.nanstd(x) / np.sqrt(len(x))


####################################################
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

    BIS = Attentional + Motor + Nonplanning\

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
        ax.plot(choices.values) 
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
    results = smf.glm("pD ~ -1 + TR + T",data = regdf, family=sm.families.Binomial()).fit()
    
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
        
    return k, logk, se_logk


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

def kmsc(data, tMax, plot_KMSC = False, Time = None):
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
    durations = [row.timeWaited if row.trialEarnings == 0 else row.scheduledDelay for index, row in data.iterrows()]
    event_observed = np.not_equal(data.trialEarnings, 0) # 1 if the participant quits and 0 otherwise 
    
    time, psurv = km(event_observed, durations)
    
    # add the first and the last datapoints
    psurv = psurv[time < tMax]
    time = time[time < tMax]
    time = np.insert(time, 0, 0);  time = np.append(time, tMax)
    psurv = np.insert(psurv, 0, 1);  psurv = np.append(psurv, np.max(psurv[-1],0))
    
    # upsample to a high resolution 
    if Time is None:
        Time = np.linspace(0, tMax, num = int(tMax / 0.1))
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
    trialdata['ready_RT'] = trialdata.eval("trialStartTime - trialReadyTime")
    blockbounds = [max(trialdata.totalTrialIdx[trialdata.blockIdx == i]) for i in np.unique(trialdata.blockIdx)]

    # plot
    fig, ax = plt.subplots(1,4)
    # ready RT timecourse
    trialdata.plot("totalTrialIdx",  "ready_RT", ax = ax[0], label = '_nolegend_')
    ax[0].set_ylabel("Ready RT (s)")
    ax[0].set_xlabel("Trial")
    ax[0].get_legend().remove()
    ax[0].vlines(blockbounds, 0, max(trialdata.ready_RT), color = "grey", linestyles = "dotted", linewidth=2)
    #  ready RT histogram
    trialdata['ready_RT'].plot.hist(ax = ax[1])
    ax[1].set_xlabel("Ready RT (s)")
    # sell RT timecourse
    trialdata.plot("totalTrialIdx", "RT", ax = ax[2])
    ax[2].set_ylabel("Sell RT(s)")
    ax[2].set_xlabel("Trial")
    ax[2].get_legend().remove()
    ax[2].vlines(blockbounds, 0, max(trialdata.RT), color = "grey", linestyles = "dotted", linewidth=2)
    # sell RT histogram
    trialdata['RT'].plot.hist(ax = ax[3])
    ax[3].set_xlabel("Sell RT (s)")

def trialplot_multiblock(trialdata):
    """Plot figures to visually check trial-by-trial behavior, for multiple blocks

    """
    fig, ax = plt.subplots()
    trialdata[trialdata.trialEarnings != 0].plot('totalTrialIdx', 'timeWaited', ax = ax, color = "blue", label = "_nolegend_")
    trialdata[trialdata.trialEarnings != 0].plot.scatter('totalTrialIdx', 'timeWaited', ax = ax, color = "blue", label = "rwd")

    trialdata[trialdata.trialEarnings == 0].plot('totalTrialIdx', 'timeWaited', ax = ax, color = "red", label = "_nolegend_")
    trialdata[trialdata.trialEarnings == 0].plot.scatter('totalTrialIdx', 'timeWaited', ax = ax, color = "red", label = "unrwd")

    trialdata[trialdata.trialEarnings == 0].plot.scatter('totalTrialIdx', 'scheduledDelay', ax = ax, color = "black", label = "scheduled")
    blockbounds = [max(trialdata.totalTrialIdx[trialdata.blockIdx == i]) for i in np.unique(trialdata.blockIdx)]
    ax.vlines(blockbounds, 0, max(expParas.tMaxs), color = "grey", linestyles = "dotted")
    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Trial")
    ax.set_ylim([-1, max(expParas.tMaxs) + 1])

def wtwTS(data, tMax,  Time, plot_WTW = False):
    """
    """
    # ensure timeWaited = scheduledWait on rewarded trials
    data.timeWaited =  [row.timeWaited if row.trialEarnings == 0 else row.scheduledDelay for index, row in data.iterrows()]
    
    # For trials before the first quit trial, wtw = the largest timeWaited value 
    if any(data.trialEarnings == 0):
        first_quit_idx = data.query('trialEarnings == 0').index[0] # in case there is no quitting 
    else:
        first_quit_idx = data.shape[0]-1
    wtw_now = max(data.timeWaited[:first_quit_idx+1])
    wtw = [wtw_now for i in range(first_quit_idx+1)]

    # For trials after the first quit trial, quitting indicates a new wtw value 
    # Otherwise, only update wtw if the current timeWaited is larger 
    for i, row in data.iloc[first_quit_idx+1:,:].iterrows():
        if row.trialEarnings == 0:
            wtw_now = row.timeWaited
        else:
            wtw_now = max(row.timeWaited, wtw_now)
        wtw.append(wtw_now)

    ############# sanity check ############### 
    # code.interact(local=dict(globals(), **locals()))
    # plotdf = pd.DataFrame({
    #     "wtw": wtw,
    #     "timeWaited": data.timeWaited,
    #     "trialEarnings": data.trialEarnings,
    #     "trialIdx": data.totalTrialIdx
    #     })
    # fig, ax = plt.subplots()
    # plotdf.plot("trialIdx", "wtw", ax = ax)
    # plotdf.plot("trialIdx", "timeWaited", ax = ax, color = "black")
    # ax.scatter(data.totalTrialIdx[data.trialEarnings == 0], data.timeWaited[data.trialEarnings == 0], color = "red")

    # cut off
    wtw = np.array([min(x, tMax) for x in wtw])

    # upsample 
    WTW = resample(wtw, data.accumSellTime, Time)

    # plot
    if plot_WTW:
        fig, ax = plt.subplots()
        ax.plot(data.totalTrialIdx, wtw)
        blockbounds = [max(data.totalTrialIdx[data.blockIdx == i]) for i in np.unique(data.blockIdx)]
        ax.vlines(blockbounds, 0, tMax, color = "grey", linestyles = "dotted", linewidth = 2)
        ax.set_ylabel("WTW (s)")
        ax.set_xlabel("Trial")
        ax.set_ylim([-0.5, tMax + 0.5]) 

    return wtw, WTW, Time


def desc_RT(trialdata):
    """Return descriptive stats of sell_RT and ready_RT, pooling all data together
    """
    # calc ready_RT
    trialdata['ready_RT'] = trialdata.eval("trialStartTime - trialReadyTime")
    # calc summary stats
    out = trialdata.agg({
            "ready_RT": ["median", "mean", calc_se],
            "RT": ["median", "mean", calc_se]
        })
    # reorganize the results
    ready_RT_median, ready_RT_mean, ready_RT_se = out.ready_RT
    sell_RT_median,  sell_RT_mean, sell_RT_se = out.RT

    return ready_RT_median, ready_RT_mean, ready_RT_se, sell_RT_median, sell_RT_mean, sell_RT_se

