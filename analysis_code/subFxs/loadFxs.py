import pandas as pd
import numpy as np
import os
import glob
import importlib
import re
import itertools
import copy # pay attention to copy 
import code
import os
import subFxs
from subFxs import expParas
from subFxs import analysisFxs
from subFxs import modelFxs
from datetime import datetime as dt
import matplotlib.pyplot as plt
import rpy2.robjects as robjects



datadir = "data"

############################## load task data ##################################
def loaddata(expname, sess):
    """load hdrdata and trialdata from given folders 
    """
    hdrdata  = pd.read_csv(os.path.join(datadir, expname, "hdrdata_sess%d.csv"%sess))
    print("Load %d files"%hdrdata.shape[0])
    print("Exclude %d participants who quit midway"%sum(hdrdata.quit_midway))
    print("\n")
    hdrdata = hdrdata.loc[np.logical_not(hdrdata.quit_midway)].reset_index(drop=True)

    nsub = hdrdata.shape[0]
    trialdata_  = {}
    # code.interact(local = dict(locals(), **globals()))
    for i in range(nsub):
        thisid = hdrdata.id.iloc[i]
        trialdata = pd.read_csv(os.path.join(datadir, expname, "task-" + thisid + "-sess%d.csv"%sess))
        
        # add blockIdx
        trialdata['blockIdx'] = [1 if x == "LP" else 2 for x in trialdata['condition']]

        # add totalTrialIdx
        ntrial_firstblock = int(trialdata.trialIdx[trialdata.condition == "LP"].max())
        trialdata['totalTrialIdx'] = trialdata['trialIdx'] + np.equal(trialdata['condition'], "HP") * ntrial_firstblock

        # add accumSellTime
        trialdata['accumSellTime'] = trialdata['sellTime'] + (trialdata['blockIdx'] - 1) * expParas.blocksec 

        # use the scheduledWait as timeWaited on rewarded trials
        trialdata.loc[trialdata.trialEarnings != 0, 'timeWaited'] = trialdata.loc[trialdata.trialEarnings != 0, 'scheduledDelay']

        # fill in 
        trialdata_[(hdrdata.id.iloc[i], hdrdata.sess.iloc[i])] =  trialdata

    return hdrdata, trialdata_

################# check quality of task data #################

def group_quality_check(expname, sess, plot_quality_check = False):
    # code.interact(local = dict(locals(), **globals()))
    # quality check the data
    hdrdatafile = os.path.join("data", 'hdrdata_sess%d.csv'%sess)
    trialdatafiles = glob.glob(os.path.join("data", "task*sess%d.csv"%sess))
    hdrdata, trialdata_ = loaddata(expname, sess)
    logdir = "analysis_log"
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # check sample size 
    nsub = hdrdata.shape[0]
    print("Check %d participants in SESS%d"%(nsub, sess))

    # # check data quality 
    stats_ = []
    for idx, row in hdrdata.iterrows():
        this_trialdata = trialdata_[(row.id, row.sess)]
        stats = pd.DataFrame({
            "id": row.id,
            "batch": row.batch,
            "cb": row.cb,
            'ntrial_block1': this_trialdata.trialIdx[this_trialdata.blockIdx == 1].max(),
            'ntrial_block2': this_trialdata.trialIdx[this_trialdata.blockIdx == 2].max(),
            'total_timeWaited_block1': np.sum(this_trialdata.timeWaited[this_trialdata.blockIdx == 1]), # this is to ensure the minimal total engaed time, excluding when one sell RT or one Ready RT is really long
            'total_timeWaited_block2': np.sum(this_trialdata.timeWaited[this_trialdata.blockIdx == 2]),
        }, index = [0])
        sell_RT_median, sell_RT_mean, sell_RT_se = analysisFxs.desc_RT(this_trialdata)
        stats['sell_RT_median'] = sell_RT_median # to make sure participants are alert in general
        stats['sell_RT_mean'] = sell_RT_mean # not very useful actually
        stats_.append(stats)

    # plot tasktime and medium RT 
    statsdf = pd.concat(stats_)
    statsdf['tasktime_block1'] = statsdf.eval("ntrial_block1 * @expParas.iti + total_timeWaited_block1")
    statsdf['tasktime_block2'] = statsdf.eval("ntrial_block2 * @expParas.iti + total_timeWaited_block2")
    
    if plot_quality_check:
        _, ax_tasktime = plt.subplots(1,2)
        statsdf.hist('tasktime_block1', ax = ax_tasktime[0])
        ax_tasktime[0].set_xlabel("Block1 Tasktime (s)")
        statsdf.hist('tasktime_block2', ax = ax_tasktime[1])
        ax_tasktime[1].set_xlabel("Block 2 Tasktime (s)")
        _, ax_sellRT = plt.subplots(1,2)
        statsdf.hist("sell_RT_median", ax = ax_sellRT[0])
        statsdf.hist("sell_RT_mean", ax = ax_sellRT[1])

    # exclude participants with low quality data 
    excluded = statsdf.loc[np.logical_or.reduce(
        [statsdf.sell_RT_median > 1.2,
        statsdf.tasktime_block1 < 450,
        statsdf.tasktime_block2 < 450]
        ), :].drop(['ntrial_block1', 'ntrial_block2', 'total_timeWaited_block1', 'total_timeWaited_block2'], axis=1)

    # code.interact(local = dict(locals(), **globals()))
    consentdata = pd.read_csv(os.path.join(datadir, expname, "consent_sess%d.csv"%sess))
    # code.interact(local = dict(globals(), **locals()))
    excluded = pd.merge(excluded, right = consentdata[['id', 'worker_id']], how = "left", left_on = "id", right_on = "id")
    excluded.to_csv(os.path.join("..", "analysis_results", expname, "excluded", "excluded_participants_sess%d.csv"%sess), index = False)
    # excluded = pd.read_csv(os.path.join("..", "analysis_results", "excluded", "excluded_participants_sess%d.csv"%sess))

    # filter excluded data 
    hdrdata = hdrdata.loc[~np.isin(hdrdata.id, excluded.id), :]
    hdrdata.reset_index(drop=True, inplace  = True)
    # code.interact(local = dict(locals(), **globals()))
    trialdata_ = {x:y for x, y in trialdata_.items() if x[0] not in excluded.id.values}
    print("Exclude %d participants with low data quality!"%excluded.shape[0])
    print("\n")
    # code.interact(local = dict(locals(), **globals()))
    return hdrdata, trialdata_

############################ parse selfreport data ##########################
# currently we don't have good way to quality check selfreport data, except for MCQ
def parse_ind_selfreport(sess, row, isplot):
    """ process selfreport data for a single participant 

    Inputs:
        row: an entry of individual selfreport data
        plot_k: whether to plot diagnostic figures for the delayed reward discounting questionaire
        plot_upps: whether to plot diagnostic figures for UPPS
        plot_BIS: whether to plot diagnostic figures for BIS

    Outputs:
        out: a panda dataframe that contains parameters and scores for the given individual

    """
    # the hyperbolic discounting parameter is calculated in R scripts under the MCQ folder

    # score BIS
    # code.interact(local = dict(locals(), **globals()))
    if len(row.filter(like = "BIS")) > 0:
        BIS_subscores, Attentional, Motor, Nonplanning, BIS = analysisFxs.score_BIS(row.filter(like = 'BIS'), isplot)
    else:
        Attentional = np.nan
        Motor = np.nan
        Nonplanning = np.nan
        BIS = np.nan
        BIS_subscores = np.full(6, np.nan)

    if len(row.filter(like = "UP")) > 0:
        NU, PU, PM, PS, SS, UPPS  = analysisFxs.score_upps(row.filter(like = 'UP'), isplot)
    else:
        NU = np.nan
        PU = np.nan
        PM = np.nan
        PS = np.nan
        SS = np.nan
        UPPS = np.nan

    # score PANAS
    if not np.isnan(row['PA-1']): # the first batch doesn't have PANAS choices
        PAS, NAS = analysisFxs.score_PANAS(row.filter(like = 'PA'), isplot)
    else:
        PAS = np.nan
        NAS = np.nan

    # assumeble outputs

    out = pd.DataFrame({
        "id": row.id,
        "duration": row.selfreport_duration,
        "NU": NU,
        "PU": PU,
        "PM": PM,
        "PS": PS,
        "SS": SS,
        "UPPS": UPPS,
        "Attentional": Attentional,
        "Motor": Motor,
        "Nonplanning": Nonplanning,
        "attention": BIS_subscores[0], 
        "cogstable": BIS_subscores[1], 
        "motor": BIS_subscores[2],
        "perseverance": BIS_subscores[3],
        "selfcontrol": BIS_subscores[4],
        "cogcomplex": BIS_subscores[5], 
        "BIS": BIS,
        "PAS": PAS,
        "NAS": NAS
        }, index = [row.id])
    # else:
    #   # code.interact(local = dict(locals(), **globals()))
    #   try:
    #       if not np.isnan(row[4]):
    #           PAS, NAS = analysisFxs.score_PANAS(row[4:24], isplot)
    #       else:
    #           PAS = np.nan
    #           NAS = np.nan
    #   except:
    #       print(row)
    #   out = pd.DataFrame({
    #       "id": row.id,
    #       "duration": row.selfreport_duration,            
    #       "PAS": PAS,
    #       "NAS": NAS
    #       }, index = [row.id])
    return out

def parse_group_selfreport(expname, sess, isplot):
    """ parse selfreport data for the group
    """
    # read the input file
    selfreportfile = os.path.join(datadir, expname, 'selfreport_sess%d.csv'%sess)
    selfreport = pd.read_csv(selfreportfile)

    # score all other questionaires except MCQ
    selfdata = pd.DataFrame()
    for i, row in selfreport.iterrows():
        out  = parse_ind_selfreport(sess, row, isplot)
        selfdata = pd.concat([selfdata, out])
        if isplot and (sess == 1 or (sess == 2 and not np.isnan(row[4]))):
            plt.show()
            input("Press Enter to continue...")
            plt.clf()
    # read the MCQ file
    mcqfile = os.path.join("..", "analysis_results", expname, "selfreport", 'MCQ_sess%d.csv'%sess)
    if os.path.exists(mcqfile):
        # code.interact(local = dict(locals(), **globals()))
        mcqdata = pd.read_csv(mcqfile)
        k_filter = np.logical_and.reduce([mcqdata.SmlCons >= 0.8, mcqdata.MedCons >= 0.8, mcqdata.LrgCons > 0.8])
        n_nonvalid_k = (~k_filter).sum()
        print("k estimates for %d participants are not valid! Didn't record them."%n_nonvalid_k)
        mcqdata = mcqdata.loc[k_filter,:] 
        selfdata = selfdata.merge(mcqdata[['GMK', 'SubjID']], how = 'outer', right_on = 'SubjID', left_on = 'id').drop("SubjID", axis = 1)
    selfdata.reset_index(inplace = True, drop = True)
    return selfdata 

# Maybe I need a function to load MCQ
############################## load model parameter estimates######################
def load_parameter_estimates(expname, sess, hdrdata, modelname, foldername):
    paranames = modelFxs.getModelParas(modelname)

    paradf = []
    for i, subj_id in enumerate(hdrdata['id']):
        # load parameter estimates
        try:
            fit_summary = pd.read_csv(os.path.join("..", "analysis_results", expname, "modelfit", foldername, '%s_sess%d_summary.txt'%(subj_id, sess)), header = None)
        except:
            print("can't find the file for %s, sess%d"%(subj_id, sess))
            continue
            # code.interact(local = dict(locals(), **globals()))
        # currently I didn't save index and columns 
        fit_summary.index = paranames + ['totalLL']
        fit_summary.columns = ['mean', 'se_mean', 'sd', '2.5%', '25%', '50%', '75%', '97.5%', 'n_effe', 'Rhat', 'n_divergent']

        # check quality
        if not modelFxs.check_stan_diagnosis(fit_summary):
            print(subj_id, 'not valid')
            continue

        # load waic 
        try:
            robjects.r['load'](os.path.join("..", "analysis_results", expname, "modelfit", foldername, '%s_sess%d_waic.RData'%(subj_id, sess)))
            waic = robjects.r['WAIC'][4][0]
        except:
            print("can't find the WAIC file for %s, sess%d"%(subj_id, sess))
            continue
        # 
        this_row = pd.DataFrame(dict(zip(paranames, fit_summary.iloc[:-1, 0])), index = [0])
        this_row['id'] = subj_id
        this_row['sess'] = sess
        this_row['waic'] = waic
        paradf.append(this_row)
    paradf = pd.concat(paradf)

    return paradf

