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
    if modelname == 'QL1':
        return ['alpha', 'tau', 'gamma', 'eta']
    elif modelname == 'QL2':
        return ['alpha', 'nu', 'tau', 'gamma', 'eta']
    elif modelname == 'RL1':
        return ['alpha', 'tau', 'eta', 'beta']
    elif modelname == 'RL2':
        return ['alpha', 'nu', 'tau', 'eta', 'beta']

def ind_model_fit(trialdata, modelname, paranames, config, outfile_stem):
    """Function to fit a model for a single partiicipant 
    """
    # prepare inputs for fitting the model
    ts = np.arange(0, np.max(expParas.tMaxs), expParas.stepsize) 

    # nMadeActions determines actions and credit assignment issues 
    # notice this index starts from 1, since it follows stan conventions
    nMadeActions = np.ceil(trialdata['timeWaited'] / expParas.stepsize).astype(int).tolist() # this is different from the R version

    # prepare inputs 
    inputs = {
        "iti": expParas.iti,
        "nt": len(ts),
        "ts": ts,
        "N": trialdata.shape[0],
        "Rs": trialdata.trialEarnings.values.astype('int').tolist(),
        "Ts": trialdata['timeWaited'].values.tolist(),
        "nMadeActions": nMadeActions
    }

    if modelname == 'QL1' or modelname == 'QL2':
        inputs['Qquit_ini'] = np.mean(expParas.optimRewardRates) / (1 -  0.85) # I don't think stepsize matters here
    elif modelname == 'RL1' or modelname == 'RL2':
        inputs['Qquit_ini'] = 0 # out of date
    
    # code.interact(local = dict(globals(), **locals()))

    # read model_code
    if modelname == 'QL1':
        model_code = stancode.QL1

    # compile the stan model
    model = stan.build(model_code, data = inputs, random_seed = 1)

    # sample posterior
    fit = model.sample(num_chains = config['nChain'], num_samples = config['nSample'], num_warmup = config['nWarmup'])

    # check fit quality 
    """
    according to this ref: https://mc-stan.org/rstan/reference/Rhat.html
    ess_bulk needs to be at least 100 per chain for reliable mean and median estimates
    ess_tail needs to be at least 100 per chain for reliable variance and tail quantile estimates
    both are important for estimates of respective posterior quantiles are reliable
    """
    # code.interact(local = dict(globals(), **locals()))
    fit_summary =  az.summary(fit, paranames + ['totalLL'])
    divergent = np.sum(fit['divergent__'] != 0)
    fit_summary['divergent'] = np.full(fit_summary.shape[0], divergent)

    # save fit_summary
    fit_summary.to_csv(outfile_stem + '_summary.csv')

    # save WAIC 
    waic = az.waic(fit, 'totalLL')
    waic.to_csv(outfile_stem + '_waic.csv')


def group_model_fit(trialdata_, modelname, config, outdir, isTrct = True):
    """ loop over participants
    """
    # get model parameters
    paranames = getModelParas(modelname)

    for key, trialdata in trialdata_.items():
        if isTrct:
            # code.interact(local = dict(globals(), **locals()))
            outfile_stem = os.path.join(outdir, '%s_sess%s'%key)  
            if isTrct:        
                trialdata = trialdata[trialdata.sellTime <= expParas.blocksec - np.max(expParas.tMaxs)]
        ind_model_fit(trialdata, modelname, paranames, config, outfile_stem)

   
def check_stan_diagnosis(fit_summary):
    if any(fit_summary['n_divergent'] > 0) or any(fit_summary['Rhat'] > 1.05) or any(fit_summary['n_effe'] < 400):
        return False
    else:
        return True

#################### functions to replicate choice data using individually fitted parameters and analyze them##########
def ind_model_rep(modelname, paras, trialdata, key, nsim, plot_each):
    # make sure to use values so that it is not affected by indices
    scheduledDelays = trialdata['scheduledDelay'].values
    scheduledRewards = np.full(scheduledDelays.shape, expParas.tokenValue)
    conditions = trialdata['condition'].values
    blockIdxs = trialdata['blockIdx'].values

    # initialize analysis results of simulated data to return
    # I think I probably only need mean and se?

    # loop over simulations 
    simdata_ = []
    for i in range(nsim):
        simdata =  simFxs.ind_sim(modelname, paras, conditions, blockIdxs, scheduledDelays, scheduledRewards)
        simdata_.append(simdata)

    # analyze simulated datasets
    stats_, Psurv_block1_, Psurv_block2_, WTW_ = analysisFxs.group_sim_MF(simdata_, trialdata)

    # return summary statistics
    stats = stats_.groupby(['block', 'condition']).agg('mean')
    Psurv_block1 = np.nanmean(Psurv_block1_, axis = 0)
    Psurv_block2 = np.nanmean(Psurv_block2_, axis = 0)
    WTW = np.nanmean(WTW_, axis = 0)

    # code.interact(local = dict(globals(), **locals()))

    # I can plot comparison here honestly
    if plot_each:
        emp_stats, emp_objs = analysisFxs.ind_MF(trialdata, key, isTrct = True, plot_RT = False, plot_trial = False, plot_KMSC = False, plot_WTW = False)
        fig, ax = plt.subplots()
        ax.plot(expParas.TaskTime, emp_objs['WTW'], label = 'Observed')
        ax.plot(expParas.TaskTime, WTW, label = 'Simulated')
    return stats, Psurv_block1, Psurv_block2, WTW



#############
# hmm I kinda want to rewrite it later, ... but whatever
def group_model_rep(trialdata_, paradf, modelname, isTrct = True, plot_each = False):
    # set random seed
    random.seed(10)

    # get parameter names for this model
    paranames = getModelParas(modelname)

    # initialize outputs
    stats_ = []
    WTW_ = []
    # loop over participants
    for key, trialdata in trialdata_.items():
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

        # replicate original behaviors
        paras = dict(zip(paranames, paravals))
        if plot_each:
            stats, _, _, WTW = ind_model_rep(modelname, paras, trialdata, key, 10, plot_each = True)
            plt.show()
            input("Press Enter to continue or ESC to exit...")
            plt.clf()
        else:
            stats, _, _, WTW = ind_model_rep(modelname, paras, trialdata, key, 10, plot_each = False)
        stats['id'] = np.full(2, key[0])
        stats['sess'] = np.full(2, key[1])
        stats['key'] = np.full(2, str(key))
        stats_.append(stats)
        WTW_.append(WTW)
        # add more code tomorrow.
        # emp_stats, emp_obj = analysisFxs.ind_MF(trialdata, key, plot_trial = True, plot_KMSC = True, plot_WTW = True)
    stats_ = pd.concat(stats_)
    stats_ = stats_.reset_index()
    WTW_ = np.array([e for e in WTW_])
    # code.interact(local = dict(locals(), **globals()))
    # temporarily let's save it to save time
    # stats_.to_csv(os.path.join('..', 'analysis_results', 'taskstats', 'rep_%s_sess%d.csv'%(modelname, key[1])), index = None)
    return stats_, WTW_
       






