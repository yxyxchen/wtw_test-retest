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
import os
import importlib
from datetime import datetime as dt
import importlib

# plot styles
plt.style.use('classic')
sns.set(font_scale = 2)
sns.set_style("white")
condition_palette = ["#762a83", "#1b7837"]


# generate output results 
def generate_output_dirs(expname):
    if not os.path.isdir(os.path.join("..", 'analysis_results')):
        os.makedirs(os.path.join("..", 'analysis_results'))

    if not os.path.isdir(os.path.join("..", 'analysis_results', expname)):
        os.makedirs(os.path.join("..", 'analysis_results', expname))

    if not os.path.isdir(os.path.join("..", "analysis_results", expname, "excluded")):
        os.makedirs(os.path.join("..", "analysis_results", expname, "excluded"))

    if not os.path.isdir(os.path.join('..', 'analysis_results', expname, 'taskstats')):
        os.makedirs(os.path.join('..', 'analysis_results', expname, 'taskstats'))

    if not os.path.isdir(os.path.join("..", 'analysis_results', expname, 'modelfit')):
        os.makedirs(os.path.join("..", 'analysis_results',expname, 'modelfit'))

    if not os.path.isdir(os.path.join("..", 'analysis_results', expname, 'crossvalid')):
        os.makedirs(os.path.join("..", 'analysis_results', expname, 'crossvalid'))

    if not os.path.isdir(os.path.join("..", 'analysis_results', expname, 'selfreport')):
        os.makedirs(os.path.join("..", 'analysis_results', expname, 'selfreport'))

    if not os.path.isdir(os.path.join("..", 'analysis_results', expname, 'correlation')):
        os.makedirs(os.path.join("..", 'analysis_results', expname, 'correlation'))

    if not os.path.isdir(os.path.join("..", "figures")):
        os.makedirs(os.path.join("..", "figures"))

    if not os.path.isdir(os.path.join("..", "figures", expname)):
        os.makedirs(os.path.join("..", "figures", expname))


def plot_parameter_selfreport_corr(modelname, hdrdata_sess1, hdrdata_sess2):
    # get parameter names
    paranames = modelFxs.getModelParas(modelname)

    # load model parameters
    paradf_sess1 = loadFxs.load_parameter_estimates(1, hdrdata_sess1, modelname)
    paradf_sess2 = loadFxs.load_parameter_estimates(2, hdrdata_sess2, modelname)
    paradf = paradf_sess1.merge(paradf_sess2, how = 'outer', left_on = "id", right_on = 'id', suffixes = ['_sess1', '_sess2'])

    # load selfreport data
    self_sess1 = pd.read_csv(os.path.join("..", "analysis_results", "selfreport", "selfreport_sess1.csv"))
    self_sess1['batch'] = [1 if np.isnan(x) else 2 for x in self_sess1.PAS]
    self_sess1['sess'] = 1
    MCQ = pd.read_csv(os.path.join("..",  "analysis_results", 'selfreport', "MCQ.csv"))
    MCQ = MCQ.loc[np.logical_and.reduce([MCQ.SmlCons >= 0.8, MCQ.MedCons >= 0.8, MCQ.LrgCons > 0.8]),:] # filter 
    self_sess1 = self_sess1.merge(MCQ[['GMK', 'SubjID']], how = 'outer', right_on = 'SubjID', left_on = 'id').drop("SubjID", axis = 1)

    self_sess2 = pd.read_csv(os.path.join("..", "analysis_results", "selfreport", "selfreport_sess2.csv"))
    self_sess2['batch'] = [1 if np.isnan(x) else 2 for x in self_sess2.PAS]
    self_sess2['sess'] = 2

    code.interact(local = dict(locals(), **globals()))
    # conduct correlation analyses
    df = paradf.merge(self_sess1, how = 'outer', left_on = 'id', right_on = 'id')

    col_vars = ['NU', 'PU', 'PM', 'PS', 'SS', 'attention', 'cogstable', 'motor', 'perseverance', 'selfcontrol', 'cogcomplex', 'GMK']
    row_vars = [x + '_sess1' for x in paranames] + [x + '_sess2' for x in paranames]

    r_table = np.zeros((len(row_vars), len(col_vars))) # r for each comb of task measure and self-report measure 
    p_table = np.zeros((len(row_vars), len(col_vars))) 

    for i, row_var in enumerate(row_vars):
        tmp = [spearmanr(df[row_var].values, df[col_var].values, nan_policy = 'omit') for col_var in col_vars]
        r_table[i,:] = [x[0] for x in tmp]
        p_table[i,:] = [x[1] for x in tmp]

    r_table = pd.DataFrame(r_table, columns = col_vars, index = row_vars)
    r_table.to_csv(os.path.join("..", "analysis_results", "%s_para_selfreport_corr_r.csv"%modelname))
    p_table = pd.DataFrame(p_table, columns = col_vars, index = row_vars)
    p_table.to_csv(os.path.join("..", "analysis_results", "%s_para_selfreport_corr_p.csv"%modelname))

    # for i, x in enumerate(col_vars):
    #     tmp = [spearmanr(x.values, y, nan_policy = 'omit') for _, x in df[self_vars].iteritems()]
    #     r_table[i,:] = [x[0] for x in tmp]
    #     p_table[i,:] = [x[1] for x in tmp]
    # code.interact(local = dict(locals(), **globals()))

#################################
def selfreport_selfcorr(expname):
    # load selfreport data
    self_sess1 = pd.read_csv(os.path.join("..", "analysis_results", expname, "selfreport", "selfreport_sess1.csv"))
    self_sess1['batch'] = [1 if np.isnan(x) else 2 for x in self_sess1.PAS]
    self_sess1['sess'] = 1
    MCQ = pd.read_csv(os.path.join("..", "analysis_results", expname, "selfreport", "MCQ.csv"))
    MCQ = MCQ.loc[np.logical_and.reduce([MCQ.SmlCons >= 0.8, MCQ.MedCons >= 0.8, MCQ.LrgCons > 0.8]),:] # filter 
    self_sess1 = self_sess1.merge(MCQ[['GMK', 'SubjID']], how = 'outer', right_on = 'SubjID', left_on = 'id').drop("SubjID", axis = 1)

    self_sess2 = pd.read_csv(os.path.join("..", "analysis_results", expname, "selfreport",  "selfreport_sess2.csv"))
    self_sess2['batch'] = [1 if np.isnan(x) else 2 for x in self_sess2.PAS]
    self_sess2['sess'] = 2

    # code.interact(local = dict(locals(), **globals()))
    # self_sess1.merge(MCQ, how = '')

    # I want MCQ here, right...
    # durations  
    # fig, ax = plt.subplots()
    # self_sess1.loc[self_sess1.batch == 1,'duration'].hist(ax = ax)
    # ax.set_xlabel('Duration (s)')
    # fig.savefig(os.path.join("..", "figures", "self_duration_sess1_batch1.png"))

    # fig, ax = plt.subplots()
    # self_sess1.loc[self_sess1.batch == 2,'duration'].hist(ax = ax)
    # ax.set_xlabel('Duration (s)')
    # fig.savefig(os.path.join("..", "figures", "self_duration_sess1_batch2.png"))

    # fig, ax = plt.subplots()
    # self_sess2.loc[self_sess2.batch == 2,'duration'].hist(ax = ax)
    # ax.set_xlabel('Duration (s)')
    # fig.savefig(os.path.join("..", "figures", "self_duration_sess2_batch2.png"))
        
    # pairplot among selfreport measures
    # sns.pairplot(self_sess1[['UPPS', 'BIS']])
    # plt.savefig(os.path.join("..", "figures", "UPPS_BIS_corr.png"))

    # sns.pairplot(self_sess1[['NU', 'PU', 'PM', 'PS', 'SS']], kind="reg", corner=True, plot_kws={'line_kws':{'color':'red'}})
    # plt.savefig(os.path.join("..", "figures", "UPPS_corr.png"))

    # sns.pairplot(self_sess1[['Attentional', 'Motor', 'Nonplanning']], kind="reg", corner=True, plot_kws={'line_kws':{'color':'red'}})
    # plt.savefig(os.path.join("..", "figures", "BIS_corr.png"))

    # # not very informative
    # sns.pairplot(self_sess1[['attention', 'cogstable', 'motor', 'perseverance', 'selfcontrol', 'cogcomplex']], kind="reg", corner=True, plot_kws={'line_kws':{'color':'red'}})


# I need to write new functions to load selfreport measures as well
def selfreport_task_corr():
    # load selfreport data
    self_sess1 = pd.read_csv(os.path.join(logdir, "selfreport_sess1.csv"))
    MCQ = pd.read_csv(os.path.join(logdir, "MCQ.csv"))
    MCQ = MCQ.loc[np.logical_and.reduce([MCQ.SmlCons >= 0.8, MCQ.MedCons >= 0.8, MCQ.LrgCons > 0.8]),:] # filter 
    self_sess1 = self_sess1.merge(MCQ[['GMK', 'SubjID']], how = 'outer', right_on = 'SubjID', left_on = 'id').drop("SubjID", axis = 1)
    
    # inside items 
    stats_sess1 = pd.read_csv(os.path.join(logdir, "stats_sess1.csv"))
    stats_sess1['sess'] = 1
    stats_sess2 = pd.read_csv(os.path.join(logdir, "stats_sess2.csv"))
    stats_sess2['sess'] = 2
    stats = pd.concat([stats_sess1, stats_sess2])
    stats = stats.pivot_table(values = 'auc', columns = ['sess', "condition"], index = 'id')

    # well let's merge these two, so that data are aligned by ID
    df = self_sess1.merge(stats, how = 'outer', on = "id")

    ################# plot for both sessions and both conditions ###########
    # x_vars = ['NU', 'PU', 'PM', 'PS', 'SS', 'attention', 'cogstable', 'motor', 'perseverance', 'selfcontrol', 'cogcomplex', 'GMK']
    self_vars = ['NU', 'PU', 'PM', 'PS', 'SS', 'attention', 'cogstable', 'motor', 'perseverance', 'selfcontrol', 'cogcomplex', 'GMK']
    stats_vars = stats.columns
    n_perm = 5000

    r_table = np.zeros((len(stats_vars), len(self_vars))) # r for each comb of task measure and self-report measure 
    p_table = np.zeros((len(stats_vars), len(self_vars))) # uncorrected p for each comb of task measure and self-report measure 
    perm_r_ = np.zeros((len(stats_vars), len(self_vars), n_perm)) # permutated r for each comb of task measure and self-report measure 

    # loop over all stats_vars
    for i in range(len(stats_vars)):
        # select stats_var 
        stats_var = stats.columns[i]
        sess = stats.columns[i][0]
        condition = stats.columns[i][1]
        y = df[stats_var].values
        # loop over selfreport_vars
        # code.interact(local = dict(locals(), **globals()))
        tmp = [spearmanr(x.values, y, nan_policy = 'omit') for _, x in df[self_vars].iteritems()]
        r_table[i,:] = [x[0] for x in tmp]
        p_table[i,:] = [x[1] for x in tmp]
        # loop over permuations. within each permutation, calc for each self_var
        for j in range(n_perm):
            np.random.shuffle(y)
            tmp = [spearmanr(x.values, y, nan_policy = 'omit') for _, x in df[self_vars].iteritems()]
            perm_r_[i, :, j] = [x[0] for x in tmp]
    
    # calc max dist
    perm_max_r_dist = perm_r_.max(axis = 1)

    perm_multict_p_table = np.zeros((len(stats_vars), len(self_vars)))
    perm_p_table = np.zeros((len(stats_vars), len(self_vars)))
    for i in range(len(stats_vars)):
        for j in range(len(self_vars)):
            perm_multict_p_table[i, j] = np.mean(abs(r_table[i, j]) < abs(perm_max_r_dist[i,:]))
            perm_p_table[i, j] = np.mean(abs(r_table[i, j]) < abs(perm_r_[i, j, :]))


    pd.DataFrame(perm_multict_p_table, columns = self_vars, index = stats_vars).to_csv(os.path.join(logdir, 'auc_selfreport_corr_perm_multip.csv'))
    pd.DataFrame(perm_p_table, columns = self_vars, index = stats_vars).to_csv(os.path.join(logdir, 'auc_selfreport_corr_perm_p.csv'))
    pd.DataFrame(r_table, columns = self_vars, index = stats_vars).to_csv(os.path.join(logdir, 'auc_selfreport_corr_r.csv'))
    pd.DataFrame(p_table, columns = self_vars, index = stats_vars).to_csv(os.path.join(logdir, 'auc_selfreport_corr_p.csv'))
    # code.interact(local = dict(locals(), **globals()))
    # generate two dfs: corrdf for math calculations, and plotdf with jitters for plotting 
    # tmp = self_sess1.drop(['sess', 'batch', 'duration', 'PAS', 'NAS'], axis = 1)
    # corrdf = stats.merge(tmp, on = "id") # use exact values
    # tmp.loc[:, tmp.columns != 'id'] = tmp.loc[:, tmp.columns != 'id'] + np.random.normal(0, 0.2, tmp.loc[:, tmp.columns != 'id'].shape)
    # plotdf = stats.merge(tmp, on = "id")

    ############### loop over x_vars and y_vars ##############
    # code.interact(local = dict(locals(), **globals()))

    # def my_permutation_test(xs, y, n = 5000):
    #   """


    #   """

    #   r = [speamanr(x, y)[0] for x in xs] 

    #   spearmanr(x, y)
    #   r, p = res

    #   perm_r = np.empty((n, 1))
    #   for i in range(n):
    #       np.random.shuffle(y)
    #       perm_r = [speamanr(x, y)[0] for x in xs]
    #       res = spearmanr(x, y)
    #       r_dist[i] = res[0]
    #   perm_p = np.sum(abs(perm_r) > abs(r)) / n
    #   return r, p, perm_p, perm_r

    # y_var = 'auc' # I don't think it will differ that much for auc1, auc2, auc3 and auc4. # I can add more loops here
    
    # for i in range(4):
    #   sess = sesses[i]
    #   condition = conditions[i]
    #   y = stats[np.logical_and(state.conditions == condition, ), 'auc']


    #   r = [speamanr(x, y)[0] for x in xs] 

    #   res = my_permutation_test(corrdf.loc[np.logical_and(corrdf.condition == condition, corrdf.sess == sess), x_var].values, corrdf.loc[np.logical_and(corrdf.condition == condition, corrdf.sess == sess), y_var].values, n_perm)
    
    #   r_table[i, j] = res[0]
    #   p_table[i, j] = res[1]
    #   perm_p_table[i, j] = res[2]
    #   perm_r_[i, j, :] = res[3]
    
    # for j, x_var in enumerate(x_vars):
    #   y_var_mean = np.mean(corrdf[y_var])
    #   y_var_sd = np.std(corrdf[y_var])
    #   x_var_mean = np.mean(corrdf.loc[corrdf.sess == 1, x_var])
    #   x_var_sd = np.std(corrdf.loc[corrdf.sess == 1, x_var])


    # code.interact(local = dict(locals(), **globals()))
    # r_str_table = np.char.array(np.round(r_table, 4).astype('str'))
    # p_str_table = np.char.array(np.round(p_table, 4).astype('str'))
    # rp_out = 'r = ' + r_str_table + ', ' + 'p = ' + p_str_table
    # p_out = 'p = ' + p_str_table
    # final_out = p_out
    # final_out[p_table < 0.05] = rp_out[p_table < 0.05]
    # final_out = pd.DataFrame(final_out, columns = x_vars, index = [ x + '_AUC_' + str(y) for x, y in zip(conditions, sesses)])
    # final_out.to_csv(os.path.join(logdir, 'auc_selfreport_corr.csv'))


    # so for visualizations I would like to only plot for significant results.
    # plot 
    # x_var = 'motor'
    # y_var = 'auc' 
    # y_var_mean = np.mean(corrdf[y_var])
    # y_var_sd = np.std(corrdf[y_var])
    # x_var_mean = np.mean(corrdf.loc[corrdf.sess == 1, x_var])
    # x_var_sd = np.std(corrdf.loc[corrdf.sess == 1, x_var])
    # g = sns.relplot(data=plotdf, x= x_var, y= y_var, col = "condition", row = "sess", kind = 'scatter', hue = 'condition')
    # # code.interact(local = dict(locals(), **globals()))
    # for i, ax in enumerate(g.axes.flat):
    #   ax.set_ylim(y_var_mean - y_var_sd*3, y_var_mean + y_var_sd*3)
    #   ax.set_xlim(x_var_mean - x_var_sd * 3, x_var_mean + x_var_sd * 3)
    #   # code.interact(local = dict(locals(), **globals()))
    #   sess = sesses[i]
    #   condition = conditions[i]
    #   res = spearmanr(corrdf.loc[np.logical_and(corrdf.condition == condition, corrdf.sess == sess), x_var], corrdf.loc[np.logical_and(corrdf.condition == condition, corrdf.sess == sess), y_var])
    #   ax.text(x_var_mean + x_var_sd * 1.5, 2, 'r = %.3f \n p = %.3f'%(res[0], res[1]))
    #   ax.set_xlabel(x_var)
    #   ax.set_ylabel(y_var + " (s)")
    # plt.show()

    ############ PANAS ###########
    # selfdf_wide = self_sess1.loc[self_sess1.batch == 2].merge(self_sess2.loc[self_sess2.batch == 2], how = 'left', on = 'id', suffixes = ['_sess1', '_sess2'])
    # selfdf_wide['DeltaP'] = selfdf_wide['PAS_sess2'] - selfdf_wide['PAS_sess1'] 
    # selfdf_wide['DeltaN'] = selfdf_wide['NAS_sess2'] - selfdf_wide['NAS_sess1'] 
    # selfdf_wide[['PAS_sess1', 'NAS_sess1', 'PAS_sess2', 'NAS_sess2', 'DeltaP', 'DeltaN']].hist()

    
    # stats_wide = stats_sess1.merge(stats_sess2, how = "left", on = ['condition', 'id', 'block'], suffixes = ['_sess1', '_sess2'])
    # stats_wide['delta_auc'] = stats_wide['auc_sess2'] - stats_wide['auc_sess1']
    # code.interact(local = dict(locals(), **globals()))
    # sns.pairplot(selfdf_wide[['PAS_sess1', 'NAS_sess1', 'PAS_sess2', 'NAS_sess2']])

    # plotdf = stats_wide.merge(selfdf_wide, on = "id")
    # g = sns.relplot(data=plotdf, x="DeltaP", y="delta_auc", col = "condition")
    # count = 1
    # for k in g.axes.flat:
    #   res = spearmanr(plotdf.loc[plotdf.block == count, 'DeltaN'], plotdf.loc[plotdf.block == count, 'delta_auc'], nan_policy = 'omit')
    #   k.text(0, 5, 'r = %.3f, p = %.3f'%(res[0], res[1]))
    #   count = count+1
    plt.show()
    # sns.relplot(data=plotdf, x="DeltaN", y="delta_auc", col = "condition")

def corr_analysis():
    # load summary statistics 
    stats_sess1 = pd.read_csv(os.path.join("..", "analysis_results", "taskstats", "emp_sess1.csv"))
    stats_sess1['sess'] = 1
    stats_sess2 = pd.read_csv(os.path.join("..", "analysis_results", "taskstats", "emp_sess2.csv"))
    stats_sess2['sess'] = 2
    stats = stats_sess1.merge(stats_sess2, on = ["id", "block", "condition"], suffixes = ['_sess1', '_sess2'])
    # code.interact(local = dict(locals(), **globals()))


    # 
    k_df = pd.read_csv("k.csv")
    k_df['gm_logk'] = np.log(k_df['GMK'])
    k_df = k_df.loc[np.logical_and.reduce([k_df.SmlCons >= 0.8, k_df.LrgCons >= 0.8, k_df.MedCons >= 0.8])]
    fig,ax = plt.subplots(1, 1)
    ax.hist(x = np.log(k_df['GMK']))
    ax.set_xlabel("log k")

    # correlations with log_k
    plotdf = k_df.merge(stats_sess2, left_on = "SubjID", right_on = "id")
    g = sns.FacetGrid(
        data=plotdf, col = 'condition'
    )
    g.map(sns.scatterplot, "gm_logk", "auc")

    # AUC timelines
    plotdf1 = pd.melt(stats_sess1.loc[stats_sess1.block == 1, ['auc1', 'auc2', 'auc3', 'auc4', 'id']], id_vars = "id")
    plotdf2 = pd.melt(stats_sess1.loc[stats_sess1.block == 2, ['auc1', 'auc2', 'auc3', 'auc4', 'id']], id_vars = "id")
    plotdf1['time'] = [int(x[-1]) for x in plotdf1.variable]
    plotdf2['time'] = [int(x[-1]) + 4 for x in plotdf2.variable]
    plotdf = pd.concat([plotdf1, plotdf2]).reset_index()
    g = sns.lineplot(
        data=plotdf, x="time", y="value", err_style="bars", ci=95
    )
    g.set(xlim=(0, 9))
    g.set(ylabel='AUC (s)')

    plotdf1 = pd.melt(stats_sess2.loc[stats_sess2.block == 1, ['auc1', 'auc2', 'auc3', 'auc4', 'id']], id_vars = "id")
    plotdf2 = pd.melt(stats_sess2.loc[stats_sess2.block == 2, ['auc1', 'auc2', 'auc3', 'auc4', 'id']], id_vars = "id")
    plotdf1['time'] = [int(x[-1]) for x in plotdf1.variable]
    plotdf2['time'] = [int(x[-1]) + 4 for x in plotdf2.variable]
    plotdf = pd.concat([plotdf1, plotdf2]).reset_index()
    g = sns.lineplot(
        data=plotdf, x="time", y="value", err_style="bars", ci=68
    )
    g.set(xlim=(0, 9))
    g.set(ylabel='AUC (s)')
    plt.savefig(os.path.join("..", "figures", "auc_timeline.png"))

    # correlations
    g = sns.FacetGrid(stats, col = "condition", hue = "condition")
    g.map(sns.scatterplot, "auc4_sess1", "auc4_sess2")
    g.axes.flat[0].plot(np.linspace(0, 12, 20), np.linspace(0, 12, 20), color = 'black')
    g.axes.flat[1].plot(np.linspace(0, 12, 20), np.linspace(0, 12, 20), color = 'black')
    LPcorr, _ = pearsonr(stats.auc4_sess1.loc[stats.block == 1], stats.auc4_sess2.loc[stats.block == 1])
    LPcorr
    HPcorr, _ = pearsonr(stats.auc4_sess1.loc[stats.block == 2], stats.auc4_sess2.loc[stats.block == 2])
    HPcorr

    # plot correlations 
    # fig,ax = plt.subplots(1, 2)
    # df_block1 = pd.merge(stats_sess1.loc[stats_sess1.block == 1], stats_sess2.loc[stats_sess1.block == 1], how = 'right', on = "id", suffixes = ['_s1', '_s2'])
    # ax[0].scatter(df_block1.auc_s1, df_block1.auc_s2)
    # ax[0].set_xlabel("LP AUC (s) SESS1")
    # ax[0].set_ylabel("LP AUC (s) SESS2")
    # m, b = np.polyfit(df_block1.auc_s1, df_block1.auc_s2, 1)
    # plt.plot(df.adapt_sess1, m*df.adapt_sess1 + b)

    # df_block2 = pd.merge(stats_sess1.loc[stats_sess1.block == 2], stats_sess2.loc[stats_sess1.block == 2], how = 'right', on = "id", suffixes = ['_s1', '_s2'])
    # ax[1].scatter(df_block2.auc_s1, df_block2.auc_s2)
    # ax[1].set_xlabel("HP AUC (s) SESS1")
    # ax[1].set_ylabel("HP AUC (s) SESS2")

    # auc adaption 
    fig,ax = plt.subplots(1, 1)
    adapt1 = stats_sess1['auc4'].loc[np.equal(stats_sess1.block, 1)].values - stats_sess1['auc1'].loc[stats_sess1.block == 2].values
    adapt2 = stats_sess2['auc4'].loc[stats_sess1.block == 1].values - stats_sess2['auc1'].loc[stats_sess1.block == 2].values
    df1 = pd.DataFrame({"id": stats_sess1.id[stats_sess1.block == 1].values, "adapt": adapt1})
    junk = {"id": stats_sess2.id[stats_sess2.block == 1].values, "adapt": adapt2}
    df2 = pd.DataFrame(junk)
    df = df1.merge(df2, on = "id", suffixes = ["_sess1", "_sess2"])
    ax.scatter(df.adapt_sess1, df.adapt_sess2)
    ax.set_xlabel("LP AUC - HP AUC, SESS1")
    ax.set_ylabel("LP AUC - HP AUC, SESS2")
    m, b = np.polyfit(df.adapt_sess1, df.adapt_sess2, 1)
    # plt.plot(df.adapt_sess1, m*df.adapt_sess1 + b)
    corr, _ = pearsonr(df.adapt_sess1, df.adapt_sess2)
    print('Pearsons correlation: %.3f' % corr)

    # with 
    plotdf = k_df.merge(df, left_on = "SubjID", right_on = "id")
    fig,ax = plt.subplots(1, 1)
    ax.scatter("gm_logk", "adapt_sess1", data = plotdf)
    ax.set_ylabel("LP AUC - HP AUC")
    ax.set_xlabel("log k")

    plotdf = k_df.merge(df, left_on = "SubjID", right_on = "id")
    fig,ax = plt.subplots(1, 1)
    ax.scatter("gm_logk", "adapt_sess2", data = plotdf)
    ax.set_ylabel("LP AUC - HP AUC")
    ax.set_xlabel("log k")

    ################## Yeah I might want to look at these auc directly 
    fig,ax = plt.subplots(1, 2)
    ax[0].scatter(stats_sess1.auc.loc[stats_sess1.block == 1], stats_sess1.auc.loc[stats_sess1.block == 2])
    ax[0].set_xlabel("SESS1 LP AUC (s)")
    ax[0].set_ylabel("SESS1 HP AUC (s)")
    ax[1].scatter(stats_sess2.auc.loc[stats_sess2.block == 1], stats_sess2.auc.loc[stats_sess2.block == 2])
    ax[1].set_xlabel("SESS2 LP AUC (s)")
    ax[1].set_ylabel("SESS2 HP AUC (s)")
############################# group-level analysis functions #####################


############################## main ##########################
if __name__ == "__main__":
    ##############################
    ###        load data       ###
    ##############################
    expname = 'passive'
    hdrdata_sess1, trialdata_sess1_ = loadFxs.group_quality_check(expname, 1, plot_quality_check = True)
    hdrdata_sess2, trialdata_sess2_ = loadFxs.group_quality_check(expname, 2, plot_quality_check = True)
    s1_stats = pd.read_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'emp_sess1.csv'))
    s2_stats = pd.read_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'emp_sess2.csv'))
    code.interact(local = dict(globals(), **locals()))

    #################################
    ## conduct model-free analysis ##
    #################################
    s1_stats, s1_Psurv_b1_, s1_Psurv_b2_, s1_WTW_ = analysisFxs.group_MF(trialdata_sess1_, plot_each = False)   
    s1_stats.to_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'emp_sess1.csv'), index = None)

    s2_stats, s2_Psurv_b1_, s2_Psurv_b2_, s2_WTW_ = analysisFxs.group_MF(trialdata_sess2_, plot_each = False)   
    s2_stats.to_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'emp_sess2.csv'), index = None)
    

    # I want to plot example survival curve
    # pltdata = pd.DataFrame({
    #     "time": np.tile(expParas.Time, 2),
    #     "Psurve": np.concatenate([s1_Psurv_b1_[0,:], s1_Psurv_b1_[1,:]]),
    #     "condition": np.repeat(["LP", "HP"], len(expParas.Time))
    #     })

    # ax = sns.lineplot(data = pltdata, x="time", y="Psurve", hue = "condition", lw = 2, palette = expParas.conditionColors.values())
    # ax.set_xlabel("Elapased time (s)")
    # ax.set_ylabel("Survival rate")
    # ax.set_ylim([-0.05, 1.05])
    # ax.legend(frameon = False, title = "")
    # plt.tight_layout()
    # plt.show()

    ############################
    ## group-level behavior  ##
    ############################
    # replicate Wtw 



    ####################
    # calc reliability # 
    ######################
colvars = ['auc', 'auc1', 'auc2', "auc_rh", 'std_wtw', 'std_wtw1', 'std_wtw2', "std_wtw_rh"]
s1_HP = s1_stats.loc[s1_stats['condition'] == 'HP', colvars + ['id']]
s1_LP = s1_stats.loc[s1_stats['condition'] == 'LP', colvars + ['id']]
s1_df = s1_HP.merge(s1_LP, left_on = 'id', right_on = 'id', suffixes = ['_HP', "_LP"])

s2_HP = s2_stats.loc[s2_stats['condition'] == 'HP', colvars + ['id']]
s2_LP = s2_stats.loc[s2_stats['condition'] == 'LP', colvars + ['id']]
s2_df = s2_HP.merge(s2_LP, left_on = 'id', right_on = 'id', suffixes = ['_HP', "_LP"])

# add delta variables
auc_vars = ['auc', 'auc1', 'auc2', "auc_rh"]
s1_df = pd.concat([s1_df, s1_df.filter(like = "auc", axis = 1).filter(like='HP', axis=1).set_axis([x+'_delta' for x in auc_vars], axis = 1) - s1_df.filter(like = "auc", axis = 1).filter(like='LP', axis=1).set_axis([x+'_delta' for x in auc_vars], axis = 1)], axis = 1)
s2_df = pd.concat([s2_df, s2_df.filter(like = "auc", axis = 1).filter(like='HP', axis=1).set_axis([x+'_delta' for x in auc_vars], axis = 1) - s2_df.filter(like = "auc", axis = 1).filter(like='LP', axis=1).set_axis([x+'_delta' for x in auc_vars], axis = 1)], axis = 1)

# add mean variables 
s1_df = pd.concat([s1_df, (s1_df.filter(like = "auc", axis = 1).filter(like='HP', axis=1).set_axis([x+'_ave' for x in auc_vars], axis = 1) + s1_df.filter(like = "auc", axis = 1).filter(like='LP', axis=1).set_axis([x+'_ave' for x in auc_vars], axis = 1)) / 2], axis = 1) 
s2_df = pd.concat([s2_df, (s2_df.filter(like = "auc", axis = 1).filter(like='HP', axis=1).set_axis([x+'_ave' for x in auc_vars], axis = 1) + s2_df.filter(like = "auc", axis = 1).filter(like='LP', axis=1).set_axis([x+'_ave' for x in auc_vars], axis = 1)) / 2], axis = 1)

# add init_wtw variables
s1_df = s1_df.merge(s1_stats.loc[s1_stats.block == 1, ["id", "init_wtw"]], on = "id")
#s1_df = s1_df.merge(s2_stats.loc[s2_stats.block == 1, ["id", "init_wtw"]], on = "id") 
s1_df['init_wtw'] = s1_df['init_wtw'] + 1
s2_df = s2_df.merge(s2_stats.loc[s2_stats.block == 1, ["id", "init_wtw"]], on = "id")

        # merge
df = s1_df.merge(s2_df, on = 'id', suffixes = ['_sess1', '_sess2']) 


    # variable names 
vars = ['auc_delta', 'auc1_delta', 'auc2_delta'] + ['auc_ave', 'auc1_ave', 'auc2_ave'] + [x + "_HP" for x in colvars] + [x + "_LP" for x in colvars] + ['init_wtw']
rows = ['spearman_rho', 'pearson_rho', 'abs_icc', 'con_icc', "ssbs", "ssbm", "sse", "msbs", "msbm", "mse"]
reliable_df = np.zeros([len(rows), len(vars)])
for i, var in enumerate(vars):
    reliable_df[:,i] = analysisFxs.calc_reliability(df.loc[:, var + '_sess1'], df.loc[:, var + '_sess2'])

reliable_df = pd.DataFrame(reliable_df, columns = vars, index = rows)
    reliable_df.to_csv(os.path.join("..", "analysis_results", expname, "mf_reliability.csv"))

    ######################
    ## plot reliability ##
    ######################
    # AUC reliability #
    fig, ax = plt.subplots()
    figFxs.AUC_reliability(s1_stats, s2_stats) # maybe I don't need it yet 
    plt.gcf().set_size_inches(8, 6)
    plt.savefig(os.path.join("..", "figures", expname, "AUC_reliability.pdf"))

    # delta AUC 
    fig, ax = plt.subplots()
    figFxs.my_regplot(df.loc[:, 'auc_delta_sess1'], df.loc[:, 'auc_delta_sess2'], color = "grey")
    fig.gca().set_ylabel(r'$\Delta$' + 'AUC SESS2 (s)')
    fig.gca().set_xlabel(r'$\Delta$' + 'AUC SESS1 (s)')
    plt.gcf().set_size_inches(4, 4)
    plt.tight_layout()
    plt.savefig(os.path.join("..", "figures", expname, "delta_auc_reliability.pdf"))


    ##############
    ## modelrep ##
    ##############
    for modelname in modelnames:
        s1_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, modelname)
        s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname, modelname)
        s1_stats_rep, s1_WTW_rep = modelFxs.group_model_rep(trialdata_sess1_, s1_paradf, modelname, isTrct = True, plot_each = False)
        s2_stats_rep, s2_WTW_rep = modelFxs.group_model_rep(trialdata_sess2_, s2_paradf, modelname, isTrct = True, plot_each = False)
        s1_stats_rep.to_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_sess1.csv'%modelname), index = None)
        s2_stats_rep.to_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_sess2.csv'%modelname), index = None)


        figFxs.plot_group_emp_rep(modelname, s1_stats_rep, s2_stats_rep, s1_stats, s2_stats)
        plt.gcf().set_size_inches(8, 4)
        plt.savefig(os.path.join('..', 'figures', expname, 'auc_emp_rep_%s.pdf'%modelname))

        figFxs.plot_group_emp_rep_wtw(modelname, s1_WTW_rep, s2_WTW_rep, s1_WTW_, s2_WTW_, hdrdata_sess1, hdrdata_sess2, s1_paradf, s2_paradf)
        plt.gcf().set_size_inches(8, 4)
        plt.savefig(os.path.join('..', 'figures', expname, 'wtw_emp_rep_%s.pdf'%modelname))

    ##########################
    ## parameter histograms ##
    ##########################
    for modelname in modelnames:
        s1_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, modelname)
        s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname, modelname)
        figFxs.plot_parameter_distribution(modelname, s1_paradf, s2_paradf)
        plt.gcf().set_size_inches(4 * len(paranames), 10)
        plt.savefig(os.path.join("..", 'figures', expname, "%s_para_hist.pdf"%modelname))
        # 
        figFxs.plot_parameter_compare(modelname, s1_paradf.iloc[:,:-1], s2_paradf.iloc[:,:-1], subtitles)

    ###################### parameter reliability #########
    for modelname in ['QL1', 'QL2']:
        paranames = modelFxs.getModelParas(modelname)
        s1_paradf  = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, modelname)
        s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname, modelname)
        if modelname == "QL1":
            subtitles = [r'$\mathbf{log(\alpha)}$', r'$\mathbf{\tau}$', r'$\mathbf{\gamma}$', r'$\mathbf{log(\eta)}$']
        elif modelname == "QL2":
            subtitles = [r'$\mathbf{log(\alpha)}$', r'$\mathbf{log(\nu)}$', r'$\mathbf{\tau}$', r'$\mathbf{\gamma}$', r'$\mathbf{log(\eta)}$']
        figFxs.plot_parameter_reliability('QL2', s1_paradf.iloc[:,:-1], s2_paradf.iloc[:,:-1], subtitles)
        plt.gcf().set_size_inches(4 * len(paranames), 5)
        plt.savefig(os.path.join("..", 'figures', expname, "%s_para_reliability.pdf"%modelname))





    ##########
    fig, ax = plt.subplots()
    figFxs.plot_group_KMSC_both(s1_Psurv_b1_, s1_Psurv_b2_, s2_Psurv_b1_, s2_Psurv_b2_, hdrdata_sess1, hdrdata_sess2, ax)
    plt.savefig(os.path.join('..', 'figures', expname, 'KMSC.pdf'))


    fig, ax = plt.subplots()
    figFxs.plot_group_WTW_both(s1_WTW_, s2_WTW_, hdrdata_sess1, hdrdata_sess2, ax)
    fig.savefig(os.path.join('..', 'figures', expname, 'WTW.pdf'))

    fig, ax = plt.subplots()
    analysisFxs.plot_group_KMSC(s1_Psurv_b1_, s1_Psurv_b2_, expParas.Time, ax)
    plt.savefig(os.path.join('..', 'figures', expname, 'sess1_KMSC.pdf'))

    fig, ax = plt.subplots()
    analysisFxs.plot_group_KMSC(s2_Psurv_b1_, s2_Psurv_b2_, expParas.Time, ax)
    plt.savefig(os.path.join('..', 'figures', expname, 'sess2_KMSC.pdf'))

    fig, ax = plt.subplots()
    analysisFxs.plot_group_WTW(s1_WTW_, expParas.TaskTime, ax)
    plt.savefig(os.path.join('..', 'figures', expname, 'sess1_WTW.pdf'))


    fig, ax = plt.subplots()
    analysisFxs.plot_group_WTW(s2_WTW_, expParas.TaskTime, ax)
    plt.savefig(os.path.join('..', 'figures', expname, 'sess2_WTW.pdf'))

    ######################## reliability of AUC measures ##############
    fig, ax = plt.subplots()
    figFxs.AUC_reliability(s1_stats, s2_stats) # maybe I don't need it yet 
    plt.gcf().set_size_inches(8, 6)
    plt.savefig(os.path.join("..", "figures", expname, "AUC_reliability.pdf"))

    ######################## reliability of AUc 
    # maybe later I want to write a function to automatically pivot this table
    colvars = ['auc', 'auc1', 'auc2']
    s1_HP = s1_stats.loc[s1_stats['condition'] == 'HP', colvars + ['id']]
    s1_LP = s1_stats.loc[s1_stats['condition'] == 'LP', colvars + ['id']]
    s1_aucdf = s1_HP.merge(s1_LP, left_on = 'id', right_on = 'id', suffixes = ['_HP', "_LP"])
    s1_aucdf = pd.concat([s1_aucdf, s1_aucdf.filter(like='HP', axis=1).set_axis([x+'_delta' for x in colvars], axis = 1) - s1_aucdf.filter(like='LP', axis=1).set_axis([x+'_delta' for x in colvars], axis = 1)], axis = 1)

    s2_HP = s2_stats.loc[s2_stats['condition'] == 'HP', colvars + ['id']]
    s2_LP = s2_stats.loc[s2_stats['condition'] == 'LP', colvars + ['id']]
    s2_aucdf = s2_HP.merge(s2_LP, left_on = 'id', right_on = 'id', suffixes = ['_HP', "_LP"])
    s2_aucdf = pd.concat([s2_aucdf, s2_aucdf.filter(like='HP', axis=1).set_axis([x+'_delta' for x in colvars], axis = 1) - s2_aucdf.filter(like='LP', axis=1).set_axis([x+'_delta' for x in colvars], axis = 1)], axis = 1)

    # plot auc reliability 
    df = s1_aucdf.merge(s2_aucdf, on = 'id', suffixes = ['_sess1', '_sess2']) 
    fig, ax = plt.subplots()
    # figFxs.my_regplot(df['auc_LP_sess1'], df['auc_LP_sess2'], axs[0], color = condition_palette[0])
    # figFxs.my_regplot(df['auc1_LP_sess1'], df['auc1_LP_sess2'], axs[0], color = condition_palette[0], line_kws={"linestyle": ":"})
    # figFxs.my_regplot(df['auc_HP_sess1'], df['auc_HP_sess2'], axs[1], color = condition_palette[1])
    figFxs.my_regplot(df['auc_delta_sess1'], df['auc_delta_sess2'], ax, color = 'grey')
    plt.tight_layout()
    plt.savefig(os.path.join("..", "figures",expname, "delta_auc_reliability.pdf"))

    ######################## reliability of selfreport measures ##############
    s1_selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
    s1_selfdf.to_csv(os.path.join("..", "analysis_results", expname, "selfreport", "selfreport_sess1.csv"), index = None)

    s2_selfdf = loadFxs.parse_group_selfreport(expname, 2, isplot = False)
    s2_selfdf.to_csv(os.path.join("..", "analysis_results", expname, "selfreport", "selfreport_sess2.csv"), index = None)

    selfdf = s2_selfdf.merge(s1_selfdf, on = "id", suffixes = ["_sess1", "_sess2"])
        # selfreport_vars = ['NU', 'PU', 'PM', 'PS', 'SS', 'attention', 'cogstable', 'motor', 'perseverance', 'selfcontrol', 'cogcomplex', 'UPPS', 'BIS', 'GMK'] 
    to_be_tested_vars = list(zip([x + "_sess1" for x in expParas.selfreport_vars], [x + "_sess2" for x in expParas.selfreport_vars]))
    spearman_rho_, pearson_rho_, abs_icc_, con_icc_, n_, report = analysisFxs.calc_zip_reliability(selfdf, to_be_tested_vars)
    report.sort_values(by = "spearman_rho")

    
    ################################ effect of temporary mood, no effects at all##################
    selfdf['NAS_change'] = selfdf['NAS_sess2'] - selfdf['NAS_sess1']
    stats_df = s1_df.merge(s2_df, on = "id", suffixes = ["_sess1", "_sess2"])
    stats_df['auc_HP_change'] = stats_df['auc_HP_sess2'] -  stats_df['auc_HP_sess1']
    stats_df['auc_LP_change'] = stats_df['auc_LP_sess2'] -  stats_df['auc_LP_sess1']
    stats_df['auc_delta_change'] = stats_df['auc_delta_sess2'] -  stats_df['auc_delta_sess1']
    df = selfdf.merge(stats_df, on = "id")
    stats.spearmanr(df['NAS_change'], df['auc_delta_change'])

    #################### correlations between model parameters and self-reports##########
    paradf  = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess2, 'QL2', 'QL2')
    selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
    tmp = paradf.merge(selfdf , on = "id")
    row_vars = ['alpha', "nu", "tau", "gamma", "eta"]
    col_vars = ['NU', 'PU', 'PM', 'PS', 'SS', 'attention', 'cogstable', 'motor', 'perseverance', 'selfcontrol', 'cogcomplex', 'UPPS', 'BIS', 'GMK'] 
    r_, p_ = analysisFxs.calc_prod_correlations(tmp, row_vars, col_vars)
    r_.to_csv(os.path.join("..", "analysis_results", expname, "correlation", "r_para_selfreport_sess1.csv"))
    p_.to_csv(os.path.join("..", "analysis_results", expname, "correlation", "p_para_selfreport_sess1.csv"))
    
    row_vars = ['GMK']
    col_vars = ['NU', 'PU', 'PM', 'PS', 'SS', 'attention', 'cogstable', 'motor', 'perseverance', 'selfcontrol', 'cogcomplex', 'UPPS', 'BIS']
    r_, p_ = analysisFxs.calc_prod_correlations(selfdf, row_vars, col_vars)


    row_vars = ['NU', 'PU', 'PM', 'PS', 'SS']
    col_vars = ['NU', 'PU', 'PM', 'PS', 'SS']
    r_, p_ = analysisFxs.calc_prod_correlations(selfdf, row_vars, col_vars)


    row_vars = ['attention', 'cogstable', 'motor', 'perseverance', 'selfcontrol', 'cogcomplex']
    col_vars = ['attention', 'cogstable', 'motor', 'perseverance', 'selfcontrol', 'cogcomplex']
    r_, p_ = analysisFxs.calc_prod_correlations(selfdf, row_vars, col_vars)

    #################### correlations between task measures and self-report measures ###########
    stats_df = s1_stats
    selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)

    task_var = ['auc']
    aucdf = stats_df.pivot(index = 'id', columns = 'condition', values = task_var)
    aucdf = aucdf.rename_axis(columns = [None] * 2)
    aucdf = aucdf.droplevel(0, axis=1) 
    aucdf = aucdf.reset_index()

    tmp = aucdf.merge(selfdf, on = "id")

    row_vars = ["HP", "LP"]
    col_vars = ['NU', 'PU', 'PM', 'PS', 'SS', 'attention', 'cogstable', 'motor', 'perseverance', 'selfcontrol', 'cogcomplex', 'UPPS', 'BIS', 'GMK'] 
    r_, p_ = analysisFxs.calc_prod_correlations(tmp, row_vars, col_vars)
    r_.to_csv(os.path.join("..", "analysis_results", expname, "correlation", "r_auc_selfreport_sess1.csv"))
    p_.to_csv(os.path.join("..", "analysis_results", expname, "correlation", "p_auc_selfreport_sess1.csv"))


    ################### correlations between self-report measures and ###########
    
    ################### correlations between self-report measures and ###########
    # how to align 
    task_var = ['auc']
    row_vars = ['HP', "LP"]
    col_vars = ['NU', 'PU', 'PM', 'PS', 'SS', 'attention', 'cogstable', 'motor', 'perseverance', 'selfcontrol', 'cogcomplex', 'UPPS', 'BIS', 'GMK'] 

    for sess in [1, 2]:
        if sess == 1:
            stats_df = s1_stats
        else:
            stats_df = s2_stats
        aucdf = stats_df.pivot(index = 'id', columns = 'condition', values = task_var)
        aucdf = aucdf.rename_axis(columns = [None] * 2)
        aucdf = aucdf.droplevel(0, axis=1) 
        aucdf = aucdf.reset_index()
        if sess == 1:
            tmp = aucdf.merge(s1_selfdf, on = "id")
        else:
            tmp = aucdf.merge(s2_selfdf, on = "id")
        # update them later?
        r_table, p_table, perm_r_, perm_p_table = figFxs.corr_analysis(tmp[row_vars], tmp[col_vars], 1000)
        p_table.to_csv(os.path.join("..", "analysis_results", expname, "correlation", "p_selfreport_simple_" + "auc" + "_sess%d.csv"%sess))
        r_table.to_csv(os.path.join("..", "analysis_results", expname, "correlation", "r_selfreport_simple_" + "auc" + "_sess%d.csv"%sess))
        perm_p_table.to_csv(os.path.join("..", "analysis_results", expname, "correlation", "perm_p_selfreport_simple_" + "auc" + "_sess%d.csv"%sess))
        r_table.columns = ['NU', 'PU', 'PM', 'PS', 'SS', 'attention', 'cogstable', 'motor', 'perseverance', 'selfcontrol', 'cogcomplex', 'UPPS', 'BIS', 'Discount'] 

    # plot all 
    # col_vars = ['NU', 'PU', 'PM', 'PS', 'SS', 'attention', 'cogstable', 'motor', 'perseverance', 'selfcontrol', 'cogcomplex', 'UPPS', 'BIS', 'Discounting'] 
    # plotdf = pd.DataFrame()
    # for i, row_var in enumerate(row_vars):
    #     for j, col_var in enumerate(col_vars):
    #         # this_row = pd.DataFrame({
    #         #     "mean": perm_r_[i, j,:].mean(),
    #         #     "q2.5": np.quantile(perm_r_[i, j,:], 0.025),
    #         #     "q97.5": np.quantile(perm_r_[i, j,:], 0.975),
    #         #     "row_var": row_var,
    #         #     "col_var": col_var,
    #         #     "observed": r_table.loc[row_var, col_var],
    #         #     }, index = [i * len(col_vars) + j])
    #         this_df = pd.DataFrame({
    #             "null": perm_r_[i, j,:],
    #             "observed": np.repeat(r_table.loc[row_var, col_var],len(perm_r_[i, j,:])),
    #             "col_var": np.repeat(col_var, len(perm_r_[i, j,:])),
    #             "row_var": np.repeat(row_var, len(perm_r_[i, j,:]))
    #             })
    #         plotdf = pd.concat([plotdf, this_df]) 

    #     g = sns.FacetGrid(plotdf.loc[np.isin(plotdf['col_var'], ['UPPS', 'BIS', 'Discounting'])], col = "row_var")
    #     g.map(sns.violinplot, "col_var", "null", color = "grey", vertical = True, label = "H0")
    #     g.map(sns.pointplot, "col_var", "observed", join=False, color = "red", label = "Observed")
    #     g.set(xlabel = "", ylabel = "Spearman's rho")

    #     plotdf = pd.DataFrame({
    #         "col_var":np.tile(['NU', 'PU', 'PM', 'PS', 'SS', 'attention', 'cogstable', 'motor', 'perseverance', 'selfcontrol', 'cogcomplex', 'UPPS', 'BIS', 'Discounting'], 2),
    #         "row_var": np.repeat(["HP_AUC", "LP_AUC"], len(r_table.columns)),
    #         "correlation": np.concatenate([r_table.loc['HP',:].values, r_table.loc['LP',:].values])})
    #     g = sns.FacetGrid(plotdf.loc[np.isin(plotdf['col_var'], ['UPPS', 'BIS', 'Discounting k'])], col = 'row_var')
    #     g.map(sns.barplot, "col_var", "correlation")
    #     g.set(xlabel = "", ylabel = "Spearman's rho with AUC")


    # s1_stats.loc[s1_stats.condition == "LP",'init_wtw']
    # tmp = s1_selfdf.merge(s1_stats.loc[s1_stats.condition == "LP",['init_wtw', "id", "auc"]], on = "id")
    # r_table, p_table, perm_r_ = figFxs.corr_analysis(tmp.loc[:,['init_wtw', 'auc']],tmp[col_vars], 10)
    # p_table.to_csv(os.path.join("..", "analysis_results", expname, "correlation", "p_selfreport_" + "init_wtw" + "_sess1.csv"))
    # r_table.to_csv(os.path.join("..", "analysis_results", expname, "correlation", "r_selfreport_" + "init_wtw" + "_sess1.csv"))

    ###############################################################################################
    ########### load data #############
    expname = 'active'
    hdrdata_sess1, trialdata_sess1_ = loadFxs.group_quality_check(expname, 1, plot_quality_check = False)
    hdrdata_sess2, trialdata_sess2_ = loadFxs.group_quality_check(expname, 2, plot_quality_check = False)
    s1_stats = pd.read_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'emp_sess1.csv'))
    s2_stats = pd.read_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'emp_sess2.csv'))
    code.interact(local = dict(globals(), **locals()))
    # s1_stats, s1_Psurv_b1_, s1_Psurv_b2_, s1_WTW_ = analysisFxs.group_MF(trialdata_sess1_, plot_each = False)
    # s2_stats, s2_Psurv_b1_, s2_Psurv_b2_, s2_WTW_ = analysisFxs.group_MF(trialdata_sess2_, plot_each = False)
    
    ############## get demographic infomation about the final sample ########
    # consent_sess1 = pd.read_csv(os.path.join("data", expname, "consent_sess1.csv"))
    # tmp = hdrdata_sess2.merge(consent_sess1, on = "id")
    # tmp['batch_x'].value_counts()
    # tmp['gender'].value_counts()
    # tmp['age'].mean()
    
    # ############### model free analysis ###########
    s1_stats, s1_Psurv_b1_, s1_Psurv_b2_, s1_WTW_ = analysisFxs.group_MF(trialdata_sess1_, plot_each = False)
    s2_stats, s2_Psurv_b1_, s2_Psurv_b2_, s2_WTW_ = analysisFxs.group_MF(trialdata_sess2_, plot_each = False)

    # s1_stats.to_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'emp_sess1.csv'), index = None)
    # s2_stats.to_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'emp_sess2.csv'), index = None)
    s1_stats = pd.read_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'emp_sess1.csv'))
    s2_stats = pd.read_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'emp_sess2.csv'))

    # plot for KMSC 
    fig, ax = plt.subplots()
    figFxs.plot_group_KMSC_both(s1_Psurv_b1_, s1_Psurv_b2_, s2_Psurv_b1_, s2_Psurv_b2_, hdrdata_sess1, hdrdata_sess2, ax)
    plt.savefig(os.path.join('..', 'figures', expname, 'KMSC.pdf'))

    # plot for WTW 
    fig, ax = plt.subplots()
    figFxs.plot_group_WTW_both(s1_WTW_, s2_WTW_, hdrdata_sess1, hdrdata_sess2, ax)
    fig.savefig(os.path.join('..', 'figures', expname, 'WTW.pdf'))

    ##### compare auc in different sessions and conditions
    ##### I think this part can be wrapped up #############
    stats_df = pd.concat([s1_stats.loc[np.isin(s1_stats['id'], s2_stats['id'])], s2_stats], axis = 0)
    stats_df['sess_condition'] = (stats_df.sess - 1) * 2 + (stats_df.condition == 'HP') * 1

    # statistical tests/ maybe later I would like to switch to bootstrap
    stats_df.groupby(['sess', 'condition']).agg({'auc':['median', lambda x: np.quantile(x, 0.25), lambda x: np.quantile(x, 0.75)]})
    stats_df.groupby(['sess', 'condition']).agg({'auc1':['median', lambda x: np.quantile(x, 0.25), lambda x: np.quantile(x, 0.75)]})

    # add statistic tests
    stats.wilcoxon(stats_df.loc[np.logical_and(stats_df.condition == "HP", stats_df.sess == 1), 'auc'],\
        stats_df.loc[np.logical_and(stats_df.condition == "LP", stats_df.sess == 1), 'auc'])
    stats.wilcoxon(stats_df.loc[np.logical_and(stats_df.condition == "HP", stats_df.sess == 2), 'auc'],\
        stats_df.loc[np.logical_and(stats_df.condition == "LP", stats_df.sess == 2), 'auc'])
    stats.wilcoxon(stats_df.loc[np.logical_and(stats_df.condition == "HP", stats_df.sess == 1), 'auc'],\
        stats_df.loc[np.logical_and(stats_df.condition == "HP", stats_df.sess == 2), 'auc'])
    stats.wilcoxon(stats_df.loc[np.logical_and(stats_df.condition == "LP", stats_df.sess == 1), 'auc'],\
        stats_df.loc[np.logical_and(stats_df.condition == "LP", stats_df.sess == 2), 'auc'])

    stats.wilcoxon(stats_df.loc[np.logical_and(stats_df.condition == "HP", stats_df.sess == 1), 'auc'],\
        stats_df.loc[np.logical_and(stats_df.condition == "HP", stats_df.sess == 2), 'auc1'])

    stats.wilcoxon(stats_df.loc[np.logical_and(stats_df.condition == "LP", stats_df.sess == 1), 'auc2'],\
        stats_df.loc[np.logical_and(stats_df.condition == "LP", stats_df.sess == 2), 'auc2'])

    stats.wilcoxon(stats_df.loc[np.logical_and(stats_df.condition == "HP", stats_df.sess == 1), 'auc1'],\
        stats_df.loc[np.logical_and(stats_df.condition == "HP", stats_df.sess == 2), 'auc1'])
    stats.wilcoxon(stats_df.loc[np.logical_and(stats_df.condition == "HP", stats_df.sess == 1), 'auc2'],\
        stats_df.loc[np.logical_and(stats_df.condition == "HP", stats_df.sess == 2), 'auc2'])
    
    #################### compare multiple reliability #############
    s1_pivot_stats = analysisFxs.pivot_by_condition(s1_stats, ['auc'], ['id'])
    s2_pivot_stats = analysisFxs.pivot_by_condition(s2_stats, ['auc'], ['id'])
    s1_pivot_stats = s1_pivot_stats[np.isin(s1_pivot_stats.id, s2_pivot_stats.id)]


# hmm I don't know what does it mean though
    for varname in ['auc_HP', 'auc_LP', 'auc_delta']:
        r, ci, resampled_r = analysisFxs.my_bootstrap_corr(s1_pivot_stats.loc[:, varname], s2_pivot_stats.loc[:, varname], n = 150)
        var_ = np.concatenate([var_, [varname] * 1000])
        model_ = np.concatenate([model_, ['MF']  * 1000])
        resampled_r_  = np.concatenate([resampled_r_, resampled_r])



    resampled_r_ = np.array([])
    var_ = np.array([])
    model_ = np.array([])

    s1_paradf_QL2  = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, 'QL2', 'QL2')
    s2_paradf_QL2 = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, 'QL2', 'QL2')
    s1_paradf_QL2 = s1_paradf_QL2[np.isin(s1_paradf_QL2.id, s2_paradf_QL2.id)]
    s2_paradf_QL2 = s2_paradf_QL2[np.isin(s2_paradf_QL2.id, s1_paradf_QL2.id)]

    
    common_ids = list(set(s1_paradf_QL1.id) & set(s1_paradf_QL2.id))
    s1_tmp = pd.melt(s1_paradf_QL2.loc[np.isin(s1_paradf_QL2.id, common_ids)], id_vars = "id", value_vars = ['alpha', 'nu', 'tau', 'gamma', 'eta'])
    s2_tmp = pd.melt(s2_paradf_QL2.loc[np.isin(s2_paradf_QL2.id, common_ids)], id_vars = "id", value_vars = ['alpha', 'nu', 'tau', 'gamma', 'eta'])
    s1_tmp.groupby('variable').agg(["median", lambda x: np.quantile(x, 0.25), lambda x: np.quantile(x, 0.75)])
    s2_tmp.groupby('variable').agg(["median", lambda x: np.quantile(x, 0.25), lambda x: np.quantile(x, 0.75)])
    


    s1_paradf_QL2 = s1_paradf_QL2.set_index("id")
    s2_paradf_QL2 = s2_paradf_QL2.set_index("id")

    # summarize

    for paraname in ['alpha', 'nu', 'tau', 'gamma', 'eta']:
        r, ci, resampled_r = analysisFxs.my_bootstrap_corr(s1_paradf_QL2.loc[:, paraname], s2_paradf_QL2.loc[:, paraname], n = 150)
        print(round(r,3))
        print((round(ci[0], 3), round(ci[1], 3)))
        var_ = np.concatenate([var_, [paraname]  * 1000])
        model_ = np.concatenate([model_, ['QL2']  * 1000])
        resampled_r_  = np.concatenate([resampled_r_, resampled_r])



    # ... make sure it is mutally included, merge is better
    s1_paradf_QL1  = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, 'QL1', 'QL1')
    s2_paradf_QL1 = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, 'QL1', 'QL1')
    s1_paradf_QL1 = s1_paradf_QL1[np.isin(s1_paradf_QL1.id, s2_paradf_QL1.id)]
    s2_paradf_QL1 = s2_paradf_QL1[np.isin(s2_paradf_QL1.id, s1_paradf_QL1.id)]
    s1_paradf_QL1 = s1_paradf_QL1.set_index("id")
    s2_paradf_QL1 = s2_paradf_QL1.set_index("id")
    for paraname in ['alpha', 'tau', 'gamma', 'eta']:
        r, p, resampled_r = analysisFxs.my_bootstrap_corr(s1_paradf_QL1.loc[:, paraname], s2_paradf_QL1.loc[:, paraname], n = 150)
        var_ = np.concatenate([var_, [paraname] * 1000])
        model_ = np.concatenate([model_, ['QL1']  * 1000])
        resampled_r_  = np.concatenate([resampled_r_, resampled_r])

    plotdf = pd.DataFrame(dict({"var":var_ , "model": model_, "r":resampled_r_}))
    sns.violinplot(data = plotdf, x = "r", y = "var", hue = 'model', scale="count", inner="quartile", split = True, palette = sns.color_palette(["#2171b5", '#6baed6']))
    # ax.get_legend().remove()
    plt.gca().set_xlabel("Bootstrapped Reliability")
    plt.gcf().set_size_inches(7, 9)
    plt.savefig(os.path.join("..", "figures", "active", "all_reliability.pdf"))

    # if I want to compare correlations, using the same number of participants
    # I think I need to match participants
    common_ids = list(set(s1_paradf_QL1.id) & set(s1_paradf_QL2.id))
    r_QL2, r_QL1, p_val = analysisFxs.my_compare_correlations(s1_paradf_QL2.loc[pd.Series(common_ids), 'alpha'], s2_paradf_QL2.loc[pd.Series(common_ids), 'alpha'], s1_paradf_QL1.loc[pd.Series(common_ids), 'alpha'], s2_paradf_QL1.loc[pd.Series(common_ids), 'alpha'], n_perm = 5000)


    # ################## reliability analysis of model-free measures, n = 144 always ###################
    # reliability of WTW
    fig, ax = plt.subplots()
    figFxs.WTW_reliability(s1_WTW_, s2_WTW_, hdrdata_sess1, hdrdata_sess2, ax)
    fig.savefig(os.path.join("..", "figures", expname, "WTW_reliability.pdf"))

    # reliability of AUC
    fig, ax = plt.subplots()
    figFxs.AUC_reliability(s1_stats, s2_stats)
    plt.gcf().set_size_inches(8, 6)
    plt.savefig(os.path.join("..", "figures", expname, "AUC_reliability.pdf"))

    # maybe later I want to write a function to automatically pivot this table
    colvars = ['auc', 'auc1', 'auc2']
    s1_HP = s1_stats.loc[s1_stats['condition'] == 'HP', colvars + ['id']]
    s1_LP = s1_stats.loc[s1_stats['condition'] == 'LP', colvars + ['id']]
    s1_aucdf = s1_HP.merge(s1_LP, left_on = 'id', right_on = 'id', suffixes = ['_HP', "_LP"])
    s1_aucdf = pd.concat([s1_aucdf, s1_aucdf.filter(like='HP', axis=1).set_axis([x+'_delta' for x in colvars], axis = 1) - s1_aucdf.filter(like='LP', axis=1).set_axis([x+'_delta' for x in colvars], axis = 1)], axis = 1)

    s2_HP = s2_stats.loc[s2_stats['condition'] == 'HP', colvars + ['id']]
    s2_LP = s2_stats.loc[s2_stats['condition'] == 'LP', colvars + ['id']]
    s2_aucdf = s2_HP.merge(s2_LP, left_on = 'id', right_on = 'id', suffixes = ['_HP', "_LP"])
    s2_aucdf = pd.concat([s2_aucdf, s2_aucdf.filter(like='HP', axis=1).set_axis([x+'_delta' for x in colvars], axis = 1) - s2_aucdf.filter(like='LP', axis=1).set_axis([x+'_delta' for x in colvars], axis = 1)], axis = 1)

    # plot auc reliability 
    df = s1_aucdf.merge(s2_aucdf, on = 'id', suffixes = ['_sess1', '_sess2']) 
    fig, ax = plt.subplots()
    # figFxs.my_regplot(df['auc_LP_sess1'], df['auc_LP_sess2'], axs[0], color = condition_palette[0])
    # figFxs.my_regplot(df['auc1_LP_sess1'], df['auc1_LP_sess2'], axs[0], color = condition_palette[0], line_kws={"linestyle": ":"})
    # figFxs.my_regplot(df['auc_HP_sess1'], df['auc_HP_sess2'], axs[1], color = condition_palette[1])
    figFxs.my_regplot(df['auc_delta_sess1'], df['auc_delta_sess2'], ax, color = 'grey')
    plt.savefig(os.path.join("..", "figures",expname, "delta_auc_reliability.pdf"))

    # plot reliability as a function of time
    # in this wide format, all condition information is stored within the column name
    conditions = ['LP', 'HP']
    measures = ['auc'] # if it is only one variable maybe it is easy enough to use summary functions...
    r_vals = []
    ci_vals = []
    # r_se_vals = []
    condition_vals = []
    measure_vals = []
    n_vals = []
    for condition, measure in itertools.product(conditions, measures):
        x = df[measure + '_' + condition + '_sess1']
        y = df[measure + '_' + condition + '_sess2']
        r, ci, _ = analysisFxs.my_bootstrap_corr(x, y)
        print(round(r, 3))
        print(round(ci[0], 3), round(ci[1], 3))
        r_vals.append(r)
        ci_vals.append(ci)
        # r_se_vals.append()
        n_vals.append(x.shape[0])
        condition_vals.append(condition)
        measure_vals.append(measure)



# plotdf = plotdf.loc[np.isin(plotdf.measure, ['auc1', 'auc2']),:]
def my_errorbar(x, y, yerr, **kwargs):
    ax = plt.gca()
    data = kwargs.pop("data")
    data.plot(x=x, y=y, yerr=yerr, kind="bar", ax=ax, **kwargs)
plotdf = pd.DataFrame({"Spearman's r": r_vals, "condition": condition_vals, "measure": measure_vals, "ci": ci_vals})
plotdf['measure'] = plotdf['measure'].replace(['auc1', 'auc2'], ['1', '2'])
# g = sns.FacetGrid(data = plotdf, col = "condition", hue = "condition", palette = condition_palette)
# g.map_dataframe(my_errorbar, "measure", "Spearman's r", "se")
# plt.gcf().set_size_inches(10, 5)
# plt.savefig(os.path.join("..", "figures",expname, "auc_reliability_compare.pdf"))


two_tail_p_vals = {}
for condition in ['HP', 'LP']:
    sess1_se = plotdf.loc[np.logical_and(plotdf.condition == condition, plotdf.measure == '1'),'se'].values[0]
    sess2_se = plotdf.loc[np.logical_and(plotdf.condition == condition, plotdf.measure == '2'),'se'].values[0]
    se_comb = np.sqrt((sess1_se **2 + sess2_se**2)/ 2)
    r_diff = plotdf.loc[np.logical_and(plotdf.condition == condition, plotdf.measure == '1'),"Spearman's r"].values[0] - plotdf.loc[np.logical_and(plotdf.condition == condition, plotdf.measure == '2'),"Spearman's r"].values[0]
    zscore = r_diff / se_comb
    p_two_tail = 2 * (1 - stats.norm.cdf(abs(zscore)))
    two_tail_p_vals[condition] = p_two_tail


    # calculate z score; I think eventually I want to use some bootstraps or permutations...


    ################## modeling analysis #############
    modelnames = ['QL1', 'QL2', 'RL1', 'RL2']
    s1_stats = pd.read_csv(os.path.join("..", "analysis_results", expname, "taskstats", "emp_sess1.csv"))
    s2_stats = pd.read_csv(os.path.join("..", "analysis_results", expname, "taskstats", "emp_sess2.csv"))

    ################ load model parameters and waic estimates ##########
# for 
modelnames = ['QL1', 'QL2', 'RL1', 'RL2']
s1_paradf_ = dict()
s2_paradf_ = dict()
for modelname in modelnames:
    s1_paradf  = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, modelname)
    s2_paradf  = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname, modelname)
    s1_paradf_[modelname] = s1_paradf 
    s2_paradf_[modelname] = s2_paradf

# 
s1_waic_df = s1_paradf_['QL1'][['id', 'waic']].rename(columns = {'waic':'waic_QL1'})
s2_waic_df = s2_paradf_['QL1'][['id', 'waic']].rename(columns = {'waic':'waic_QL1'})
for modelname in ['QL2']:
    s1_waic_df = s1_waic_df.merge(s1_paradf_[modelname][['id', 'waic']], on = 'id').rename(columns = {'waic':'waic_' + modelname})
    s2_waic_df = s2_waic_df.merge(s2_paradf_[modelname][['id', 'waic']], on = 'id').rename(columns = {'waic':'waic_' + modelname})
    
    s1_waic_df.to_csv(os.path.join('..', 'analysis_results', expname, 'modelfit', 'waic_sess1.csv'), index = None)
    s2_waic_df.to_csv(os.path.join('..', 'analysis_results', expname, 'modelfit', 'waic_sess2.csv'), index = None)
    # code.interact(local = dict(locals(), **globals()))
    s1_waic_df.melt(id_vars = 'id', value_vars = ['waic_' + x for x in modelnames]).groupby('variable').agg({'value':['mean', lambda x: np.std(x) / np.sqrt(len(x))]}).round(2)
    s2_waic_df.melt(id_vars = 'id', value_vars = ['waic_' + x for x in modelnames]).groupby('variable').agg({'value':['mean', lambda x: np.std(x) / np.sqrt(len(x))]}).round(2)

    # calc n_best_fit
s1_waic_df = s1_waic_df.iloc[:, 1:3]
s2_waic_df = s2_waic_df.iloc[:, 1:3]
s1_lowest_waic = s1_waic_df.apply(min, axis = 1)
s2_lowest_waic = s2_waic_df.apply(min, axis = 1)
s1_n_bestfit = dict()
s2_n_bestfit = dict()
for i, modelname in enumerate(['QL1', 'QL2']):
    s1_n_bestfit[modelname] = np.sum(s1_lowest_waic  == s1_waic_df['waic_' + modelname])
    s2_n_bestfit[modelname] = np.sum(s2_lowest_waic  == s2_waic_df['waic_' + modelname])
   
    ################ modeling replication ###########
    for modelname in modelnames:
        # s1_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, modelname)
        # s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname, modelname)
        # s1_stats_rep = modelFxs.group_model_rep(trialdata_sess1_, s1_paradf, modelname, isTrct = True, plot_each = False)
        # s2_stats_rep = modelFxs.group_model_rep(trialdata_sess2_, s2_paradf, modelname, isTrct = True, plot_each = False)
        s1_stats_rep.to_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_sess1.csv'%modelname), index = None)
        s2_stats_rep.to_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_sess2.csv'%modelname), index = None)

        s1_stats_rep = pd.read_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_sess1.csv'%modelname))
        s2_stats_rep = pd.read_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_sess2.csv'%modelname))
        # plot replication 
        figFxs.plot_group_emp_rep(modelname, s1_stats_rep, s2_stats_rep, s1_stats, s2_stats)
        plt.gcf().set_size_inches(10, 6)
        plt.savefig(os.path.join("..", "figures", expname, "emp_rep_%s.pdf"%modelname))

    ################ model comparison ################3
    # the results look reasonable

    ################ parameter reliability ##############
    # loop over models 
    for modelname in modelnames:
        sns.set(font_scale = 1)
        sns.set_style('white')
        # code.interact(local = dict(locals(), **globals()))
        paranames = modelFxs.getModelParas(modelname)
        npara = len(paranames)
        # plot parameter distributions
        figFxs.plot_parameter_distribution(modelname, s1_paradf_[modelname].iloc[:,:-1], s2_paradf_[modelname].iloc[:,:-1])
        plt.gcf().set_size_inches(5 * npara, 5 * 2)
        plt.savefig(os.path.join("..", 'figures', expname, "%s_para_dist.pdf"%modelname))

        # plot parameter correlations
        figFxs.plot_parameter_reliability(modelname, s1_paradf_[modelname].iloc[:,:-1], s2_paradf_[modelname].iloc[:,:-1])
        plt.gcf().set_size_inches(5 * npara, 5)
        plt.savefig(os.path.join("..", 'figures', expname, "%s_para_reliability.pdf"%modelname))
        
    ### for late fitting method
    s1_paradf  = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, 'QL2', 'QL2')
    s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, 'QL2', 'QL2')
    figFxs.plot_parameter_reliability('QL2', s1_paradf.iloc[:,:-1], s2_paradf.iloc[:,:-1])
    plt.gcf().set_size_inches(5 * 5, 5)
    plt.savefig(os.path.join("..", 'figures', expname, "QL2_para_reliability.pdf"))

    ################## correlations between self-report measures and ########
    s1_selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
    s1_selfdf.to_csv(os.path.join("..", "analysis_results", "active", "selfreport", "selfreport_sess1.csv"))
    s2_selfdf = loadFxs.parse_group_selfreport(expname, 2, isplot = False)
    s2_selfdf.to_csv(os.path.join("..", "analysis_results", "active", "selfreport", "selfreport_sess2.csv"))

    # add MCQ here, I want to improve that piece of code in the future...

    ######## correlations between AUC and self-report measures
    # prepare the aucdf
    s1_aucdf = s1_stats.pivot(index = 'id', columns = 'condition', values = ['auc'])
    s1_aucdf = s1_aucdf.rename_axis(columns = [None] * 2)
    s1_aucdf = s1_aucdf.droplevel(0, axis=1) 
    s1_aucdf = s1_aucdf.reset_index()
    s2_aucdf = s2_stats.pivot(index = 'id', columns = 'condition', values = ['auc'])
    s2_aucdf = s2_aucdf.rename_axis(columns = [None] * 2)
    s2_aucdf = s2_aucdf.droplevel(0, axis=1) 
    s2_aucdf = s2_aucdf.reset_index()

    aucdf = s1_aucdf.merge(s2_aucdf, how = 'left', on = "id", suffixes = ["_sess1", "_sess2"])

    # prepare for the corr analysis
    row_vars = ['HP_sess1', 'LP_sess1', 'HP_sess2', 'LP_sess2']
    col_vars = ['NU', 'PU', 'PM', 'PS', 'SS', 'attention', 'cogstable', 'motor', 'perseverance', 'selfcontrol', 'cogcomplex', 'UPPS', 'BIS', 'GMK'] # hmm I might want to add several more later
    r_table, p_table, perm_r_ = figFxs.corr_analysis(aucdf[row_vars], s1_selfdf.loc[np.isin(s1_selfdf.id, aucdf.id), col_vars], 1000)

    ##################### correlations between init_wtw and selfreport measures ###########
    s1_init_wtw_df = s1_stats.loc[s1_stats['block'] == 1, ['id', 'init_wtw']]
    s2_init_wtw_df = s2_stats.loc[s2_stats['block'] == 1, ['id', 'init_wtw']]
    init_wtw_df = s1_init_wtw_df.merge(s2_init_wtw_df, how = 'left', on = 'id', suffixes = ['_sess1', '_sess2'])

    # prepare for the corr analysis
    row_vars = ['init_wtw_sess1', 'init_wtw_sess2']
    col_vars = ['NU', 'PU', 'PM', 'PS', 'SS', 'attention', 'cogstable', 'motor', 'perseverance', 'selfcontrol', 'cogcomplex', 'UPPS', 'BIS', 'GMK'] # hmm I might want to add several more later
    r_table, p_table, perm_r_ = figFxs.corr_analysis(init_wtw_df[row_vars], s1_selfdf.loc[np.isin(s1_selfdf.id, init_wtw_df.id), col_vars], 10)

    # maybe disscociate batch 1 and batch 2 later?
    
    ##################### correlations between late parameters and selfreport measures #############
    row_vars = modelFxs.getModelParas('QL2')
    col_vars = ['NU', 'PU', 'PM', 'PS', 'SS', 'attention', 'cogstable', 'motor', 'perseverance', 'selfcontrol', 'cogcomplex']
    s1_paradf  = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, 'QL2', 'QL2')
    s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, 'QL2', 'QL2')
    r_table, p_table, perm_r_ = figFxs.corr_analysis(s1_paradf[row_vars], s1_selfdf.loc[np.isin(s1_selfdf.id, s1_paradf.id), col_vars], 10)
    p_table.to_csv(os.path.join("..", "analysis_results", "active", "correlation", "selfreport_para_sess1.csv"))

    r_table, p_table, perm_r_ = figFxs.corr_analysis(s2_paradf[row_vars], s1_selfdf.loc[np.isin(s1_selfdf.id, s2_paradf.id), col_vars], 10)
    p_table.to_csv(os.path.join("..", "analysis_results", "active", "correlation", "selfreport_para_sess2.csv"))
    ############

    
