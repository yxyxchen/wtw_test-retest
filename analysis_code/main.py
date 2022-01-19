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
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from subFxs import simFxs 
from subFxs import normFxs
from subFxs import loadFxs
from subFxs import figFxs
import os
import importlib
from datetime import datetime as dt
import importlib

# plot styles
plt.style.use('classic')
sns.set(font_scale = 1.5)
sns.set_style("white")
condition_palette = ["#762a83", "#1b7837"]


# Create an array with the colors you want to use
condition_colors = ["#762a83", "#1b7837"]
# Set your custom color palette
sns.set_palette(sns.color_palette(condition_colors))


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
def selfreport_selfcorr():
    # load selfreport data
    self_sess1 = pd.read_csv(os.path.join(logdir, "selfreport_sess1.csv"))
    self_sess1['batch'] = [1 if np.isnan(x) else 2 for x in self_sess1.PAS]
    self_sess1['sess'] = 1
    MCQ = pd.read_csv(os.path.join(logdir, "MCQ.csv"))
    MCQ = MCQ.loc[np.logical_and.reduce([MCQ.SmlCons >= 0.8, MCQ.MedCons >= 0.8, MCQ.LrgCons > 0.8]),:] # filter 
    self_sess1 = self_sess1.merge(MCQ[['GMK', 'SubjID']], how = 'outer', right_on = 'SubjID', left_on = 'id').drop("SubjID", axis = 1)

    self_sess2 = pd.read_csv(os.path.join(logdir, "selfreport_sess2.csv"))
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
    
    # expname = 'passive'

    # hdrdata_sess1, trialdata_sess1_ = loadFxs.group_quality_check(expname, 1, plot_quality_check = True)

    # fig, ax = plt.subplots()
    # analysisFxs.plot_group_KMSC(s1_Psurv_b1_, s1_Psurv_b2_, expParas.Time, ax)
    # plt.savefig(os.path.join('..', 'figures', expname, 'sess1_KMSC.pdf'))

    # fig, ax = plt.subplots()
    # analysisFxs.plot_group_WTW(s1_WTW_, expParas.TaskTime, ax)
    # plt.savefig(os.path.join('..', 'figures', expname, 'sess1_WTW.pdf'))


    
    ########### load data #############
    expname = 'active'
    hdrdata_sess1, trialdata_sess1_ = loadFxs.group_quality_check(expname, 1, plot_quality_check = False)
    hdrdata_sess2, trialdata_sess2_ = loadFxs.group_quality_check(expname, 2, plot_quality_check = False)
    code.interact(local = dict(globals(), **locals()))
    # ############### model free analysis ###########
    s1_stats, s1_Psurv_b1_, s1_Psurv_b2_, s1_WTW_ = analysisFxs.group_MF(trialdata_sess1_, plot_each = False)
    s2_stats, s2_Psurv_b1_, s2_Psurv_b2_, s2_WTW_ = analysisFxs.group_MF(trialdata_sess2_, plot_each = False)

    s1_stats.to_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'emp_sess1.csv'), index = None)
    s2_stats.to_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'emp_sess2.csv'), index = None)
    
    # # plot for KMSC 
    # fig, ax = plt.subplots()
    # figFxs.plot_group_KMSC_both(s1_Psurv_b1_, s1_Psurv_b2_, s2_Psurv_b1_, s2_Psurv_b2_, hdrdata_sess1, hdrdata_sess2, ax)
    # plt.savefig(os.path.join('..', 'figures', expname, 'KMSC.pdf'))


    # # plot for WTW 
    # fig, ax = plt.subplots()
    # figFxs.plot_group_WTW_both(s1_WTW_, s2_WTW_, hdrdata_sess1, hdrdata_sess2, ax)
    # plt.savefig(os.path.join('..', 'figures', expname, 'WTW.pdf'))

    # ################## reliability analysis of model-free measures ###################
    # # reliability of WTW
    # fig, ax = plt.subplots()
    # figFxs.WTW_reliability(s1_WTW_, s2_WTW_, hdrdata_sess1, hdrdata_sess2, ax)
    # fig.savefig(os.path.join("..", "figures", expname, "WTW_reliability.pdf"))

    # # reliability of AUC
    # figFxs.AUC_reliability(s1_stats, s2_stats)
    # plt.gcf().set_size_inches(10, 5)
    # plt.savefig(os.path.join("..", "figures", expname, "AUC_reliability.pdf"))

    ################## modeling analysis #############
    modelnames = ['QL1', 'QL2', 'RL1', 'RL2']
    s1_stats = pd.read_csv(os.path.join("..", "analysis_results", expname, "taskstats", "emp_sess1.csv"))
    s2_stats = pd.read_csv(os.path.join("..", "analysis_results", expname, "taskstats", "emp_sess2.csv"))

    ################ load model parameters and waic estimates ##########
    s1_paradf_ = dict()
    s2_paradf_ = dict()
    for modelname in modelnames:
        s1_paradf  = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname)
        s2_paradf  = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname)
        s1_paradf_[modelname] = s1_paradf 
        s2_paradf_[modelname] = s2_paradf

    s1_waic_df = s1_paradf_['QL1'][['id', 'waic']].rename(columns = {'waic':'waic_QL1'})
    s2_waic_df = s2_paradf_['QL1'][['id', 'waic']].rename(columns = {'waic':'waic_QL1'})
    for modelname in ['QL2', 'RL1', 'RL2']:
        s1_waic_df = s1_waic_df.merge(s1_paradf_[modelname][['id', 'waic']], on = 'id').rename(columns = {'waic':'waic_' + modelname})
        s2_waic_df = s2_waic_df.merge(s2_paradf_[modelname][['id', 'waic']], on = 'id').rename(columns = {'waic':'waic_' + modelname})
    
    s1_waic_df.to_csv(os.path.join('..', 'analysis_results', expname, 'modelfit', 'waic_sess1.csv'), index = None)
    s2_waic_df.to_csv(os.path.join('..', 'analysis_results', expname, 'modelfit', 'waic_sess2.csv'), index = None)
    # code.interact(local = dict(locals(), **globals()))


    ################ modeling replication ###########
    # for modelname in modelnames:
    #     s1_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname)
    #     s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname)
    #     s1_stats_rep = modelFxs.group_model_rep(trialdata_sess1_, s1_paradf, modelname, isTrct = True, plot_each = False)
    #     s2_stats_rep = modelFxs.group_model_rep(trialdata_sess2_, s2_paradf, modelname, isTrct = True, plot_each = False)
    #     s1_stats_rep.to_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_sess1.csv'%modelname), index = None)
    #     s2_stats_rep.to_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_sess2.csv'%modelname), index = None)

    #     # plot replication 
    #     figFxs.plot_group_emp_rep(modelname, s1_stats_rep, s2_stats_rep, s1_stats, s2_stats)
    #     plt.savefig(os.path.join("..", "figures",expname, "emp_rep_%s.eps"%modelname))
    
    ################ model comparison ################3
    # the results look reasonable

    ################ parameter reliability ##############
    # loop over models 
    # for modelname in modelnames:
    #     sns.set(font_scale = 1)
    #     sns.set_style('white')
    #     # code.interact(local = dict(locals(), **globals()))
    #     paranames = modelFxs.getModelParas(modelname)
    #     npara = len(paranames)
    #     # plot parameter distributions
    #     figFxs.plot_parameter_distribution(modelname, s1_paradf_[modelname].iloc[:,:-1], s2_paradf_[modelname].iloc[:,:-1])
    #     plt.gcf().set_size_inches(5 * npara, 5 * 2)
    #     plt.savefig(os.path.join("..", 'figures', expname, "%s_para_dist.pdf"%modelname))

    #     # plot parameter correlations
    #     figFxs.plot_parameter_reliability(modelname, s1_paradf_[modelname].iloc[:,:-1], s2_paradf_[modelname].iloc[:,:-1])
    #     plt.gcf().set_size_inches(5 * npara, 5)
    #     plt.savefig(os.path.join("..", 'figures', expname, "%s_para_reliability.pdf"%modelname))
        
    ### for late fitting method
    s1_paradf  = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, 'QL2', 'QL2_late')
    s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, 'QL2', 'QL2_late')



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
    col_vars = ['NU', 'PU', 'PM', 'PS', 'SS', 'attention', 'cogstable', 'motor', 'perseverance', 'selfcontrol', 'cogcomplex'] # hmm I might want to add several more later
    r_table, p_table, perm_r_ = figFxs.corr_analysis(aucdf[row_vars], s1_selfdf.loc[np.isin(s1_selfdf.id, aucdf.id), col_vars], 1000)

    ##################### correlations between init_wtw and selfreport measures ###########
    s1_init_wtw_df = s1_stats.loc[s1_stats['block'] == 1, ['id', 'init_wtw']]
    s2_init_wtw_df = s2_stats.loc[s2_stats['block'] == 1, ['id', 'init_wtw']]
    init_wtw_df = s1_init_wtw_df.merge(s2_init_wtw_df, how = 'left', on = 'id', suffixes = ['_sess1', '_sess2'])

    # prepare for the corr analysis
    row_vars = ['init_wtw_sess1', 'init_wtw_sess2']
    col_vars = ['NU', 'PU', 'PM', 'PS', 'SS', 'attention', 'cogstable', 'motor', 'perseverance', 'selfcontrol', 'cogcomplex'] # hmm I might want to add several more later
    r_table, p_table, perm_r_ = figFxs.corr_analysis(init_wtw_df[row_vars], s1_selfdf.loc[np.isin(s1_selfdf.id, init_wtw_df.id), col_vars], 10)

    ##################### correlations between parameters and selfreport measures
    
############

    
