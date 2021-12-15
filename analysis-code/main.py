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
from datetime import datetime as dt

# plot styles
plt.style.use('classic')
sns.set(font_scale = 1.5)
sns.set_style("white")


# Create an array with the colors you want to use
condition_colors = ["#762a83", "#1b7837"]
# Set your custom color palette
sns.set_palette(sns.color_palette(condition_colors))


# generate output results 
if not os.path.isdir(os.path.join("..", "analysis_results", "excluded")):
    os.makedirs(os.path.join("..", "analysis_results", "excluded"))

if not os.path.isdir(os.path.join('..', 'analysis_results', 'taskstats')):
    os.makedirs(os.path.join('..', 'analysis_results', 'taskstats'))

if not os.path.isdir(os.path.join("..", 'analysis_results')):
    os.makedirs(os.path.join("..", 'analysis_results'))

if not os.path.isdir(os.path.join("..", 'analysis_results', 'modelfit')):
    os.makedirs(os.path.join("..", 'analysis_results', 'modelfit'))

if not os.path.isdir(os.path.join("..", 'analysis_results', 'crossvalid')):
    os.makedirs(os.path.join("..", 'analysis_results', 'crossvalid'))

if not os.path.isdir(os.path.join("..", 'analysis_results', 'figures')):
    os.makedirs(os.path.join("..", 'analysis_results', 'figures'))

if not os.path.isdir(os.path.join("..", 'analysis_results', 'figures')):
    os.makedirs(os.path.join("..", 'analysis_results', 'selfreport'))

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

################ 
def plot_parameter_reliability(modelname, hdrdata_sess1, hdrdata_sess2):
    # get model parameters
    paranames = modelFxs.getModelParas(modelname)
    npara = len(paranames)

    # merge sess1 and sess2 data
    paradf_sess1 = loadFxs.load_parameter_estimates(1, hdrdata_sess1, modelname)
    paradf_sess2 = loadFxs.load_parameter_estimates(2, hdrdata_sess2, modelname)
    paradf = paradf_sess1.drop('sess', axis = 1).merge(paradf_sess2.drop('sess', axis = 1), left_on = 'id', right_on = 'id', suffixes = ['_sess1', '_sess2'])
    # plot reliability 
    fig, axes = plt.subplots(1, npara)
    for i, ax in enumerate(axes.flat):
        paraname = paranames[i]
        # filter = np.logical_and(paradf['alpha_sess1'] < 0.02, paradf['alpha_sess2'] < 0.02)
        if paraname == 'alpha':
            sess1_vals = np.log(paradf[paraname + '_sess1'].values)
            sess2_vals = np.log(paradf[paraname + '_sess2'].values)
        else:
            sess1_vals = paradf[paraname + '_sess1']
            sess2_vals = paradf[paraname + '_sess2']
        sns.regplot(sess1_vals, sess2_vals, ax = ax)
        # set x and y limits 
        corr_res = spearmanr(sess1_vals, sess2_vals, nan_policy = 'omit')
        ax.text(0.7, 0.2, 'r = %.3f\np =%.3f'%corr_res, size=15, color='red', transform=ax.transAxes)
        ax.set_xlabel("SESS1 Value")
        ax.set_ylabel("SESS2 Value")
        if paraname == 'alpha':
            ax.set_title('log(alpha)')
        else:
            ax.set_title(paraname)
    plt.gcf().set_size_inches(5 * npara, 6)
    fig.savefig(os.path.join("..", 'figures', 'para_reliability_%s.pdf'%modelname))
    return fig, axes

def plot_parameter_distribution(modelname, hdrdata_sess1, hdrdata_sess2, ax):
    # get model parameters
    paranames = modelFxs.getModelParas(modelname)
    npara = len(paranames)

    # merge sess1 and sess2 data
    paradf_sess1 = loadFxs.load_parameter_estimates(1, hdrdata_sess1, modelname)
    paradf_sess2 = loadFxs.load_parameter_estimates(2, hdrdata_sess2, modelname)
    tmp = pd.concat([paradf_sess1, paradf_sess2])
    # paradf = paradf_sess1.drop('sess', axis = 1).merge(paradf_sess2.drop('sess', axis = 1), left_on = 'id', right_on = 'id', suffixes = ['_sess1', '_sess2'])
    plotdf = tmp.melt(id_vars = ['id', 'sess'], value_vars = paranames)
    g = sns.FacetGrid(plotdf, col= "variable", row = 'sess', sharex = 'col', ax = ax)
    g.map(sns.histplot, "value", bins = 10) # this works, but low flexibility
    # plt.gcf().set_size_inches(10, 5)
    # plt.savefig(os.path.join("..", "figures", "para_dist_%s.eps"%modelname))

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
    code.interact(local = dict(locals(), **globals()))


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
    
    ########### load data #############
    hdrdata_sess1, trialdata_sess1_ = loadFxs.group_quality_check(1, plot_quality_check = False)
    hdrdata_sess2, trialdata_sess2_ = loadFxs.group_quality_check(2, plot_quality_check = False)

    ############### model free analysis ###########
    # s1_stats, s1_Psurv_b1_, s1_Psurv_b2_, s1_WTW_ = analysisFxs.group_MF(trialdata_sess1_, plot_each = False)

    # # plot for session 1
    # fig, ax = plt.subplots()
    # analysisFxs.plot_group_WTW(s1_WTW_, expParas.TaskTime, ax)
    # plt.savefig(os.path.join('..', 'figures', 'WTW_sess1.pdf'))
    
    # fig, ax = plt.subplots()
    # analysisFxs.plot_group_AUC(s1_stats, ax)
    # fig.savefig(os.path.join('..', 'figures', 'AUC_sess1.pdf'))

    # fig, ax = plt.subplots()
    # analysisFxs.plot_group_KMSC(s1_Psurv_b1_, s1_Psurv_b2_, expParas.Time, ax)
    # plt.savefig(os.path.join('..', 'figures', 'KMSC_sess1.pdf'))

    # s2_stats, s2_Psurv_b1_, s2_Psurv_b2_, s2_WTW_ = analysisFxs.group_MF(trialdata_sess2_, plot_each = False)

    # # plot for session 2
    # fig, ax = plt.subplots()
    # analysisFxs.plot_group_WTW(s2_WTW_, expParas.TaskTime, ax)
    # plt.savefig(os.path.join('..', 'figures', 'WTW_sess2.pdf'))
    
    # fig, ax = plt.subplots()
    # analysisFxs.plot_group_AUC(s2_stats, ax)
    # fig.savefig(os.path.join('..', 'figures', 'AUC_sess2.pdf'))

    # fig, ax = plt.subplots()
    # analysisFxs.plot_group_KMSC(s2_Psurv_b1_, s2_Psurv_b2_, expParas.Time, ax)
    # plt.savefig(os.path.join('..', 'figures', 'KMSC_sess2.pdf'))

    # # plot AUC values


    # # plot WTW of both sessions together
    # code.interact(local = dict(locals(), **globals()))
    # fig, ax = plt.subplots()
    # figFxs.plot_group_WTW_both(s1_WTW_, s2_WTW_, hdrdata_sess1, hdrdata_sess2, ax)
    # analysisFxs.plot_group_WTW(s1_WTW_[np.isin(hdrdata_sess1['id'], hdrdata_sess2['id'])], expParas.TaskTime, ax)
    # analysisFxs.plot_group_WTW(s2_WTW_, expParas.TaskTime, ax)
    # line1 = ax.get_lines()[0]
    # line1.set_color("black")
    # line2 = ax.get_lines()[1]
    # line2.set_color("#999999")
    # ax.legend([line1, line2], ['SESS1', 'SESS2'])
    # fig.savefig(os.path.join("..", "figures", "WTW_both.pdf"))

    # # reliability of WTW
    # fig, ax = plt.subplots()
    # WTW_reliability(sess1_WTW_, sess2_WTW_, ax)
    # fig.savefig(os.path.join("..", "figures", "WTW_reliability.pdf"))

    ######################## 
    modelnames = ['RL1', 'RL2']
    # emp_sess1 = pd.read_csv(os.path.join("..", "analysis_results", "taskstats", "emp_sess1.csv"))
    # emp_sess2 = pd.read_csv(os.path.join("..", "analysis_results", "taskstats", "emp_sess2.csv"))
    ################ modeling replication ###########
    # for modelname in modelnames:
    #     # code.interact(local = dict(locals(), **globals()))
    #     rep_sess1 = modelFxs.group_model_rep(trialdata_sess1_, modelname, isTrct = True, plot_each = False)
    #     rep_sess2 = modelFxs.group_model_rep(trialdata_sess2_, modelname, isTrct = True, plot_each = False)
    #     figFxs.plot_group_emp_rep(modelname, rep_sess1, rep_sess2, emp_sess1, emp_sess2)
    #     plt.savefig(os.path.join("..", "figures", "emp_rep_%s.eps"%modelname))
    
    ################ modeling parameters ############
    # loop over models 
    for modelname in modelnames:
        sns.set(font_scale = 1)
        sns.set_style('white')
        # # plot parameter distributions
        # g = figFxs.plot_parameter_distribution(modelname, hdrdata_sess1, hdrdata_sess2)
        # plt.gcf().set_size_inches(10, 5)
        # plt.savefig(os.path.join("..", 'figures', "%s_para_dist.pdf"%modelname))

        # # compare estimates in sess1 vs estimtes in sess2
        # g = figFxs.plot_parameter_distribution(modelname, hdrdata_sess1, hdrdata_sess2)
        # plt.savefig(os.path.join("..", "figures", "%s_para_compare.pdf"))


        # plot parameter reliability 
        # figFxs.plot_parameter_compare(modelname, hdrdata_sess1, hdrdata_sess2)
        g = figFxs.plot_parameter_reliability(modelname, hdrdata_sess1, hdrdata_sess2)
        plt.savefig(os.path.join("..", "figures", "para_reliability_%s.pdf"%modelname))




    # APT 
    # stats_, Psurv_block1_, Psurv_block2_, WTW_ = analysisFxs.group_MF(trialdata_sess2_, plot_each = False)
    # analysisFxs.plot_group_WTW(WTW_, expParas.TaskTime)
    # analysisFxs.plot_group_KMSC(Psurv_block1_, Psurv_block2_, expParas.Time)
    # code.interact(local = dict(locals(), **globals()))

    ############ model parameter reliability ################
    # plot_parameter_reliability('QL1', hdrdata_sess1, hdrdata_sess2)

    ############ model rep #############
    # # load emp_stats


    # for modelname in modelnames:
    #   # rep_sess1 = modelFxs.group_model_rep(trialdata_sess1_, modelname, isTrct = True, plot_each = False)
    #   # rep_sess2 = modelFxs.group_model_rep(trialdata_sess2_, modelname, isTrct = True, plot_each = False)
    #   rep_sess1 = pd.read_csv(os.path.join("..", "analysis_results", "taskstats", "rep_%s_sess1.csv"%modelname))
    #   rep_sess2 = pd.read_csv(os.path.join("..", "analysis_results", "taskstats", "rep_%s_sess2.csv"%modelname))
    #   plot_group_emp_rep(modelname, rep_sess1, rep_sess2, emp_sess1, emp_sess2)

    # plot_parameter_distribution(modelname, hdrdata_sess1, hdrdata_sess2)
    # fit is good for some participants
    # rep_stats_ = modelFxs.group_model_rep(trialdata_sess1_, modelname, isTrct = True, plot_each = False)
    # rep_stats_ = rep_stats_.reset_index()
    # I think it is better to read this one right?
    # emp_stats_, _, _, _ = analysisFxs.group_MF(trialdata_sess1_, plot_each = False)
    # # compare 
    # plotdf = rep_stats_[['auc', 'id', 'condition']].merge(emp_stats_[['auc', 'id', 'condition']], left_on = ('id', 'condition'), right_on = ('id', 'condition'), suffixes = ('_rep', '_emp'))
    # # until 57 is fine
    # fig, ax = plt.subplots()
    # ax.scatter(plotdf['auc_emp'], plotdf['auc_rep'])
    # rep_stats_ = pd.read_csv(os.path.join("..", "analysis_results", "taskstats", "rep_QL1_sess1.csv"))
    # emp_stats_ = 
    # fig, ax = plt.subplots()
    # fig, ax = plot_group_emp_rep(rep_stats_, emp_stats_)
# latter I want to calc correlations between 
# compare differences later
############

    
