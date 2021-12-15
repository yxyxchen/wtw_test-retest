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
import os
from datetime import datetime as dt

# plot styles
plt.style.use('classic')
sns.set(font_scale = 1.5)
sns.set_style("white")



######## 

def plot_group_emp_rep(modelname, rep_sess1, rep_sess2, emp_sess1, emp_sess2):
    # code.interact(local = dict(locals(), **globals()))

    # plot AUC against AUC
    rep = pd.concat([rep_sess1[['auc', 'id', 'condition', 'sess']], rep_sess2[['auc', 'id', 'condition', 'sess']]])
    emp = pd.concat([emp_sess1[['auc', 'id', 'condition', 'sess']], emp_sess2[['auc', 'id', 'condition', 'sess']]])
    plotdf = rep.merge(emp, left_on = ('id', 'condition', 'sess'), right_on = ('id', 'condition', 'sess'), suffixes = ('_rep', '_emp'))
    g = sns.FacetGrid(plotdf, col= "sess", row = 'condition', sharex = True, sharey = True)
    g.set(ylim=(-0.5, expParas.tMax + 0.5), xlim = (-0.5, expParas.tMax + 0.5))
    g.map(sns.scatterplot, 'auc_emp', 'auc_rep', color = 'grey')
    for ax in g.axes.flat:
        ax.set_xlabel('Observed AUC (s)')
        ax.set_ylabel('Generated AUC (s)')
        ax.plot([0, expParas.tMax], [0, expParas.tMax], ls = '--', color = 'red')
    plt.gcf().set_size_inches(10, 10)
    plt.savefig(os.path.join("..", "figures", "emp_rep_%s.eps"%modelname))


def plot_group_WTW_both(sess1_WTW_, sess2_WTW_, hdrdata_sess1, hdrdata_sess2, ax):
    analysisFxs.plot_group_WTW(sess1_WTW_[np.isin(hdrdata_sess1['id'], hdrdata_sess2['id'])], expParas.TaskTime, ax)
    analysisFxs.plot_group_WTW(sess2_WTW_, expParas.TaskTime, ax)
    line1 = ax.get_lines()[0]
    line1.set_color("black")
    line2 = ax.get_lines()[1]
    line2.set_color("#999999")
    ax.legend([line1, line2], ['SESS1', 'SESS2'])

def log_transform_parameter(paradf, selected_paranames):
    """ log transform certain parameters in the paradf dataframe
    """
    # log transforms 
    # code.interact(local = dict(locals(), **globals()))
    for paraname in selected_paranames:
        if paraname in paradf:
            paradf[paraname] = np.log(paradf[paraname])
            # paradf.drop(paraname, axis = 1, inplace = True)
            paradf.rename(columns = {paraname : 'log_' + paraname}, inplace = True)

def WTW_reliability(sess1_WTW_, sess2_WTW_, hdrdata_sess1, hdrdata_sess2, ax):
    sess1_df = pd.DataFrame(sess1_WTW_)
    sess2_df = pd.DataFrame(sess2_WTW_)

    sess1_df['id'] = hdrdata_sess1['id']
    sess2_df['id'] = hdrdata_sess2['id']

    df = sess1_df.merge(sess2_df, on = "id", suffixes = ['_sess1', '_sess2'])

    WTW_rs = np.empty(len(expParas.TaskTime))
    WTW_ps = np.empty(len(expParas.TaskTime))
    for i in range(len(WTW_rs)):
        tmp = spearmanr(df[str(i) + '_sess1'], df[str(i) + '_sess2'])
        WTW_rs[i] = tmp[0]
        WTW_ps[i] = tmp[1]

    # fig, ax = plt.subplots()
    ax.plot(expParas.TaskTime, WTW_rs, color = 'black')
    ax.vlines(expParas.blocksec, 0, 1, color = "red", linestyles = "dotted")
    ax.set_ylim([0, 1])
    ax.set_xlabel('Task time (s)')
    ax.set_ylabel("Reliability of WTW")

def plot_parameter_compare(modelname, hdrdata_sess1, hdrdata_sess2):
    # load model parameters, and log transform some of them
    paradf_sess1 = loadFxs.load_parameter_estimates(1, hdrdata_sess1, modelname)
    log_transform_parameter(paradf_sess1, ['alpha', 'rho', 'eta'])
    paradf_sess2 = loadFxs.load_parameter_estimates(2, hdrdata_sess2, modelname)
    log_transform_parameter(paradf_sess2, ['alpha', 'rho', 'eta'])
    # melt and merge
    paradf_sess1 = pd.melt(paradf_sess1, id_vars = ('id', 'sess'), value_vars = paradf_sess1.columns.drop(['id', 'sess']))
    paradf_sess2 = pd.melt(paradf_sess2, id_vars = ('id', 'sess'), value_vars = paradf_sess2.columns.drop(['id', 'sess']))
    paradf = paradf_sess1.merge(paradf_sess2, left_on = ("id", "variable"), right_on = ("id", "variable"), suffixes = ['_sess1', '_sess2'])

    # plot
    def my_scatterplot(x, y, **kwargs):  
        ax = plt.gca()
        ax.scatter(x, y, **kwargs)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        lims = [min(x0, y0), max(x1, y1)]
        ax.plot(lims, lims, '-r')
        ax.set_ylim(lims)
        ax.set_xlim(lims)

    g = sns.FacetGrid(paradf, col= "variable", sharex = False, sharey = False)
    g.map(my_scatterplot, 'value_sess1', "value_sess2")
    return g

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


# I am refining code for these several analyses
def corr_analysis(row_df, col_df):
    """ calculate correlations for all combinations of variables in row_df and col_df
    """
    row_vars = row_df.columns
    col_vars = col_df.columns

    # 
    # initialize outputs
    r_table = np.zeros([len(row_vars), len(col_vars)])
    p_table = np.zeros([len(row_vars), len(col_vars)])

    # loop 
    for i, row_var in enumerate(row_vars):
        for j, col_var in enumerate(col_vars):
            res = spearmanr(row_df[row_var].values, col_df[col_var].values)
            r_table[i, j] = res[0]
            p_table[i, j] = res[1]
    return r_table, p_table

################ 
def plot_parameter_reliability(modelname, hdrdata_sess1, hdrdata_sess2):
    # load and log transform parameter data
    paradf_sess1 = loadFxs.load_parameter_estimates(1, hdrdata_sess1, modelname)
    log_transform_parameter(paradf_sess1, ['alpha', 'rho', 'eta'])
    paradf_sess2 = loadFxs.load_parameter_estimates(2, hdrdata_sess2, modelname)
    log_transform_parameter(paradf_sess2, ['alpha', 'rho', 'eta'])

    # reorganize and merge sess1 and sess2 data
    paradf_sess1 = pd.melt(paradf_sess1, id_vars = ('id', 'sess'), value_vars = paradf_sess1.columns.drop(['id', 'sess']))
    paradf_sess2 = pd.melt(paradf_sess2, id_vars = ('id', 'sess'), value_vars = paradf_sess2.columns.drop(['id', 'sess']))
    paradf = paradf_sess1.merge(paradf_sess2, left_on = ("id", "variable"), right_on = ("id", "variable"), suffixes = ['_sess1', '_sess2'])

    # plot
    def my_regplot(x, y, **kwargs):  
        ax = plt.gca()
        sns.regplot(x, y, **kwargs, ax = ax)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        lims = [min(x0, y0), max(x1, y1)]
        corr_res = spearmanr(x, y, nan_policy = 'omit')
        ax.text(0.7, 0.2, 'r = %.3f\np =%.3f'%corr_res, size=15, color='red', transform=ax.transAxes)
        ax.set_xlabel("SESS1 value")
        ax.set_ylabel("SESS2 value")
        ax.set_ylim(lims)
        ax.set_xlim(lims)  

    g = sns.FacetGrid(paradf, col= "variable", sharex = False, sharey = False)
    g.map(my_regplot, 'value_sess1', "value_sess2")
    # code.interact(local = dict(locals(), **globals()))
    return g

def plot_parameter_distribution(modelname, hdrdata_sess1, hdrdata_sess2):
    # get model parameters
    paranames = modelFxs.getModelParas(modelname)
    npara = len(paranames)

    # merge sess1 and sess2 data
    paradf_sess1 = loadFxs.load_parameter_estimates(1, hdrdata_sess1, modelname)
    paradf_sess2 = loadFxs.load_parameter_estimates(2, hdrdata_sess2, modelname)
    tmp = pd.concat([paradf_sess1, paradf_sess2])
    # paradf = paradf_sess1.drop('sess', axis = 1).merge(paradf_sess2.drop('sess', axis = 1), left_on = 'id', right_on = 'id', suffixes = ['_sess1', '_sess2'])
    plotdf = tmp.melt(id_vars = ['id', 'sess'], value_vars = paranames)
    g = sns.FacetGrid(plotdf, col= "variable", row = 'sess', sharex = 'col')
    g.map(sns.histplot, "value", bins = 10) # this works, but low flexibility
    return g
    # plt.gcf().set_size_inches(10, 5)
    # plt.savefig(os.path.join("..", "figures", "para_dist_%s.eps"%modelname))