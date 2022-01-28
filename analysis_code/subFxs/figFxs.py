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
from scipy import stats
from scipy.stats import ttest_rel

# plot styles
plt.style.use('classic')
sns.set(font_scale = 2)
sns.set_style("white")
condition_palette = ["#762a83", "#1b7837"]


######## 
def plot_group_emp_rep(modelname, rep_sess1, rep_sess2, emp_sess1, emp_sess2):
    # code.interact(local = dict(locals(), **globals()))

    # plot AUC against AUC
    # code.interact(local = dict(locals(), **globals()))
    rep = pd.concat([rep_sess1[['auc', 'id', 'condition', 'sess']], rep_sess2[['auc', 'id', 'condition', 'sess']]])
    emp = pd.concat([emp_sess1[['auc', 'id', 'condition', 'sess']], emp_sess2[['auc', 'id', 'condition', 'sess']]])
    plotdf = rep.merge(emp, left_on = ('id', 'condition', 'sess'), right_on = ('id', 'condition', 'sess'), suffixes = ('_rep', '_emp'))
    g = sns.FacetGrid(plotdf, col= "sess", hue = 'condition', sharex = True, sharey = True, palette = condition_palette)
    g.set(ylim=(-0.5, expParas.tMax + 0.5), xlim = (-0.5, expParas.tMax + 0.5))
    g.map(sns.scatterplot, 'auc_emp', 'auc_rep', s = 50, marker = "+", alpha = 0.8)
    for ax in g.axes.flat:
        ax.set_xlabel('Observed AUC (s)')
        ax.set_ylabel('Generated AUC (s)')
        ax.set_aspect("equal")
        ax.plot([0, expParas.tMax], [0, expParas.tMax], ls = '--', color = 'grey', zorder = 10)

def plot_group_KMSC_both(s1_Psurv_b1_, s1_Psurv_b2_, s2_Psurv_b1_, s2_Psurv_b2_, hdrdata_sess1, hdrdata_sess2, ax):
    analysisFxs.plot_group_KMSC(s1_Psurv_b1_[np.isin(hdrdata_sess1['id'], hdrdata_sess2['id'])], s1_Psurv_b2_[np.isin(hdrdata_sess1['id'], hdrdata_sess2['id'])], expParas.Time, ax)
    analysisFxs.plot_group_KMSC(s2_Psurv_b1_, s2_Psurv_b2_, expParas.Time, ax, linestyle = '--')

def plot_group_WTW_both(sess1_WTW_, sess2_WTW_, hdrdata_sess1, hdrdata_sess2, ax):
    # code.interact(local = dict(locals(), **globals()))

    # filter
    sess1_WTW_ = sess1_WTW_[np.isin(hdrdata_sess1['id'], hdrdata_sess2['id'])]

    # calc p values
    # observed_t_, permutated_abs_t_max_, p_ = analysisFxs.my_paired_multiple_permuation(sess1_WTW_, sess2_WTW_, lambda x, y: ttest_rel(x, y)[0], n_perm = 100)
    p_ = [stats.wilcoxon(sess1_WTW_[:,i], sess2_WTW_[:,i])[1] for i in range(sess1_WTW_.shape[1])]
    import statsmodels.stats.multitest as multitest
    _, p_corrected =  multitest.fdrcorrection(p_)
    sig_ = [11 if p < 0.005 else None for p in p_corrected]

    # fig, ax = plt.subplots()
    analysisFxs.plot_group_WTW(sess1_WTW_, expParas.TaskTime, ax)
    analysisFxs.plot_group_WTW(sess2_WTW_, expParas.TaskTime, ax, linestyle = ':')
    ax.plot(expParas.TaskTime, sig_, marker = ".", color = "red")
    # line1 = ax.get_lines()[0]
    # line1.set_color("black")
    # line2 = ax.get_lines()[1]
    # line2.set_color("#999999")
    # ax.legend([line1, line2], ['SESS1', 'SESS2'])

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

def AUC_reliability(sess1_stats, sess2_stats):
    colnames = ['id', 'condition', 'auc']
    df = sess1_stats.loc[:, colnames].merge(sess2_stats.loc[:, colnames], on = ["id", "condition"], suffixes = ['_sess1', '_sess2'])
    g = sns.FacetGrid(df, col= "condition", hue = 'condition', sharex = True, sharey = True, palette = condition_palette)
    g.map(sns.scatterplot, 'auc_sess1', 'auc_sess2')
    g.map(my_regplot, 'auc_sess1', "auc_sess2")
    g.set(ylim=(-0.5, expParas.tMax + 0.5), xlim = (-0.5, expParas.tMax + 0.5))

# def plot_reliability(sess1_df, sess2_df, varname, ax, **kwargs):
#     df = sess1_df.merge(sess2_df, on = 'id', suffixes = ['_sess1', '_sess2']) 
#     my_regplot(df[varname + '_sess1'], df[varname + '_sess2'], **kwargs)


######################### parameter analysis ###############
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

def plot_parameter_reliability(modelname, paradf_sess1, paradf_sess2):
    # log transform parameter data
    log_transform_parameter(paradf_sess1, ['alpha', 'nu', 'eta'])
    log_transform_parameter(paradf_sess2, ['alpha', 'nu', 'eta'])

    # reorganize and merge sess1 and sess2 data
    paradf_sess1 = pd.melt(paradf_sess1, id_vars = ('id', 'sess'), value_vars = paradf_sess1.columns.drop(['id', 'sess']))
    paradf_sess2 = pd.melt(paradf_sess2, id_vars = ('id', 'sess'), value_vars = paradf_sess2.columns.drop(['id', 'sess']))
    paradf = paradf_sess1.merge(paradf_sess2, left_on = ("id", "variable"), right_on = ("id", "variable"), suffixes = ['_sess1', '_sess2'])

    g = sns.FacetGrid(paradf, col= "variable", sharex = False, sharey = False)
    g.map(my_regplot, 'value_sess1', "value_sess2")
    return g

def plot_parameter_distribution(modelname, paradf_sess1, paradf_sess2):
    # get model parameters
    paranames = modelFxs.getModelParas(modelname)
    npara = len(paranames)

    # merge sess1 and sess2 data
    tmp = pd.concat([paradf_sess1, paradf_sess2])
    plotdf = tmp.melt(id_vars = ['id', 'sess'], value_vars = paranames)
    g = sns.FacetGrid(plotdf, col= "variable", row = 'sess', sharex = 'col')
    g.map(sns.histplot, "value", bins = 10) # this works, but low flexibility
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
def corr_analysis(row_df, col_df, n_perm):
    """ calculate correlations for all combinations of variables in row_df and col_df
    """
    row_vars = row_df.columns
    col_vars = col_df.columns

    # initialize outputs
    r_table = np.zeros([len(row_vars), len(col_vars)])
    p_table = np.zeros([len(row_vars), len(col_vars)])
    perm_r_ = np.zeros((len(row_vars), len(col_vars), n_perm)) # permutated r for each comb of row var and col var

    # loop 
    for i, row_var in enumerate(row_vars):
        for j, col_var in enumerate(col_vars):
            x = row_df[row_var].values
            y = col_df[col_var].values
            res = spearmanr(x, y, nan_policy = 'omit')
            r_table[i, j] = res[0]
            p_table[i, j] = res[1]
            for k in range(n_perm):
                # fix x and shuffle y, using np.random.permutation which returns a copy of y
                perm_r_[i, j, k] = spearmanr(x, np.random.permutation(y), nan_policy = 'omit')[0]
    r_table = pd.DataFrame(r_table, index = row_vars, columns = col_vars)
    p_table = pd.DataFrame(p_table, index = row_vars, columns = col_vars)
    return r_table, p_table, perm_r_

# def all_reliability(sess1_stats, sess2_stats):



################ 
# reliability functions
def my_regplot(x, y, ax = None, **kwargs):  
    if(ax is None):
        ax = plt.gca()
    # code.interact(local = dict(locals(), **globals()))
    # calc correlation with all data points
    # r, ci, _ = analysisFxs.my_bootstrap_corr(x, y) 

    r = spearmanr(x, y, nan_policy = 'omit') 
    # set boundaries to exclude outliers: either based on the min/max value or the 1.5 iqr limit
    # x_min = x.min(); x_max = x.max()
    # y_min = y.min(); y_max = y.max()
    # code.interact(local = dict(locals(), **globals()))
    # calc boundaries and detect outliers
    x_iqr = x.quantile(0.75) - x.quantile(0.25)
    # x_upper = min(x.quantile(0.75) + 1.5 * x_iqr, x_max)
    # x_lower = max(x.quantile(0.25) - 1.5 * x_iqr, x_min)
    x_upper = x.quantile(0.75) + 1.5 * x_iqr
    x_lower = x.quantile(0.25) - 1.5 * x_iqr
    y_iqr = y.quantile(0.75) - y.quantile(0.25)
    # y_upper = min(y.quantile(0.75) + 1.5 * y_iqr, y_max)
    # y_lower = max(y.quantile(0.25) - 1.5 * y_iqr, y_min)
    y_upper = y.quantile(0.75) + 1.5 * y_iqr
    y_lower = y.quantile(0.25) - 1.5 * y_iqr
    is_outlier = np.logical_or.reduce([x > x_upper, x < x_lower, y < y_lower, y > y_upper]) 
    n_outlier = is_outlier.sum()
    # print(n_outlier)
    # scatter plot with included data
    # 
    sns.regplot(x[~is_outlier], y[~is_outlier], **kwargs, ax = ax)
    # 
    # choose equal limits for the x and y axes
    x_now_min = x[~is_outlier].min()
    y_now_min = y[~is_outlier].min()
    x_now_max = x[~is_outlier].max()
    y_now_max = y[~is_outlier].max()
    tmp = [min(x_now_min, y_now_min), max(x_now_max, y_now_max)]
    lims = [tmp[0] - (tmp[1] - tmp[0]) * 0.1, tmp[1] + (tmp[1] - tmp[0]) * 0.1]
    ax.set_xlabel("SESS1 value")
    ax.set_ylabel("SESS2 value")
    ax.set_ylim(lims)
    ax.set_xlim(lims)  
    ax.set_aspect('equal')

    # add text
    ax.text(0.7, 0.2, 'r = %.3f\n'%r, size=15, color='red', transform=ax.transAxes)
    print('n_o = %d'%n_outlier)
    # print('ci = (%.3f, %.3f)'%ci)
    # ax.text(0.7, 0.1, 'n_o = %d'%n_outlier, size=15, color='red', transform=ax.transAxes)

