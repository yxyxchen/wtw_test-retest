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
from scipy.interpolate import interp1d
from sksurv.nonparametric import kaplan_meier_estimator as km
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


def tosig(x, marginal = False):
    if x > 0.1:
        y = "ns"
    elif x > 0.05 and x <= 0.1:
        if marginal:
            y = "p = %.3f"%x
        else:
            y = "ns"
    elif x <= 0.05 and x > 0.01:
        y = "*"
    elif x <= 0.01 and x > 0.001:
        y = "**"
    else:
        y = "***"
    return y

######## 
def plot_group_emp_rep_wtw(s1_WTW_rep, s2_WTW_rep, s1_WTW_emp, s2_WTW_emp, hdrdata_sess1, hdrdata_sess2, s1_paradf, s2_paradf):
    # how do I truncate the ending part of data?
    s1_WTW_emp = s1_WTW_emp[np.isin(hdrdata_sess1.id, s1_paradf.id), ]
    s1_ave_emp = s1_WTW_emp.mean(axis = 0)
    s1_ave_rep = s1_WTW_rep.mean(axis = 0)
    s1_se_emp = np.std(s1_WTW_emp, axis = 0 ) / np.sqrt(s1_WTW_emp.shape[0])
    s1_se_rep = np.std(s1_WTW_rep, axis = 0 ) / np.sqrt(s1_WTW_rep.shape[0])

    # s1_ave_emp = np.median(s1_WTW_emp, axis = 0)
    # s1_ave_rep = np.median(s1_WTW_rep, axis = 0)

    s2_WTW_emp = s2_WTW_emp[np.isin(hdrdata_sess2.id, s2_paradf.id), ]
    s2_ave_emp = s2_WTW_emp.mean(axis = 0)
    s2_ave_rep = s2_WTW_rep.mean(axis = 0)
    s2_se_emp = np.std(s2_WTW_emp, axis = 0 ) / np.sqrt(s2_WTW_emp.shape[0])
    s2_se_rep = np.std(s2_WTW_rep, axis = 0 ) / np.sqrt(s2_WTW_rep.shape[0])
    # s2_ave_emp = np.median(s1_WTW_emp, axis = 0)
    # s2_ave_rep = np.median(s1_WTW_rep, axis = 0)

    plotdf = pd.DataFrame({
        "wtw": np.concatenate([s1_ave_emp, s1_ave_rep, s2_ave_emp, s2_ave_rep]),
        "se": np.concatenate([s1_se_emp, s1_se_rep, s2_se_emp, s2_se_rep]),
        "type": np.tile(np.repeat(["emp", "rep"], len(expParas.TaskTime)), 2),
        "time": np.tile((expParas.TaskTime / 60), 4),
        "sess": np.repeat(["SESS1", "SESS2"], len(expParas.TaskTime) * 2)
        })
    plotdf['ymin'] = plotdf['wtw'] - plotdf['se']
    plotdf['ymax'] = plotdf['wtw'] + plotdf['se']

    palette = ["black"] + [sns.color_palette("tab10")[1]]

    g = sns.FacetGrid(plotdf, col= "sess", hue = 'type', sharex = True, sharey = True, palette = palette)
    g.map(sns.lineplot, "time", "wtw")
    g.map(plt.fill_between, "time", "ymin", "ymax", facecolor='grey', edgecolor = "none",alpha = 0.4, interpolate=True, linewidth = 2)
    g.set(ylim=(3, 10), ylabel = "WTW (s)", xlabel = "Task time (min)")
    axs = g.axes_dict.values()
    for i, ax in enumerate(axs):
        ax.set_title("SESS%d"%(i+1))
        ax.axvline(expParas.blocksec/60, color = "grey", linestyle = "dashed")
    return g

def plot_group_emp_rep_wtw_multi(s1_WTW_rep_, s2_WTW_rep_, s1_WTW_emp, s2_WTW_emp, hdrdata_sess1, hdrdata_sess2, s1_paradf_, s2_paradf_, methods, estimation = "mean"):
    # 
    from functools import reduce
    s1_ids = reduce(np.intersect1d, [paradf.id for paradf in s1_paradf_])
    s2_ids = reduce(np.intersect1d, [paradf.id for paradf in s2_paradf_])

    # average empirical wtw data
    s1_WTW_emp = s1_WTW_emp[np.isin(hdrdata_sess1.id, s1_ids), :]
    s2_WTW_emp = s2_WTW_emp[np.isin(hdrdata_sess2.id, s2_ids), :]

    if estimation == "mean":
        s1_ave_emp = s1_WTW_emp.mean(axis = 0)
        s2_ave_emp = s2_WTW_emp.mean(axis = 0)
    else:
        s1_ave_emp = np.median(s1_WTW_emp, axis = 0)
        s2_ave_emp = np.median(s2_WTW_emp, axis = 0)

    s1_se_emp = np.std(s1_WTW_emp, axis = 0 ) / np.sqrt(s1_WTW_emp.shape[0])
    s2_se_emp = np.std(s2_WTW_emp, axis = 0 ) / np.sqrt(s2_WTW_emp.shape[0])

    # average replicated wtw data 
    nmethod = len(methods)
    s1_ave_rep_ = []
    s2_ave_rep_ = []
    s1_se_rep_ = []
    s2_se_rep_ = []
    for i in np.arange(nmethod):
        s1_WTW_rep = s1_WTW_rep_[i]
        s2_WTW_rep = s2_WTW_rep_[i]
        s1_paradf = s1_paradf_[i]
        s2_paradf = s2_paradf_[i]
        if estimation == "mean":
            s1_ave_rep = s1_WTW_rep[np.isin(s1_paradf.id, s1_ids)].mean(axis = 0)
            s2_ave_rep = s2_WTW_rep[np.isin(s2_paradf.id, s2_ids)].mean(axis = 0)
        else: 
            s1_ave_rep = np.median(s1_WTW_rep[np.isin(s1_paradf.id, s1_ids)], axis = 0)
            s2_ave_rep = np.median(s2_WTW_rep[np.isin(s2_paradf.id, s2_ids)], axis = 0)
        s1_se_rep = np.std(s1_WTW_rep, axis = 0 ) / np.sqrt(s1_WTW_rep.shape[0])
        s2_se_rep = np.std(s2_WTW_rep, axis = 0 ) / np.sqrt(s2_WTW_rep.shape[0])
        s1_ave_rep_.append(s1_ave_rep)
        s2_ave_rep_.append(s2_ave_rep)
        s1_se_rep_.append(s1_se_rep)
        s2_se_rep_.append(s2_se_rep)



    plotdf = pd.DataFrame({
        "wtw": np.concatenate([s1_ave_emp, s2_ave_emp, np.concatenate(s1_ave_rep_), np.concatenate(s2_ave_rep_)]),
        "se": np.concatenate([s1_se_emp, s2_se_emp, np.concatenate(s1_se_rep_), np.concatenate(s2_se_rep_)]),
        "type": ['Observed'] * len(expParas.TaskTime) * 2 + list(np.repeat(methods, len(expParas.TaskTime))) + list(np.repeat(methods, len(expParas.TaskTime))),
        "time": np.tile((expParas.TaskTime / 60).astype('float'), 2 + 2 * nmethod),
        "sess": np.concatenate([np.repeat(["SESS1", "SESS2"], len(expParas.TaskTime)), np.repeat(["SESS1", "SESS2"], len(expParas.TaskTime) * nmethod)])
        })
    plotdf['ymin'] = plotdf['wtw'] - plotdf['se']
    plotdf['ymax'] = plotdf['wtw'] + plotdf['se']

    palette = ["black"] + sns.color_palette("tab10")[:nmethod]

    g = sns.FacetGrid(plotdf, col= "sess", hue = 'type', sharex = True, sharey = True, palette  = sns.color_palette(palette, len(palette)))
    g.map(sns.lineplot, "time", "wtw")
    g.map(plt.fill_between, "time", "ymin", "ymax", facecolor='grey', edgecolor = "none",alpha = 0.4, interpolate=True, linewidth = 2)
    g.set(ylim=(3, 10), ylabel = "WTW (s)", xlabel = "Task time (min)")
    plt.legend(labels=["Observed"] + methods)
    axs = g.axes_dict.values()
    for i, ax in enumerate(axs):
        ax.set_title("SESS%d"%(i+1))
        ax.axvline(expParas.blocksec/60, color = "grey", linestyle = "dashed")
    return g

def plot_group_emp_rep(rep_sess1, rep_sess2, emp_sess1, emp_sess2, var = "auc", label = "AUC"):
    rep = pd.concat([rep_sess1[[var, 'id', 'condition', 'sess']], rep_sess2[[var, 'id', 'condition', 'sess']]])
    emp = pd.concat([emp_sess1[[var, 'id', 'condition', 'sess']], emp_sess2[[var, 'id', 'condition', 'sess']]])
    plotdf = rep.merge(emp, left_on = ('id', 'condition', 'sess'), right_on = ('id', 'condition', 'sess'), suffixes = ('_rep', '_emp'))
    g = sns.FacetGrid(plotdf, col= "sess", hue = 'condition', sharex = True, sharey = True, palette = condition_palette)
    if var == "auc":
        g.set(ylim=(-0.5, expParas.tMax + 0.5), xlim = (-0.5, expParas.tMax + 0.5))
    else:
        g.set(ylim=(-0.5, 6 + 0.5), xlim = (-0.5, 6 + 0.5))
    g.map(sns.scatterplot, var + '_emp', var + '_rep', s = 50, marker = "+", alpha = 0.8)
    for ax in g.axes.flat:
        ax.set_xlabel('Observed %s (s)'%label)
        ax.set_ylabel('Generated %s(s)'%label)
        ax.set_aspect("equal")
        if var == "auc":
            ax.plot([0, expParas.tMax], [0, expParas.tMax], ls = '--', color = 'grey', zorder = 10)
        else:
            ax.plot([0, 6], [0, 6], ls = '--', color = 'grey', zorder = 10)
    return g
def plot_group_emp_rep_diff(rep_sess1, rep_sess2, emp_sess1, emp_sess2):
    # plot AUC against AUC
    # code.interact(local = dict(locals(), **globals()))
    rep = pd.concat([rep_sess1[['auc', 'id', 'condition', 'sess']], rep_sess2[['auc', 'id', 'condition', 'sess']]])
    emp = pd.concat([emp_sess1[['auc', 'id', 'condition', 'sess']], emp_sess2[['auc', 'id', 'condition', 'sess']]])
    plotdf = rep.merge(emp, left_on = ('id', 'condition', 'sess'), right_on = ('id', 'condition', 'sess'), suffixes = ('_rep', '_emp'))
    plotdf['diff'] = plotdf['auc_emp'] - plotdf['auc_rep']
    g = sns.FacetGrid(plotdf, col= "sess", hue = 'condition', sharex = True, sharey = True, palette = condition_palette)
    g.map(sns.scatterplot, 'auc_emp', 'diff', s = 50, marker = "+", alpha = 0.8)
    for ax in g.axes.flat:
        ax.set_xlabel('Observed AUC (s)')
        ax.set_ylabel('Observed AUC - Generated AUC (s)')
        ax.axhline(0)


def plot_group_KMSC(Psurv_block1_, Psurv_block2_, Time, ax, **kwargs):
    """ Plot group-level survival curves 
    """
    # fig, ax = plt.subplots()
    df1 = pd.DataFrame({
            "mu": np.apply_along_axis(np.mean, 0, Psurv_block1_),
            "se": np.apply_along_axis(analysisFxs.calc_se, 0, Psurv_block1_),
            "Time": Time
        })
    df1 = df1.assign(ymin = lambda df: df.mu - df.se, ymax = lambda df: df.mu + df.se)
    df2 = pd.DataFrame({
            "mu": np.apply_along_axis(np.mean, 0, Psurv_block2_),
            "se": np.apply_along_axis(analysisFxs.calc_se, 0, Psurv_block2_),
            "Time": Time
        })
    df2 = df2.assign(ymin = lambda df: df.mu - df.se, ymax = lambda df: df.mu + df.se)

    df1.plot("Time", "mu", color = expParas.conditionColors['LP'], ax = ax, linewidth=3, **kwargs)
    ax.fill_between(df1.Time, df1.ymin, df1.ymax, facecolor= expParas.conditionColors['LP'], edgecolor = "none",alpha = 0.25, interpolate=True)
    df2.plot("Time", "mu", color = expParas.conditionColors['HP'], ax = ax, linewidth=3, **kwargs)
    ax.fill_between(df2.Time, df2.ymin, df2.ymax, facecolor= expParas.conditionColors['HP'], edgecolor = "none",alpha = 0.25, interpolate=True)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Survival rate")
    ax.set_ylim((0, 1))
    ax.set_xlim((0, expParas.tMax))
    ax.get_legend().remove()
    # plt.savefig(savepath)

def plot_group_KMSC_both(s1_Psurv_b1_, s1_Psurv_b2_, s2_Psurv_b1_, s2_Psurv_b2_, hdrdata_sess1, hdrdata_sess2, ax):
    plot_group_KMSC(s1_Psurv_b1_[np.isin(hdrdata_sess1['id'], hdrdata_sess2['id'])], s1_Psurv_b2_[np.isin(hdrdata_sess1['id'], hdrdata_sess2['id'])], expParas.Time, ax)
    plot_group_KMSC(s2_Psurv_b1_, s2_Psurv_b2_, expParas.Time, ax, linestyle = '--')

def plot_group_WTW(WTW_, TaskTime, ax, **kwargs):
    """Plot group-level WTW timecourse 
    """
    # fig, ax = plt.subplots()
    df = pd.DataFrame({
            "mu": np.apply_along_axis(np.mean, 0, WTW_),
            "se": np.apply_along_axis(analysisFxs.calc_se, 0, WTW_),
            "TaskTime": TaskTime
        })
    # code.interact(local = dict(globals(), **locals()))
    df = df.assign(ymin = df.mu - df.se, ymax = df.mu + df.se)
    df.plot("TaskTime", "mu", color = "black", ax = ax, label = '_nolegend_', **kwargs)
    ax.fill_between(df.TaskTime, df.ymin, df.ymax, facecolor='grey', edgecolor = "none",alpha = 0.4, interpolate=True, linewidth = 2)
    ax.set_xlabel("")
    ax.set_ylabel("WTW (s)")
    ax.set_xlabel("Task time (min)")
    ax.vlines(expParas.blocksec/60, 0, expParas.tMax, color = "red", linestyles = "dotted") # I might want to change it later
    ax.get_legend().remove()
    ax.set_ylim(3, 12)

def plot_group_WTW_both(sess1_WTW_, sess2_WTW_, hdrdata_sess1, hdrdata_sess2, ax):
    # code.interact(local = dict(locals(), **globals()))

    # filter
    sess1_WTW_ = sess1_WTW_[np.isin(hdrdata_sess1['id'], hdrdata_sess2['id'])]

    # calc p values
    # observed_t_, permutated_abs_t_max_, p_ = analysisFxs.my_paired_multiple_permuation(sess1_WTW_, sess2_WTW_, lambda x, y: ttest_rel(x, y)[0], n_perm = 100)
    # p_ = [stats.wilcoxon(sess1_WTW_[:,i], sess2_WTW_[:,i])[1] for i in range(sess1_WTW_.shape[1])]
    # import statsmodels.stats.multitest as multitest
    # _, p_corrected =  multitest.fdrcorrection(p_)
    # sig_ = [11 if p < 0.005 else None for p in p_corrected]

    # fig, ax = plt.subplots()
    # random selection
    # idxs = np.random.choice(np.arange(sess1_WTW_.shape[0]), 50)
    # people are learning faster 
    plot_group_WTW(sess1_WTW_, expParas.TaskTime / 60, ax)
    plot_group_WTW(sess2_WTW_, expParas.TaskTime / 60, ax, linestyle = ':')
    import matplotlib
    lp_rect = matplotlib.patches.Rectangle((0,0), 10, 12, color = condition_palette[0], alpha = 0.5)
    hp_rect = matplotlib.patches.Rectangle((10,0), 10, 12, color = condition_palette[1], alpha = 0.5)
    ax.add_patch(lp_rect)
    ax.add_patch(hp_rect)
    # ax.plot(expParas.TaskTime, sig_, marker = ".", color = "red")
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

# def AUC_reliability(sess1_stats, sess2_stats):
#     colnames = ['id', 'condition', 'auc']
#     df = sess1_stats.loc[:, colnames].merge(sess2_stats.loc[:, colnames], on = ["id", "condition"], suffixes = ['_sess1', '_sess2'])
#     g = sns.FacetGrid(df, col= "condition", hue = 'condition', sharex = True, sharey = True, palette = condition_palette)
#     # g.map(sns.scatterplot, 'auc_sess1', 'auc_sess2')
#     g.map(my_regplot, 'auc_sess1', "auc_sess2", exclude_outliers = False)
#     g.set(xlabel ="AUC SESS1 (s)", ylabel = "AUC SESS2 (s)")
#     g.set(ylim=(-0.5, expParas.tMax + 0.5), xlim = (-0.5, expParas.tMax + 0.5))
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
        print(paraname)
        if paraname in paradf:
            paradf[paraname] = np.log(paradf[paraname])
            # paradf.drop(paraname, axis = 1, inplace = True)
            paradf = paradf.rename(columns = {paraname : 'log_' + paraname})
    return paradf

def plot_parameter_reliability(modelname, paradf_sess1, paradf_sess2, subtitles, **kwargs):
    # log transform parameter data
    #paradf_sess1 = log_transform_parameter(paradf_sess1, ['alpha', 'nu', "tau", 'eta'])
    #paradf_sess2 = log_transform_parameter(paradf_sess2, ['alpha', 'nu', "tau", 'eta'])
    plt.style.use('classic')
    sns.set(font_scale=1.5)
    sns.set_style("white")
    paranames = modelFxs.getModelParas(modelname)
    labelnames = ["log_" + x if x in ['alpha', "tau", 'nu', 'eta'] else x for x in paranames]
    # reorganize and merge sess1 and sess2 data
    paradf_sess1 = pd.melt(paradf_sess1, id_vars = ('id', 'sess'), value_vars = labelnames)
    paradf_sess2 = pd.melt(paradf_sess2, id_vars = ('id', 'sess'), value_vars = labelnames)
    paradf = paradf_sess1.merge(paradf_sess2, left_on = ("id", "variable"), right_on = ("id", "variable"), suffixes = ['_sess1', '_sess2'])
    g = sns.FacetGrid(paradf, col = "variable", sharex = False, sharey = False, **kwargs)
    g.map(my_regplot, 'value_sess1', "value_sess2", equal_aspect = False)
    g.set(xlabel ="SESS1", ylabel = "SESS2")
    for i, ax in enumerate(g.axes_dict.values()):
        ax.set_title(subtitles[i], fontdict= { 'fontsize': 24, 'weight':'bold'})
    return g

def plot_parameter_practice(modelname, paradf_sess1, paradf_sess2, subtitles, **kwargs):
    log_transform_parameter(paradf_sess1, ['alpha', 'nu', 'eta'])
    log_transform_parameter(paradf_sess2, ['alpha', 'nu', 'eta'])
    plt.style.use('classic')
    sns.set(font_scale=1.5)
    sns.set_style("white")
    paranames = modelFxs.getModelParas(modelname)
    npara = len(paranames)

    # merge sess1 and sess2 data
    tmp = pd.concat([paradf_sess1, paradf_sess2])
    plotdf = tmp.melt(id_vars = ['id', 'sess'], value_vars = paradf_sess1.columns.values[:npara])
    sns.set(font_scale=1.5)
    sns.set_style("white")

    g = sns.FacetGrid(plotdf, col= "variable", sharey = False)
    g.map(sns.swarmplot, "sess", "value",  color = "grey", edgecolor = "black", alpha = 0.4, linewidth=1, size = 3) # this works, but low flexibility
    g.map(sns.boxplot, "sess", "value",  boxprops={'facecolor':'None'}, medianprops={"linestyle":"--", "color": "red"})
    for i, ax in enumerate(g.axes.flatten()):
        ax.set_title(subtitles[i], fontdict= { 'fontsize': 24, 'weight':'bold'})
    return g

def plot_parameter_distribution(modelname, paradf_sess1, paradf_sess2, **kwargs):
    # get model parameters
    # hmmm maybe i will think again 
    paranames = modelFxs.getModelParas(modelname)
    npara = len(paranames)

    # merge sess1 and sess2 data
    tmp = pd.concat([paradf_sess1, paradf_sess2])
    plotdf = tmp.melt(id_vars = ['id', 'sess'], value_vars = paranames)
    plotdf["sess"] = ["Session 1" if x == 1 else "Session 2" for x in plotdf["sess"]]
    plotdf["variable"] = [r'$\%s$'%x for x in plotdf["variable"]]
    plt.style.use('classic')
    sns.set(font_scale=1.5)
    sns.set_style("white")
    g = sns.FacetGrid(plotdf, col= "variable", row = 'sess', sharex = 'col', margin_titles = True)
    g.map(sns.histplot, "value", bins = 10, **kwargs) # this works, but low flexibility
    for (row_key, col_key), ax in g.axes_dict.items():
        # ax.set_title(r'SESS%d, $\%s$'%(row_key, col_key), fontdict= { 'fontsize': 24, 'weight':'bold'})
        median_val = np.median(plotdf.loc[np.logical_and(plotdf["sess"] == row_key, plotdf["variable"] == col_key), "value"])
        ax.axvline(median_val, color = "red")
        ax.text(median_val, 40, "%.2f"%median_val, color = "red")

    for i in np.arange(g.axes.shape[1]):
        ax1 = g.axes[0, i]
        ax2 = g.axes[1, i]
        if i == 0:
            ax1.set_xlim([0, 0.8])
            ax2.set_xlim([0, 0.8])
        elif i == 1:
            ax1.set_xlim([0, 9])
            ax2.set_xlim([0, 9])
        elif i == 2:
            ax1.set_xlim([0, 40])
            ax2.set_xlim([0, 40])
        elif i == 3:
            ax1.set_xlim([0.5, 1])
            ax2.set_xlim([0.5, 1])
        elif i == 4:
            ax1.set_xlim([0, 12])
            ax2.set_xlim([0, 12])
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    return g

def plot_parameter_density(modelname, paradf_sess1, paradf_sess2, **kwargs):
    paranames = modelFxs.getModelParas(modelname)
    npara = len(paranames)

    # merge sess1 and sess2 data
    tmp = pd.concat([paradf_sess1, paradf_sess2])
    plotdf = tmp.melt(id_vars = ['id', 'sess'], value_vars = paranames)
    plt.style.use('classic')
    sns.set(font_scale=1.5)
    sns.set_style("white")
    g = sns.FacetGrid(plotdf, col= "variable", sharex = False, sharey = False, hue = "sess", palette=sns.diverging_palette(220, 20, n=2))
    g.map(sns.kdeplot, "value") # this works, but low flexibility
    for i, ax in enumerate(g.axes.flatten()):
        ax.set_title(paranames[i], fontdict= { 'fontsize': 24, 'weight':'bold'})
        ax.axvline(np.median(paradf_sess1[paranames[i]]), color = "#d6604d", linestyle = "--")
        ax.axvline(np.median(paradf_sess2[paranames[i]]), color = "#4393c3", linestyle = "--")
    return g

def plot_parameter_compare(modelname, paradf_sess1, paradf_sess2, subtitles):
    # compare session-by-session differences
    log_transform_parameter(paradf_sess1, ['alpha', 'nu', 'eta'])
    log_transform_parameter(paradf_sess2, ['alpha', 'nu', 'eta'])
    paradf_sess1 = pd.melt(paradf_sess1, id_vars = ('id', 'sess'), value_vars = paradf_sess1.columns.drop(['id', 'sess']))
    paradf_sess2 = pd.melt(paradf_sess2, id_vars = ('id', 'sess'), value_vars = paradf_sess2.columns.drop(['id', 'sess']))
    plotdf = pd.concat([paradf_sess1.loc[np.isin(paradf_sess1.id, paradf_sess2.id)], paradf_sess2])
    # compare differences
    plt.style.use('classic')
    sns.set(font_scale=1.5)
    sns.set_style("white")
    g = sns.FacetGrid(plotdf, col= "variable", sharey = False)
    g.map(sns.boxplot, "sess", "value")
    print(g.axes_dict.values())
    for i, ax in enumerate(g.axes_dict.values()):
        ax.set_title(subtitles[i], fontdict= { 'fontsize': 24, 'weight':'bold'})
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

# yeah I also need to recalc this part 
def corr_analysis(row_df, col_df, n_perm):
    """ calculate correlations for all combinations of variables in row_df and col_df
    """
    row_vars = row_df.columns
    col_vars = col_df.columns

    # initialize outputs
    r_table = np.zeros([len(row_vars), len(col_vars)])
    p_table = np.zeros([len(row_vars), len(col_vars)])
    perm_r_ = np.zeros((len(row_vars), len(col_vars), n_perm)) # permutated r for each comb of row var and col var
    perm_p_table = np.zeros([len(row_vars), len(col_vars)])
    # loop 
    for i, row_var in enumerate(row_vars):
        for j, col_var in enumerate(col_vars):
            x = row_df[row_var].values
            y = col_df[col_var].values
            res = spearmanr(x, y, nan_policy = 'omit')
            r_table[i, j] = res[0]
            p_table[i, j] = res[1]
            for k in range(n_perm):
                perm_r_[i, j, k] = spearmanr(x, np.random.permutation(y), nan_policy = 'omit')[0]
            perm_p_table[i, j] = np.mean(np.abs(perm_r_[i, j, :]) > np.abs(r_table[i, j]))
    r_table = pd.DataFrame(r_table, index = row_vars, columns = col_vars)
    p_table = pd.DataFrame(p_table, index = row_vars, columns = col_vars)
    perm_p_table = pd.DataFrame(perm_p_table, index = row_vars, columns = col_vars)
    return r_table, p_table, perm_r_, perm_p_table

# def all_reliability(sess1_stats, sess2_stats):


# reliability functions

def annotate_reg(x, y, test = "spearman", ax = None, **kwargs):
    x, y = x[np.logical_and(~np.isnan(x), ~np.isnan(y))], y[np.logical_and(~np.isnan(x), ~np.isnan(y))]
    print(len(x))
    print(len(y))
    if(ax is None):
        ax = plt.gca()
    spearman_rho, pearson_rho, abs_icc, con_icc, ssbs, ssbm, sse, msbs, msbm, mse = analysisFxs.calc_reliability(x, y)
    print(test)
    if test == "spearman":
        r, p = spearmanr(x, y)
        ax.text(0.4, 0.1, "r = %.3f\n"%r, size=18, color='red', transform=ax.transAxes)
    elif test == "pearson":
        r, p = pearsonr(x, y)
        ax.text(0.4, 0.1, "pearson's r = %.3f"%r, size=18, color='red', transform=ax.transAxes)
    elif test == "icc":
        ax.text(0.4, 0.1, "icc = %.3f\n"%abs_icc, size=18, color='red', transform=ax.transAxes)


def my_regplot(x, y, ax = None, exclude_outliers = False, equal_aspect = True, rsquared = False, **kwargs):  
    if(ax is None):
        ax = plt.gca()
    # cacl realibility 
    spearman_rho, pearson_rho, abs_icc, con_icc, ssbs, ssbm, sse, msbs, msbm, mse = analysisFxs.calc_reliability(x, y)
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
    if exclude_outliers:
        sns.regplot(x[~is_outlier], y[~is_outlier], scatter_kws={"color": "#bdbdbd", "s": 40, "alpha":0.7, "edgecolor":'black'}, line_kws={"color": "#cb181d", "linestyle":"--"}, **kwargs, ax = ax)
        print('n_o = %d'%n_outlier)
        x_now_min = x[~is_outlier].min()
        y_now_min = y[~is_outlier].min()
        x_now_max = x[~is_outlier].max()
        y_now_max = y[~is_outlier].max()
        tmp = [min(x_now_min, y_now_min), max(x_now_max, y_now_max)]
        lims = [tmp[0] - (tmp[1] - tmp[0]) * 0.1, tmp[1] + (tmp[1] - tmp[0]) * 0.1]
        ax.set_ylim(lims)
        ax.set_xlim(lims)  
    else:
        sns.regplot(x = x, y = y, scatter_kws={"color": "#bdbdbd", "s": 40, "alpha":0.7, "edgecolor":'black'}, line_kws={"color": "#cb181d", "linestyle":"--"}, **kwargs, ax = ax)
    # 
    # choose equal limits for the x and y axes

    # ax.set_xlabel("SESS1 value")
    # ax.set_ylabel("SESS2 value")
    if equal_aspect:
        ax.set_aspect('equal')

    # add text
    # code.interact(local = dict(locals(), **globals()))
    # ax.text(0.4, 0.1, 'ICC = %.3f\n'%abs_icc, size=16, color='red', transform=ax.transAxes)
    if not rsquared:
        ax.text(0.3, 0.1, r'$\rho$ = %.3f'%spearman_rho, size=20, color='red', transform=ax.transAxes)
    else:
        r2 = pearson_rho ** 2
        ax.text(0.3, 0.1, r'$R^2$ = %.3f'%r2, size=16, color='red', transform=ax.transAxes)
    # print('ci = (%.3f, %.3f)'%ci)
    # ax.text(0.7, 0.1, 'n_o = %d'%n_outlier, size=15, color='red', transform=ax.transAxes)

