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
from datetime import datetime as dt
import pickle

# plot styles
plt.style.use('classic')
sns.set(font_scale = 1)
sns.set_style("white")
condition_palette = ["#762a83", "#1b7837"]


expname = "passive"

# load data 
hdrdata_sess1, trialdata_sess1_ = loadFxs.group_quality_check(expname, 1, plot_quality_check = False)
hdrdata_sess2, trialdata_sess2_ = loadFxs.group_quality_check(expname, 2, plot_quality_check = False)

## only exclude valid participants 
hdrdata_sess1 = hdrdata_sess1[np.isin(hdrdata_sess1["id"], hdrdata_sess2["id"])]
trialdata_sess1_ = {x: y for x,y in trialdata_sess1_.items() if x[0] in hdrdata_sess2["id"].values}

###
s1_stats, s1_Psurv_b1_, s1_Psurv_b2_, s1_WTW_emp = analysisFxs.group_MF(trialdata_sess1_, plot_each = False)   
s2_stats, s2_Psurv_b1_, s2_Psurv_b2_, s2_WTW_emp = analysisFxs.group_MF(trialdata_sess2_, plot_each = False)   

for modelname in ['QL2reset']:
    # modelname = 'QL2reset_slope'
    fitMethod = "whole"
    stepsize = 0.5
    # subtitles = [r'$\mathbf{log(\alpha)}$', r'$\mathbf{log(\nu)}$', r'$\mathbf{\tau}$', r'$\mathbf{\gamma}$', r'$\mathbf{log(\eta)}$']
    paranames = modelFxs.getModelParas(modelname)
    npara = len(paranames)
    # replicate task data 
    s1_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, fitMethod, stepsize)
    s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname, fitMethod, stepsize)
    # if it is the second time
    s1_stats_rep = pd.read_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_sess1_%s_stepsize%.2f.csv'%(modelname, fitMethod, stepsize)))
    s2_stats_rep = pd.read_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_sess2_%s_stepsize%.2f.csv'%(modelname, fitMethod, stepsize)))
    dbfile = open(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_%s_stepsize%.2f'%(modelname, fitMethod, stepsize)), "rb")
    dbobj = pickle.load(dbfile)
    s1_WTW_rep = dbobj['s1_WTW_rep']
    s2_WTW_rep = dbobj['s2_WTW_rep']


    # plot WTW 
    sns.set(font_scale = 1.5)
    sns.set_style("white")
    g = figFxs.plot_group_emp_rep_wtw(s1_WTW_rep, s2_WTW_rep, s1_WTW_emp, s2_WTW_emp, hdrdata_sess1, hdrdata_sess2, s1_paradf, s2_paradf)
    plt.tight_layout()
    plt.gcf().set_size_inches(3 * 3, 4)
    g.savefig(os.path.join("..", "figures", expname, "emp_rep_%s_wtw.pdf"%modelname))

    # plot task measures, combining both sessions and both conditions 
    s1_df_emp, s2_df_emp = analysisFxs.pivot_by_condition(s1_stats), analysisFxs.pivot_by_condition(s2_stats)
    emp_df = analysisFxs.agg_across_sessions(s1_df_emp, s2_df_emp)
    #emp_df = pd.concat([s1_df_emp, s2_df_emp], axis = 0)
    s1_df_rep, s2_df_rep = analysisFxs.pivot_by_condition(s1_stats_rep), analysisFxs.pivot_by_condition(s2_stats_rep)
    rep_df = analysisFxs.agg_across_sessions(s1_df_rep, s2_df_rep)
    # rep_df = pd.concat([s1_df_rep, s2_df_rep], axis = 0)
    plotdf = emp_df.merge(rep_df, on = "id", suffixes = ["_emp", "_rep"])
    vars = ['auc', 'std_wtw', "auc_delta"]
    labels = ['AUC (s)', r'$\sigma_{wtw}$ (s)', r"$\Delta$ AUC (s)"]
    # let me get a report 
    _, _, _, _, _, report = analysisFxs.calc_zip_reliability(plotdf, [(x,y) for x, y in zip([x + "_emp" for x in vars], [x + "_rep" for x in vars])])
    report['rsquared'] = report['pearson_rho']**2
    report.round(3)
    # plot
    plt.style.use('classic')
    sns.set(font_scale = 1.5)
    sns.set_style("white")
    long_plotdf = pd.DataFrame({
        "Observed": plotdf[["auc_emp", "std_wtw_emp", "auc_delta_emp"]].values.flatten("F"),
        "Model-generated": plotdf[["auc_rep", "std_wtw_rep", "auc_delta_rep"]].values.flatten("F"),
        "var": np.repeat(labels, plotdf.shape[0])
        })

    g = sns.FacetGrid(long_plotdf, col = "var", sharex = False, sharey = False)
    g.map(figFxs.my_regplot, "Observed", "Model-generated", rsquared = True, equal_aspect = False)
    g.set_titles(col_template="{col_name}")
    limits = [(-0.5, 12.5), (0, 5), (-4, 8)]
    for i, ax in enumerate(g.axes.flatten()):
        limit = limits[i]
        ax.set_xlim(limit)
        ax.set_ylim(limit)
        ax.axline([limit[0], limit[0]], slope = 1, color = "#252525", linestyle = "dashed")
    plt.gcf().set_size_inches(3 * 3, 4)
    g.savefig(os.path.join("..", "figures", expname, "cb_emp_rep_stats_%s_%d.pdf"%(modelname, long_plotdf.shape[0] / 3)), bbox_inches = "tight")


