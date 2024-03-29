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
import scipy as sp 
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib as mpl

# plot styles
sns.set_theme(style="white", font_scale = 1)
condition_palette = ["#762a83", "#1b7837"]


report_df_ = []
for expname in ["passive", "active"]:
    # load data 
    hdrdata_sess1, trialdata_sess1_ = loadFxs.group_quality_check(expname, 1, plot_quality_check = False)
    hdrdata_sess2, trialdata_sess2_ = loadFxs.group_quality_check(expname, 2, plot_quality_check = False)
    ## only exclude valid participants 
    hdrdata_sess1 = hdrdata_sess1[np.isin(hdrdata_sess1["id"], hdrdata_sess2["id"])]
    trialdata_sess1_ = {x: y for x,y in trialdata_sess1_.items() if x[0] in hdrdata_sess2["id"].values}
    ###
    s1_stats, s1_Psurv_b1_, s1_Psurv_b2_, s1_WTW_emp = analysisFxs.group_MF(trialdata_sess1_, plot_each = False)   
    s2_stats, s2_Psurv_b1_, s2_Psurv_b2_, s2_WTW_emp = analysisFxs.group_MF(trialdata_sess2_, plot_each = False)   
    modelnames = ['QL2reset_slope', 'QL2reset_slope_two', 'QL2reset_slope_two_simple', 'QL2reset']
    fitMethod = "whole"
    stepsize = 0.5
    ### load model parameters ##
    s1_paradf_ = []
    s2_paradf_ = []
    for modelname in modelnames:
        s1_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, fitMethod, stepsize)
        s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname, fitMethod, stepsize)
        s1_paradf["model"] = modelname
        s2_paradf["model"] = modelname
        s1_paradf_.append(s1_paradf)
        s2_paradf_.append(s2_paradf)
    ##### common ids ###
    from functools import reduce
    s1_ids = reduce(np.intersect1d, [paradf.id for paradf in s1_paradf_])
    s2_ids = reduce(np.intersect1d, [paradf.id for paradf in s2_paradf_])
    ######## model fit ######### 
    s1_WTW_rep_ = []
    s2_WTW_rep_ = []
    s1_stats_rep_ = []
    s2_stats_rep_ = []
    for i, modelname in enumerate(modelnames):
        s1_paradf = s1_paradf_[i]
        s2_paradf = s2_paradf_[i]
        s1_stats_rep = pd.read_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_sess1_%s_stepsize%.2f.csv'%(modelname, fitMethod, stepsize)))
        s2_stats_rep = pd.read_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_sess2_%s_stepsize%.2f.csv'%(modelname, fitMethod, stepsize)))
        s1_stats_rep_.append(s1_stats_rep)
        s2_stats_rep_.append(s2_stats_rep)
        dbfile = open(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_%s_stepsize%.2f'%(modelname, fitMethod, stepsize)), "rb")
        dbobj = pickle.load(dbfile)
        s1_WTW_rep = dbobj['s1_WTW_rep']
        s2_WTW_rep = dbobj['s2_WTW_rep']
        s1_WTW_rep_.append(s1_WTW_rep)
        s2_WTW_rep_.append(s2_WTW_rep)
    figFxs.plot_group_emp_rep_wtw_multi(s1_WTW_rep_, s2_WTW_rep_, s1_WTW_emp, s2_WTW_emp, hdrdata_sess1, hdrdata_sess2, s1_paradf_, s2_paradf_, modelnames)
    # plt.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left')
    plt.legend("", frameon = False)
    plt.gcf().set_size_inches(12, 6)
    plt.tight_layout()
    plt.savefig(os.path.join("..", "figures", expname, "emp_rep_para.pdf"))
    ######## compare variance explained
    s1_df_emp, s2_df_emp = analysisFxs.pivot_by_condition(s1_stats), analysisFxs.pivot_by_condition(s2_stats)
    emp_df = analysisFxs.agg_across_sessions(s1_df_emp, s2_df_emp)
    vars = ['auc', 'std_wtw', "auc_delta"]
    var_ = []
    r2_ = []
    modelname_ = []
    for s1_stats_rep, s2_stats_rep, modelname in zip(s1_stats_rep_, s2_stats_rep_, modelnames):
        s1_df_rep, s2_df_rep = analysisFxs.pivot_by_condition(s1_stats_rep), analysisFxs.pivot_by_condition(s2_stats_rep)
        rep_df = analysisFxs.agg_across_sessions(s1_df_rep, s2_df_rep)
        plotdf = emp_df.merge(rep_df, on = "id", suffixes = ["_emp", "_rep"])
        _, _, _, _, _, report = analysisFxs.calc_zip_reliability(plotdf, [(x,y) for x, y in zip([x + "_emp" for x in vars], [x + "_rep" for x in vars])])
        report['rsquared'] = report['pearson_rho']**2
        report.round(3)
        var_ = var_ + vars
        r2_ = r2_ + report['rsquared'].values.tolist();
        modelname_ = modelname_ + [modelname] * len(vars)
    report_df_.append(pd.DataFrame({
        "var": var_,
        "r2": r2_,
        "modelname": modelname_,
        "exp": np.repeat(expname, len(var_))
        }))

reportdf = pd.concat(report_df_)
reportdf["modelname"] = np.tile(np.repeat(["M2", "M3", "M4", "M1"], 3), 2)
reportdf["var"] = pd.Categorical(np.tile(["AUC", r"$\sigma_{wtw}$", r"$\Delta$AUC"], 8), categories = ["AUC", r"$\sigma_{wtw}$", r"$\Delta$AUC"], ordered = True)
from plotnine import ggplot, aes, geom_bar, geom_errorbar, facet_grid, position_dodge, theme, theme_classic, ylim, labs
p = ggplot(reportdf) + facet_grid(facets="exp~var") + aes(x="modelname", y="r2") + geom_bar(stat = "identity") + theme_classic() + ylim([0, 1]) + labs( y=r"$R^{2}$", x = "")
p.save(os.path.join("..", "figures", "combined", "variance-explained_M1-4.pdf"))





