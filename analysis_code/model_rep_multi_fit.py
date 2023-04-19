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
    # I lack a passive subj and also the color is not right ..
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
p = ggplot(reportdf) + facet_grid(facets="~var") + aes(x="modelname", y="r2") + geom_bar(stat = "identity") + theme_classic() + ylim([0, 1]) + labs( y=r"$R^{2}$", x = "")
p.save(os.path.join("..", "figures", "combined", "variance-explained_M1-4.pdf"))
######### compare WAIC ##############
# for i in np.arange(len(modelnames)):
#     s1_paradf = s1_paradf_[i]
#     s2_paradf = s2_paradf_[i]
#     s1_paradf = s1_paradf[np.isin(s1_paradf["id"], s1_ids)]
#     s2_paradf = s2_paradf[np.isin(s2_paradf["id"], s2_ids)]
#     s1_paradf_[i] = s1_paradf
#     s2_paradf_[i] = s2_paradf
# s1_paradf = pd.concat([x for x in s1_paradf_])
# s2_paradf = pd.concat([x for x in s2_paradf_])
# paradf = pd.concat([s1_paradf, s2_paradf])

# fig, ax = plt.subplots()
# sns.barplot(data = paradf, x = "model", y = "waic", ax = ax, palette = sns.color_palette("tab10"))
# fig.savefig(os.path.join("..", "figures", expname, "waic_multi.pdf"))
# paradf.groupby(["sess", "model"]).agg({"waic":[np.median, lambda x: np.std(x) / np.sqrt(len(x))]})

# paradf.groupby(["sess", "model"]).agg({"neg_ippd":[np.median, lambda x: np.std(x) / np.sqrt(len(x))]})
# # let me calculate the best fit 
# waicdf = paradf.pivot_table(values = ['waic'], index = ['id', "sess"], columns = "model")
# waicdf.to_csv(os.path.join("..", "figures", expname, "waic_multi.csv"), index = None, header = None)

# a = waicdf.reset_index()
# for sess in [1, 2]:
#     a.loc[a[("sess", "")] == sess].iloc[:,2:].to_csv(os.path.join("..", "figures", expname, "waic_multi_sess%d.csv"%sess), index = None, header = None)
#     a.loc[a[("sess", "")] == sess].iloc[:,2:].apply(np.argmin, axis = 1).value_counts()
    
# # use neg_lppd
# neglppd_df = paradf.pivot_table(values = ['neg_ippd'], index = ['id', "sess"], columns = "model")
# a = neglppd_df.reset_index()
# for sess in [1, 2]:
#     a.loc[a[("sess", "")] == sess].iloc[:,2:].to_csv(os.path.join("..", "figures", expname, "waic_multi_sess%d.csv"%sess), index = None, header = None)
#     a.loc[a[("sess", "")] == sess].iloc[:,2:].apply(np.argmin, axis = 1).value_counts()

# # let me calculate the waic differences
# fig, axes = plt.subplots(1, 3)
# for i, ax in zip(np.arange(3), axes):
#     ax.hist(waicdf.iloc[:,i+1] - waicdf.iloc[:,i])
#     ax.set_title(modelnames[i+1] + ' - ' + modelnames[i])

# plt.hist(waicdf.iloc[:,1] - waicdf.iloc[:,0])
# sp.stats.wilcoxon(waicdf.iloc[:,1], waicdf.iloc[:,0])



######### compare WAIC ##############
fig, ax = plt.subplots()
sns.barplot(data = paradf, x = "model", y = "waic", ax = ax, palette = sns.color_palette("tab10"))
fig.savefig(os.path.join("..", "figures", expname, "waic_multi.pdf"))
paradf.groupby(["sess", "model"]).agg({"waic":[np.median, lambda x: np.std(x) / np.sqrt(len(x))]})

paradf.groupby(["sess", "model"]).agg({"neg_ippd":[np.median, lambda x: np.std(x) / np.sqrt(len(x))]})
# let me calculate the best fit 
waicdf = paradf.pivot_table(values = ['waic'], index = ['id', "sess"], columns = "model")
waicdf.to_csv(os.path.join("..", "figures", expname, "waic_multi.csv"), index = None, header = None)

a = waicdf.reset_index()
for sess in [1, 2]:
    a.loc[a[("sess", "")] == sess].iloc[:,2:].to_csv(os.path.join("..", "figures", expname, "waic_multi_sess%d.csv"%sess), index = None, header = None)
    a.loc[a[("sess", "")] == sess].iloc[:,2:].apply(np.argmin, axis = 1).value_counts()
    
# use neg_lppd
neglppd_df = paradf.pivot_table(values = ['neg_ippd'], index = ['id', "sess"], columns = "model")
a = neglppd_df.reset_index()
for sess in [1, 2]:
    a.loc[a[("sess", "")] == sess].iloc[:,2:].to_csv(os.path.join("..", "figures", expname, "waic_multi_sess%d.csv"%sess), index = None, header = None)
    a.loc[a[("sess", "")] == sess].iloc[:,2:].apply(np.argmin, axis = 1).value_counts()

# let me calculate the waic differences
fig, axes = plt.subplots(1, 3)
for i, ax in zip(np.arange(3), axes):
    ax.hist(waicdf.iloc[:,i+1] - waicdf.iloc[:,i])
    ax.set_title(modelnames[i+1] + ' - ' + modelnames[i])

plt.hist(waicdf.iloc[:,1] - waicdf.iloc[:,0])
sp.stats.wilcoxon(waicdf.iloc[:,1], waicdf.iloc[:,0])

# let me calculate the waic difference for two sessions 
delta_waic = waicdf.iloc[:,1] - waicdf.iloc[:,0]
delta_waic = delta_waic.reset_index()
delta_waic.columns = ['id', "sess", "waic"]
delta_waic['rank'] = delta_waic.groupby("sess")['waic'].rank()

g = sns.FacetGrid(delta_waic, col = "sess")
g.map(sns.barplot, "rank", "waic") # it dosen't work this way ....



# vars = ['auc', 'std_wtw', "auc_delta"]
# labels = ['AUC (s)', r'$\sigma_{wtw}$ (s)', r"$\Delta$ AUC (s)"]
# for s1_stats_rep, s2_stats_rep, modelname in zip(s1_stats_rep_, s2_stats_rep_, modelnames):
#     s1_df_rep, s2_df_rep = analysisFxs.pivot_by_condition(s1_stats_rep), analysisFxs.pivot_by_condition(s2_stats_rep)
#     rep_df = analysisFxs.agg_across_sessions(s1_df_rep, s2_df_rep)
#     plotdf = emp_df.merge(rep_df, on = "id", suffixes = ["_emp", "_rep"])
#     fig, axes = plt.subplots(1, 3)
#     for (var, label), ax in zip(zip(vars, labels), axes.flatten()):
#         sns.regplot(x = var + '_emp', y = var + '_rep', data = plotdf, scatter_kws={"color": "grey", "s": 40, "alpha":0.7, "edgecolor":'black'}, line_kws={"color": "black", "linestyle":"--"}, ax = ax)
#         ax.set_xlabel("Observed")
#         ax.set_ylabel("Model-generated")
#         ax.set_title(label)
#     fig.tight_layout()
#     fig.set_size_inches(14, 4)
#     fig.savefig(os.path.join("..", "figures", expname, "cb_emp_rep_%s_%s.pdf"%(modelname, var)))




