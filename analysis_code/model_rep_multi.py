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

# plot styles
sns.set_theme(style="white", font_scale = 1)
condition_palette = ["#762a83", "#1b7837"]
expname = 'passive'

# load data 
hdrdata_sess1, trialdata_sess1_ = loadFxs.group_quality_check(expname, 1, plot_quality_check = False)
hdrdata_sess2, trialdata_sess2_ = loadFxs.group_quality_check(expname, 2, plot_quality_check = False)


## only exclude valid participants 
hdrdata_sess1 = hdrdata_sess1[np.isin(hdrdata_sess1["id"], hdrdata_sess2["id"])]
trialdata_sess1_ = {x: y for x,y in trialdata_sess1_.items() if x[0] in hdrdata_sess2["id"].values}


###
s1_stats, s1_Psurv_b1_, s1_Psurv_b2_, s1_WTW_emp = analysisFxs.group_MF(trialdata_sess1_, plot_each = False)   
s2_stats, s2_Psurv_b1_, s2_Psurv_b2_, s2_WTW_emp = analysisFxs.group_MF(trialdata_sess2_, plot_each = False)   


# modelnames = ['QL2reset', 'QL2reset_slope', 'QL2reset_slope_two', 'QL2reset_slope_two_simple']
modelnames = ['QL2', 'QL2reset']
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
    #s1_stats_rep, s1_WTW_rep, s1_dist_vals_ = modelFxs.group_model_rep(trialdata_sess1_, s1_paradf, modelname, 'whole', stepsize, isTrct = True, plot_each = False)
    #s2_stats_rep, s2_WTW_rep, s2_dist_vals_ = modelFxs.group_model_rep(trialdata_sess2_, s2_paradf, modelname, 'whole', stepsize, isTrct = True, plot_each = False)
    #s1_stats_rep.to_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_sess1_%s_stepsize%.2f.csv'%(modelname, fitMethod, stepsize)), index = None)
    #s2_stats_rep.to_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_sess2_%s_stepsize%.2f.csv'%(modelname, fitMethod, stepsize)), index = None)
    #s1_WTW_rep_.append(s1_WTW_rep)
    #s2_WTW_rep_.append(s2_WTW_rep)
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
plt.savefig(os.path.join("..", "figures", expname, "emp_rep_multi.pdf"))


######### compare WAIC ##############
for i in np.arange(len(modelnames)):
    s1_paradf = s1_paradf_[i]
    s2_paradf = s2_paradf_[i]
    s1_paradf = s1_paradf[np.isin(s1_paradf["id"], s1_ids)]
    s2_paradf = s2_paradf[np.isin(s2_paradf["id"], s2_ids)]
    s1_paradf_[i] = s1_paradf
    s2_paradf_[i] = s2_paradf
s1_paradf = pd.concat([x for x in s1_paradf_])
s2_paradf = pd.concat([x for x in s2_paradf_])
paradf = pd.concat([s1_paradf, s2_paradf])

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


######## compare variance explained
s1_df_emp, s2_df_emp = analysisFxs.pivot_by_condition(s1_stats), analysisFxs.pivot_by_condition(s2_stats)
emp_df = analysisFxs.agg_across_sessions(s1_df_emp, s2_df_emp)
vars = ['auc', 'std_wtw', "auc_delta"]
for s1_stats_rep, s2_stats_rep in zip(s1_stats_rep_, s2_stats_rep_):
    s1_df_rep, s2_df_rep = analysisFxs.pivot_by_condition(s1_stats_rep), analysisFxs.pivot_by_condition(s2_stats_rep)
    rep_df = analysisFxs.agg_across_sessions(s1_df_rep, s2_df_rep)
    plotdf = emp_df.merge(rep_df, on = "id", suffixes = ["_emp", "_rep"])
    _, _, _, _, _, report = analysisFxs.calc_zip_reliability(plotdf, [(x,y) for x, y in zip([x + "_emp" for x in vars], [x + "_rep" for x in vars])])
    report['rsquared'] = report['pearson_rho']**2
    report.round(3)


vars = ['auc', 'std_wtw', "auc_delta"]
labels = ['AUC (s)', r'$\sigma_{wtw}$ (s)', r"$\Delta$ AUC (s)"]
for s1_stats_rep, s2_stats_rep, modelname in zip(s1_stats_rep_, s2_stats_rep_, modelnames):
    s1_df_rep, s2_df_rep = analysisFxs.pivot_by_condition(s1_stats_rep), analysisFxs.pivot_by_condition(s2_stats_rep)
    rep_df = analysisFxs.agg_across_sessions(s1_df_rep, s2_df_rep)
    plotdf = emp_df.merge(rep_df, on = "id", suffixes = ["_emp", "_rep"])
    fig, axes = plt.subplots(1, 3)
    for (var, label), ax in zip(zip(vars, labels), axes.flatten()):
        sns.regplot(x = var + '_emp', y = var + '_rep', data = plotdf, scatter_kws={"color": "grey", "s": 40, "alpha":0.7, "edgecolor":'black'}, line_kws={"color": "black", "linestyle":"--"}, ax = ax)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Model-generated")
        ax.set_title(label)
    fig.tight_layout()
    fig.set_size_inches(14, 4)
    fig.savefig(os.path.join("..", "figures", expname, "cb_emp_rep_%s_%s.pdf"%(modelname, var)))




################ compare structural correlation ##########
r_ = []
pair_ = []
id_ = []
sess_ = []
model_ = []
for i, row in paradf.iterrows():
    parafile = os.path.join("../analysis_results", expname, "modelfit", fitMethod, "stepsize%.2f"%stepsize, row["model"], "%s_sess%d_sample.txt"%(row['id'], row["sess"]))
    parasamples = pd.read_csv(parafile, header = None)
    if row["model"] == "QL2reset_slope_two_simple":
        parasamples = parasamples.iloc[:, :4]
        paranames = ["alpha", "nu", "tau", "eta"]
    else:
        parasamples = parasamples.iloc[:, :5]
        paranames = modelFxs.getModelParas("QL2reset")
    # print(paranames)
    parasamples =  parasamples.rename(columns = dict(zip(parasamples.columns, paranames)))
    for x, y in itertools.combinations(paranames, 2):
        pair_.append((x, y))
        id_.append(row["id"])
        r_.append(spearmanr(parasamples[x], parasamples[y])[0])
        sess_.append(row["sess"])
        model_.append(row["model"])

paralabels = ['$\%s$'%x for x in modelFxs.getModelParas("QL2reset")]
structural_corr_df = pd.DataFrame({
    "r": r_,
    "pair": pair_,
    "id": id_,
    "sess": sess_,
    "model": model_,
    "var1": pd.Categorical(['$\%s$'%x[0] for x in pair_], categories = paralabels, ordered = True),
    "var2": pd.Categorical(['$\%s$'%x[1] for x in pair_], categories = paralabels, ordered = True)
    })

sns.set_theme(style="white", font_scale = 1)
fig, axes = plt.subplots(1, len(modelnames))
for i in np.arange(len(modelnames)):
    tmp = structural_corr_df[structural_corr_df["model"] == modelnames[i]].pivot_table(columns = "var2", index = "var1", values = "r", aggfunc='mean', margins_name = "None")
    plotdf = pd.DataFrame(np.nan, columns = paralabels, index = paralabels)
    plotdf.loc[tmp.index, tmp.columns] = tmp
    if i == len(modelnames) - 1:
        sns.heatmap(plotdf, annot=True, square=True, linewidths=1, ax = axes.flatten()[i], vmin=-1, vmax=1, center = 0, cmap = "RdBu_r")
    else:
        sns.heatmap(plotdf, annot=True, square=True, linewidths=1, ax = axes.flatten()[i], vmin=-1, vmax=1, center = 0, cmap = "RdBu_r", cbar = False)
    axes.flatten()[i].set_xlabel("")
    axes.flatten()[i].set_ylabel("")

fig.suptitle('Median correlation')
fig.set_size_inches(16, 6)
fig.savefig(os.path.join("..", "figures", expname, "structural_corr_multi.pdf"))


################ compare gross correlation ##########
r_ = []
var1_ = []
var2_ = []
model_ = []
for modelname in modelnames:
    this_paradf = paradf[paradf['model'] == modelname]
    paranames = modelFxs.getModelParas('QL2reset')
    if modelname == "QL2reset_slope_two" or modelname == "QL2reset_slope_two_simple":
        this_paradf['nu'] = this_paradf['alphaU']
    for x, y in itertools.combinations(paranames, 2):
        print(x)
        print(y)
        r_.append(spearmanr(this_paradf[x], this_paradf[y])[0])
        var1_.append(x)
        var2_.append(y)
        model_.append(modelname)


paralabels = ['$\%s$'%x for x in modelFxs.getModelParas("QL2reset")]
gross_corr_df = pd.DataFrame({
    "r": r_,
    "model": model_,
    "var1": pd.Categorical(['$\%s$'%x for x in var1_], categories = paralabels, ordered = True),
    "var2": pd.Categorical(['$\%s$'%x for x in var2_], categories = paralabels, ordered = True)
    })

sns.set_theme(style="white", font_scale = 1)
fig, axes = plt.subplots(1, len(modelnames))
for i in np.arange(len(modelnames)):
    tmp = gross_corr_df[gross_corr_df["model"] == modelnames[i]].pivot_table(columns = "var2", index = "var1", values = "r", aggfunc='mean', margins_name = "None")
    plotdf = pd.DataFrame(np.nan, columns = paralabels, index = paralabels)
    plotdf.loc[tmp.index, tmp.columns] = tmp
    if i == len(modelnames) - 1:
        sns.heatmap(plotdf, annot=True, square=True, linewidths=1, ax = axes.flatten()[i], vmin=-1, vmax=1, center = 0, cmap = "RdBu_r")
    else:
        sns.heatmap(plotdf, annot=True, square=True, linewidths=1, ax = axes.flatten()[i], vmin=-1, vmax=1, center = 0, cmap = "RdBu_r", cbar = False)
    axes.flatten()[i].set_xlabel("")
    axes.flatten()[i].set_ylabel("")


fig.set_size_inches(16, 6)
fig.savefig(os.path.join("..", "figures", expname, "gross_corr_multi.pdf"))


################ compare test-retest reliability ##########
r_ = []
var_ = []
model_ = []
for modelname in modelnames:
    this_paradf = paradf[paradf['model'] == modelname]
    paranames = modelFxs.getModelParas('QL2reset')
    if modelname == "QL2reset_slope_two" or modelname == "QL2reset_slope_two_simple":
        this_paradf['nu'] = this_paradf['alphaU']
    # calculate test-retest reliaility 
    s1_paradf = this_paradf[this_paradf['sess'] == 1]
    s2_paradf = this_paradf[this_paradf['sess'] == 2]
    s1_paradf_log  = figFxs.log_transform_parameter(s1_paradf, ['alpha', 'nu',  "tau", 'eta'])
    s2_paradf_log = figFxs.log_transform_parameter(s2_paradf, ['alpha', 'nu', "tau", 'eta'])
    s1_paradf_log = s1_paradf_log[np.isin(s1_paradf_log['id'], s2_paradf_log['id'])]
    s2_paradf_log = s2_paradf_log[np.isin(s2_paradf_log['id'], s1_paradf_log['id'])]
    if modelname == "QL2reset_slope_two_simple":
        subtitles = [r'$\mathbf{log(\alpha)}$', r'$\mathbf{log(\nu)}$', r'$\mathbf{\tau}$',  r'$\mathbf{log(\eta)}$']
        for para in ['log_alpha', 'log_nu', 'log_tau', 'log_eta']:
            r, _, _,  _, _, _, _, _, _, _ = analysisFxs.calc_reliability(s1_paradf_log[para].values, s2_paradf_log[para].values)
            r_.append(r)
            var_.append(para)
            model_.append(modelname)
    else:
        subtitles = [r'$\mathbf{log(\alpha)}$', r'$\mathbf{log(\nu)}$', r'$\mathbf{\tau}$', r'$\mathbf{\gamma}$', r'$\mathbf{log(\eta)}$']
        for para in ['log_alpha', 'log_nu', 'log_tau', 'gamma', 'log_eta']:
            r, _, _,  _, _, _, _, _, _, _ = analysisFxs.calc_reliability(s1_paradf_log[para].values, s2_paradf_log[para].values)
            r_.append(r)
            var_.append(para)
            model_.append(modelname)

plotdf = pd.DataFrame({
    "r": r_,
    "var": var_,
    "model": model_
    })
g = sns.FacetGrid(data = plotdf, col = "var")
g.map(sns.barplot, "model", "r")

g.savefig(os.path.join("..", "figures", expname, "reliability_multi.pdf"))

