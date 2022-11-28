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



modelnames = ['QL2reset', 'QL2reset_slope']
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
for i in np.arange(len(modelnames)):
    s1_paradf = s1_paradf_[i]
    s2_paradf = s2_paradf_[i]
    s1_paradf = s1_paradf[np.isin(s1_paradf["id"], s1_ids)]
    s2_paradf = s2_paradf[np.isin(s2_paradf["id"], s2_ids)]
    s1_paradf_[i] = s1_paradf
    s2_paradf_[i] = s2_paradf

######## model fit ######### 
s1_WTW_rep_ = []
s2_WTW_rep_ = []
for i, modelname in enumerate(modelnames):
    s1_paradf = s1_paradf_[i]
    s2_paradf = s2_paradf_[i]
    s1_stats_rep, s1_WTW_rep, s1_dist_vals_ = modelFxs.group_model_rep(trialdata_sess1_, s1_paradf, modelname, 'whole', stepsize, isTrct = True, plot_each = False)
    s2_stats_rep, s2_WTW_rep, s2_dist_vals_ = modelFxs.group_model_rep(trialdata_sess2_, s2_paradf, modelname, 'whole', stepsize, isTrct = True, plot_each = False)
    s1_stats_rep.to_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_sess1_%s_stepsize%.2f.csv'%(modelname, fitMethod, stepsize)), index = None)
    s2_stats_rep.to_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_sess2_%s_stepsize%.2f.csv'%(modelname, fitMethod, stepsize)), index = None)
    s1_WTW_rep_.append(s1_WTW_rep)
    s2_WTW_rep_.append(s2_WTW_rep)

figFxs.plot_group_emp_rep_wtw_multi(s1_WTW_rep_, s2_WTW_rep_, s1_WTW_emp, s2_WTW_emp, hdrdata_sess1, hdrdata_sess2, s1_paradf_, s2_paradf_, modelnames)
plt.gcf().set_size_inches(10, 6)
plt.savefig(os.path.join("..", "figures", expname, "emp_rep_multi.pdf"))


######### compare WAIC ##############
s1_paradf = pd.concat([x for x in s1_paradf_])
s2_paradf = pd.concat([x for x in s2_paradf_])
paradf = pd.concat([s1_paradf, s2_paradf])


fig, ax = plt.subplots()
sns.barplot(data = paradf, x = "model", y = "waic", ax = ax, palette = sns.color_palette("tab10"))
fig.savefig(os.path.join("..", "figures", expname, "waic_multi.pdf"))
waic_df.groupby("model").agg({"waic":[np.median, lambda x: np.std(x) / np.sqrt(len(x))]})


################ compare structural correlation ##########
r_ = []
pair_ = []
id_ = []
sess_ = []
model_ = []
for i, row in paradf.iterrows():
    parafile = os.path.join("../analysis_results", expname, "modelfit", fitMethod, "stepsize%.2f"%stepsize, row["model"], "%s_sess%d_sample.txt"%(row['id'], row["sess"]))
    parasamples = pd.read_csv(parafile, header = None)
    parasamples = parasamples.iloc[:, :5]
    paranames = modelFxs.getModelGroupParas(modelname)
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

sns.set_theme(style="white", font_scale = 1.5)
fig, axes = plt.subplots(1, len(modelnames))
for i in np.arange(len(modelnames)):
    tmp = structural_corr_df[structural_corr_df["model"] == modelnames[i]].pivot_table(columns = "var2", index = "var1", values = "r", aggfunc='mean', margins_name = "None")
    plotdf = pd.DataFrame(np.nan, columns = paralabels, index = paralabels)
    plotdf.loc[tmp.index, tmp.columns] = tmp
    sns.heatmap(plotdf, annot=True, square=True, linewidths=1, ax = axes[i], vmin=-1, vmax=1, center = 0, cmap = "RdBu_r")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
fig.suptitle('Median correlation')
fig.set_size_inches(12, 6)
fig.savefig(os.path.join("..", "figures", expname, "structural_corr_multi.pdf"))

