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
from matplotlib.collections import PatchCollection
from plotnine import ggplot, aes, geom_bar, geom_errorbar

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


modelnames = ['QL2reset', 'QL2reset_slope', 'QL2reset_slope_two', 'QL2reset_slope_two_simple']
modellabels = ["M1", "M2", "M3", "M4"]
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


##### load things ##
for i, modelname in enumerate(modelnames):
    s1_paradf = s1_paradf_[i]
    s2_paradf = s2_paradf_[i]
    s1_paradf = s1_paradf[np.isin(s1_paradf["id"], s1_ids)]
    s2_paradf = s2_paradf[np.isin(s2_paradf["id"], s2_ids)]
    if modelname == "QL2reset_slope_two_simple" or modelname == "QL2reset_slope_two":
        s1_paradf = s1_paradf.rename(columns = dict(zip(["alphaU"], ["kappa/nu"])))
        s2_paradf = s2_paradf.rename(columns = dict(zip(["alphaU"], ["kappa/nu"])))
    else:
        s1_paradf = s1_paradf.rename(columns = dict(zip(["nu"], ["kappa/nu"])))
        s2_paradf = s2_paradf.rename(columns = dict(zip(["nu"], ["kappa/nu"])))
    s1_paradf_[i] = copy.copy(s1_paradf)
    s2_paradf_[i] = copy.copy(s2_paradf)


s1_paradf = pd.concat([x for x in s1_paradf_])
s2_paradf = pd.concat([x for x in s2_paradf_])
paradf = pd.concat([s1_paradf, s2_paradf])
log_paradf = figFxs.log_transform_parameter(paradf, ["alpha", "kappa/nu", "tau", "eta"])


log_paranames = ["log_alpha", "log_kappa/nu", "log_tau", "gamma", "log_eta"]
log_paralabels = [r"$log(\alpha)$", r"$log(\nu)/log(\kappa)$", r"$log(\tau)$", r"$\gamma$", "$log(\eta)$"]
para_name_label_mapping = dict(zip(log_paranames, log_paralabels))
para_name_limits_mapping = dict(zip(log_paranames, [(-12, 2), (-8, 4), (-3, 5), (0.4, 1.1), (-4, 4)]))
para_name_ticks_mapping = dict(zip(log_paranames, [(-12, -6, 0), (-8, 0, 4), (-3, 0, 3), (0.5, 1), (-4, 0, 4)]))

# focused pairs 
focused_pairs = [("log_alpha", "log_kappa/nu"), ("log_tau", "log_eta")]


# color bar settings 
norm = Normalize(vmin=-0.90, vmax=0.90)
cmap = cm.get_cmap('RdBu_r')

################ compare gross correlation ##########
############ for all pairs ############
model_ = []
pair_ = []
mu_ = []
for i, modelname in enumerate(modelnames):
    plotdf = log_paradf[log_paradf["model"] == modelname]
    for j, pair in enumerate(itertools.combinations(log_paranames, 2)):
        rho = spearmanr(plotdf[pair[0]], plotdf[pair[1]])[0]
        # save the data 
        mu_.append(rho)
        model_.append(modelname)
        pair_.append(pair)

gross_corr_summary_df = pd.DataFrame({
    "model": model_,
    "pair": pair_,
    "mean": mu_,
    })

# plot the scatter plot for only the focused pairs 
fig, axes = plt.subplots(len(focused_pairs), len(modelnames))
for i, modelname in enumerate(modelnames):
    plotdf = log_paradf.loc[log_paradf["model"] == modelname]
    for j, pair in enumerate(focused_pairs):
        rho = spearmanr(plotdf[pair[0]], plotdf[pair[1]])[0]
        sns.regplot(plotdf[pair[0]], plotdf[pair[1]], ax = axes[j, i], scatter_kws = {"color":"grey", "edgecolor":"black", "alpha": 0.8},line_kws = {"color": cmap(norm(rho))})
        axes[j,i].text(0.2, 0.2, r"$\rho = $%.2f"%rho, fontsize = 15, color = "orange", transform = axes[j,i].transAxes)
        axes[j,i].set_title(modellabels[i])
        axes[j,i].set_xlabel(para_name_label_mapping[pair[0]])
        axes[j,i].set_ylabel(para_name_label_mapping[pair[1]])
        for spine in axes[j,i].spines.values():
            spine.set_edgecolor(cmap(norm(rho)))
            spine.set_linewidth(2.5)

plt.tight_layout(h_pad = 2, w_pad = 1)
fig.set_size_inches(w = 3 * 4, h = 2 * 3)
fig.savefig(os.path.join("..", "figures", expname, "gross_corr_M1-4.pdf"))


# plot the heatmap for all the pairs 
sns.set_theme(style="white", font_scale = 1)
fig, axes = plt.subplots(1, len(modelnames))
for i, modelname in enumerate(modelnames):
    plotdf = pd.DataFrame(np.nan, columns = log_paralabels, index = log_paralabels)
    tmp = gross_corr_summary_df[gross_corr_summary_df["model"] == modelname]
    tmp["var1"] = [para_name_label_mapping[x[0]] for x in tmp["pair"]]
    tmp["var2"] = [para_name_label_mapping[x[1]] for x in tmp["pair"]]
    tmp = tmp.pivot_table(values = "mean", index = "var2", columns = "var1", margins_name = "None")
    plotdf.loc[tmp.index, tmp.columns] = tmp
    sns.heatmap(plotdf, annot=True, square=True, linewidths=1, ax = axes.flatten()[i], norm = norm, cmap = cmap, cbar = False)
    axes.flatten()[i].set_xlabel("")
    axes.flatten()[i].set_ylabel("")

fig.set_size_inches(16, 6)
fig.savefig(os.path.join("..", "figures", expname, "gross_corr_all_M1-4.pdf"))


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
        paranames = ["log_alpha", "log_kappa/nu", "log_tau", "log_eta"]
    else:
        parasamples = parasamples.iloc[:, :5]
        paranames = paranames = ["log_alpha", "log_kappa/nu", "log_tau",  "gamma", "log_eta"]
    # print(paranames)
    parasamples =  parasamples.rename(columns = dict(zip(parasamples.columns, paranames)))
    for x, y in itertools.combinations(paranames, 2):
        pair_.append((x, y))
        id_.append(row["id"])
        r_.append(spearmanr(parasamples[x], parasamples[y])[0])
        sess_.append(row["sess"])
        model_.append(row["model"])

structural_corr_df = pd.DataFrame({
    "r": r_,
    "pair": pair_,
    "id": id_,
    "sess": sess_,
    "model": model_,
    "var1": pd.Categorical([para_name_label_mapping[x[0]] for x in pair_], categories = log_paralabels, ordered = True),
    "var2": pd.Categorical([para_name_label_mapping[x[1]] for x in pair_], categories = log_paralabels, ordered = True),
    })

# maybe get a summary df first ?
# plot only focused pairs
mu_ = []
model_ = []
pair_ = []
for i, modelname in enumerate(modelnames):
    tmp = structural_corr_df[structural_corr_df["model"] == modelname]
    for j, pair in enumerate(itertools.combinations(log_paranames, 2)):
        mean = np.mean(tmp.loc[tmp["pair"] == pair, "r"])
        mu_.append(mean)
        model_.append(modelname)
        pair_.append(pair)

structural_corr_summary_df = pd.DataFrame({
    "mean": mu_,
    "model": model_,
    "pair": pair_
    })

###### plot the  heatmap for all pairs 
sns.set_theme(style="white", font_scale = 1)
fig, axes = plt.subplots(1, len(modelnames))
for i in np.arange(len(modelnames)):
    tmp = structural_corr_df[structural_corr_df["model"] == modelnames[i]].pivot_table(columns = "var1", index = "var2", values = "r", aggfunc='mean', margins_name = "None")
    plotdf = pd.DataFrame(np.nan, columns = log_paralabels, index = log_paralabels)
    plotdf.loc[tmp.index, tmp.columns] = tmp
    if i == len(modelnames) - 1:
        sns.heatmap(plotdf, annot=True, square=True, linewidths=1, ax = axes.flatten()[i], vmin=-1, vmax=1, center = 0, cmap = "RdBu_r", cbar = False)
    else:
        sns.heatmap(plotdf, annot=True, square=True, linewidths=1, ax = axes.flatten()[i], vmin=-1, vmax=1, center = 0, cmap = "RdBu_r", cbar = False)
    axes.flatten()[i].set_xlabel("")
    axes.flatten()[i].set_ylabel("")

fig.suptitle('Median correlation')
fig.set_size_inches(16, 6)
fig.savefig(os.path.join("..", "figures", expname, "structural_corr_all_M1-4.pdf.pdf"))


# plot only focused pairs
fig, axes = plt.subplots(len(focused_pairs), len(modelnames))
for i, modelname in enumerate(modelnames):
    plotdf = structural_corr_df[structural_corr_df["model"] == modelname]
    for j, pair in enumerate(focused_pairs):
        median = np.median(plotdf.loc[plotdf["pair"] == pair, "r"])
        sns.histplot(plotdf.loc[plotdf["pair"] == pair, "r"], ax = axes[j, i], color = "grey")
        axes[j,i].set_xlim([-1.05, 1.05])
        axes[j,i].axvline(median, color = cmap(norm(median)), linewidth = 3)
        axes[j,i].axvline(0, color = "black", linestyle = "dotted") # "#238b45", 
        for spine in axes[j,i].spines.values():
            spine.set_edgecolor(cmap(norm(median)))
            spine.set_linewidth(2.5)

plt.tight_layout(h_pad = 2, w_pad = 1)
fig.set_size_inches(w = 3 * 4, h = 2 * 3)
fig.savefig(os.path.join("..", "figures", expname, "structure_corr_M1-4.pdf"))

################################ plot the circle version for focused pairs 


structural_corr_summary_df["type"] = "structure"
gross_corr_summary_df["type"] = "gross"
tmp = pd.concat([structural_corr_summary_df, gross_corr_summary_df], axis = 0)

for i in np.arange(len(focused_pairs)):
    plotdf = tmp[tmp["pair"] == focused_pairs[i]].pivot_table(values = "mean", index = "type", columns = "model")
    N = 2
    M = len(modelnames)
    ylabels = plotdf.index.values
    xlabels = modellabels
    x, y = np.meshgrid(np.arange(M), np.arange(N))
    s = np.abs(plotdf.values) # size vector
    c = np.array(plotdf.values) # color vector
    fig, ax = plt.subplots()
    # R = np.sqrt(s)/np.sqrt(s.max())/2
    R = s/s.max()/2 * 0.8
    circles = [plt.Circle((j,i), radius=r) for r, j, i in zip(R.flat, x.flat, y.flat)]
    for x_val, y_val in zip(x.flat, y.flat):
        ax.text(x_val - 0.12, y_val-0.05, '%2.2f'%plotdf.iloc[y_val, x_val], color = "#f03b20", fontsize = 15)
    col = PatchCollection(circles, array=c.flatten(), edgecolor = "black", cmap="RdBu_r", norm = norm)
    ax.add_collection(col)
    ax.set(xticks=np.arange(M), yticks=np.arange(N),
           xticklabels=xlabels, yticklabels=ylabels)
    ax.set_xticks(np.arange(M+1)-0.5, minor=True)
    ax.set_yticks(np.arange(N+1)-0.5, minor=True)
    ax.grid(which='minor')
    fig.colorbar(col)
    ax.set_aspect("equal")
    fig.set_size_inches(w = M * 3, h = N * 3)
    fig.savefig(os.path.join("..", "figures", expname, "pair%d_corr_circle_M1-4.pdf"%i))


################ compare test-retest reliability ##########
r_ = []
var_ = []
model_ = []
for i, modelname in enumerate(modelnames):
    tmp = log_paradf.loc[log_paradf["model"] == modelname]
    s1_tmp = tmp[tmp["sess"] == 1]
    s2_tmp = tmp[tmp["sess"] == 2]
    s1_tmp, s2_tmp = s1_tmp[np.isin(s1_tmp["id"], s2_tmp["id"])], s2_tmp[np.isin(s2_tmp["id"], s1_tmp["id"])]
    print(s1_tmp.shape[0])
    print(s2_tmp.shape[0])
    for para in log_paranames:
        r_.append(spearmanr(s1_tmp[para], s2_tmp[para])[0])
        var_.append(para)
        model_.append(modellabels[i])


# please check again 
plotdf = pd.DataFrame({
    "r": r_,
    "var": pd.Categorical([para_name_label_mapping[x] for x in var_], categories = log_paralabels, ordered = True),
    "model": model_,
    "z": [0.5 * np.log((1 + r) / (1-r)) for r in r_],
    "upper_z": [0.5 * np.log((1 + r) / (1-r)) + 1 / np.sqrt(s1_tmp.shape[0] - 3) for r in r_],
    "lower_z": [0.5 * np.log((1 + r) / (1-r)) - 1 / np.sqrt(s1_tmp.shape[0] - 3) for r in r_]
    })
plotdf["upper_r"] = [(np.exp(2*z)-1)/(np.exp(2*z)+1) for z in plotdf["upper_z"]]
plotdf["lower_r"] = [(np.exp(2*z)-1)/(np.exp(2*z)+1) for z in plotdf["lower_z"]]


from plotnine import ggplot, aes, geom_bar, geom_errorbar, facet_grid, position_dodge, theme, theme_classic, ylim
ggplot(plotdf) + facet_grid(facets="~var") + aes(x="model", y="r") + geom_bar(stat = "identity") +\
geom_errorbar(aes(ymin = "lower_r", ymax = "upper_r"), width=.2, position=position_dodge(.9)) +\
theme_classic() + ylim([0, 1])



