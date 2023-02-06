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


expname = "active"

# load data 
hdrdata_sess1, trialdata_sess1_ = loadFxs.group_quality_check(expname, 1, plot_quality_check = False)
hdrdata_sess2, trialdata_sess2_ = loadFxs.group_quality_check(expname, 2, plot_quality_check = False)

## only exclude valid participants 
hdrdata_sess1 = hdrdata_sess1[np.isin(hdrdata_sess1["id"], hdrdata_sess2["id"])]
trialdata_sess1_ = {x: y for x,y in trialdata_sess1_.items() if x[0] in hdrdata_sess2["id"].values}


# for modelname in ['QL2', 'QL2reset']:
modelname = 'QL2reset'
fitMethod = "whole"
stepsize = 0.5

# subtitles = [r'$\mathbf{log(\alpha)}$', r'$\mathbf{log(\nu)}$', r'$\mathbf{\tau}$', r'$\mathbf{\gamma}$', r'$\mathbf{log(\eta)}$']
paranames = modelFxs.getModelParas(modelname)
npara = len(paranames)
# load model parameters
s1_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, fitMethod, stepsize)
s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname, fitMethod, stepsize)
s1_logparadf = figFxs.log_transform_parameter(s1_paradf, ["alpha", "nu", "tau", "eta"])
s2_logparadf = figFxs.log_transform_parameter(s2_paradf, ["alpha", "nu", "tau", "eta"])
# log transform 
log_paradf = pd.concat([s1_logparadf, s2_logparadf])


log_paranames = log_paradf.columns[:5].values
log_paralabels = [r'$log(\alpha)$', r'$log(\nu)$', r'$log(\tau)$', r'$\gamma$', r'$log(\eta)$']
log_paralabel_mapping = dict(zip(log_paranames, log_paralabels))
log_paralimits = dict(zip(log_paranames, [(-12, 2), (-8, 4), (-3, 5), (0.4, 1.1), (-4, 4)]))
log_paraticks = dict(zip(log_paranames, [(-12, -6, 0), (-8, 0, 4), (-3, 0, 3), (0.5, 1), (-4, 0, 4)]))


# I need to combine both conditions
#### parameter reliabiliy ########
subtitles = [r'$\mathbf{log(\alpha)}$', r'$\mathbf{log(\nu)}$', r'$\mathbf{\tau}$', r'$\mathbf{\gamma}$', r'$\mathbf{log(\eta)}$']
paranames = modelFxs.getModelParas(modelname)
npara = len(paranames)


######## plot parameter distributions ########
figFxs.plot_parameter_distribution(modelname, s1_paradf.iloc[:,:-1], s2_paradf.iloc[:,:-1], color = "grey", edgecolor = "black")
plt.gcf().set_size_inches(5 * npara, 5 * 2)
plt.savefig(os.path.join("..", 'figures', expname, "%s_%s_stepsize%.2f_para_dist.pdf"%(modelname, fitMethod, stepsize)))

# parameter reliability
g = figFxs.plot_parameter_reliability(modelname, s1_logparadf.iloc[:, :(npara+2)], s2_logparadf.iloc[:, :(npara+2)], log_paralabels)
for para, ax in zip(log_paranames, g.axes.flatten()):
    ax.set_xlim(log_paralimits[para])
    ax.set_xticks(log_paraticks[para])
    ax.set_ylim(log_paralimits[para])
    ax.set_yticks(log_paraticks[para])
plt.savefig(os.path.join("..", 'figures', expname, "%s_para_reliability.pdf"%(modelname)))
########## bootstrapped structural noise ########
########## should I combine data here???? #### rethink
r_ = []
pair_ = []
id_ = []
cv_ = []
std_ = []
para_ = []
id_para_ = []
sess_ = []
for sess in [1, 2]:
    paradf = s1_paradf if sess == 1 else s2_paradf
    for id in paradf["id"]:
        parafile = os.path.join("../analysis_results", expname, "modelfit", fitMethod, "stepsize%.2f"%stepsize, modelname, "%s_sess%d_sample.txt"%(id, sess))
        parasamples = pd.read_csv(parafile, header = None)
        parasamples = parasamples.iloc[:, :npara]
        parasamples =  parasamples.rename(columns = dict(zip(parasamples.columns, paranames)))
        for x, y in itertools.combinations(paranames, 2):
            pair_.append((log_paralabels[paranames.index(x)], log_paralabels[paranames.index(y)]))
            id_.append(id)
            r_.append(spearmanr(parasamples[x], parasamples[y])[0])
            sess_.append(sess)
        for x in paranames:
            cv_.append(parasamples[x].std()/parasamples[x].mean())
            std_.append(parasamples[x].std())
            para_.append(x)
            id_para_.append(x)

structure_noise_df = pd.DataFrame({
    "r": r_,
    "pair": pair_,
    "id": id_,
    "sess": sess_
    })

structure_noise_summary_df = structure_noise_df.groupby(["pair", "sess"]).agg({"r":np.median}).reset_index()

structure_noise_summary_df = structure_noise_df.groupby(["pair", "sess"]).agg({"r":np.mean}).reset_index()


# plot one participant example 
fig, ax = plt.subplots()
figFxs.my_regplot(np.log(parasamples['tau']), np.log(parasamples['eta']), equal_aspect = False, ax = ax)
ax.set_xlabel(r"$log(\tau)$", fontsize = 25)
ax.set_ylabel(r"$log(\eta)$", fontsize = 25)
fig.set_size_inches(w = 6, h = 6)
fig.tight_layout()
fig.savefig(os.path.join("..", "figures", expname, "sample_structure_corr_%s.pdf"%modelname))


# plot the heatmap version 
structure_noise_df['var1'] = pd.Categorical([x[0] for x in structure_noise_df['pair']], log_paralabels)
structure_noise_df['var2'] = pd.Categorical([x[1] for x in structure_noise_df['pair']], log_paralabels)
structure_noise_matrix = pd.DataFrame(np.full((len(paranames), len(paranames)), np.nan), columns = log_paralabels, index = log_paralabels)
structure_noise_matrix.loc[log_paralabels[1:], log_paralabels[:5]] = structure_noise_df.pivot_table('r', 'var2', 'var1', np.mean)


from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib as mpl
norm = Normalize(vmin=-0.90, vmax=0.90)
cmap = cm.get_cmap('RdBu_r')
rgba_values = cmap(norm(structure_noise_matrix))


fig, ax = plt.subplots()
g = sns.heatmap(structure_noise_matrix, ax = ax, annot=True, square=True, linewidths=1, cmap = cmap, norm = norm, cbar = True) 
#cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=[-0.80, -0.40, 0, 0.40, 0.80], orientation='vertical', label='Correlation')
plt.savefig(os.path.join("..", "figures", expname, "heatmap_structure_corr_%s.pdf"%modelname))


plt.style.use('classic')
sns.set(font_scale = 1)
sns.set_style("white")
condition_palette = ["#762a83", "#1b7837"]
fig, axes = plt.subplots(len(paranames), len(paranames))
for (i, x), (j,y) in itertools.product(enumerate(log_paralabels), enumerate(log_paralabels)):
    if (x, y) in itertools.combinations(log_paralabels, 2):
        axes[j,i].hist(structure_noise_df.loc[structure_noise_df["pair"] == (x,y), "r"].values, bins = 15, color = "#bdbdbd", edgecolor = "#bdbdbd")
        axes[j,i].set_xlim((-1.1, 1.1))
        median_val = np.median(structure_noise_df.loc[structure_noise_df["pair"] == (x,y), "r"].values)
        print(median_val)
        axes[j,i].axvline(median_val, color = rgba_values[j, i, :], linewidth = 3)
        # axes[j,i].axvline(median_val, color = "black", linewidth = 3)
        axes[j,i].text(0.2, 50, "%.2f"%median_val, color = "black")
        axes[j,i].axvline(0, color = "black", linestyle = "dotted") # "#238b45", 
        axes[j,i].set_ylim([0, 105])
        for spine in axes[j,i].spines.values():
                spine.set_edgecolor(rgba_values[j, i, :])
                spine.set_linewidth(2.5)
    else:
        for spine in axes[j,i].spines.values():
                spine.set_edgecolor("white")
    if i == (npara-1):
       axes[i,j].set_xlabel(y, fontsize=25)
    if i != (npara-1):
        axes[i,j].set_xticklabels([])
    if j == 0:
        axes[i,j].set_ylabel(x, fontsize=25)
    if j != 0:
        axes[i,j].set_yticklabels([])
    if j == 0:
        axes[i,j].set_yticks([0, 50, 100])
        axes[i,j].set_yticklabels([0, 50, 100])

fig.tight_layout()
fig.set_size_inches(w = 5 * 1.5, h = 5 * 1.7)
fig.savefig(os.path.join("..", "figures", expname, "para_structure_corr_%s_v2.pdf"%modelname))




############ estimation uncertainty #########
plotdf = pd.DataFrame({
    "std": std_,
    "cv": cv_,
    "para": para_,
    "id": id_para_
    })
plotdf.groupby("para").agg({"std":np.median})
plotdf.groupby("para").agg({"cv":np.median}) # nu has strong uncertainty 

g = sns.FacetGrid(plotdf, col = "para")
g.map(plt.hist, "cv") 

####### among participant correlations ####
# log_paradf = analysisFxs.agg_across_sessions(figFxs.log_transform_parameter(s1_paradf, ["alpha", "tau", "eta"]), figFxs.log_transform_parameter(s2_paradf, ["alpha", "tau", "eta"]))
# g = sns.pairplot(data = log_paradf.iloc[:,1:(npara+1)], kind = "reg", diag_kind = "None", corner = True,diag_kws = {"color": "grey", "edgecolor": "black"}, plot_kws ={'line_kws':{'color':'red'}, "scatter_kws": {"color": "grey", "edgecolor": "black"}})
g = sns.pairplot(data = log_paradf.iloc[:,:npara], kind = "reg", diag_kind = "None", corner = True,diag_kws = {"color": "grey", "edgecolor": "black"}, plot_kws ={'line_kws':{'color':'red'}, "scatter_kws": {"color": "grey", "edgecolor": "black"}})
g.map_lower(figFxs.annotate_reg)



# plot the heatmap version
plt.style.use('classic')
sns.set(font_scale = 1.5)
sns.set_style("white")
gross_corr_matrix = pd.DataFrame(np.full((len(paranames), len(paranames)), np.nan), columns = log_paralabels, index = log_paralabels)
for (i, x), (j,y) in itertools.product(enumerate(log_paranames), enumerate(log_paranames)):
    if i > j:
        gross_corr_matrix.iloc[i, j] = spearmanr(log_paradf[x], log_paradf[y])[0]

norm = Normalize(vmin=-0.90, vmax=0.90)
cmap = cm.get_cmap('RdBu_r')
rgba_values = cmap(norm(gross_corr_matrix))
fig, ax = plt.subplots()
g = sns.heatmap(gross_corr_matrix, ax = ax, annot=True, square=True, linewidths=1, cmap = cmap, norm = norm, cbar = True) 
#cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=[-0.80, -0.40, 0, 0.40, 0.80], orientation='vertical', label='Correlation')
plt.savefig(os.path.join("..", "figures", expname, "heatmap_gross_corr_%s.pdf"%modelname))


plt.style.use('classic')
sns.set(font_scale = 1)
sns.set_style("white")
fig, axes = plt.subplots(len(log_paranames), len(log_paranames))
for (i, x), (j,y) in itertools.product(enumerate(log_paranames), enumerate(log_paranames)):
    if (x, y) in itertools.combinations(log_paranames, 2):
        sns.regplot(log_paradf[x], log_paradf[y], ax = axes[j,i], line_kws={"color": rgba_values[j, i, :], "linestyle":"--"}, \
            scatter_kws={"color": "#bdbdbd", "s": 40, "alpha":0.7, "edgecolor":'black'})
        axes[j,i].text(0.2, 0.1, r"$\rho$ = %.3f"%gross_corr_matrix.iloc[j,i], size=12, color='black', transform=axes[j,i].transAxes)
        axes[j,i].set_xlim(log_paralimits[x])
        axes[j,i].set_ylim(log_paralimits[y])
        for spine in axes[j,i].spines.values():
                spine.set_edgecolor(rgba_values[j, i, :])
                spine.set_linewidth(2.5)
    else:
        for spine in axes[j,i].spines.values():
                spine.set_edgecolor("white")
    if i == (npara-1):
        axes[i,j].set_xlabel(log_paralabels[np.where(log_paranames == y)[0][0]], fontsize=18)
        axes[i,j].set_xticks(log_paraticks[y])
        axes[i,j].set_xticklabels(log_paraticks[y])
    else:
        axes[i,j].set_xticklabels([])
        axes[i,j].set_xlabel("")
    if j == 0:
        axes[i,j].set_ylabel(log_paralabels[np.where(log_paranames == x)[0][0]], fontsize=18)
        axes[i,j].set_yticks(log_paraticks[x])
        axes[i,j].set_yticklabels(log_paraticks[x])
    else:
        axes[i,j].set_yticklabels([])
        axes[i,j].set_ylabel("")

fig.tight_layout()
fig.set_size_inches(w = 5 * 1.5, h = 5 * 1.7)
fig.savefig(os.path.join("..", "figures", expname, "scatter_gross_corr_%s.pdf"%modelname))


# #####################################################
# ##################### split half reliability ########
# modelname = 'QL2reset_FL2'
# stepsize = 0.5
# s1_even_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, 'even', stepsize)
# s1_odd_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, 'odd', stepsize)
# figFxs.plot_parameter_reliability(modelname, s1_even_paradf.iloc[:,:-1], s1_odd_paradf.iloc[:,:-1], subtitles)
# plt.savefig(os.path.join("..", 'figures', expname, "%s_stepsize%.2f_para_split_hald.pdf"%(modelname, stepsize)))

# # is the reliability superficial? 
# a = pd.merge(s1_paradf, s1_stats, how = 'inner', on = 'id')
# spearmanr(a.loc[a.block == 1, 'tau'], a.loc[a.block == 1, 'auc'])
# spearmanr(a.loc[a.block == 2, 'tau'], a.loc[a.block == 2, 'auc'])
# # .... hmmm
# spearmanr(a.loc[a.block == 1, 'std_wtw'], a.loc[a.block == 1, 'tau'])
# spearmanr(a.loc[a.block == 2, 'std_wtw'], a.loc[a.block == 2, 'tau'])

# # prior has high correlation with AUC
# s1_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, fitMethod, stepsize)
# s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname, fitMethod, stepsize)
# figFxs.log_transform_parameter(s1_paradf, ['alpha', 'nu',  "tau", 'eta'])
# figFxs.log_transform_parameter(s2_paradf, ['alpha', 'nu', "tau", 'eta'])
# sns.pairplot(s1_paradf.iloc[:, :5])
# r_, p_ = analysisFxs.calc_prod_correlations(s1_paradf, ["log_alpha", "log_nu", "log_tau", "log_eta", "gamma"], ["log_alpha", "log_nu", "log_tau", "log_eta", "gamma"])

