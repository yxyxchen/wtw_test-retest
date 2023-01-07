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
plt.style.use('classic')
sns.set(font_scale = 1)
sns.set_style("white")
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


# modelnames = ['QL2reset_FL3']
modelname = 'QL2reset'
fitMethod = "whole"
stepsize = 0.5

# subtitles = [r'$\mathbf{log(\alpha)}$', r'$\mathbf{log(\nu)}$', r'$\mathbf{\tau}$', r'$\mathbf{\gamma}$', r'$\mathbf{log(\eta)}$']
paranames = modelFxs.getModelParas(modelname)
npara = len(paranames)

# replicate task data 
s1_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, fitMethod, stepsize)
s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname, fitMethod, stepsize)
s1_stats_rep, s1_WTW_rep, s1_dist_vals_ = modelFxs.group_model_rep(trialdata_sess1_, s1_paradf, modelname, 'whole', stepsize, isTrct = True, plot_each = False)
s2_stats_rep, s2_WTW_rep, s2_dist_vals_ = modelFxs.group_model_rep(trialdata_sess2_, s2_paradf, modelname, 'whole', stepsize, isTrct = True, plot_each = False)
s1_stats_rep.to_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_sess1_%s_stepsize%.2f.csv'%(modelname, fitMethod, stepsize)), index = None)
s2_stats_rep.to_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_sess2_%s_stepsize%.2f.csv'%(modelname, fitMethod, stepsize)), index = None)

# plot WTW 
sns.set(font_scale = 1.5)
sns.set_style("white")
g = figFxs.plot_group_emp_rep_wtw(s1_WTW_rep, s2_WTW_rep, s1_WTW_emp, s2_WTW_emp, hdrdata_sess1, hdrdata_sess2, s1_paradf, s2_paradf)
plt.tight_layout()
plt.gcf().set_size_inches(12, 6)
g.savefig(os.path.join("..", "figures", expname, "emp_rep_%s_wtw_%s_stepsize%.2f.pdf"%(modelname, fitMethod, stepsize)))
# plot task measures separately for both conditions
vars = ['auc', 'std_wtw']
labels = ['AUC', 'std_wtw']
for var, label in zip(vars, labels):
    g = figFxs.plot_group_emp_rep(s1_stats_rep, s2_stats_rep, s1_stats, s2_stats, "std_wtw", "std_wtw")
    plt.gcf().set_size_inches(10, 6)
    plt.savefig(os.path.join("..", "figures", expname, "emp_rep_%s_%s_%s_stepsize%.2f.pdf"%(modelname, fitMethod, var, stepsize)))

# plot task measures, combining both sessions and both conditions 
s1_df_emp, s2_df_emp = analysisFxs.pivot_by_condition(s1_stats), analysisFxs.pivot_by_condition(s2_stats)
emp_df = analysisFxs.agg_across_sessions(s1_df_emp, s2_df_emp)
s1_df_rep, s2_df_rep = analysisFxs.pivot_by_condition(s1_stats_rep), analysisFxs.pivot_by_condition(s2_stats_rep)
rep_df = analysisFxs.agg_across_sessions(s1_df_rep, s2_df_rep)
plotdf = emp_df.merge(rep_df, on = "id", suffixes = ["_emp", "_rep"])
vars = ['auc', 'std_wtw', "auc_delta"]
labels = ['AUC (s)', r'$\sigma_{wtw}$ (s)', r"$\Delta$ AUC (s)"]
fig, axes = plt.subplots(1, 3)
for (var, label), ax in zip(zip(vars, labels), axes.flatten()):
    sns.regplot(x = var + '_emp', y = var + '_rep', data = plotdf, scatter_kws={"color": "grey", "s": 40, "alpha":0.7, "edgecolor":'black'}, line_kws={"color": "black", "linestyle":"--"}, ax = ax)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Model-generated")
    ax.set_title(label)

fig.tight_layout()
fig.set_size_inches(14, 4)
fig.savefig(os.path.join("..", "figures", expname, "cb_emp_rep_%s_%s.pdf"%(modelname, var)))

# let me get a report 
_, _, _, _, _, report = analysisFxs.calc_zip_reliability(plotdf, [(x,y) for x, y in zip([x + "_emp" for x in vars], [x + "_rep" for x in vars])])
report['rsquared'] = report['pearson_rho']**2
report.round(3)


# I need to combine both conditions
#### parameter reliabiliy ########
subtitles = [r'$\mathbf{log(\alpha)}$', r'$\mathbf{log(\nu)}$', r'$\mathbf{\tau}$', r'$\mathbf{\gamma}$', r'$\mathbf{log(\eta)}$']
paranames = modelFxs.getModelParas(modelname)
npara = len(paranames)


######## plot parameter distributions ########
figFxs.plot_parameter_distribution(modelname, s1_paradf.iloc[:,:-1], s2_paradf.iloc[:,:-1], color = "grey", edgecolor = "black")
plt.gcf().set_size_inches(5 * npara, 5 * 2)
plt.savefig(os.path.join("..", 'figures', expname, "%s_%s_stepsize%.2f_para_dist.pdf"%(modelname, fitMethod, stepsize)))

# maybe log transform first?
figFxs.plot_parameter_reliability(modelname, s1_paradf.iloc[:,:-1], s2_paradf.iloc[:,:-1], subtitles)

########## bootstrapped structural noise ########
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
        parasamples = parasamples.iloc[:, :5]
        parasamples =  parasamples.rename(columns = dict(zip(parasamples.columns, paranames)))
        for x, y in itertools.combinations(paranames, 2):
            pair_.append((x, y))
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

plt.style.use('classic')
sns.set(font_scale = 1)
sns.set_style("white")
condition_palette = ["#762a83", "#1b7837"]
para_label_mapping = dict(zip(paranames, [r'$\alpha$', r'$\nu$', r'$\tau$', r'$\gamma$', r'$\eta$']))
fig, axes = plt.subplots(len(paranames), len(paranames))
for (i, y), (j,x) in itertools.product(enumerate(paranames), enumerate(paranames)):
    if (x, y) in itertools.combinations(paranames, 2):
        axes[i,j].hist(structure_noise_df.loc[structure_noise_df["pair"] == (x,y), "r"].values, bins = 15, color = "grey")
        axes[i,j].set_xlim((-1, 1))
        median_val = np.median(structure_noise_df.loc[structure_noise_df["pair"] == (x,y), "r"].values)
        axes[i,j].axvline(median_val, color = "red")
        axes[i,j].text(median_val, 40, "%.2f"%median_val, color = "red")
        axes[i,j].axvline(0, color = "black", linestyle = "dotted")
    if i == (npara-1):
       axes[i,j].set_xlabel(para_label_mapping[x])
    if i != (npara-1):
        axes[i,j].set_xticklabels([])
    if j == 0:
        axes[i,j].set_ylabel(para_label_mapping[y])
    if j != 0:
        axes[i,j].set_yticklabels([])
fig.savefig(os.path.join("..", "figures", expname, "para_structure_corr_%s.pdf"%modelname))


# plot the heatmap version 
structure_noise_df['var1'] = pd.Categorical([x[0] for x in structure_noise_df['pair']], paranames)
structure_noise_df['var2'] = pd.Categorical([x[1] for x in structure_noise_df['pair']], paranames)
structure_noise_matrix = pd.DataFrame(np.full((len(paranames), len(paranames)), np.nan), columns = paranames, index = paranames)
structure_noise_matrix.loc[paranames[:5], paranames[1:]] = structure_noise_df.pivot_table('r', 'var1', 'var2', np.mean)


g = sns.heatmap(structure_noise_matrix, vmin = -0.6, vmax = 0.6, center = 0, cmap = 'PiYG', annot = True)
plt.savefig(os.path.join("..", "figures", expname, "heatmap_structure_corr_%s.pdf"%modelname))

### flatten version
g = sns.FacetGrid(data = structure_noise_df, col = "pair")
g.map(plt.hist, "r", bins = 15, color = "grey")
for i, pair in enumerate(g.col_names):
    median_val = np.median(structure_noise_df.loc[structure_noise_df["pair"] == pair, "r"].values)
    g.axes.flatten()[i].axvline(median_val, color = "red")
    g.axes.flatten()[i].axvline(0, color = "black", linestyle = "dotted")
    g.axes.flatten()[i].text(median_val, 40, "%.2f"%median_val, color = "red")
g.savefig(os.path.join("..", "figures", expname, "para_structure_corr_flattern_%s.pdf"%modelname))





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
# maybe I want to log transform first ....
s1_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, fitMethod, stepsize)
s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname, fitMethod, stepsize)
log_paradf = pd.concat([figFxs.log_transform_parameter(s1_paradf, ["alpha", "tau", "eta"]), figFxs.log_transform_parameter(s2_paradf, ["alpha", "tau", "eta"])])
g = sns.pairplot(data = log_paradf.iloc[:,:npara], kind = "reg", diag_kind = "None", corner = True,\
    diag_kws = {"color": "grey", "edgecolor": "black"}, plot_kws ={'line_kws':{'color':'red'}, "scatter_kws": {"color": "grey", "edgecolor": "black"}})
g.map_lower(figFxs.annotate_reg)
g.savefig(os.path.join("..", 'figures', expname, "%s_%s_stepsize%.2f_para_correlation.pdf"%(modelname, fitMethod, stepsize)))

plt.style.use('classic')
sns.set(font_scale = 2)
sns.set_style("white")
log_para_mapping = dict(zip(log_paradf.columns[:5], subtitles))
log_paradf = log_paradf.rename(columns=log_para_mapping)
fig, axes = plt.subplots(1, 5*2)
for i, (x, y) in enumerate(itertools.combinations(log_paradf.columns[:5], 2)):
    figFxs.my_regplot(log_paradf[x], log_paradf[y], ax = axes.flatten()[i])
    axes.flatten()[i].set_xlabel(x)
    axes.flatten()[i].set_ylabel(y)
fig.set_size_inches(5 * npara * (npara-1) / 2, 5)
fig.savefig(os.path.join("..", 'figures', expname, "%s_%s_stepsize%.2f_para_correlation_flattern.pdf"%(modelname, fitMethod, stepsize)))


#####################################################
##################### split half reliability ########
modelname = 'QL2reset_FL2'
stepsize = 0.5
s1_even_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, 'even', stepsize)
s1_odd_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, 'odd', stepsize)
figFxs.plot_parameter_reliability(modelname, s1_even_paradf.iloc[:,:-1], s1_odd_paradf.iloc[:,:-1], subtitles)
plt.savefig(os.path.join("..", 'figures', expname, "%s_stepsize%.2f_para_split_hald.pdf"%(modelname, stepsize)))

# is the reliability superficial? 
a = pd.merge(s1_paradf, s1_stats, how = 'inner', on = 'id')
spearmanr(a.loc[a.block == 1, 'tau'], a.loc[a.block == 1, 'auc'])
spearmanr(a.loc[a.block == 2, 'tau'], a.loc[a.block == 2, 'auc'])
# .... hmmm
spearmanr(a.loc[a.block == 1, 'std_wtw'], a.loc[a.block == 1, 'tau'])
spearmanr(a.loc[a.block == 2, 'std_wtw'], a.loc[a.block == 2, 'tau'])

# prior has high correlation with AUC
s1_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, fitMethod, stepsize)
s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname, fitMethod, stepsize)
figFxs.log_transform_parameter(s1_paradf, ['alpha', 'nu',  "tau", 'eta'])
figFxs.log_transform_parameter(s2_paradf, ['alpha', 'nu', "tau", 'eta'])
sns.pairplot(s1_paradf.iloc[:, :5])
r_, p_ = analysisFxs.calc_prod_correlations(s1_paradf, ["log_alpha", "log_nu", "log_tau", "log_eta", "gamma"], ["log_alpha", "log_nu", "log_tau", "log_eta", "gamma"])

