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
import code
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
from scipy.stats import mannwhitneyu
import scipy as sp
import scipy

# plot styles
plt.style.use('classic')
sns.set(font_scale = 1)
sns.set_style("white")
condition_palette = ["#762a83", "#1b7837"]

#############   
def var2label(var):
    if var == "auc":
        label = "AUC (s)"
    elif var == "std_wtw":
        label = r"$\sigma_\mathrm{wtw}$ (s)"
    elif var == "auc_delta":
        label = r'$\Delta$ AUC (s)'
    else:
        return var
    return label

vars = ['auc', 'std_wtw', 'auc_delta']
labels = ["AUC (s)", r"$\sigma_\mathrm{wtw}$ (s)", r'$\Delta$ AUC (s)']

################
s1_df_ = []
s2_df_ = []
s1_stats_ = []
s2_stats_ = []
trialdata_sess1_list = []
trialdata_sess2_list = []
for expname in ['active', 'passive', "combined"]:
    if expname == "combined":
        s1_df, s2_df, s1_stats, s2_stats = pd.concat(s1_df_), pd.concat(s2_df_), pd.concat(s1_stats_), pd.concat(s2_stats_)
        trialdata_sess1_, trialdata_sess2_ = copy.copy(trialdata_sess1_list[0]), copy.copy(trialdata_sess2_list[0])
        trialdata_sess1_.update(trialdata_sess1_list[1])
        trialdata_sess2_.update(trialdata_sess2_list[1])
    else:
        hdrdata_sess1, trialdata_sess1_ = loadFxs.group_quality_check(expname, 1, plot_quality_check = True)
        hdrdata_sess2, trialdata_sess2_ = loadFxs.group_quality_check(expname, 2, plot_quality_check = True)
        # only include participants who completed two sessions
        hdrdata_sess1 = hdrdata_sess1[np.isin(hdrdata_sess1["id"], hdrdata_sess2["id"])]
        trialdata_sess1_ = {x: y for x,y in trialdata_sess1_.items() if x[0] in hdrdata_sess2["id"].values}
        s1_stats, s1_Psurv_b1_, s1_Psurv_b2_, s1_WTW_ = analysisFxs.group_MF(trialdata_sess1_, plot_each = False)   
        s2_stats, s2_Psurv_b1_, s2_Psurv_b2_, s2_WTW_ = analysisFxs.group_MF(trialdata_sess2_, plot_each = False)   
        s1_df = analysisFxs.pivot_by_condition(s1_stats)
        s2_df = analysisFxs.pivot_by_condition(s2_stats)
        s1_df_.append(s1_df)
        s2_df_.append(s2_df)
        s1_stats_.append(s1_stats)
        s2_stats_.append(s2_stats)
        trialdata_sess1_list.append(trialdata_sess1_)
        trialdata_sess2_list.append(trialdata_sess2_)

    # across-condition convergence
    tmp1 = s1_stats[["auc", "std_wtw", "id", "condition"]].melt(id_vars = ["id", "condition"], value_vars = ["auc", "std_wtw"])
    tmp1 = tmp1[tmp1["condition"] == "HP"].drop(["condition"], axis = 1).merge(tmp1[tmp1["condition"] == "LP"].drop(["condition"], axis = 1), on = ["id", "variable"], suffixes = ["_HP", "_LP"])
    tmp2 = s2_stats[["auc", "std_wtw", "id", "condition"]].melt(id_vars = ["id", "condition"], value_vars = ["auc", "std_wtw"])
    tmp2 = tmp2[tmp2["condition"] == "HP"].drop(["condition"], axis = 1).merge(tmp2[tmp2["condition"] == "LP"].drop(["condition"], axis = 1), on = ["id", "variable"], suffixes = ["_HP", "_LP"])
    df = analysisFxs.vstack_sessions(tmp1, tmp2)
    df["variable"] = df["variable"].apply(var2label)
    g = sns.FacetGrid(df, col = "variable", hue = "sess", sharex  = False, sharey  = False, palette = ["#b2182b", "#2166ac"])
    g.map(sns.regplot, "value_HP", "value_LP")
    for ax, label in zip(g.axes.flatten(), labels):
        for sess in [1, 2]:
            spearman_rho, pearson_rho, abs_icc, con_icc, ssbs, ssbm, sse, msbs, msbm, mse = analysisFxs.calc_reliability(df.loc[np.logical_and(df['sess'] == 'Session %d'%sess, df['variable'] == label), 'value_HP'], df.loc[np.logical_and(df['sess'] == 'Session %d'%sess, df['variable'] == label), "value_LP"])
            ax.text(0.4, 0.3 - sess * 0.1, 'SESS%d r = %.3f\n'%(sess, spearman_rho), size=10, color = "red", transform = ax.transAxes)
    g.set_titles(col_template = "{col_name}")
    g.set(xlabel = "HP block", ylabel = "LP block")
    g.savefig(os.path.join('..', 'figures', expname, 'across-condition_convergence.pdf'))
    
    # summary statistics
    df = analysisFxs.vstack_sessions(s1_df, s2_df)
    df.groupby(["sess"]).agg({"auc":[np.median, scipy.stats.iqr, lambda x: np.median(x) - scipy.stats.iqr(x), lambda x: np.median(x) + scipy.stats.iqr(x)],\
        "std_wtw": [np.median, scipy.stats.iqr, lambda x: np.median(x) - scipy.stats.iqr(x), lambda x: np.median(x) + scipy.stats.iqr(x)],
        "auc_delta": [np.median, scipy.stats.iqr, lambda x: np.median(x) - scipy.stats.iqr(x), lambda x: np.median(x) + scipy.stats.iqr(x)]})
    df.groupby(["sess"])['auc'].describe()
    df.groupby(["sess"])['std_wtw'].describe()
    df.groupby(["sess"])['auc_delta'].describe()


    # reliability table
    df = analysisFxs.hstack_sessions(s1_df, s2_df)
    _, _, _, _, _, report = analysisFxs.calc_zip_reliability(df, [(x + '_sess1', x + '_sess2') for x in ["auc", "std_wtw", "auc_delta"]])

    print(report.round(3))

    #reliability 
    vars = ["auc", "std_wtw", "auc_delta"]
    df = s1_df.melt(id_vars = "id", value_vars = vars).merge(s2_df.melt(id_vars = "id", value_vars = vars), on = ["id", "variable"], suffixes = ["_sess1", "_sess2"])
    df["variable"] = df["variable"].apply(var2label)
    g = sns.FacetGrid(data = df, col = "variable", sharex = False, sharey = False)
    g.map(figFxs.my_regplot, "value_sess1", "value_sess2")
    g.set_titles(col_template = "{col_name}")
    g.set(xlabel = "Session 1", ylabel = "Session 2")
    g.savefig(os.path.join('..', 'figures', expname, 'task_reliability.pdf'))

    # practice effect 
    s1_df["sess"] = "Session 1"
    s2_df["sess"] = "Session 2"
    df = analysisFxs.vstack_sessions(s1_df.melt(id_vars = ["id", "sess"], value_vars = vars),
        s2_df.melt(id_vars = ["id", "sess"], value_vars = vars))
    df["variable"] = df["variable"].apply(var2label)
    g = sns.FacetGrid(data = df, col = "variable", sharex = False, sharey = False)
    g.map(sns.boxplot, "sess", "value", boxprops={'facecolor':'None'}, medianprops={"linestyle":"--", "color": "red"})
    g.map(sns.swarmplot, "sess", "value",  color = "grey", edgecolor = "black", alpha = 0.4, linewidth=1,  size = 3)
    for ax, var in zip(g.axes.flatten(), labels):
        sig = figFxs.tosig(sp.stats.wilcoxon(df.loc[np.logical_and(df["sess"] == "Session 1", df["variable"] == var), "value"], df.loc[np.logical_and(df["sess"] == "Session 2", df["variable"] == var), "value"]).pvalue)
        ymax = df.loc[df["variable"] == var, "value"].max()
        ax.plot([0, 0, 1, 1], [ymax * 1.1, ymax * 1.2, ymax * 1.2, ymax * 1.1], lw=1.5, color = "black")
        ax.text((0+1)*.5, ymax * 1.2, sig, ha='center', va='bottom', size = 10)
    g.set_titles(col_template = "{col_name}")
    g.set(xlabel = "", ylabel = "")
    g.savefig(os.path.join('..', 'figures', expname, 'task_practice.pdf'))


    ######################## spilt half reliability #############
    df_ = []
    for sess in [1, 2]:
        if sess == 1:
            trialdata_ = trialdata_sess1_
        else:
            trialdata_ = trialdata_sess2_
        odd_trialdata_, even_trialdata_ = analysisFxs.split_odd_even(trialdata_)
        stats_odd, _, _, _ = analysisFxs.group_MF(odd_trialdata_, plot_each = False)  
        stats_even, _, _, _ = analysisFxs.group_MF(even_trialdata_, plot_each = False) 
        odd_df = analysisFxs.pivot_by_condition(stats_odd)
        even_df = analysisFxs.pivot_by_condition(stats_even)
        df = odd_df.melt(id_vars = "id", value_vars = vars).merge(even_df.melt(id_vars = "id", value_vars = vars), on = ["variable", "id"], suffixes = ["_odd", "_even"])
        df_.append(df)
    df = analysisFxs.vstack_sessions(*df_)
    df["variable"] = df["variable"].apply(var2label)
    g = sns.FacetGrid(data = df, col = "variable", hue = "sess", sharex = False, sharey = False, palette = ["#b2182b", "#2166ac"])
    g.map(sns.regplot, "value_odd", "value_even", scatter_kws={"s": 40, "alpha":0.5}, line_kws={"linestyle":"--"})
    g.set_titles(col_template = "{col_name}")
    g.set(xlabel = "Odd", ylabel = "Even")
    for ax, label in zip(g.axes.flatten(), labels):
        for sess in [1, 2]:
            spearman_rho, pearson_rho, abs_icc, con_icc, ssbs, ssbm, sse, msbs, msbm, mse = analysisFxs.calc_reliability(df.loc[np.logical_and(df['sess'] == 'Session %d'%sess, df['variable'] == label), 'value_odd'], df.loc[np.logical_and(df['sess'] == 'Session %d'%sess, df['variable'] == label), "value_even"])
            ax.text(0.4, 0.3 - sess * 0.1, 'SESS%d r = %.3f\n'%(sess, spearman_rho), size=10, color = "red", transform = ax.transAxes)
    g.savefig(os.path.join('..', 'figures', expname, 'split_half.pdf'))

    g = sns.FacetGrid(data = df, col = "variable", sharex = False, sharey = False)
    g.map(figFxs.my_regplot, "value_odd", "value_even")
    g.set_titles(col_template = "{col_name}")
    g.set(xlabel = "Odd", ylabel = "Even")
    g.savefig(os.path.join('..', 'figures', expname, 'split_half_session-combined.pdf'))

    ###### correlations among measures #########
    statsdf = analysisFxs.agg_across_sessions(s1_df, s2_df)
    statsdf = statsdf.rename(columns = dict(zip(vars, labels)))
    g = sns.pairplot(statsdf[labels], kind = "reg", diag_kws = {"color": "grey", "edgecolor": "black"},\
    plot_kws ={'line_kws':{'color':'red'}, "scatter_kws": {"color": "grey", "edgecolor": "black"}})
    g.map_lower(figFxs.annotate_reg)
    g.savefig(os.path.join('..', 'figures', expname, 'among_measures.pdf'))
############ model parameter analysis ###########
# expname = 'passive'
# modelname = 'QL2reset_HM_short'
# fitMethod = "whole"
# stepsize = 0.5
# s1_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, fitMethod, stepsize)
# s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname, fitMethod, stepsize)
# paradf = analysisFxs.hstack_sessions(s1_paradf, s2_paradf)
# subtitles = [r'$\mathbf{log(\alpha)}$', r'$\mathbf{log(\nu)}$', r'$\mathbf{\tau}$', r'$\mathbf{\gamma}$', r'$\mathbf{log(\eta)}$']
# paranames = modelFxs.getModelParas(modelname)
# npara = len(paranames)

# # prepare data for combined reliability 
# paranames = ['alpha', 'nu', 'tau', 'gamma', 'eta']
# paralabels = [r"$\alpha$", r"$\nu$", r"$\tau$", r"$\gamma$", r"$\eta$"]
# for paraname, paralabel in zip(paranames, paralabels):
#     icc_vals, _, _ = analysisFxs.calc_bootstrap_reliability(paradf[paraname + '_sess1'], paradf[paraname + '_sess2'], n = 150)
#     var_vals = np.full(len(icc_vals), paralabel)
#     exp_vals = np.full(len(icc_vals), expname)
#     icc_vals_.append(icc_vals)
#     var_vals_.append(var_vals)
#     exp_vals_.append(exp_vals)




# # plot parameter distributions
# figFxs.plot_parameter_distribution(modelname, s1_paradf.iloc[:,:-1], s2_paradf.iloc[:,:-1], color = "grey", edgecolor = "black")
# plt.gcf().set_size_inches(5 * npara, 5 * 2)
# plt.savefig(os.path.join("..", 'figures', expname, "%s_%s_stepsize%.2f_para_dist.pdf"%(modelname, fitMethod, stepsize)))
# # plot parameter correlations
# figFxs.plot_parameter_reliability(modelname, s1_paradf.iloc[:,:-1], s2_paradf.iloc[:,:-1], subtitles)
# plt.gcf().set_size_inches(5 * npara, 5)
# plt.savefig(os.path.join("..", 'figures', expname, "%s_%s_stepsize%.2f_para_reliability.pdf"%(modelname, fitMethod, stepsize)))
# # plot parameter practice
# figFxs.plot_parameter_practice(modelname, s1_paradf.iloc[:,:-1], s2_paradf.iloc[:,:-1], subtitles)
# plt.gcf().set_size_inches(5 * npara, 5)
# plt.savefig(os.path.join("..", 'figures', expname, "%s_%s_stepsize%.2f_para_practice.pdf"%(modelname, fitMethod, stepsize)))


# #################
# expname = 'passive'
# modelname = 'QL2reset_HM_short'
# fitMethod = "whole"
# stepsize = 0.5
# s1_paradf = loadFxs.load_hm_parameter_estimates(expname, 1, hdrdata_sess1, modelname, fitMethod, stepsize)
# s2_paradf = loadFxs.load_hm_parameter_estimates(expname, 2, hdrdata_sess2, modelname, fitMethod, stepsize)

# figFxs.plot_parameter_reliability(modelname, s1_paradf, s2_paradf, subtitles)
# plt.gcf().set_size_inches(5 * npara, 5)
# plt.savefig(os.path.join("..", 'figures', expname, "%s_%s_stepsize%.2f_para_reliability_short.pdf"%(modelname, fitMethod, stepsize)))

# HM_ids = list(set(s1_paradf.id) and set(s2_paradf.id))

# # it doesn't seem to increase defre... either 

# expname = 'passive'
# modelname = 'QL2reset'
# fitMethod = "whole"
# stepsize = 0.5
# s1_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, fitMethod, stepsize)
# s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname, fitMethod, stepsize)

# figFxs.plot_parameter_reliability(modelname, s1_paradf[np.isin(s1_paradf.id, HM_ids)].iloc[:,:-1], s2_paradf[np.isin(s2_paradf.id, HM_ids)].iloc[:,:-1], subtitles)
# plt.gcf().set_size_inches(5 * npara, 5)
# plt.savefig(os.path.join("..", 'figures', expname, "%s_%s_stepsize%.2f_para_reliability_short.pdf"%(modelname, fitMethod, stepsize)))



# ##################### reliability about selfreport measures
#  if expname == "passive":
#     s1_selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
#     s2_selfdf = loadFxs.parse_group_selfreport(expname, 2, isplot = False)
#     s1_selfdf = s1_selfdf[np.isin(s1_selfdf["id"], hdrdata_sess1["id"])]
#     ###################################### reliability of selfreport data #################
#     selfreport_vars = ["UPPS", "BIS", "Motor", "Nonplanning", "Attentional", "attention", "cogstable", "motor", "perseverance", "selfcontrol", "cogcomplex"]
#     df = pd.melt(s1_selfdf, id_vars = ["id"], value_vars = selfreport_vars).merge(
#         pd.melt(s2_selfdf, id_vars = ["id"], value_vars = selfreport_vars), on = ["id", "variable"],
#         suffixes = ["_sess1", "_sess2"])
#     g = sns.FacetGrid(data = df, col = "variable", sharex= False, sharey = False)
#     g.map(figFxs.my_regplot, "value_sess1", "value_sess2")
#     g.savefig(os.path.join("..", 'figures', expname, "selfreport_reliability.pdf"))

#     # let's do practice effects 
#     df = analysisFxs.vstack_sessions(pd.melt(s1_selfdf[np.isin(s1_selfdf.id, s2_selfdf)], id_vars = ["id"], value_vars = selfreport_vars),
#         pd.melt(s2_selfdf, id_vars = ["id"], value_vars = selfreport_vars))
#     g = sns.FacetGrid(data = df, col = "variable", sharex= False, sharey = False)
#     g.map(sns.swarmplot, "sess", "value")

#     # this might be a useful table to show ....
#         # selfreport_vars = ['NU', 'PU', 'PM', 'PS', 'SS', 'attention', 'cogstable', 'motor', 'perseverance', 'selfcontrol', 'cogcomplex', 'UPPS', 'BIS', 'GMK'] 
#     selfdf = analysisFxs.hstack_sessions(s1_selfdf, s2_selfdf)
#     to_be_tested_vars = list(zip([x + "_sess1" for x in expParas.selfreport_vars], [x + "_sess2" for x in expParas.selfreport_vars]))
#     spearman_rho_, pearson_rho_, abs_icc_, con_icc_, n_, report = analysisFxs.calc_zip_reliability(selfdf, to_be_tested_vars)
#     report.sort_values(by = "spearman_rho")


