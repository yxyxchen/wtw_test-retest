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
sns.set(font_scale = 2)
sns.set_style("white")
condition_palette = ["#762a83", "#1b7837"]

vars = ['auc', 'std_wtw', 'auc_delta']
labels = ['AUC (s)', r"$\sigma_\mathrm{wtw}$ (s)", r'$\Delta$AUC (s)']
icc_vals_ = []
var_vals_ = []
exp_vals_ = []
for expname in ['active', 'passive']:
	s1_selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
	hdrdata_sess1, trialdata_sess1_ = loadFxs.group_quality_check(expname, 1, plot_quality_check = True)
	hdrdata_sess2, trialdata_sess2_ = loadFxs.group_quality_check(expname, 2, plot_quality_check = True)
	s1_stats, s1_Psurv_b1_, s1_Psurv_b2_, s1_WTW_ = analysisFxs.group_MF(trialdata_sess1_, plot_each = False)   
	s2_stats, s2_Psurv_b1_, s2_Psurv_b2_, s2_WTW_ = analysisFxs.group_MF(trialdata_sess2_, plot_each = False)   
	s1_df = analysisFxs.pivot_by_condition(s1_stats)
	s2_df = analysisFxs.pivot_by_condition(s2_stats)
	####### reliability analysis ######
	df = analysisFxs.hstack_sessions(s1_df, s2_df)
	for var, label in zip(vars, labels):
		icc_vals, _, _ = analysisFxs.calc_bootstrap_reliability(df[var + '_sess1'], df[var + '_sess2'], n = 150)
		var_vals = np.full(len(icc_vals), label)
		exp_vals = np.full(len(icc_vals), expname)
		icc_vals_.append(icc_vals)
		var_vals_.append(var_vals)
		exp_vals_.append(exp_vals)

# load selfreport data for the passive exp
expname = "passive"
s1_selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
s2_selfdf = loadFxs.parse_group_selfreport(expname, 2, isplot = False)
df = analysisFxs.hstack_sessions(s1_selfdf, s2_selfdf)
for var, label in zip(["UPPS", "BIS", "Motor", "Nonplanning", "Attentional", "attention", "cogstable", "motor", "perseverance", "selfcontrol", "cogcomplex"], ["UPPS", "BIS", "Motor", "Nonplanning", "Attentional", "attention", "cogstable", "motor", "perseverance", "selfcontrol", "cogcomplex"]):
	icc_vals, _, _ = analysisFxs.calc_bootstrap_reliability(df[var + '_sess1'], df[var + '_sess2'], n = 150)
	var_vals = np.full(len(icc_vals), label)
	exp_vals = np.full(len(icc_vals), expname)
	icc_vals_.append(icc_vals)
	var_vals_.append(var_vals)
	exp_vals_.append(exp_vals)

######## put all reliability measures together ######
df = pd.DataFrame({
	"icc": np.array(icc_vals_).flatten(),
	"var": np.array(var_vals_).flatten(),
	"exp": np.array(exp_vals_).flatten(),
	})
df.loc[~np.isin(df["var"], ["AUC (s)", "$\\sigma_\\mathrm{wtw}$ (s)", "$\\Delta$AUC (s)"]), "exp"] = "survey"
vars = ["UPPS", "BIS", "Motor", "Nonplanning", "Attentional", "attention", "cogstable", "motor", "perseverance", "selfcontrol", "cogcomplex",  "AUC (s)", "$\\sigma_\\mathrm{wtw}$ (s)", "$\\Delta$AUC (s)"]
fig, ax = plt.subplots()
sns.set_palette(sns.color_palette(["#d73027", "#4575b4", "#a1d76a"]))
sns.violinplot(x = "icc", y = "var", data = df, hue = "exp", ax = ax, order = vars)
summarydf = df[~np.isin(df["var"], ["AUC (s)", "$\\sigma_\\mathrm{wtw}$ (s)", "$\\Delta$AUC (s)"])].groupby(["var", "exp"]).agg({"icc":[np.median, "max"]})
for i, var in enumerate(["UPPS", "BIS", "Motor", "Nonplanning", "Attentional", "attention", "cogstable", "motor", "perseverance", "selfcontrol", "cogcomplex"]):
	ax.text(summarydf.loc[(var, "survey"), ('icc', 'max')] * 1.1, i + 0.5, round(summarydf.loc[(var, "survey"),('icc', "median")], 2), color = "green")

summarydf = df[np.isin(df["var"], ["AUC (s)", "$\\sigma_\\mathrm{wtw}$ (s)", "$\\Delta$AUC (s)"])].groupby(["var", "exp"]).agg({"icc":[np.median, "max"]})
for i, var in enumerate(["AUC (s)", "$\\sigma_\\mathrm{wtw}$ (s)", "$\\Delta$AUC (s)"]):
	for j, exp in enumerate(["active", "passive"]):
		c =  "blue" if exp == "passive" else "red"
		max_icc = max(summarydf.loc[(var, "passive"), ('icc', 'max')], summarydf.loc[(var, "active"), ('icc', 'max')])
		if exp == "active":
			ax.text(max_icc * 1.1, 11.25 + i  , round(summarydf.loc[(var, exp),('icc', "median")], 2), color = c)
		if exp == "passive":
			ax.text(max_icc * 1.1 + 0.08,  11.25 + i  , "/ %.2f"%summarydf.loc[(var, exp),('icc', "median")], color = c)

plt.tight_layout()
ax.legend(loc='upper left')
ax.set_xlabel("Bootstrapped ICC")
ax.set_ylabel("")
fig.set_size_inches(18.5, 10.5)
fig.savefig(os.path.join('..', 'figures', expname, 'all_reliability.pdf'))

#############	
vars = ['auc', 'std_wtw', 'auc_delta']
labels = ['AUC (s)', r"$\sigma_\mathrm{wtw}$ (s)", r'$\Delta$AUC (s)']
icc_vals_ = []
var_vals_ = []
exp_vals_ = []
for expname in ['active', 'passive']:
	s1_selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
	hdrdata_sess1, trialdata_sess1_ = loadFxs.group_quality_check(expname, 1, plot_quality_check = True)
	hdrdata_sess2, trialdata_sess2_ = loadFxs.group_quality_check(expname, 2, plot_quality_check = True)
	s1_stats, s1_Psurv_b1_, s1_Psurv_b2_, s1_WTW_ = analysisFxs.group_MF(trialdata_sess1_, plot_each = False)   
	s2_stats, s2_Psurv_b1_, s2_Psurv_b2_, s2_WTW_ = analysisFxs.group_MF(trialdata_sess2_, plot_each = False)   
	s1_df = analysisFxs.pivot_by_condition(s1_stats)
	s2_df = analysisFxs.pivot_by_condition(s2_stats)
	df = analysisFxs.hstack_sessions(s1_df, s2_df)
	for var, label in zip(vars, labels):
		fig, ax = plt.subplots()
		figFxs.my_regplot(df[var + '_sess1'], df[var + '_sess2'], ax = ax)
		ax.set_xlabel("Session 1")
		ax.set_ylabel("Session 2")
		ax.set_title(label)
		plt.tight_layout()
		fig.savefig(os.path.join('..', 'figures', expname, var + '_reliability.pdf'))
	######## parctice effects ###########
	df = analysisFxs.vstack_sessions(s1_df, s2_df)
	for var, label in zip(vars, labels):
		fig, ax = plt.subplots()
		sns.swarmplot(data = df, x = "sess", y = var, color = "grey", edgecolor = "black", alpha = 0.4, linewidth=1, ax = ax, size = 3)
		sns.boxplot(x="sess", y=var, data=df, boxprops={'facecolor':'None'}, medianprops={"linestyle":"--", "color": "red"}, ax=ax)
		ax.set_xlabel("")
		ax.set_ylabel(label)
		plt.gcf().set_size_inches(4.5, 6)
		plt.tight_layout()
		fig.savefig(os.path.join('..', 'figures', expname, var + '_practice.pdf'))
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
		df = analysisFxs.hstack_sessions(odd_df, even_df, suffixes = ["_odd", "_even"])
		df_.append(df)
	df = analysisFxs.vstack_sessions(*df_)
	for var, label in zip(vars, labels):
		g = sns.lmplot(data = df, x = var+'_odd', y = var+'_even', hue = "sess", scatter_kws={"s": 40, "alpha":0.5}, line_kws={"linestyle":"--"})
		for sess in [1, 2]:
			spearman_rho, pearson_rho, abs_icc, con_icc, ssbs, ssbm, sse, msbs, msbm, mse = analysisFxs.calc_reliability(df.loc[df['sess'] == 'Session %d'%sess, var+'_odd'], df.loc[df['sess'] == 'Session %d'%sess, var+'_even'])
			g.axes[0,0].text(0.4, 0.3 - sess * 0.1, 'SESS%d r = %.3f\n'%(sess, spearman_rho), size=20, color = "red", transform = g.axes[0,0].transAxes)
		g.axes[0,0].set_xlabel("Odd")
		g.axes[0,0].set_ylabel("Even")
		g.axes[0,0].set_title(label)
		g.axes[0,0].set_aspect('equal')
		plt.tight_layout()
		g._legend.remove()
		plt.savefig(os.path.join('..', 'figures', expname, var + '_split_half.pdf'))

############ model parameter analysis ###########
expname = 'passive'
modelname = 'QL2reset_HM_short'
fitMethod = "whole"
stepsize = 0.5
s1_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, fitMethod, stepsize)
s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname, fitMethod, stepsize)
paradf = analysisFxs.hstack_sessions(s1_paradf, s2_paradf)
subtitles = [r'$\mathbf{log(\alpha)}$', r'$\mathbf{log(\nu)}$', r'$\mathbf{\tau}$', r'$\mathbf{\gamma}$', r'$\mathbf{log(\eta)}$']
paranames = modelFxs.getModelParas(modelname)
npara = len(paranames)

# prepare data for combined reliability 
paranames = ['alpha', 'nu', 'tau', 'gamma', 'eta']
paralabels = [r"$\alpha$", r"$\nu$", r"$\tau$", r"$\gamma$", r"$\eta$"]
for paraname, paralabel in zip(paranames, paralabels):
	icc_vals, _, _ = analysisFxs.calc_bootstrap_reliability(paradf[paraname + '_sess1'], paradf[paraname + '_sess2'], n = 150)
	var_vals = np.full(len(icc_vals), paralabel)
	exp_vals = np.full(len(icc_vals), expname)
	icc_vals_.append(icc_vals)
	var_vals_.append(var_vals)
	exp_vals_.append(exp_vals)




# plot parameter distributions
figFxs.plot_parameter_distribution(modelname, s1_paradf.iloc[:,:-1], s2_paradf.iloc[:,:-1], color = "grey", edgecolor = "black")
plt.gcf().set_size_inches(5 * npara, 5 * 2)
plt.savefig(os.path.join("..", 'figures', expname, "%s_%s_stepsize%.2f_para_dist.pdf"%(modelname, fitMethod, stepsize)))
# plot parameter correlations
figFxs.plot_parameter_reliability(modelname, s1_paradf.iloc[:,:-1], s2_paradf.iloc[:,:-1], subtitles)
plt.gcf().set_size_inches(5 * npara, 5)
plt.savefig(os.path.join("..", 'figures', expname, "%s_%s_stepsize%.2f_para_reliability.pdf"%(modelname, fitMethod, stepsize)))
# plot parameter practice
figFxs.plot_parameter_practice(modelname, s1_paradf.iloc[:,:-1], s2_paradf.iloc[:,:-1], subtitles)
plt.gcf().set_size_inches(5 * npara, 5)
plt.savefig(os.path.join("..", 'figures', expname, "%s_%s_stepsize%.2f_para_practice.pdf"%(modelname, fitMethod, stepsize)))


#################
expname = 'passive'
modelname = 'QL2reset_HM_short'
fitMethod = "whole"
stepsize = 0.5
s1_paradf = loadFxs.load_hm_parameter_estimates(expname, 1, hdrdata_sess1, modelname, fitMethod, stepsize)
s2_paradf = loadFxs.load_hm_parameter_estimates(expname, 2, hdrdata_sess2, modelname, fitMethod, stepsize)

figFxs.plot_parameter_reliability(modelname, s1_paradf, s2_paradf, subtitles)
plt.gcf().set_size_inches(5 * npara, 5)
plt.savefig(os.path.join("..", 'figures', expname, "%s_%s_stepsize%.2f_para_reliability_short.pdf"%(modelname, fitMethod, stepsize)))

HM_ids = list(set(s1_paradf.id) and set(s2_paradf.id))

# it doesn't seem to increase defre... either 

expname = 'passive'
modelname = 'QL2reset'
fitMethod = "whole"
stepsize = 0.5
s1_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, fitMethod, stepsize)
s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname, fitMethod, stepsize)

figFxs.plot_parameter_reliability(modelname, s1_paradf[np.isin(s1_paradf.id, HM_ids)].iloc[:,:-1], s2_paradf[np.isin(s2_paradf.id, HM_ids)].iloc[:,:-1], subtitles)
plt.gcf().set_size_inches(5 * npara, 5)
plt.savefig(os.path.join("..", 'figures', expname, "%s_%s_stepsize%.2f_para_reliability_short.pdf"%(modelname, fitMethod, stepsize)))



##################### reliability about selfreport measures
 if expname == "passive":
	s1_selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
	s2_selfdf = loadFxs.parse_group_selfreport(expname, 2, isplot = False)
	s1_selfdf = s1_selfdf[np.isin(s1_selfdf["id"], hdrdata_sess1["id"])]
	###################################### reliability of selfreport data #################
	selfreport_vars = ["UPPS", "BIS", "Motor", "Nonplanning", "Attentional", "attention", "cogstable", "motor", "perseverance", "selfcontrol", "cogcomplex"]
	df = pd.melt(s1_selfdf, id_vars = ["id"], value_vars = selfreport_vars).merge(
		pd.melt(s2_selfdf, id_vars = ["id"], value_vars = selfreport_vars), on = ["id", "variable"],
		suffixes = ["_sess1", "_sess2"])
	g = sns.FacetGrid(data = df, col = "variable", sharex= False, sharey = False)
	g.map(figFxs.my_regplot, "value_sess1", "value_sess2")
	g.savefig(os.path.join("..", 'figures', expname, "selfreport_reliability.pdf"))

	# let's do practice effects 
	df = analysisFxs.vstack_sessions(pd.melt(s1_selfdf[np.isin(s1_selfdf.id, s2_selfdf)], id_vars = ["id"], value_vars = selfreport_vars),
		pd.melt(s2_selfdf, id_vars = ["id"], value_vars = selfreport_vars))
	g = sns.FacetGrid(data = df, col = "variable", sharex= False, sharey = False)
	g.map(sns.swarmplot, "sess", "value")

	# this might be a useful table to show ....
		# selfreport_vars = ['NU', 'PU', 'PM', 'PS', 'SS', 'attention', 'cogstable', 'motor', 'perseverance', 'selfcontrol', 'cogcomplex', 'UPPS', 'BIS', 'GMK'] 
	selfdf = analysisFxs.hstack_sessions(s1_selfdf, s2_selfdf)
	to_be_tested_vars = list(zip([x + "_sess1" for x in expParas.selfreport_vars], [x + "_sess2" for x in expParas.selfreport_vars]))
	spearman_rho_, pearson_rho_, abs_icc_, con_icc_, n_, report = analysisFxs.calc_zip_reliability(selfdf, to_be_tested_vars)
	report.sort_values(by = "spearman_rho")


	########## additional measures ###########
	# n_sub = 4
	# # calc within-block adaptation using the non-parametric method
	# s1_stats['wb_adapt_np'] = s1_stats['end_wtw'] - s1_stats['init_wtw']
	# s2_stats['wb_adapt_np'] = s2_stats['end_wtw'] - s2_stats['init_wtw']

	# # calc within-block adaptation using AUC values
	# s1_stats['wb_adapt'] = s1_stats['auc'+str(n_sub)] - s1_stats['auc1']
	# s2_stats['wb_adapt'] = s2_stats['auc'+str(n_sub)] - s2_stats['auc1']

	# # calc std_wtw using the moving window method
	# s1_stats['std_wtw_mw'] = np.mean(s1_stats[['std_wtw' + str(i+1) for i in np.arange(n_sub)]]**2,axis = 1)**0.5
	# s2_stats['std_wtw_mw'] = np.mean(s2_stats[['std_wtw' + str(i+1) for i in np.arange(n_sub)]]**2,axis = 1)**0.5

	# # colvars = ['auc_end_start', 'auc', 'auc1', 'auc2', "auc_rh", 'std_wtw', 'std_wtw1', 'std_wtw2', "std_wtw_rh"]
	# # colvars = ['auc', "std_wtw", "std_wtw_mw", "init_wtw", "wb_adapt_np", 'wb_adapt']
	# colvars = ['auc', "std_wtw", "std_wtw_mw", "wb_adapt_np", "wb_adapt", "init_wtw"]
	# s1_HP = s1_stats.loc[s1_stats['condition'] == 'HP', colvars + ['id']]
	# s1_LP = s1_stats.loc[s1_stats['condition'] == 'LP', colvars + ['id']]
	# s1_df = s1_HP.merge(s1_LP, left_on = 'id', right_on = 'id', suffixes = ['_HP', "_LP"])


	# s2_HP = s2_stats.loc[s2_stats['condition'] == 'HP', colvars + ['id']]
	# s2_LP = s2_stats.loc[s2_stats['condition'] == 'LP', colvars + ['id']]
	# s2_df = s2_HP.merge(s2_LP, left_on = 'id', right_on = 'id', suffixes = ['_HP', "_LP"])

	# # add auc_delta and auc_ave
	# auc_vars = ['auc']
	# for var in ['auc']:
	#     s1_df[var + '_delta'] = s1_df.apply(func = lambda x: x[var + '_HP'] - x[var + '_LP'], axis = 1)
	#     s2_df[var + '_delta'] = s2_df.apply(func = lambda x: x[var + '_HP'] - x[var + '_LP'], axis = 1)
	#     s1_df[var + '_ave'] = (s1_df.apply(func = lambda x: x[var + '_HP'] + x[var + '_LP'], axis = 1)) / 2
	#     s2_df[var + '_ave'] = (s2_df.apply(func = lambda x: x[var + '_HP'] + x[var + '_LP'], axis = 1)) / 2


	######################
	## plot reliability for variables I am interested in ##
	######################







#########################
# plot practice effects 
########################


######### combine all conditions #########, also I might want to compare with other tasks ...


