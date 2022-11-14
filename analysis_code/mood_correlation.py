
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
import scipy 
import statsmodels.formula.api as smf
from plotnine import ggplot, aes, facet_grid, labs, geom_point, geom_errorbar, geom_text, position_dodge, scale_fill_manual, labs, theme_classic, ggsave, geom_bar, scale_x_discrete


# plot styles
plt.style.use('classic')
sns.set(font_scale = 2)
sns.set_style("white")
condition_palette = ["#762a83", "#1b7837"]

# passive version
expname = "passive"
hdrdata_sess1, trialdata_sess1_ = loadFxs.group_quality_check(expname, 1, plot_quality_check = True)
hdrdata_sess2, trialdata_sess2_ = loadFxs.group_quality_check(expname, 2, plot_quality_check = True)

########### let me only include participants complete both sessions
hdrdata_sess1 = hdrdata_sess1[np.isin(hdrdata_sess1["id"], hdrdata_sess2["id"])]
trialdata_sess1_ = {x: y for x,y in trialdata_sess1_.items() if x[0] in hdrdata_sess2["id"].values}

############ conduct behavioral analysis ######
s1_stats, s1_Psurv_b1_, s1_Psurv_b2_, s1_WTW_ = analysisFxs.group_MF(trialdata_sess1_, plot_each = False)   
s2_stats, s2_Psurv_b1_, s2_Psurv_b2_, s2_WTW_ = analysisFxs.group_MF(trialdata_sess2_, plot_each = False)   
s1_df = analysisFxs.pivot_by_condition(s1_stats)
s2_df = analysisFxs.pivot_by_condition(s2_stats)
# s1_df = s1_df.merge(s1_stats[["id", "sell_RT_mean"]], on = "id")
# s2_df = s2_df.merge(s2_stats[["id", "sell_RT_mean"]], on = "id")
statsdf = analysisFxs.agg_across_sessions(s1_df, s2_df)


####################### analyze only selfreport data ####################
if expname == "passive":
	s1_selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
	s2_selfdf = loadFxs.parse_group_selfreport(expname, 2, isplot = False)
	selfdf = analysisFxs.agg_across_sessions(s1_selfdf, s2_selfdf)
else:
	s1_selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
	s1_selfdf[np.isin(s1_selfdf["id"], hdrdata_sess1["id"])]
	selfdf = s1_selfdf	

######## load model parameters ########
modelname = 'QL2reset'
fitMethod = "whole"
stepsize = 0.5
subtitles = [r'$\mathbf{log(\alpha)}$', r'$\mathbf{log(\nu)}$', r'$\mathbf{\tau}$', r'$\mathbf{\gamma}$', r'$\mathbf{log(\eta)}$']
paranames = modelFxs.getModelParas(modelname)
npara = len(paranames)
s1_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, fitMethod, stepsize)
s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname, fitMethod, stepsize)
s1_paradf["alphaU"] = s1_paradf["alpha"] * s1_paradf["nu"]
s2_paradf["alphaU"] = s2_paradf["alpha"] * s2_paradf["nu"]
figFxs.log_transform_parameter(s1_paradf, ["alpha", "nu", "eta", "tau", "alphaU"])
figFxs.log_transform_parameter(s2_paradf, ["alpha", "nu", "eta", "tau", "alphaU"])
paradf = analysisFxs.agg_across_sessions(s1_paradf, s2_paradf)

################### self autocorrelations ###############


g = sns.pairplot(statsdf[["auc", "std_wtw", "auc_delta"]], kind = "reg", diag_kws = {"color": "grey", "edgecolor": "black"},\
	plot_kws ={'line_kws':{'color':'red'}, "scatter_kws": {"color": "grey", "edgecolor": "black"}})
g.map_lower(figFxs.annotate_reg)
plt.savefig(os.path.join("..", "figures", expname, "task_corr.pdf"))





############
df = statsdf.merge(selfdf, on = "id").merge(paradf, on = "id")
r_, p_ = analysisFxs.calc_prod_correlations(df, ['Motor', "Nonplanning", "Attentional"], ['auc', "std_wtw", "auc_delta"])
r_, p_ = analysisFxs.calc_prod_correlations(df, ['motor', "perseverance", "cogstable", "selfcontrol", "attention", "cogcomplex"], ['auc', "std_wtw", "auc_delta"])
r_, p_ = analysisFxs.calc_prod_correlations(df, ['BIS', "UPPS", "log_GMK"], ['auc', "std_wtw", "auc_delta"])
r_, p_ = analysisFxs.calc_prod_correlations(df, ["NU", "PU", "PM", "PS", "SS"], ["NU", "PU", "PM", "PS", "SS"])

r_, p_ = analysisFxs.calc_prod_correlations(df, ['BIS', "UPPS", "discount_logk"], ['BIS', "UPPS", "discount_logk"])
r_, p_ = analysisFxs.calc_prod_correlations(df, ["log_alpha", "log_alphaU", "tau", "gamma", "log_eta"], ["log_alpha", "log_alphaU", "tau", "gamma", "log_eta"])



r_, p_ = analysisFxs.calc_prod_correlations(df, ['auc', "std_wtw", "auc_delta"], ['auc', "std_wtw", "auc_delta"])
r_, p_ = analysisFxs.calc_prod_correlations(df, ['motor', "selfcontrol", "cogstable"], ['auc', "std_wtw", "auc_delta"])

r_, p_ = analysisFxs.calc_prod_correlations(df, ["Motor", "Nonplanning", "Attentional"], ['auc', "std_wtw", "auc_delta"])

r_, p_ = analysisFxs.calc_prod_correlations(df, ["BIS", "UPPS", "log_GMK"], ['auc', "std_wtw", "auc_delta"])

r_, p_ = analysisFxs.calc_prod_correlations(df, ['motor', "perseverance",  "cogcomplex", "selfcontrol", "attention", "cogstable"], ['auc', "std_wtw", "auc_delta"])

sns.pairplot(df[["Motor", "Nonplanning", "Attentional"]])
plt.savefig(os.path.join("..", "figures", expname, "BIS_corr.pdf"))

sns.pairplot(df[["log_alpha", "log_nu", "log_tau", "gamma", "log_eta"]]) # strong auto-correlations among parameters; 
plt.savefig(os.path.join("..", "figures", expname, "para_corr_%s_wtw_%s_stepsize%.2f.pdf"%(modelname, fitMethod, stepsize)))



################### regression correlations among selfreports and model measures ###############
source_ = []
sp_var_ = []
bh_var_ = []
coef_ = [] 
se_ = []
pval_ = []
impulse_vars = ["BIS", "UPPS",  "survey_impulsivity", "discount_logk"]
bh_vars = ["auc", "std_wtw", "auc_delta", "age"]
if expname == "passive":
	sources = ["sess1", "sess2", "combined"]
elif expname == "active":
	sources = ["sess1", "combined"]

# impulse_vars = ["NU", "PU", "PM", "PS", "SS"]
for source in sources:
	if source == "sess1":
		df = s1_df.merge(s1_selfdf, on = "id").merge(hdrdata_sess1[["id", "age", "gender"]], on = "id")
	elif source == "sess2":
		df = s2_df.merge(s2_selfdf, on = "id").merge(hdrdata_sess1[["id", "age", "gender"]], on = "id")
	elif source == "combined":
		df = statsdf.merge(selfdf, on = "id").merge(hdrdata_sess1[["id", "age", "gender"]], on = "id")
	df = df[df["gender"] != "Neither/Do not wish to disclose"]
	for impulse_var in impulse_vars:
		df[df.select_dtypes('number').columns] = df.select_dtypes('number').apply(lambda x:scipy.stats.zscore(x, nan_policy = "omit")) 
		results = smf.ols(impulse_var + ' ~ auc + auc_delta + std_wtw + age * gender', data=df).fit()
		source_ = source_ + [source] * len(results.params[1:].index.values)
		sp_var_ = sp_var_ + [impulse_var] * len(results.params[1:].index.values)
		bh_var_ = bh_var_ + list(results.params[1:].index.values)
		coef_ = coef_ + list(results.params[1:].values)
		se_ = se_ + list(results.bse[1:].values)
		pval_ = pval_ + list(results.pvalues[1:].values)

plotdf = pd.DataFrame({
	"source": source_,
	"bh_var": bh_var_,
	"sp_var": sp_var_,
	"coef": coef_,
	"se": se_,
	"ymax": [x + y for x, y in zip(coef_, se_)],
	"ymin": [x - y for x, y in zip(coef_, se_)],
	"pval": [round(x,3) for x in pval_],
	"label_y": [x + 0.03 * np.sign(x) for x in coef_]
	})
plotdf["source"] =  pd.Categorical(plotdf["source"], categories = ["sess1", "sess2", "combined"])
plotdf["bh_var"] =  pd.Categorical(plotdf["bh_var"], categories = results.params[1:].index.values)


p = 
(ggplot(plotdf[plotdf["source"] == "combined"]) \
	+ aes(x="bh_var", y="coef") \
	+ geom_bar(stat = "identity", position="dodge", width=0.75) \
	# + geom_errorbar(aes(ymin = "ymin", ymax = "ymax"), position = "dodge", width = 0.9)\
	+ facet_grid(facets="~sp_var") 
	+ geom_text(aes(y="label_y", label = "pval"), position=position_dodge(width=0.75)) 
	+ scale_fill_manual(values = ["#ca0020", "#0571b0", "#bababa"]) 
	+ labs(x = "", y = "Standardized coef") + theme_classic())

##################################
# basically all the results are gone ...
# ok survey impulsivity doesn't work 
source_ = []
sp_var_ = []
bh_var_ = []
coef_ = [] 
se_ = []
pval_ = []
impulse_vars = ["BIS", "UPPS",  "survey_impulsivity", "discount_logk"]
bh_vars = ["auc", "std_wtw", "auc_delta"]
if expname == "passive":
	sources = ["sess1", "sess2", "combined"]
elif expname == "active":
	sources = ["sess1", "combined"]

# impulse_vars = ["NU", "PU", "PM", "PS", "SS"]
for source in sources:
	if source == "sess1":
		df = s1_df.merge(s1_selfdf, on = "id").merge(hdrdata_sess1[["id", "age", "gender", "seq"]], on = "id")
	elif source == "sess2":
		df = s2_df.merge(s2_selfdf, on = "id").merge(hdrdata_sess1[["id", "age", "gender", "seq"]], on = "id")
	elif source == "combined":
		df = statsdf.merge(selfdf, on = "id").merge(hdrdata_sess1[["id", "age", "gender", "seq"]], on = "id")
	df = df[df["gender"] != "Neither/Do not wish to disclose"]
	# df = df[df["age"] < 50]
	# df = df[1:140]
	for bh_var in bh_vars:
		df[df.select_dtypes('number').columns] = df.select_dtypes('number').apply(lambda x:scipy.stats.zscore(x, nan_policy = "omit")) 
		results = smf.ols(bh_var + ' ~ motor + age * gender * seq', data=df).fit()
		source_ = source_ + [source] * len(results.params[1:])
		sp_var_ = sp_var_ + list(results.params[1:].index.values)
		bh_var_ = bh_var_ + [bh_var] * len(results.params[1:])
		coef_ = coef_ + list(results.params[1:].values)
		se_ = se_ + list(results.bse[1:].values)
		pval_ = pval_ + list(results.pvalues[1:].values)

plotdf = pd.DataFrame({
	"source": source_,
	"bh_var": bh_var_,
	"sp_var": sp_var_,
	"coef": coef_,
	"se": se_,
	"ymax": [x + y for x, y in zip(coef_, se_)],
	"ymin": [x - y for x, y in zip(coef_, se_)],
	"pval": [round(x,3) for x in pval_],
	"label_y": [x + 0.03 * np.sign(x) for x in coef_]
	})
plotdf["source"] =  pd.Categorical(plotdf["source"], categories = ["sess1", "sess2", "combined"])
plotdf["sp_var"] =  pd.Categorical(plotdf["sp_var"], categories = results.params[1:].index.values)
plotdf[np.logical_and(plotdf["source"] == "combined", plotdf["sp_var"] == "motor")]



# only combined sessions
(ggplot(plotdf[plotdf["source"] == "combined"]) \
	+ aes(x="sp_var", y="coef") \
	+ geom_bar(stat = "identity", position="dodge", width=0.75) \
	# + geom_errorbar(aes(ymin = "ymin", ymax = "ymax"), position = "dodge", width = 0.9)\
	+ facet_grid(facets="~bh_var") 
	+ geom_text(aes(y="label_y", label = "pval"), position=position_dodge(width=0.75)) 
	+ scale_fill_manual(values = ["#ca0020", "#0571b0", "#bababa"]) 
	+ labs(x = "", y = "Standardized coef") + theme_classic())
+ 
	  scale_x_discrete(labels= ["BUP", "log_GMK"]))
ggsave(plot = p, filename= "../figures/%s/coef_combined_BIS.pdf"%(expname))

results = smf.ols('auc_delta ~ age * gender + seq', data=df[df["age"]<50]).fit()


# all sessions 
p = (ggplot(plotdf) \
	+ aes(x="sp_var", y="coef", fill = "source") \
	+ geom_bar(stat = "identity", position="dodge", width=0.75) \
	# + geom_errorbar(aes(ymin = "ymin", ymax = "ymax"), position = "dodge", width = 0.9)\
	+ facet_grid(facets="~bh_var") 
	+ geom_text(aes(y="label_y", label = "pval"), position=position_dodge(width=0.75)) 
	+ scale_fill_manual(values = ["#ca0020", "#0571b0", "#bababa"]) 
	+ labs(x = "", y = "Standardized coef") +
	scale_x_discrete(labels= ["BUP", "log_GMK", "age"]))
ggsave(plot = p, filename= "../figures/%s/coef_all-sessions_BIS.pdf"%(expname))
