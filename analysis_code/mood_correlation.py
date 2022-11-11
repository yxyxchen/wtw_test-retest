
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
from plotnine import ggplot, aes, facet_grid, labs, geom_point, geom_errorbar, geom_text, position_dodge, scale_fill_manual, labs, theme_classic, ggsave, geom_bar

def tosig(x):
	if x > 0.05:
		y = ""
	elif x <= 0.05 and x > 0.01:
		y = "*"
	elif x <= 0.01 and x > 0.001:
		y = "**"
	else:
		y = "***"
	return y

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
s1_df = s1_df.merge(s1_stats[["id", "sell_RT_mean"]], on = "id")
s2_df = s2_df.merge(s2_stats[["id", "sell_RT_mean"]], on = "id")
statsdf = analysisFxs.agg_across_sessions(s1_df, s2_df)


####################### analyze only selfreport data ####################
if expname == "passive":
	s1_selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
	s2_selfdf = loadFxs.parse_group_selfreport(expname, 2, isplot = False)
	s1_selfdf = s1_selfdf[np.isin(s1_selfdf["id"], hdrdata_ssess1["id"])]
###################################### reliability of selfreport data #################
	selfreport_vars = ["UPPS", "BIS", "GMK", "PAS", "NAS"]
	df = pd.melt(s1_selfdf, id_vars = ["id"], value_vars = selfreport_vars).merge(
		pd.melt(s2_selfdf, id_vars = ["id"], value_vars = selfreport_vars), on = ["id", "variable"],
		suffixes = ["_sess1", "_sess2"])
	g = sns.FacetGrid(data = df, col = "variable", sharex= False, sharey = False)
	g.map(figFxs.my_regplot, "value_sess1", "value_sess2")

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

	selfdf = analysisFxs.agg_across_sessions(s1_selfdf, s2_selfdf)
else:
	s1_selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
	s1_selfdf[np.isin(s1_selfdf["id"], hdrdata_sess1["id"])]
	selfdf = s1_selfdf	
	selfdf['log_GMK'] = np.log(selfdf['GMK'])
	selfdf["BUP"] = selfdf["BIS"] + selfdf["UPPS"]
	


##### showing demographic results 
tmp = hdrdata_sess2["gender"].value_counts()
fig, ax = plt.subplots()
ax.pie(tmp.values, labels = tmp.index, autopct='%.0f%%')
fig.savefig(os.path.join("..", "figures", expname, "gender_pie.pdf"))


fig, ax = plt.subplots()
ax.hist(hdrdata_sess2["age"], color = "grey", edgecolor = "black")
fig.savefig(os.path.join("..", "figures", expname, "age_dist.pdf"))


df = statsdf.merge(selfdf, on = "id").merge(hdrdata_sess1[["id", "age", "gender", "cb"]], on = "id")
df["seq"] = ["seq1" if x in ("A", "C") else "seq2" for x in df["cb"]]
smf.ols('auc  ~ gender + age + sell_RT_mean + seq', data=df).fit().summary()
df.loc[df["gender"] == "Neither/Do not wish to disclose", "gender"] = np.nan
plotdf = df.melt(id_vars = ["id", "gender", "cb"], value_vars = ["auc", "std_wtw", "auc_delta", "age", "BUP", "log_GMK", "sell_RT_mean"])
plotdf["gender"] = pd.Categorical(plotdf["gender"],categories = ["Male", "Female"] )
plotdf["seq"] = [1 if x in ("A", "C") else 2 for x in plotdf["cb"]]
fig, axes = plt.subplots(1, 7)
for ax, var in zip(axes, ["auc", "std_wtw", "auc_delta", "age", "BUP", "log_GMK", "sell_RT_mean"]):
	sns.barplot(data = plotdf[plotdf['variable'] == var], x = "gender", y = "value", ax = ax)
	ax.set_ylabel(var)

# ...
var = "auc_delta"
mannwhitneyu(df.loc[df["seq"] == "seq1", var], df.loc[df["seq"] == "seq2", var], alternative="two-sided")


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
g = sns.pairplot(selfdf[["UPPS", "BIS", "log_GMK"]], kind = "reg", diag_kws = {"color": "grey", "edgecolor": "black"},\
	plot_kws ={'line_kws':{'color':'red'}, "scatter_kws": {"color": "grey", "edgecolor": "black"}})
g.map_lower(figFxs.annotate_reg)
plt.savefig(os.path.join("..", "figures", expname, "impulsivity_corr.pdf"))

g = sns.pairplot(selfdf[["NU", "PU", "PM", "PS", "SS"]], kind = "reg", diag_kws = {"color": "grey", "edgecolor": "black"},\
	plot_kws ={'line_kws':{'color':'red'}, "scatter_kws": {"color": "grey", "edgecolor": "black"}})
g.map_lower(figFxs.annotate_reg)
plt.savefig(os.path.join("..", "figures", expname, "UPPS_corr.pdf"))

g = sns.pairplot(selfdf[["Motor", "Nonplanning", "Attentional"]], kind = "reg", diag_kws = {"color": "grey", "edgecolor": "black"},\
	plot_kws ={'line_kws':{'color':'red'}, "scatter_kws": {"color": "grey", "edgecolor": "black"}})
g.map_lower(figFxs.annotate_reg)
plt.savefig(os.path.join("..", "figures", expname, "BIS_corr.pdf"))

g = sns.pairplot(selfdf[['motor', "perseverance", "cogstable", "selfcontrol", "attention", "cogcomplex"]], kind = "reg", diag_kws = {"color": "grey", "edgecolor": "black"},\
	plot_kws ={'line_kws':{'color':'red'}, "scatter_kws": {"color": "grey", "edgecolor": "black"}})
g.map_lower(figFxs.annotate_reg)
plt.savefig(os.path.join("..", "figures", expname, "BIS_sub_corr.pdf"))

############

df = statsdf.merge(selfdf, on = "id").merge(paradf, on = "id")
r_, p_ = analysisFxs.calc_prod_correlations(df, ['Motor', "Nonplanning", "Attentional"], ['auc', "std_wtw", "auc_delta"])
r_, p_ = analysisFxs.calc_prod_correlations(df, ['motor', "perseverance", "cogstable", "selfcontrol", "attention", "cogcomplex"], ['auc', "std_wtw", "auc_delta"])
r_, p_ = analysisFxs.calc_prod_correlations(df, ['BIS', "UPPS", "log_GMK"], ['auc', "std_wtw", "auc_delta"])
r_, p_ = analysisFxs.calc_prod_correlations(df, ["NU", "PU", "PM", "PS", "SS"], ["NU", "PU", "PM", "PS", "SS"])

r_, p_ = analysisFxs.calc_prod_correlations(df, ['BIS', "UPPS", "log_GMK"], ["log_alpha", "log_alphaU", "tau", "gamma", "log_eta"])
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

####### PCA ##### 
# let me use original items maybe ?
from sklearn.decomposition import PCA
pca = PCA(n_components=5,svd_solver='auto')
# s1_selfdf_norm = s1_selfdf.select_dtypes(include = "number")[['Motor', "Nonplanning", "Attentional"]]
# s1_selfdf_norm = s1_selfdf_norm.apply(lambda x:scipy.stats.zscore(x[~np.isnan(x)]))
# s1_selfdf_norm = s1_selfdf_norm.dropna(axis = 0)
s1_paradf_norm = copy.copy(s1_paradf.select_dtypes(include = "number"))
s1_paradf_norm =s1_paradf_norm.apply(lambda x:scipy.stats.zscore(x[~np.isnan(x)])).drop(["waic", "log_alphaU"], axis = 1)
survey_pca = pd.DataFrame(pca.fit_transform(s1_paradf_norm ),columns= ["PC" + str(x) for x in np.arange(5)])
print(pca.explained_variance_ratio_)

loadings = pd.DataFrame(pca.components_.T, columns = ["PC" + str(x) for x in np.arange(5)], index=s1_paradf_norm.columns.values.tolist())
loadings


################### regression correlations among selfreports and model measures ###############
source_ = []
sp_var_ = []
bh_var_ = []
coef_ = [] 
se_ = []
pval_ = []
impulse_vars = ["BIS", "UPPS",  "BUP", "log_GMK"]
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
		results = smf.ols(impulse_var + ' ~ auc + std_wtw + auc_delta + sell_RT_mean + age', data=df).fit()
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
	"pval": [tosig(x) for x in pval_],
	"label_y": [x + 0.03 * np.sign(x) for x in coef_]
	})
plotdf["source"] =  pd.Categorical(plotdf["source"], categories = ["sess1", "sess2", "combined"])
plotdf["bh_var"] =  pd.Categorical(plotdf["bh_var"], categories = results.params[1:].index.values)



#########################
source_ = []
sp_var_ = []
bh_var_ = []
coef_ = [] 
se_ = []
pval_ = []
impulse_vars = ["BIS", "UPPS",  "BUP", "log_GMK"]
bh_vars = ["auc", "std_wtw", "auc_delta"]
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
	for bh_var in bh_vars:
		df[df.select_dtypes('number').columns] = df.select_dtypes('number').apply(lambda x:scipy.stats.zscore(x, nan_policy = "omit")) 
		results = smf.ols(bh_var + ' ~ BUP + log_GMK', data=df).fit()
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
	"pval": [tosig(x) for x in pval_],
	"label_y": [x + 0.03 * np.sign(x) for x in coef_]
	})
plotdf["source"] =  pd.Categorical(plotdf["source"], categories = ["sess1", "sess2", "combined"])
plotdf["sp_var"] =  pd.Categorical(plotdf["sp_var"], categories = results.params[1:].index.values)


# only combined sessions
p = 
(ggplot(plotdf[plotdf["source"] == "combined"]) \
	+ aes(x="sp_var", y="coef") \
	+ geom_bar(stat = "identity", position="dodge", width=0.75) \
	# + geom_errorbar(aes(ymin = "ymin", ymax = "ymax"), position = "dodge", width = 0.9)\
	+ facet_grid(facets="~bh_var") 
	+ geom_text(aes(y="label_y", label = "pval"), position=position_dodge(width=0.75)) 
	+ scale_fill_manual(values = ["#ca0020", "#0571b0", "#bababa"]) 
	+ labs(x = "", y = "Standardized coef") + theme_classic() + 
	  scale_x_discrete(labels= ["BUP", "log_GMK"]))
ggsave(plot = p, filename= "../figures/%s/coef_combined_BIS.pdf"%(expname))


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
