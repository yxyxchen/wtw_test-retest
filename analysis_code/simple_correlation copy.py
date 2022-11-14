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
from scipy.stats import mannwhitneyu


# plot styles
plt.style.use('classic')
sns.set(font_scale = 1)
sns.set_style("white")
condition_palette = ["#762a83", "#1b7837"]

# passive version
expname = "active"
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
statsdf = analysisFxs.agg_across_sessions(s1_df, s2_df)


# effect of seq, color and cb on gender 
df = hdrdata_sess2[np.isin(hdrdata_sess2["gender"], ["Female", "Male"])]
chi2, p, dof, ex = scipy.stats.chi2_contingency(df.groupby(["seq"])["gender"].value_counts().rename("sum").reset_index().pivot(index = "seq", columns = "gender",values = "sum"))
p
chi2, p, dof, ex = scipy.stats.chi2_contingency(df.groupby(["color"])["gender"].value_counts().rename("sum").reset_index().pivot(index = "color", columns = "gender",values = "sum"))
p
chi2, p, dof, ex = scipy.stats.chi2_contingency(df.groupby(["cb"])["gender"].value_counts().rename("sum").reset_index().pivot(index = "cb", columns = "gender",values = "sum"))
p


fig, axes = plt.subplots(1,3)
for var, ax in zip(["seq", "color", "cb"], axes): 
	var_levels = np.unique(hdrdata_sess2[var])
	sns.violinplot(data = hdrdata_sess2, x = var, y = "age", order = var_levels, ax = ax)
	sns.boxplot(data = hdrdata_sess2, x=var, y='age', order = var_levels, saturation=0.5, width=0.4, boxprops={'zorder': 2}, ax=ax)
	ax.set_xlabel("")
	for i, j in itertools.combinations(np.arange(len(var_levels)), 2):
		sig = figFxs.tosig(mannwhitneyu(hdrdata_sess2.loc[hdrdata_sess2[var] == var_levels[i], "age"], hdrdata_sess2.loc[hdrdata_sess2[var] == var_levels[j], "age"]).pvalue)
		ax.plot([i, i, i+1, i+1], [80, 84, 84, 80], lw=1.5)
		plt.text((2*i + 1)*.5, 81, sig, ha='center', va='bottom')
fig.savefig(os.path.join("..", "figures", expname, "age_seq.pdf"))


fig, axes = plt.subplots(1,3)
for var, ax in zip(["seq", "color", "cb"], axes):
	df = hdrdata_sess2[np.isin(hdrdata_sess2["gender"], ["Female", "Male"])]
	df = df.groupby(var)["gender"].value_counts(normalize = True).rename("percent").reset_index().pivot(index = var,  columns = "gender")
	df.plot(kind='bar', stacked=True, ax = ax)
fig.savefig(os.path.join("..", "figures", expname, "gender_seq.pdf"))


########################## effects of seq on the tasks 
for groupvar in ["seq", "color", "cb"]:
	plt.style.use('classic')
	sns.set(font_scale = 1)
	sns.set_style("white")
	condition_palette = ["#762a83", "#1b7837"]
	df = statsdf.merge(hdrdata_sess2, on = "id")
	tmp = df.melt(id_vars = ["id", groupvar], value_vars = ["auc", "std_wtw", "auc_delta"])
	var_levels = np.unique(df[groupvar])
	g = sns.FacetGrid(data = tmp, col = "variable", sharey = False)
	g.map(sns.barplot, groupvar, "value", color = "grey")
	for var, ax in zip(["auc", "std_wtw", "auc_delta"], g.axes.flatten()):
		for i, j in itertools.combinations(np.arange(len(var_levels)), 2):
			sig = figFxs.tosig(mannwhitneyu(tmp.loc[np.logical_and(tmp[groupvar] == var_levels[i], tmp["variable"] == var), "value"], tmp.loc[np.logical_and(tmp[groupvar] == var_levels[j], tmp["variable"] == var), "value"]).pvalue)
			ymax = tmp.loc[tmp["variable"] == var].groupby(groupvar)["value"].mean().max()
			ax.plot([i, i, i+1, i+1], [ymax, ymax*1.2, ymax*1.2, ymax], lw=1.5)
			ax.text((2*i + 1)*.5, ymax, sig, ha='center', va='bottom')
	g.savefig(os.path.join("..", "figures", expname, groupvar + "_task_comparison.pdf"))
	# for ax, var in zip(g.axes.flatten(), ["auc", "std_wtw", "auc_delta"]):
	# 	sig = figFxs.tosig(mannwhitneyu(df.loc[df[groupvar] == "seq1", var].dropna(), df.loc[df["seq"] == "seq2", var].dropna()).pvalue)
	# 	
	# 	print(ymax)
	# 	ax.plot([0, 0, 1, 1], [ymax * 1.1, ymax * 1.2, ymax * 1.2, ymax * 1.1], lw=1.5)
	# 	ax.text(.5, ymax * 1.2, sig, ha='center', va='bottom')
	

################## task vars ##################
# there might be an interaction, I don't think it will be that important
# plot styles
plt.style.use('classic')
sns.set(font_scale = 1)
sns.set_style("white")
condition_palette = ["#762a83", "#1b7837"]
df = statsdf.merge(hdrdata_sess2, on = "id")
df = df[np.isin(df["gender"], ["Male", "Female"])]
tmp = df.melt(id_vars = ["id", "gender"], value_vars = ["auc", "std_wtw", "auc_delta"])
g = sns.FacetGrid(data = tmp, col = "variable", sharey = False)
g.map(sns.barplot, "gender", "value", color = "grey")
for ax, var in zip(g.axes.flatten(), ["auc", "std_wtw", "auc_delta"]):
	sig = figFxs.tosig(mannwhitneyu(df.loc[df["gender"] == "Female", var].dropna(), df.loc[df["gender"] == "Male", var].dropna()).pvalue)
	ymax = df.loc[~np.isnan(df[var])].groupby(["gender"])[var].mean().max()
	print(ymax)
	ax.plot([0, 0, 1, 1], [ymax * 1.1, ymax * 1.2, ymax * 1.2, ymax * 1.1], lw=1.5)
	ax.text(.5, ymax * 1.2, sig, ha='center', va='bottom')
g.savefig(os.path.join("..", "figures", expname, "gender_task_comparison.pdf"))


df = statsdf.merge(hdrdata_sess2, on = "id")
df = df[np.isin(df["gender"], ["Male", "Female"])]
# df = df[df["age"] < 50]
tmp = df.melt(id_vars = ["id", "age", "gender"], value_vars = ["auc", "std_wtw", "auc_delta"])
g = sns.FacetGrid(data = tmp, col = "variable", sharey = False, margin_titles=True )
g.map(sns.regplot, "age", "value", line_kws = {'color':'red'}, scatter_kws = {"color": "grey", "edgecolor": "black"})
g.map(figFxs.annotate_reg, "age", "value")
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.savefig(os.path.join("..", "figures", expname, "age_task_corr.pdf"))


df = statsdf.merge(hdrdata_sess2, on = "id")
df = df[np.isin(df["gender"], ["Male", "Female"])]
tmp = df.melt(id_vars = ["id", "age", "gender"], value_vars = ["auc", "std_wtw", "auc_delta"])
g = sns.FacetGrid(data = tmp, row = "gender", col = "variable", sharey = False, margin_titles=True)
g.map(sns.regplot, "age", "value", line_kws = {'color':'red'}, scatter_kws = {"color": "grey", "edgecolor": "black"})
g.map(figFxs.annotate_reg, "age", "value")
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.savefig(os.path.join("..", "figures", expname, "age_gender_task_corr.pdf"))


df = statsdf.merge(hdrdata_sess2, on = "id")
df = df[np.isin(df["gender"], ["Female", "Male"])]
df = df[df["age"] < 50]
yvals = [ "auc", "std_wtw", "auc_delta"]
coef = []
for yval in yvals:
	results = smf.ols(yval + " ~ age * gender + seq", data = df).fit()
	coef.append(["%.3f( "%x + "p=%.3f"%y + " )" for x, y in zip(results.params[1:].values, results.pvalues[1:].values)])

predictors = results.pvalues[1:].index.values
coef_report = pd.DataFrame(coef).rename(index = dict(zip(np.arange(len(yvals)), yvals)), columns = dict(zip(np.arange(len(predictors)), predictors)))
coef_report



############## load model parameters ###########
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


################### simple correlations among selfreports and model measures ###############
source_ = []
sp_var_ = []
bh_var_ = []
coef_ = [] 
pval_ = []
impulse_vars = ["BIS", "UPPS", "discount_logk"]
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
	r_, p_ = analysisFxs.calc_prod_correlations(df, impulse_vars, ['auc', "std_wtw", "auc_delta"])
	coef_ = coef_ + list(r_.reset_index().melt(id_vars = ["index"])["value"])
	sp_var_ = sp_var_ + list(r_.reset_index().melt(id_vars = ["index"])["index"])
	bh_var_ = bh_var_ + list(r_.reset_index().melt(id_vars = ["index"])["variable"])
	pval_ = pval_ + list(p_.reset_index().melt(id_vars = ["index"])["value"])
	source_ = source_ + [source] * len(impulse_vars) * 3

plotdf = pd.DataFrame({
	"source": source_,
	"bh_var": bh_var_,
	"sp_var": sp_var_,
	"coef": coef_,
	"pval": [figFxs.tosig(x) for x in pval_],
	"label_y": [x + 0.03 * np.sign(x) for x in coef_]
	})
plotdf["source"] =  pd.Categorical(plotdf["source"], categories = ["sess1", "sess2", "combined"])
plotdf["bh_var"] =  pd.Categorical(plotdf["bh_var"], categories = ["auc", "std_wtw", "auc_delta"])

# only combined sessions
p = (ggplot(plotdf[plotdf["source"] == "combined"]) \
	+ aes(x="bh_var", y="coef") \
	+ geom_bar(stat = "identity", position="dodge", width=0.75) \
	# + geom_errorbar(aes(ymin = "ymin", ymax = "ymax"), position = "dodge", width = 0.9)\
	+ facet_grid(facets="~sp_var") 
	+ geom_text(aes(y="label_y", label = "pval"), position=position_dodge(width=0.75)) 
	+ scale_fill_manual(values = ["#ca0020", "#0571b0", "#bababa"]) 
	+ labs(x = "", y = r"$Spearman's \rho$") + theme_classic() 
	  scale_x_discrete(labels= ["AUC", r"$\sigma_{wtw}$", r"$\Delta AUC$"]))
ggsave(plot = p, filename= "../figures/%s/simple_coef_combined_BIS.pdf"%(expname))


# all sessions 
p = (ggplot(plotdf) \
	+ aes(x="bh_var", y="coef", fill = "source") \
	+ geom_bar(stat = "identity", position="dodge", width=0.75) \
	# + geom_errorbar(aes(ymin = "ymin", ymax = "ymax"), position = "dodge", width = 0.9)\
	+ facet_grid(facets="~sp_var") 
	+ geom_text(aes(y="label_y", label = "pval"), position=position_dodge(width=0.75)) 
	+ scale_fill_manual(values = ["#ca0020", "#0571b0", "#bababa"]) 
	+ labs(x = "", y = r"$Spearman's \rho$") +
	scale_x_discrete(labels= ["AUC", r"$\sigma_{wtw}$", r"$\Delta AUC$"]))
ggsave(plot = p, filename= "../figures/%s/simple_coef_all-sessions_BIS.pdf"%(expname))


######################## regression correlations among selfreports and model measures #########
source_ = []
sp_var_ = []
bh_var_ = []
coef_ = [] 
se_ = []
pval_ = []
impulse_vars = ["BIS", "UPPS", "log_GMK"]
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
	for bh_var in impulse_vars:
		df[df.select_dtypes('number').columns] = df.select_dtypes('number').apply(lambda x:scipy.stats.zscore(x, nan_policy = "omit")) 
		results = smf.ols(impulse_var + ' ~ auc + std_wtw + auc_delta + age', data=df).fit()
		source_ = source_ + [source] * len(bh_vars)
		sp_var_ = sp_var_ + [impulse_var] * len(bh_vars)
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

# only combined sessions
p = (ggplot(plotdf[plotdf["source"] == "combined"]) \
	+ aes(x="bh_var", y="coef") \
	+ geom_bar(stat = "identity", position="dodge", width=0.75) \
	# + geom_errorbar(aes(ymin = "ymin", ymax = "ymax"), position = "dodge", width = 0.9)\
	+ facet_grid(facets="~sp_var") 
	+ geom_text(aes(y="label_y", label = "pval"), position=position_dodge(width=0.75)) 
	+ scale_fill_manual(values = ["#ca0020", "#0571b0", "#bababa"]) 
	+ labs(x = "", y = "Standardized coef") + theme_classic() + 
	  scale_x_discrete(labels= ["AUC", r"$\sigma_{wtw}$", r"$\Delta AUC$", "age"]))
ggsave(plot = p, filename= "../figures/%s/coef_combined_BIS.pdf"%(expname))


# all sessions 
p = (ggplot(plotdf) \
	+ aes(x="bh_var", y="coef", fill = "source") \
	+ geom_bar(stat = "identity", position="dodge", width=0.75) \
	# + geom_errorbar(aes(ymin = "ymin", ymax = "ymax"), position = "dodge", width = 0.9)\
	+ facet_grid(facets="~sp_var") 
	+ geom_text(aes(y="label_y", label = "pval"), position=position_dodge(width=0.75)) 
	+ scale_fill_manual(values = ["#ca0020", "#0571b0", "#bababa"]) 
	+ labs(x = "", y = "Standardized coef") +
	scale_x_discrete(labels= ["AUC", r"$\sigma_{wtw}$", r"$\Delta AUC$"]))
ggsave(plot = p, filename= "../figures/%s/coef_all-sessions_BIS.pdf"%(expname))


################### regression correlations among selfreports and model measures ###############
source_ = []
sp_var_ = []
bh_var_ = []
coef_ = [] 
se_ = []
pval_ = []
impulse_vars = ["BIS", "UPPS", "log_GMK"]
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
		results = smf.ols(impulse_var + ' ~ auc + std_wtw + auc_delta + age', data=df).fit()
		source_ = source_ + [source] * len(bh_vars)
		sp_var_ = sp_var_ + [impulse_var] * len(bh_vars)
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

# only combined sessions
p = (ggplot(plotdf[plotdf["source"] == "combined"]) \
	+ aes(x="bh_var", y="coef") \
	+ geom_bar(stat = "identity", position="dodge", width=0.75) \
	# + geom_errorbar(aes(ymin = "ymin", ymax = "ymax"), position = "dodge", width = 0.9)\
	+ facet_grid(facets="~sp_var") 
	+ geom_text(aes(y="label_y", label = "pval"), position=position_dodge(width=0.75)) 
	+ scale_fill_manual(values = ["#ca0020", "#0571b0", "#bababa"]) 
	+ labs(x = "", y = "Standardized coef") + theme_classic() + 
	  scale_x_discrete(labels= ["AUC", r"$\sigma_{wtw}$", r"$\Delta AUC$", "age"]))
ggsave(plot = p, filename= "../figures/%s/coef_combined_BIS.pdf"%(expname))


# all sessions 
p = (ggplot(plotdf) \
	+ aes(x="bh_var", y="coef", fill = "source") \
	+ geom_bar(stat = "identity", position="dodge", width=0.75) \
	# + geom_errorbar(aes(ymin = "ymin", ymax = "ymax"), position = "dodge", width = 0.9)\
	+ facet_grid(facets="~sp_var") 
	+ geom_text(aes(y="label_y", label = "pval"), position=position_dodge(width=0.75)) 
	+ scale_fill_manual(values = ["#ca0020", "#0571b0", "#bababa"]) 
	+ labs(x = "", y = "Standardized coef") +
	scale_x_discrete(labels= ["AUC", r"$\sigma_{wtw}$", r"$\Delta AUC$"]))
ggsave(plot = p, filename= "../figures/%s/coef_all-sessions_BIS.pdf"%(expname))

############################# look at subscores ###########
# look at subscores 
source_ = []
sp_var_ = []
bh_var_ = []
coef_ = [] 
se_ = []
pval_ = []
impulse_vars = ["Motor", "Attentional", "Nonplanning"]
for source in ["sess1", "combined"]:
	if source == "sess1":
		df = s1_df.merge(selfdf, on = "id")
	# 	df = s1_df.merge(s1_selfdf, on = "id").merge(s1_paradf, on = "id")
	# elif source == "sess2":
	# 	df = s2_df.merge(s2_selfdf, on = "id").merge(s2_paradf, on = "id")
	else:
		df = statsdf.merge(selfdf, on = "id")
	for impulse_var in impulse_vars:
		df[df.select_dtypes('number').columns] = df.select_dtypes('number').apply(scipy.stats.zscore) 
		results = smf.ols(impulse_var + ' ~ auc + std_wtw + auc_delta', data=df).fit()
		source_ = source_ + [source] * len(impulse_vars)
		sp_var_ = sp_var_ + [impulse_var] * len(impulse_vars)
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
	"pval": [tosig(x) for x in pval_],
	"ymin": [ x - se for x, se in zip(coef_, se_)],
	"ymax": [ x + se for x, se in zip(coef_, se_)],
	"label_y": [x + 0.03 * np.sign(x) for x in coef_]
	})
plotdf["source"] =  pd.Categorical(plotdf["source"], categories = ["sess1", "sess2", "combined"])
plotdf["bh_var"] =  pd.Categorical(plotdf["bh_var"], categories = ["auc", "std_wtw", "auc_delta"])

(ggplot(plotdf) \
	+ aes(x="bh_var", y="coef", fill = "source") \
	+ geom_bar(stat = "identity", position="dodge", width=0.75) \
	# + geom_errorbar(aes(ymin = "ymin", ymax = "ymax"), position = "dodge", width = 0.9)\
	+ facet_grid(facets="~sp_var") 
	+ geom_text(aes(y="label_y", label = "pval"), position=position_dodge(width=0.75)) 
	+ scale_fill_manual(values = ["#ca0020", "#0571b0", "#bababa"]) 
	+ labs(x = "", y = "Standardized coefficient"))


(ggplot(plotdf) \
	+ aes(x="source", y="coef", fill = "bh_var") \
	+ geom_bar(stat = "identity", position="dodge", width=0.75) \
	# + geom_errorbar(aes(ymin = "ymin", ymax = "ymax"), position = "dodge", width = 0.9)\
	+ facet_grid(facets="~sp_var") 
	+ geom_text(aes(y="label_y", label = "pval"), position=position_dodge(width=0.75)) 
	+ scale_fill_manual(values = ["#ca0020", "#0571b0", "#bababa"]) 
	+ labs(x = "", y = "Standardized coefficient"))


### using only combined data 
p = (ggplot(plotdf[plotdf["source"] == "combined"]) \
	+ aes(x="bh_var", y="coef", fill = "bh_var") \
	+ geom_bar(stat = "identity", position="dodge", width=0.75) \
	# + geom_errorbar(aes(ymin = "ymin", ymax = "ymax"), position = "dodge", width = 0.9)\
	+ facet_grid(facets="~sp_var") 
	+ geom_text(aes(y="label_y", label = "pval"), position=position_dodge(width=0.75)) 
	+ scale_fill_manual(values = ["#ca0020", "#0571b0", "#bababa"]) 
	+ labs(x = "", y = "Standardized coefficient") + theme_classic())
ggsave(plot = p, filename= "../figures/%s/coef_combined.pdf"%(expname))


###################### is this results reliably? ######
############
# small motor 
# df = s2_df.merge(s2_selfdf, on = "id").merge(s2_paradf, on = "id")
df = statsdf.merge(selfdf, on = "id").merge(paradf, on = "id")
results = smf.ols('log_GMK ~ log_alpha + log_nu', data=df).fit() # 
print(results.summary())

###### ABC #####


####### 
df = s2_df.merge(s2_selfdf, on = "id").merge(s2_paradf, on = "id")
df = statsdf.merge(selfdf, on = "id").merge(paradf, on = "id")
results = smf.ols('log_GMK ~ log_alpha + log_alphaU + gamma + log_eta + tau', data=df).fit()
print(results.summary())


df = s2_df.merge(s2_selfdf, on = "id").merge(s2_paradf, on = "id")
results = smf.ols('motor ~ auc + std_wtw + auc_delta', data=df).fit()
print(results.summary())



###### attentional, works for the first session, maybe in the first session it is the exploration, and in the second session it is mainly ...?
df = s1_df.merge(s1_selfdf, on = "id").merge(s1_paradf, on = "id")
results = smf.ols('attention ~ auc + std_wtw + auc_delta', data=df).fit()
print(results.summary())


df = s2_df.merge(s2_selfdf, on = "id").merge(s2_paradf, on = "id")
results = smf.ols('attention ~ auc + std_wtw + auc_delta', data=df).fit()
print(results.summary())


############## So it seems that different aspects of impulsivity influence different aspects of ##########
df = s1_df.merge(s1_selfdf, on = "id").merge(s1_paradf, on = "id")
results = smf.ols('cogcomplex ~ auc + std_wtw + auc_delta', data=df).fit()
print(results.summary())

df = s2_df.merge(s2_selfdf, on = "id").merge(s2_paradf, on = "id")
results = smf.ols('cogcomplex ~ auc + std_wtw + auc_delta', data=df).fit()
print(results.summary())



########## XXXX ######
results = smf.ols('log_GMK ~ auc + std_wtw + auc_delta', data=df).fit()
print(results.summary())



results = smf.ols('log_GMK ~ log_nu', data=df).fit()
print(results.summary())


df = statsdf.merge(selfdf, on = "id").merge(paradf, on = "id")
results = smf.ols('log_GMK ~ log_alpha', data=df).fit() # I think there is a correlation with log nu? 
print(results.summary())


df = s1_df.merge(s1_selfdf, on = "id").merge(s1_paradf, on = "id")
results = smf.ols('Motor ~ log_alpha + log_nu  + tau + gamma + log_eta', data=df).fit()
print(results.summary())


################## there is 
r_, p_ = analysisFxs.calc_prod_correlations(df, ["auc", "std_wtw"], ["auc", "std_wtw"])


###### hmmm let me calcualte changes in PNA ..., changes in the two dimensions are not correlated 
selfdf = analysisFxs.sub_across_sessions(s1_selfdf, s2_selfdf, vars = ["PAS", "NAS", "UPPS", "BIS", "GMK"])
selfdf[["PAS_diff", "NAS_diff"]].plot.hist()
selfdf[["PAS_diff", "NAS_diff"]].plot.scatter(x = "NAS_diff", y = "PAS_diff") # is this really meaningful??? or is it just ramdom???

statsdf = analysisFxs.sub_across_sessions(analysisFxs.pivot_by_condition(s1_stats), analysisFxs.pivot_by_condition(s2_stats))


df = selfdf[["id", "PAS_diff", "NAS_diff", "UPPS_diff", "BIS_diff", "GMK_diff"]].merge(statsdf[["id", "auc_diff", "std_wtw_diff", "auc_delta_diff"]], on = "id")
results = smf.ols('PAS ~ auc_diff + std_wtw_diff + auc_delta_diff', data=df).fit()
print(results.summary())


r_, p_ = analysisFxs.calc_prod_correlations(df, ["PAS_diff", "NAS_diff"], ["auc_diff", "std_wtw_diff", "auc_delta_diff"])

# r_, p_ = analysisFxs.calc_prod_correlations(df, ["PAS_diff", "NAS_diff"], ["UPPS_diff", "BIS_diff", "GMK_diff"])
# r_, p_ = analysisFxs.calc_prod_correlations(df, ["auc_diff", "std_wtw_diff", "auc_delta_diff"], ["UPPS_diff", "BIS_diff", "GMK_diff"])
#######


