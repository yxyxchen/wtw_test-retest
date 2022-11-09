
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
s1_stats, s1_Psurv_b1_, s1_Psurv_b2_, s1_WTW_ = analysisFxs.group_MF(trialdata_sess1_, plot_each = False)   
s2_stats, s2_Psurv_b1_, s2_Psurv_b2_, s2_WTW_ = analysisFxs.group_MF(trialdata_sess2_, plot_each = False)   
s1_df = analysisFxs.pivot_by_condition(s1_stats)
s2_df = analysisFxs.pivot_by_condition(s2_stats)
statsdf = analysisFxs.agg_across_sessions(s1_df, s2_df)

####################### analyze only selfreport data ####################
if expname == "passive":
	s1_selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
	s2_selfdf = loadFxs.parse_group_selfreport(expname, 2, isplot = False)
	s1_selfdf['log_GMK'] = np.log(s1_selfdf['GMK'])
	s2_selfdf['log_GMK'] = np.log(s2_selfdf['GMK'])
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
	selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
	selfdf['log_GMK'] = np.log(selfdf['GMK'])


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

################### single correlations ###############


################### single correlations ###############
sns.pairplot(selfdf[["UPPS", "BIS", "log_GMK"]])
sns.pairplot(selfdf[["UPPS", "BIS", "log_GMK"]])

df = statsdf.merge(selfdf, on = "id").merge(paradf, on = "id")
r_, p_ = analysisFxs.calc_prod_correlations(df, ['Motor', "Nonplanning", "Attentional"], ['auc', "std_wtw", "auc_delta"])
r_, p_ = analysisFxs.calc_prod_correlations(df, ['motor', "perseverance", "cogstable", "selfcontrol", "attention", "cogcomplex"], ['auc', "std_wtw", "auc_delta"])
r_, p_ = analysisFxs.calc_prod_correlations(df, ['BIS', "UPPS", "log_GMK"], ['auc', "std_wtw", "auc_delta"])
r_, p_ = analysisFxs.calc_prod_correlations(df, ["NU", "PU", "PM", "PS", "SS"], ["NU", "PU", "PM", "PS", "SS"])

r_, p_ = analysisFxs.calc_prod_correlations(df, ['BIS', "UPPS", "log_GMK"], ["log_alpha", "log_alphaU", "tau", "gamma", "log_eta"])
r_, p_ = analysisFxs.calc_prod_correlations(df, ["log_alpha", "log_alphaU", "tau", "gamma", "log_eta"], ["log_alpha", "log_alphaU", "tau", "gamma", "log_eta"])

sns.pairplot(df[["NU", "PU", "PM", "PS", "SS"]])
plt.savefig(os.path.join("..", "figures", expname, "UPPS_corr.pdf"))
r_, p_ = analysisFxs.calc_prod_correlations(df, ['motor', "selfcontrol", "cogstable"], ['motor', "selfcontrol", "cogstable"])

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


################### correlations among selfreports and model measures ###############
source_ = []
sp_var_ = []
bh_var_ = []
coef_ = [] 
se_ = []
pval_ = []
impulse_vars = ["BIS", "UPPS", "log_GMK"]
# impulse_vars = ["NU", "PU", "PM", "PS", "SS"]
for source in ["sess1", "combined"]:
	if source == "sess1":
		df = s1_df.merge(selfdf, on = "id")
	else:
		df = statsdf.merge(selfdf, on = "id")
	for impulse_var in impulse_vars:
		df[df.select_dtypes('number').columns] = df.select_dtypes('number').apply(lambda x:scipy.stats.zscore(x, nan_policy = "omit")) 
		results = smf.ols(impulse_var + ' ~ auc + std_wtw + auc_delta', data=df).fit()
		source_ = source_ + [source] * 3
		sp_var_ = sp_var_ + [impulse_var] * 3
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


# look at UPPS, BIS and Log_gamma
p = (ggplot(plotdf[plotdf["source"] == "combined"]) \
	+ aes(x="bh_var", y="coef", fill = "bh_var") \
	+ geom_bar(stat = "identity", position="dodge", width=0.75) \
	# + geom_errorbar(aes(ymin = "ymin", ymax = "ymax"), position = "dodge", width = 0.9)\
	+ facet_grid(facets="~sp_var") 
	+ geom_text(aes(y="label_y", label = "pval"), position=position_dodge(width=0.75)) 
	+ scale_fill_manual(values = ["#ca0020", "#0571b0", "#bababa"]) 
	+ labs(x = "", y = "Standardized coefficient") + theme_classic())
ggsave(plot = p, filename= "../figures/%s/coef_combined_BIS.pdf"%(expname))


# all sessions 
(ggplot(plotdf) \
	+ aes(x="source", y="coef", fill = "bh_var") \
	+ geom_bar(stat = "identity", position="dodge", width=0.75) \
	# + geom_errorbar(aes(ymin = "ymin", ymax = "ymax"), position = "dodge", width = 0.9)\
	+ facet_grid(facets="~sp_var") 
	+ geom_text(aes(y="label_y", label = "pval"), position=position_dodge(width=0.75)) 
	+ scale_fill_manual(values = ["#ca0020", "#0571b0", "#bababa"]) 
	+ labs(x = "", y = "Standardized coefficient"))

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


