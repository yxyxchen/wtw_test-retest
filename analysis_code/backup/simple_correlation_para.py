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
UPPS_subscales = ["NU", "PU", "PM", "PS", "SS"]
BIS_l1_subscales = ["Attentional", "Motor", "Nonplanning"]
BIS_l2_subscales = ["attention", "cogstable", "motor", "perseverance", "selfcontrol", "cogcomplex"]

modelname = "QL2reset_slope_two_simple"
fitMethod = "whole"
stepsize = 0.5
# passive version
paradf_ = [] 
selfdf_ = []
for expname in ["active", "passive"]:
	hdrdata_sess1, trialdata_sess1_ = loadFxs.group_quality_check(expname, 1, plot_quality_check = True)
	hdrdata_sess2, trialdata_sess2_ = loadFxs.group_quality_check(expname, 2, plot_quality_check = True)
	########### let me only include participants complete both sessions
	hdrdata_sess1 = hdrdata_sess1[np.isin(hdrdata_sess1["id"], hdrdata_sess2["id"])]
	trialdata_sess1_ = {x: y for x,y in trialdata_sess1_.items() if x[0] in hdrdata_sess2["id"].values}
	############ conduct behavioral analysis ######
	if expname == "passive":
		s1_selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
		s2_selfdf = loadFxs.parse_group_selfreport(expname, 2, isplot = False)
		s1_selfdf = s1_selfdf[np.isin(s1_selfdf["id"], s2_selfdf["id"])]
		selfdf = analysisFxs.agg_across_sessions(s1_selfdf, s2_selfdf)
	else:
		selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
		s1_selfdf = selfdf
	selfdf["exp"] = expname
	selfdf_.append(selfdf)
	s1_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, fitMethod, stepsize)
	s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname, fitMethod, stepsize)
	paradf = analysisFxs.agg_across_sessions(s1_paradf, s2_paradf)
	paradf["exp"] = expname
	paradf_.append(paradf)

paradf = pd.concat(paradf_)
selfdf = pd.concat(selfdf_)
df = paradf.merge(selfdf, on = ["id", "exp"])
df.loc[:, df.dtypes == "float64"] = df.select_dtypes("number") - df.select_dtypes("number").apply(np.mean, axis = 0)
df["exp"] = pd.Categorical(df["exp"], categories = ["passive", "active"], ordered = True)


paranames = modelFxs.getModelParas(modelname)
self_vars = ["BIS", "UPPS", "discount_logk"] + BIS_l2_subscales + UPPS_subscales
npara = len(paranames)
nselfvar = len(self_vars)
# record whether there is an interaction of exp 
cb_p_df = pd.DataFrame(np.zeros((nselfvar, npara)), index = self_vars, columns = paranames)
ps_p_df = pd.DataFrame(np.zeros((nselfvar, npara)), index = self_vars, columns = paranames)
ac_p_df = pd.DataFrame(np.zeros((nselfvar, npara)), index = self_vars, columns = paranames)
cb_r_df = pd.DataFrame(np.zeros((nselfvar, npara)), index = self_vars, columns = paranames)
ps_r_df = pd.DataFrame(np.zeros((nselfvar, npara)), index = self_vars, columns = paranames)
ac_r_df = pd.DataFrame(np.zeros((nselfvar, npara)), index = self_vars, columns = paranames)
interaction_pvals = pd.DataFrame(np.zeros((nselfvar, npara)), index = self_vars, columns = paranames)
for para, self_var in itertools.product(paranames, self_vars):
	fit = smf.ols(para +  " ~ %s * exp" % self_var, data = df).fit()
	interaction_pvals.loc[self_var, para] = fit.pvalues[3]
	cb_r_df.loc[self_var, para],cb_p_df.loc[self_var, para]  = spearmanr(df[para], df[self_var], nan_policy = "omit")
	ps_r_df.loc[self_var, para],ps_p_df.loc[self_var, para] = spearmanr(df.loc[df["exp"]=="passive", para], df.loc[df["exp"]=="passive", self_var], nan_policy = "omit")
	ac_r_df.loc[self_var, para],ac_p_df.loc[self_var, para]  = spearmanr(df.loc[df["exp"]=="active", para], df.loc[df["exp"]=="active", self_var], nan_policy = "omit")

# select 
cb_p_df[interaction_pvals < 0.05] = np.nan
cb_p_df[cb_p_df > 0.01] = np.nan

########## visualize 
# plt.style.use('classic')
# sns.set(font_scale = 2)
# sns.set_style("white")
# fig, ax  = plt.subplots()
# ax.hist(cb_r_df.values.reshape(-1), color = "grey", edgecolor = "black")
# ax.set_xlim([-0.5, 0.5])
# ax.set_xticks([ -0.4, -0.2, 0, 0.2, 0.4])
# ax.set_xlabel("Spearman's correlation")
# fig.tight_layout()
# fig.savefig(os.path.join("../figures/combined/rho_dist_para.pdf"))



##### permutation tests #######
# shuffle within each condition # 
self_vars = ["discount_logk"] + BIS_l2_subscales + UPPS_subscales
n_perm = 5000
cb_max_abs_r_dist = []
coef_dist_ = dict(zip(['%s-%s'%(x,y) for x, y in itertools.product(self_vars, paranames)], [[]]*len(self_vars)*len(paranames)))
coef_dist_list = [] # to record all abs correlations 
for i in np.arange(n_perm):
	for para in paranames:
		df["rd_" + para] = np.random.permutation(df[para])
	r_, _ = analysisFxs.calc_prod_correlations(df, ["rd_" + x for x in paranames], self_vars)
	cb_max_abs_r_dist.append(np.max(np.abs(r_.values)))
	for self_var in self_vars:
		for para in paranames:
			coef_dist_['%s-%s'%(self_var, para)] = coef_dist_['%s-%s'%(self_var, para)] + [abs(r_.loc['rd_' + para, self_var])]
			coef_dist_list.append(abs(r_.loc['rd_' + para, self_var]))



################ compare h0 and h1 correlation dists 
plt.style.use('classic')
plt.hist(coef_dist_list)
plt.style.use('classic')
sns.set(font_scale = 2)
sns.set_style("white")
fig, ax = plt.subplots()
h0_dist = ax.hist(coef_dist_list, density = True, color = "#bdbdbd", edgecolor = "black", alpha = 0.8, label='Null dist')
h1_dist = ax.hist(np.abs(cb_r_df.values.reshape(-1)), density = True, color = "#dd1c77", edgecolor = "black", alpha = 0.6, label='Observed dist')
ax.legend(['Null dist', 'Observed dist'])
ax.set_xlabel("Spearman's correlation")
ax.set_xlim([])
ax.set_ylabel("Density")
fig.tight_layout()
fig.savefig(os.path.join("../figures", "combined", "h0_h1_para_correlations.pdf"))


################
ac_res_df = pd.DataFrame({
	""
	})
# 
smf.ols("discount_logk ~ auc_delta + exp", data = df).fit().summary() # the best
smf.ols("discount_logk ~ auc_delta", data = df).fit().summary() # the best
smf.ols("discount_logk ~ auc_delta", data = df[df["exp"] == "passive"]).fit().summary()
smf.ols("discount_logk ~ auc_delta", data = df[df["exp"] == "active"]).fit().summary()


smf.ols("discount_logk ~ std_wtw * exp", data = df).fit().summary() # the best 
smf.ols("discount_logk ~ std_wtw + exp", data = df).fit().summary() 
smf.ols("discount_logk ~ std_wtw", data = df).fit().summary() 
smf.ols("discount_logk ~ std_wtw", data = df[df["exp"] == "passive"]).fit().summary()
smf.ols("discount_logk ~ std_wtw", data = df[df["exp"] == "active"]).fit().summary()


# 
smf.ols("auc ~ selfcontrol * exp", data = df).fit().summary()
spearmanr(df.loc[df["exp"]=="passive"]["auc"], df.loc[df["exp"]=="passive"]["selfcontrol"], nan_policy = "omit")


spearmanr(df["auc"], df["PS"], nan_policy = "omit")



smf.ols("auc ~ selfcontrol", data = df[df["exp"] == "passive"]).fit().summary()
smf.ols("auc ~ selfcontrol", data = df[df["exp"] == "active"]).fit().summary()


# 
smf.ols("auc ~ Nonplanning * exp", data = df).fit().summary()
################
smf.ols("UPPS ~ std_wtw * exp", data = df).fit().summary() # the best; BIS is not ..
smf.ols("UPPS ~ std_wtw + exp", data = df).fit().summary() 
smf.ols("UPPS ~ std_wtw", data = df).fit().summary() 
smf.ols("UPPS ~ std_wtw", data = df[df["exp"] == "passive"]).fit().summary() ....
smf.ols("UPPS ~ std_wtw", data = df[df["exp"] == "active"]).fit().summary()


smf.ols("std_wtw ~ UPPS * exp", data = df).fit().summary() # the best; upps is not significant
smf.ols("std_wtw ~ UPPS + exp", data = df).fit().summary() 
smf.ols("UPPS ~ std_wtw", data = df).fit().summary() # significant ....
smf.ols("UPPS ~ std_wtw", data = df[df["exp"] == "passive"]).fit().summary() ....
smf.ols("UPPS ~ std_wtw", data = df[df["exp"] == "active"]).fit().summary()

########
smf.ols("Motor ~ std_wtw", data = df).fit().summary() 
smf.ols("motor ~ std_wtw", data = df).fit().summary() 


# why adding the interaction term makes the effect smaller 
smf.ols("discount_logk ~ auc_delta", data = df[df["exp"] == "passive"]).fit().summary()
smf.ols("UPPS ~ exp + std_wtw", data = df).fit().summary()

############
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
	+ labs(x = "", y = r"$Spearman's \rho$") + theme_classic() +
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


################### regressions
source_ = []
sp_var_ = []
bh_var_ = []
coef_ = [] 
se_ = []
pval_ = []
impulse_vars = ["BIS", "UPPS",  "discount_logk"]
bh_vars = ["auc", "std_wtw", "auc_delta"]
if expname == "passive":
	sources = ["sess1", "sess2", "combined"]
elif expname == "active":
	sources = ["sess1", "combined"]

# impulse_vars = ["NU", "PU", "PM", "PS", "SS"]
for impulse_var in impulse_vars:
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
			results = smf.ols(bh_var + ' ~ ' + impulse_var , data=df).fit()
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
	"pval": [figFxs.tosig(x, marginal = True) for x in pval_],
	"label_y": [x + 0.03 * np.sign(x) for x in coef_]
	})
plotdf["source"] =  pd.Categorical(plotdf["source"], categories = ["sess1", "sess2", "combined"])

plotdf = plotdf[np.logical_and(plotdf["source"] == "combined", np.isin(plotdf["sp_var"], ["BIS", "UPPS",  "discount_logk",  "Motor", "Nonplanning", "Attentional", "motor"]))]
plotdf["bh_var"] =  pd.Categorical(plotdf["bh_var"], categories = ["auc", "std_wtw", "auc_delta"])

# only combined sessions
p = (ggplot(plotdf) \
	+ aes(x="bh_var", y="coef") \
	+ geom_bar(stat = "identity", position="dodge", width=0.75) \
	# + geom_errorbar(aes(ymin = "ymin", ymax = "ymax"), position = "dodge", width = 0.9)\
	+ facet_grid(facets="~sp_var") 
	+ geom_text(aes(y="label_y", label = "pval"), position=position_dodge(width=0.75)) 
	+ scale_fill_manual(values = ["#ca0020", "#0571b0", "#bababa"]) 
	+ labs(x = "", y = "Standardized coef") + theme_classic() + scale_x_discrete(labels= [r"AUC", r"$\sigma_{WTW}$", r"$\Delta AUC$"]))
ggsave(plot = p, filename= "../figures/%s/coef_combined_BIS.pdf"%(expname), width = 10, height = 5)



############################# look at subscores ###########
source_ = []
sp_var_ = []
bh_var_ = []
coef_ = [] 
se_ = []
pval_ = []
impulse_vars = ["Motor", "Nonplanning", "Attentional", "attention", "cogstable", "motor", "perseverance", "selfcontrol", "cogcomplex"]
bh_vars = ["auc", "std_wtw", "auc_delta"]
if expname == "passive":
	sources = ["sess1", "sess2", "combined"]
elif expname == "active":
	sources = ["sess1", "combined"]

# impulse_vars = ["NU", "PU", "PM", "PS", "SS"]
for impulse_var in impulse_vars:
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
			results = smf.ols(bh_var + ' ~ ' + impulse_var, data=df).fit()
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
	"pval": [figFxs.tosig(x, marginal = True) for x in pval_],
	"label_y": [x + 0.03 * np.sign(x) for x in coef_]
	})
plotdf["source"] =  pd.Categorical(plotdf["source"], categories = ["sess1", "sess2", "combined"])
plotdf = plotdf[np.logical_and(plotdf["source"] == "combined", np.isin(plotdf["sp_var"], ["Motor", "Nonplanning", "Attentional", "attention", "cogstable", "motor", "perseverance", "selfcontrol", "cogcomplex"]))]
plotdf["bh_var"] =  pd.Categorical(plotdf["bh_var"], categories = ["auc", "std_wtw", "auc_delta"])

p = (ggplot(plotdf)\
	+ aes(x="bh_var", y="coef") \
	+ geom_bar(stat = "identity", position="dodge", width=0.75) \
	# + geom_errorbar(aes(ymin = "ymin", ymax = "ymax"), position = "dodge", width = 0.9)\
	+ facet_grid(facets="~sp_var") 
	+ geom_text(aes(y="label_y", label = "pval"), position=position_dodge(width=0.75)) 
	+ scale_fill_manual(values = ["#ca0020", "#0571b0", "#bababa"]) 
	+ labs(x = "", y = "Standardized coef") + theme_classic() + scale_x_discrete(labels= [r"AUC", r"$\sigma_{WTW}$", r"$\Delta AUC$"]))
ggsave(plot = p, filename= "../figures/%s/coef_combined_subscores.pdf"%(expname), width = 20, height = 5)


#  all variale regressions
source_ = []
sp_var_ = []
bh_var_ = []
coef_ = [] 
se_ = []
pval_ = []
impulse_vars = ["BIS", "UPPS",  "discount_logk"]
bh_vars = ["auc", "std_wtw", "auc_delta"]
if expname == "passive":
	sources = ["sess1", "sess2", "combined"]
elif expname == "active":
	sources = ["sess1", "combined"]

# impulse_vars = ["NU", "PU", "PM", "PS", "SS"]
for impulse_var in impulse_vars:
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
		df[df.select_dtypes('number').columns] = df.select_dtypes('number').apply(lambda x:scipy.stats.zscore(x, nan_policy = "omit")) 
		results = smf.ols(impulse_var + '~ auc + std_wtw + auc_delta', data=df).fit()
		source_ = source_ + [source] * len(results.params[1:])
		bh_var_ = bh_var_ + list(results.params[1:].index.values)
		sp_var_ = sp_var_ + [impulse_var] * len(results.params[1:])
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
	"pval": [figFxs.tosig(x, marginal = True) for x in pval_],
	"label_y": [x + 0.03 * np.sign(x) for x in coef_]
	})
plotdf["source"] =  pd.Categorical(plotdf["source"], categories = ["sess1", "sess2", "combined"])
plotdf = plotdf[np.logical_and(plotdf["source"] == "combined", np.isin(plotdf["bh_var"], ["auc", "std_wtw", "auc_delta"]))]



