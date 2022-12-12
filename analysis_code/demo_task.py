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


def full_selected(data, response):
	best_score = 0.0
	predictors = ["age", "gender", "education", "age + gender", "age + education", "education + gender",\
	"age * gender", "age * education", "education * gender", "age * gender + education", "age * education + gender", \
	"education * age + gender", "age * gender + age * education", "gender * education + gender * age", "education * age + education * gender", \
	"education * age * gender"]
	score_ = []
	model_ = []
	for predictor in predictors:
		model = smf.ols(response + '~' + predictor, data).fit()
		score_.append(model.rsquared_adj)
		model_.append(model)
		if model.rsquared_adj > best_score:
			best_score = model.rsquared_adj
	return [x for x in score_ if x == best_score], [y for x, y in zip(score_, model_) if x == best_score], score_


# plot styles
plt.style.use('classic')
sns.set(font_scale = 1)
sns.set_style("white")
condition_palette = ["#762a83", "#1b7837"]

# passive version
statsdf_ = []
hdrdata_ = []
for expname in ["passive", "active"]:
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
	hdrdata = hdrdata_sess2
	statsdf_.append(statsdf)
	hdrdata_.append(hdrdata)


hdrdata_[0]["exp"] = "passive"
hdrdata_[1]["exp"] = "active"
statsdf = pd.concat(statsdf_)
hdrdata = pd.concat(hdrdata_)


################ simple effects ###########
# plot 
task_vars = [ "auc", "auc_delta", "std_wtw"]
demo_vars = ["age", "education"]

for var in demo_vars:
	plt.style.use('classic')
	sns.set(font_scale = 1)
	sns.set_style("white")
	df = statsdf.merge(hdrdata, on = "id")
	tmp = df.melt(id_vars = ["id", var, "gender"], value_vars = task_vars )
	g = sns.FacetGrid(data = tmp, col = "variable", sharey = False, col_order = task_vars )
	g.map(sns.regplot, var, "value", line_kws = {'color':'red'}, scatter_kws = {"color": "grey", "edgecolor": "black"})
	g.map(figFxs.annotate_reg, var, "value")
	g.savefig(os.path.join("..", "figures", expname, var + "_task.pdf"))


binary_var = "gender"
plt.style.use('classic')
sns.set(font_scale = 1)
sns.set_style("white")
df = statsdf.merge(hdrdata, on = "id")
df = df[np.isin(df["gender"], ["Male", "Female"])]
tmp = df.melt(id_vars = ["id", binary_var], value_vars = task_vars )
g = sns.FacetGrid(data = tmp, col = "variable", sharey = False, col_order = task_vars)
g.map(sns.boxplot, binary_var, "value")
g.savefig(os.path.join("..", "figures", expname, "gender_task.pdf"))

for var, ax in zip(task_vars, g.axes.flatten()):
	res = mannwhitneyu(df.loc[df["gender"] == "Male", var], df.loc[df["gender"] == "Female", var], alternative = "two-sided")
	ax.plot([0, 0, 1, 1], [5, 5.5, 5.5, 5], lw=1.5)
	ax.text((0+1)*.5, 5.5, figFxs.tosig(res[1]), ha='center', va='bottom')


binary_var = "seq"
plt.style.use('classic')
sns.set(font_scale = 1)
sns.set_style("white")
df = statsdf.merge(hdrdata, on = "id")
df = df[np.isin(df["gender"], ["Male", "Female"])]
tmp = df.melt(id_vars = ["id", binary_var], value_vars = task_vars )
g = sns.FacetGrid(data = tmp, col = "variable", sharey = False, col_order = task_vars)
g.map(sns.boxplot, binary_var, "value")
for var, ax in zip(task_vars, g.axes.flatten()):
	res = mannwhitneyu(df.loc[df["seq"] == "seq1", var], df.loc[df["seq"] == "seq2", var])
	ax.plot([0, 0, 1, 1], [5, 5.5, 5.5, 5], lw=1.5)
	ax.text((0+1)*.5, 5.5, figFxs.tosig(res[1]), ha='center', va='bottom')


##################
df["age"] = df["age"] - np.mean(df["age"])
df["education"] = df["education"] - np.mean(df["education"])

full_res_ = []
for task_var in task_vars:
	full_res = smf.ols(task_var + " ~ gender * age * education", data = df[[task_var, "gender", "age", "education", "seq", "color"]]).fit()
	full_res_.append(full_res)


############## permutation test
df = statsdf.merge(hdrdata, on = "id")
df = df[np.isin(df["gender"], ["Male", "Female"])]

df["age"] = df["age"] - np.mean(df["age"])
df["education"] = df["education"] - np.mean(df["education"])

mannwhitneyu(df.loc[df["gender"] == "Male", "auc_delta"], df.loc[df["gender"] == "Female", "auc_delta"], alternative = "two-sided")

smf.ols("auc_delta ~ gender", data = df[["auc_delta", "gender", "age", "education", "seq", "color", "exp"]]).fit().summary()

smf.ols("auc_delta ~ gender", data = df[["auc_delta", "gender", "age", "education", "seq", "color", "exp"]].loc[df["exp"] == "passive"]).fit().summary()
smf.ols("auc_delta ~ gender", data = df[["auc_delta", "gender", "age", "education", "seq", "color", "exp"]].loc[df["exp"] == "active"]).fit().summary()


smf.ols("auc ~ education * exp", data = df[["auc", "gender", "age", "education", "seq", "color", "exp"]]).fit().summary()



