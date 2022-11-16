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
from scipy.stats import mannwhitneyu

# plot styles
plt.style.use('classic')
sns.set(font_scale = 2)
sns.set_style("white")
condition_palette = ["#762a83", "#1b7837"]

# passive version
hdrdata_ = []
selfdf_ = []
for expname in ["passive", "active"]:	
	hdrdata_sess2, trialdata_sess2_ = loadFxs.group_quality_check(expname, 2, plot_quality_check = True)
	hdrdata_.append(hdrdata_sess2)

selfdf = pd.concat(selfdf_, axis = 0).reset_index()

############# plot demographic data #########
for i, expname in enumerate(["passive", "active", "combined"]):	
	for var, label in zip(["age", "education"], ["Age", "Education year"]):
		if i < 2:
			this_hdrdata = hdrdata_[i]
		else:
			this_hdrdata = hdrdata
		fig, ax = plt.subplots()
		tmp = this_hdrdata["gender"].value_counts()
		fig, ax = plt.subplots()
		ax.pie(tmp.values, labels = tmp.index, autopct='%.0f%%')
		fig.savefig(os.path.join("..", "figures", expname, "gender_pie.pdf"))
		fig, ax = plt.subplots()
		tmp = this_hdrdata["gender"].value_counts()
		fig, ax = plt.subplots()
		ax.pie(tmp.values, labels = tmp.index, autopct='%.0f%%')
		fig.savefig(os.path.join("..", "figures", expname, "gender_pie.pdf"))
		fig, ax = plt.subplots()
		ax.hist(this_hdrdata[var], color = "grey", edgecolor = "black")
		ax.set_xlabel(label)
		ax.set_ylabel("Frequency")
		fig.tight_layout()
		fig.savefig(os.path.join("..", "figures", expname, var + "_hist.pdf"))
		df = this_hdrdata[np.isin(this_hdrdata["gender"],["Female", "Male"])]
		fig, ax = plt.subplots()
		sns.violinplot(data = df, x = "gender", y = var, ax = ax)
		sns.boxplot(data = df, x='gender', y = var, saturation=0.5, width=0.4, boxprops={'zorder': 2}, ax=ax)
		ax.set_xlabel("")
		sig = figFxs.tosig(mannwhitneyu(df.loc[df["gender"] == "Female", var], df.loc[df["gender"] == "Male", var]).pvalue)
		if var == "age":
			ax.plot([0, 0, 1, 1], [80, 84, 84, 80], lw=1.5)
			plt.text((0+1)*.5, 79, sig, ha='center', va='bottom')
		else:
			ax.plot([0, 0, 1, 1], [23, 24, 24, 23], lw=1.5)
			plt.text((0+1)*.5, 23, sig, ha='center', va='bottom')		
		fig.savefig(os.path.join("..", "figures", expname, var + "_gender_barplot.pdf"))
		g = sns.FacetGrid(data = df, col = "gender", hue = "gender")
		g.map(plt.hist, var)
		g.savefig(os.path.join("..", "figures", expname, var + "_gender_hist.pdf"))


#################### effects of demographic variables on selfreport ########
plt.style.use('classic')
sns.set(font_scale = 1)
sns.set_style("white")
df = selfdf.merge(hdrdata, on = "id")
tmp = df.melt(id_vars = ["id", "age", "gender"], value_vars = [ "BIS", "UPPS", "survey_impulsivity", "discount_logk"])
g = sns.FacetGrid(data = tmp, col = "variable", sharey = False, col_order = [ "BIS", "UPPS", "survey_impulsivity", "discount_logk"])
g.map(sns.regplot, "age", "value", line_kws = {'color':'red'}, scatter_kws = {"color": "grey", "edgecolor": "black"})
g.map(figFxs.annotate_reg, "age", "value")
g.savefig(os.path.join("..", "figures", "combined", "age_selfreport_corr.pdf"))


df = selfdf.merge(hdrdata, on = "id")
df = df[np.isin(df["gender"], ["Female", "Male"])]
df = df[df["age"] < 50]
tmp = df.melt(id_vars = ["id", "age", "gender"], value_vars = [ "BIS", "UPPS", "survey_impulsivity", "discount_logk"])
g = sns.FacetGrid(data = tmp, row = "gender", col = "variable", sharex = False, sharey = False, col_order = [ "BIS", "UPPS", "survey_impulsivity", "discount_logk"], margin_titles = True)
g.map(sns.regplot, "age", "value", line_kws = {'color':'red'}, scatter_kws = {"color": "grey", "edgecolor": "black"})
g.map(figFxs.annotate_reg, "age", "value")
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.savefig(os.path.join("..", "figures", "combined", "age_gender_selfreport_corr_limited.pdf"))


df = selfdf.merge(hdrdata, on = "id")
df = df[df["age"] < 50]
tmp = df.melt(id_vars = ["id", "age", "gender"], value_vars = [ "BIS", "UPPS", "survey_impulsivity", "discount_logk"])
g = sns.FacetGrid(data = tmp, col = "variable", sharey = False, col_order = [ "BIS", "UPPS", "survey_impulsivity", "discount_logk"])
g.map(sns.regplot, "age", "value", line_kws = {'color':'red'}, scatter_kws = {"color": "grey", "edgecolor": "black"})
g.map(figFxs.annotate_reg, "age", "value")
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.savefig(os.path.join("..", "figures", "combined", "age_selfreport_corr_limited.pdf"))


# with gender 
df = selfdf.merge(hdrdata, on = "id")
df = df[np.isin(df["gender"], ["Female", "Male"])]
df = df[df["age"] < 50]
tmp = df.melt(id_vars = ["id", "age", "gender"], value_vars = [ "BIS", "UPPS", "survey_impulsivity", "discount_logk"])
g = sns.FacetGrid(data = tmp, col = "variable", row = "gender", sharey = False, col_order = [ "BIS", "UPPS", "survey_impulsivity", "discount_logk"], margin_titles=True )
g.map(sns.regplot, "age", "value", line_kws = {'color':'red'}, scatter_kws = {"color": "grey", "edgecolor": "black"})
g.map(figFxs.annotate_reg, "age", "value")
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.savefig(os.path.join("..", "figures", expname, "age_gender_selfreport_corr_limited.pdf"))


######
df = selfdf.merge(hdrdata, on = "id")
df = df[np.isin(df["gender"], ["Female", "Male"])]
df[df.select_dtypes('number').columns] = df.select_dtypes('number').apply(lambda x:scipy.stats.zscore(x, nan_policy = "omit")) 
predictors = ["gender", "age", "age*gender"]
yvals = [ "BIS", "UPPS", "survey_impulsivity", "SS", "PU", "NU", "PM", "PS", "Motor", "Nonplanning", "Attentional", "discount_logk"]
coef = []
for yval in yvals:
	results = smf.ols(yval + " ~ age + gender + education", data = df).fit()
	# coef.append(["%.3f( "%x + "p=%.4f"%y + " )" for x, y in zip(results.params[1:].values, results.pvalues[1:].values)])
	coef.append(["%.3f( "%x + "p=" + figFxs.tosig(y) + " )" for x, y in zip(results.params[1:].values, results.pvalues[1:].values)])

coef_report = pd.DataFrame(coef).rename(index = dict(zip(np.arange(len(yvals)), yvals)), columns = dict(zip(np.arange(len(results.params.index)-1), results.params.index[1:])))
coef_report
coef_report.loc[["BIS", "UPPS", "discount_logk"],:]





