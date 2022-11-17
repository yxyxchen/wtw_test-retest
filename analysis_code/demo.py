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
demo_vars = ["gender", "age", "education", "race", "language"]

# passive version
hdrdata_ = []
selfdf_ = []
for expname in ["passive", "active"]:	
	hdrdata_sess2, trialdata_sess2_ = loadFxs.group_quality_check(expname, 2, plot_quality_check = True)
	hdrdata_.append(hdrdata_sess2)

hdrdata = pd.concat(hdrdata_, axis = 0).reset_index()

# print summary statistics
for i, expname in enumerate(["passive", "active", "combined"]):	
	if i < 2:
		this_hdrdata = hdrdata_[i]
	else:
		this_hdrdata = hdrdata
	# summary statistics 
	tmp = this_hdrdata[["gender", "race", "language"]].melt().value_counts()
	tmp = tmp / (this_hdrdata.shape[0] + 1)
	tmp.rename("counts").reset_index().sort_values(by = ["variable", "counts"], ascending = False)
	this_hdrdata[["age", "education"]].describe()

############# plot demographic data #########
for i, expname in enumerate(["passive", "active", "combined"]):	
	if i < 2:
		this_hdrdata = hdrdata_[i]
	else:
		this_hdrdata = hdrdata
	# interaction between age and education 
	fig, ax = plt.subplots()
	ax.scatter(this_hdrdata["age"], this_hdrdata["education"])
	fig.savefig(os.path.join("..", "figures", expname, "age_education.pdf"))
	for var in ["gender", "race", "language"]:
		fig, ax = plt.subplots()
		tmp = this_hdrdata[var].value_counts()
		fig, ax = plt.subplots()
		ax.pie(tmp.values, labels = tmp.index, autopct='%.0f%%')
		fig.savefig(os.path.join("..", "figures", expname, var + "_pie.pdf"))
	for var, label in zip(["age", "education"], ["Age", "Years of education"]):
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






