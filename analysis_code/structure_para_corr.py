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

expname = 'passive'
hdrdata_sess1, trialdata_sess1_ = loadFxs.group_quality_check(expname, 1, plot_quality_check = False)
hdrdata_sess2, trialdata_sess2_ = loadFxs.group_quality_check(expname, 2, plot_quality_check = False)

modelname = 'QL2reset_slope'
fitMethod = "whole"
stepsize = 0.5
subtitles = [r'$\mathbf{log(\alpha)}$', r'$\mathbf{log(\nu)}$', r'$\mathbf{\tau}$', r'$\mathbf{\gamma}$', r'$\mathbf{log(\eta)}$']
paranames = modelFxs.getModelParas(modelname)
npara = len(paranames)
s1_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, fitMethod, stepsize)
s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname, fitMethod, stepsize)


## look at s1_paradf alpha 
# s1_paradf = s1_paradf.set_index("id")
# s2_paradf = s2_paradf.set_index("id")

# rewarded = []
# nonrewarded = []
# id_ = []
# for key, x in trialdata_sess1_.items():
# 	rewarded.append(np.sum(x["trialEarnings"] > 0))
# 	nonrewarded.append(np.sum(x["trialEarnings"] == 0))
# 	id_.append(key[0])
# s1_tmp = pd.DataFrame({
# 	"id": id_,
# 	"rewarded": rewarded,
# 	"nonrewarded": nonrewarded
# 	})
# s1_tmp = s1_tmp.set_index("id")
# s1_tmp["ratio"] = s1_tmp["rewarded"]  / (s1_tmp["rewarded"] + s1_tmp["nonrewarded"])


# rewarded = []
# nonrewarded = []
# id_ = []
# for key, x in trialdata_sess2_.items():
# 	rewarded.append(np.sum(x["trialEarnings"] > 0))
# 	nonrewarded.append(np.sum(x["trialEarnings"] == 0))
# 	id_.append(key[0])
# s2_tmp = pd.DataFrame({
# 	"id": id_,
# 	"rewarded": rewarded,
# 	"nonrewarded": nonrewarded
# 	})
# s2_tmp = s2_tmp.set_index("id")
# s2_tmp["ratio"] = s2_tmp["rewarded"]  / (s2_tmp["rewarded"] + s2_tmp["nonrewarded"])
# spearmanr(s1_tmp.loc[s2_tmp.index, "ratio"], s2_tmp["ratio"])
# SpearmanrResult(correlation=0.692284611137064, pvalue=7.380055537069578e-38)

# s1_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, fitMethod, stepsize)
# s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname, fitMethod, stepsize)

# s1_paradf = s1_paradf.set_index("id")
# s2_paradf = s2_paradf.set_index("id")


# s1_paradf["alpha"] = s1_paradf["alpha"] * (s1_tmp["ratio"] + s1_paradf["nu"] * (1 - s1_tmp["ratio"])) 
# s2_paradf["alpha"] = s2_paradf["alpha"] * (s2_tmp["ratio"] + s2_paradf["nu"] * (1 - s2_tmp["ratio"]))


########### let me get the structure correlations #######
r_ = []
pair_ = []
id_ = []

cv_ = []
std_ = []
para_ = []
id_para_ = []
for id in s1_paradf["id"]:
	parafile = os.path.join("../analysis_results", expname, "modelfit", fitMethod, "stepsize%.2f"%stepsize, modelname, "%s_sess1_sample.txt"%id)
	parasamples = pd.read_csv(parafile, header = None)
	parasamples = parasamples.iloc[:, :5]
	parasamples =  parasamples.rename(columns = dict(zip(parasamples.columns, paranames)))
	for x, y in itertools.combinations(paranames, 2):
		pair_.append((x, y))
		id_.append(id)
		r_.append(spearmanr(parasamples[x], parasamples[y])[0])
	for x in paranames:
		cv_.append(parasamples[x].std()/parasamples[x].mean())
		std_.append(parasamples[x].std())
		para_.append(x)
		id_para_.append(x)

####### structural correlations ######
plotdf = pd.DataFrame({
	"r": r_,
	"pair": pair_,
	"id": id_
	})

plotdf.groupby("pair").agg({"r":np.median})

# let me plot the lower triangle 
plt.style.use('classic')
sns.set(font_scale = 1)
sns.set_style("white")
condition_palette = ["#762a83", "#1b7837"]
fig, axes = plt.subplots(len(paranames), len(paranames))
for (i, x), (j,y) in itertools.product(enumerate(paranames), enumerate(paranames)):
	if (x, y) in itertools.combinations(paranames, 2):
		axes[i,j].hist(plotdf.loc[plotdf["pair"] == (x,y), "r"].values, bins = 15)
		axes[i,j].set_xlabel(x)
		axes[i,j].set_ylabel(y)
		axes[i,j].set_xlim((-1, 1))
		median_val = np.median(plotdf.loc[plotdf["pair"] == (x,y), "r"].values)
		axes[i,j].axvline(median_val, color = "red")
		axes[i,j].text(median_val, 40, "%.2f"%median_val, color = "red")
fig.savefig(os.path.join("..", "figures", expname, "para_structure_corr_%s.pdf"%modelname))

####### structural uncertainty #####
plotdf = pd.DataFrame({
	"std": std_,
	"cv": cv_,
	"para": para_,
	"id": id_para_
	})
plotdf.groupby("para").agg({"std":np.median})
plotdf.groupby("para").agg({"cv":np.median}) # nu has strong uncertainty 

g = sns.FacetGrid(plotdf, col = "para")
g.map(plt.hist, "cv") 

####### among participant correlations ####
# maybe I want to log transform first ....
log_paradf = pd.concat([figFxs.log_transform_parameter(s1_paradf, ["alpha", "nu", "tau", "eta"]), figFxs.log_transform_parameter(s2_paradf, ["alpha", "nu", "tau", "eta"])])
g = sns.pairplot(data = log_paradf.iloc[:,:npara], kind = "reg", diag_kind = "None", corner = True,\
	diag_kws = {"color": "grey", "edgecolor": "black"}, plot_kws ={'line_kws':{'color':'red'}, "scatter_kws": {"color": "grey", "edgecolor": "black"}})
g.map_lower(figFxs.annotate_reg)
g.savefig(os.path.join("..", 'figures', expname, "%s_%s_stepsize%.2f_para_correlation.pdf"%(modelname, fitMethod, stepsize)))

plt.style.use('classic')
sns.set(font_scale = 2)
sns.set_style("white")
log_para_mapping = dict(zip(log_paradf.columns[:5], subtitles))
log_paradf = log_paradf.rename(columns=log_para_mapping)
fig, axes = plt.subplots(1, 5*2)
for i, (x, y) in enumerate(itertools.combinations(log_paradf.columns[:5], 2)):
	figFxs.my_regplot(log_paradf[x], log_paradf[y], ax = axes.flatten()[i])
	axes.flatten()[i].set_xlabel(x)
	axes.flatten()[i].set_ylabel(y)
fig.set_size_inches(5 * npara * (npara-1) / 2, 5)
fig.savefig(os.path.join("..", 'figures', expname, "%s_%s_stepsize%.2f_para_correlation_flattern.pdf"%(modelname, fitMethod, stepsize)))

