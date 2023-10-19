import random
import numpy as np
import pandas as pd
import subFxs
from subFxs import analysisFxs
from subFxs import expParas
from subFxs import stancode
from subFxs import simFxs
from subFxs import modelFxs
import code
import stan
import os
import arviz as az
from sksurv.nonparametric import kaplan_meier_estimator as km
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from statistics import NormalDist
# plot styles
plt.style.use('classic')
sns.set_style("white")
sns.set_context("poster")



expname = "passive"
sess = 2
stepsize = 0.5
chainIdxs = [1, 3, 4]
S = 50
modelname = "QL2reset_HM_new"
paranames = modelFxs.getModelParas(modelname)
fitMethod = "whole"

# 
chainIdx = 4
colnames = ["mu_" + x for x in paranames] + [ x + "[" + str(y) + "]" for x, y in itertools.product(paranames, np.arange(S))]+ ["totalLL"]
para_samples = pd.read_csv("../analysis_results/%s/modelfit_hm/%s/stepsize%.2f/%s/chain%d/sess%d_para_sample.txt"%(expname, fitMethod, stepsize,  modelname, chainIdx, sess),
	header = None, names = colnames)

para_summary = pd.read_csv("../analysis_results/%s/modelfit_hm/%s/stepsize%.2f/%s/chain%d/sess%d_summary.txt"%(expname, fitMethod, stepsize,  modelname, chainIdx, sess),
	header = None).transpose()

para_summary.columns = colnames

raw_mu_alpha_ = norm.ppf(para_samples['mu_alpha'] / 0.3)

# I don't think this looks right ... ok
raw_alpha_sds = norm.ppf(para_summary.iloc[0, 55:105] / 10) - norm.ppf(para_summary.iloc[0, 1] / 10)
plt.hist(raw_alpha_sds)
plt.show()

# I think I did this wrong ...
raw_alpha_mu = norm.ppf(para_summary['mu_alpha'][0] / 0.3)
raw_alpha_ = norm.ppf( para_summary.iloc[0, 5:55] / 0.3)


