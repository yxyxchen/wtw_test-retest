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


expname = "passive"

# load behavioral data
hdrdata_sess1, trialdata_sess1_ = loadFxs.group_quality_check(expname, 1, plot_quality_check = True)
hdrdata_sess2, trialdata_sess2_ = loadFxs.group_quality_check(expname, 2, plot_quality_check = True)


# load model parameters
modelname = 'QL2reset_FL2'
fitMethod = "whole"
stepsize = 0.5
paranames = modelFxs.getModelParas(modelname)
s1_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, fitMethod, stepsize)
s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname, fitMethod, stepsize)


# 
for id in s1_paradf['id']:
	sample_file = os.path.join(("../analysis_results/%s/modelfit/%s/stepsize%.2f/%s/%s_sess1_sample.txt")%(expname, fitMethod, stepsize, modelname, id))
	para_samples = pd.read_csv(sample_file, header = None)
	para_samples.columns = paranames + ['totalLL']
	g = sns.FacetGrid(data = para_samples.iloc[2000:,:-1].melt(var_name = "para"), col = "para", sharex = False)
	g.map(plt.hist, "value", bins = 20)
	plt.show()
	input("Press Enter to continue...")
	plt.clf()

########### let me calculate similiarity between 
for id in set(s1_paradf['id']) & set(s2_paradf['id']):
	s1_para_samples = pd.read_csv(os.path.join(("../analysis_results/%s/modelfit/%s/stepsize%.2f/%s/%s_sess1_sample.txt")%(expname, fitMethod, stepsize, modelname, id)), header = None)
	s1_para_samples = pd.read_csv(os.path.join(("../analysis_results/%s/modelfit/%s/stepsize%.2f/%s/%s_sess1_sample.txt")%(expname, fitMethod, stepsize, modelname, id)), header = None)
	# para_samples.columns = paranames + ['totalLL']



