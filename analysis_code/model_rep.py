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

# plot styles
plt.style.use('classic')
sns.set(font_scale = 1)
sns.set_style("white")
condition_palette = ["#762a83", "#1b7837"]


# generate output results 
def generate_output_dirs(expname):
    if not os.path.isdir(os.path.join("..", 'analysis_results')):
        os.makedirs(os.path.join("..", 'analysis_results'))

    if not os.path.isdir(os.path.join("..", 'analysis_results', expname)):
        os.makedirs(os.path.join("..", 'analysis_results', expname))

    if not os.path.isdir(os.path.join("..", "analysis_results", expname, "excluded")):
        os.makedirs(os.path.join("..", "analysis_results", expname, "excluded"))

    if not os.path.isdir(os.path.join('..', 'analysis_results', expname, 'taskstats')):
        os.makedirs(os.path.join('..', 'analysis_results', expname, 'taskstats'))

    if not os.path.isdir(os.path.join("..", 'analysis_results', expname, 'modelfit')):
        os.makedirs(os.path.join("..", 'analysis_results',expname, 'modelfit'))

    if not os.path.isdir(os.path.join("..", 'analysis_results', expname, 'crossvalid')):
        os.makedirs(os.path.join("..", 'analysis_results', expname, 'crossvalid'))

    if not os.path.isdir(os.path.join("..", 'analysis_results', expname, 'selfreport')):
        os.makedirs(os.path.join("..", 'analysis_results', expname, 'selfreport'))

    if not os.path.isdir(os.path.join("..", 'analysis_results', expname, 'correlation')):
        os.makedirs(os.path.join("..", 'analysis_results', expname, 'correlation'))

    if not os.path.isdir(os.path.join("..", "figures")):
        os.makedirs(os.path.join("..", "figures"))

    if not os.path.isdir(os.path.join("..", "figures", expname)):
        os.makedirs(os.path.join("..", "figures", expname))



# generate output directories
# I probably want to make this part easier 
expname = 'passive'
generate_output_dirs(expname)

# load data 
s1_selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
hdrdata_sess1, trialdata_sess1_ = loadFxs.group_quality_check(expname, 1, plot_quality_check = True)
hdrdata_sess2, trialdata_sess2_ = loadFxs.group_quality_check(expname, 2, plot_quality_check = True)
s1_stats, s1_Psurv_b1_, s1_Psurv_b2_, s1_WTW_emp = analysisFxs.group_MF(trialdata_sess1_, plot_each = False)   
s2_stats, s2_Psurv_b1_, s2_Psurv_b2_, s2_WTW_emp = analysisFxs.group_MF(trialdata_sess2_, plot_each = False)   

modelnames = ['QL2', 'QL2reset_FL2']
s1_WTW_rep_ = []
s2_WTW_rep_ = []
s1_paradf_ = []
s2_paradf_ = []
for modelname in modelnames:
    s1_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, modelname)
    s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname, modelname)
    s1_paradf_.append(s1_paradf)
    s2_paradf_.append(s2_paradf)
    s1_stats_rep, s1_WTW_rep = modelFxs.group_model_rep(trialdata_sess1_, s1_paradf, modelname, isTrct = True, plot_each = False)
    s2_stats_rep, s2_WTW_rep = modelFxs.group_model_rep(trialdata_sess2_, s2_paradf, modelname, isTrct = True, plot_each = False)
    s1_WTW_rep_.append(s1_WTW_rep)
    s2_WTW_rep_.append(s2_WTW_rep)
    s1_stats_rep.to_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_sess1.csv'%modelname), index = None)
    s2_stats_rep.to_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_sess2.csv'%modelname), index = None)
    #s1_stats_rep = pd.read_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_sess1.csv'%modelname))
    #s2_stats_rep = pd.read_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_sess2.csv'%modelname))
    # plot replication 
sns.set(font_scale = 2)
sns.set_style("white")
figFxs.plot_group_emp_rep_wtw(modelname, s1_WTW_rep_[1], s2_WTW_rep_[1], s1_WTW_emp, s2_WTW_emp, hdrdata_sess1, hdrdata_sess2, s1_paradf, s2_paradf)
plt.tight_layout()
plt.gcf().set_size_inches(12, 6)
plt.savefig(os.path.join("..", "figures", expname, "emp_rep_%s_wtw.pdf"%modelname))
    figFxs.plot_group_emp_rep(modelname, s1_stats_rep, s2_stats_rep, s1_stats, s2_stats)
    plt.gcf().set_size_inches(10, 6)
    plt.savefig(os.path.join("..", "figures", expname, "emp_rep_%s.pdf"%modelname))


methods = ['QL2', 'QL2_reset']
figFxs.plot_group_emp_rep_wtw_multi(modelname, s1_WTW_rep_, s2_WTW_rep_, s1_WTW_emp, s2_WTW_emp, hdrdata_sess1, hdrdata_sess2, s1_paradf_, s2_paradf_, methods, estimation = "mean")
plt.gcf().set_size_inches(10, 6)
plt.savefig(os.path.join("..", "figures", expname, "emp_rep_multi.pdf"))

# compare parameter reliabiliy
subtitles = [r'$\mathbf{log(\alpha)}$', r'$\mathbf{\nu}$', r'$\mathbf{\tau}$', r'$\mathbf{\gamma}$', r'$\mathbf{log(\eta)}$']
for modelname in modelnames:
    sns.set(font_scale = 1)
    sns.set_style('white')
s1_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, modelname)
s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname, modelname)
paranames = modelFxs.getModelParas(modelname)
npara = len(paranames)
# plot parameter distributions
figFxs.plot_parameter_distribution(modelname, s1_paradf.iloc[:,:-1], s2_paradf.iloc[:,:-1])
plt.gcf().set_size_inches(5 * npara, 5 * 2)
plt.savefig(os.path.join("..", 'figures', expname, "%s_para_dist.pdf"%modelname))
# plot parameter correlations
figFxs.plot_parameter_reliability(modelname, s1_paradf.iloc[:,:-1], s2_paradf.iloc[:,:-1], subtitles)
plt.gcf().set_size_inches(5 * npara, 5)
plt.savefig(os.path.join("..", 'figures', expname, "%s_para_reliability.pdf"%modelname))

    # pd.merge(s1_paradf, s2_paradf, how = 'inner', on = 'id')
