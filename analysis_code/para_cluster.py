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
from sklearn.cluster import KMeans

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
# generate_output_dirs(expname)

# load data 
hdrdata_sess1, trialdata_sess1_ = loadFxs.group_quality_check(expname, 1, plot_quality_check = False)
hdrdata_sess2, trialdata_sess2_ = loadFxs.group_quality_check(expname, 2, plot_quality_check = False)
s1_stats, s1_Psurv_b1_, s1_Psurv_b2_, s1_WTW_emp = analysisFxs.group_MF(trialdata_sess1_, plot_each = False)   
s2_stats, s2_Psurv_b1_, s2_Psurv_b2_, s2_WTW_emp = analysisFxs.group_MF(trialdata_sess2_, plot_each = False)   
s1_df = analysisFxs.pivot_by_condition(s1_stats)
s2_df = analysisFxs.pivot_by_condition(s2_stats)

plt.plot(s1_df["auc"], s1_df["auc_delta"])

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter3D(s1_df["auc"], s1_df["auc_delta"], s1_df["std_wtw"])
ax.set_xlabel("auc")
ax.set_ylabel("std_wtw")
ax.set_zlabel("auc_delta")
plt.show()

###### 
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter3D(np.log(s2_paradf["alpha"]), s2_paradf["tau"], s2_paradf["eta"])
ax.set_xlabel("log_alpha")
ax.set_ylabel("tau")
ax.set_zlabel("eta")
plt.show()

######## mf analysis #######
stats_df = s1_df
kmeans = KMeans(n_clusters=2, random_state=0).fit(stats_df[["auc", "auc_delta", "std_wtw"]])
stats_df["label"] = kmeans.labels_
stats_df['label'].value_counts()

stats_df = s2_df
kmeans = KMeans(n_clusters=2, random_state=0).fit(stats_df[["auc", "auc_delta", "std_wtw"]])
stats_df["label"] = kmeans.labels_
stats_df['label'].value_counts()



colors = ['red', "green", "blue"]
fig = plt.figure()
ax = fig.gca(projection='3d')
for i in np.arange(2):
    ax.scatter3D(stats_df.loc[stats_df["label"] == i, "auc"], stats_df.loc[stats_df["label"] == i, "std_wtw"], stats_df.loc[stats_df["label"] == i, "auc_delta"], c = colors[i])
ax.set_xlabel("auc")
ax.set_ylabel("std_wtw")
ax.set_zlabel("auc_delta")
plt.show()

s1_selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
# s1_selfdf.to_csv(os.path.join("..", "analysis_results", expname, "selfreport", "selfreport_sess1.csv"), index = None)
s2_selfdf = loadFxs.parse_group_selfreport(expname, 2, isplot = False)
s1_tmp = s1_df.merge(s1_selfdf, on = "id")
s1_tmp.groupby("label").agg({"BIS":"mean"})
sns.boxplot(data = s1_tmp, x = "label", y ="GMK")

##################### parameter analysis ###############
paradf = s1_paradf_tf
kmeans = KMeans(n_clusters=4, random_state=0).fit(paradf[["log_alpha", "tau", "log_eta", "log_nu", "gamma"]])
paradf["label"] = kmeans.labels_
paradf['label'].value_counts()
colors = ['red', "green", "blue"]

fig = plt.figure()
ax = fig.gca(projection='3d')
for i in np.arange(3):
    ax.scatter3D(paradf.loc[paradf["label"] == i, "log_alpha"], paradf.loc[paradf["label"] == i, "tau"], paradf.loc[paradf["label"] == i, "log_eta"], c = colors[i])
ax.set_xlabel("log_alpha")
ax.set_ylabel("tau")
ax.set_zlabel("eta")
plt.show()


m_paradf.loc[m_paradf["label_x"] == 0, "label_y"].value_counts()
m_paradf.loc[m_paradf["label_x"] == 2, "label_y"].value_counts()


s1_selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
# s1_selfdf.to_csv(os.path.join("..", "analysis_results", expname, "selfreport", "selfreport_sess1.csv"), index = None)
s2_selfdf = loadFxs.parse_group_selfreport(expname, 2, isplot = False)
s1_tmp = s1_paradf_tf.merge(s1_selfdf, on = "id")
s1_tmp.groupby("label").agg({"GMK":"mean"})
sns.boxplot(data = s1_tmp, x = "label", y ="GMK")
######



############# let me try the #####
# modelnames = ['QL2reset_FL3']
modelname = 'QL2reset'
fitMethod = "whole"
stepsize = 0.5

s1_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, fitMethod, stepsize)
s2_paradf = loadFxs.load_parameter_estimates(expname, 2, hdrdata_sess2, modelname, fitMethod, stepsize)

s1_paradf_tf = copy.copy(s1_paradf)
s2_paradf_tf = copy.copy(s2_paradf)
figFxs.log_transform_parameter(s1_paradf_tf, ["alpha", "nu", "eta"])
figFxs.log_transform_parameter(s2_paradf_tf, ["alpha", "nu", "eta"])


##########
kmeans = KMeans(n_clusters=3, random_state=0).fit(s1_stats[["auc", "std_wtw"]])


# s1_paradf_.append(s1_paradf)
# s2_paradf_.append(s2_paradf)
s1_stats_rep, s1_WTW_rep, s1_dist_vals_ = modelFxs.group_model_rep(trialdata_sess1_, s1_paradf, modelname, 'whole', stepsize, isTrct = True, plot_each = False)
s2_stats_rep, s2_WTW_rep, s2_dist_vals_ = modelFxs.group_model_rep(trialdata_sess2_, s2_paradf, modelname, 'whole', stepsize, isTrct = True, plot_each = False)
# s1_WTW_rep_.append(s1_WTW_rep)
# s2_WTW_rep_.append(s2_WTW_rep)
s1_stats_rep.to_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_sess1_%s_stepsize%.2f.csv'%(modelname, fitMethod, stepsize)), index = None)
s2_stats_rep.to_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_sess2_%s_stepsize%.2f.csv'%(modelname, fitMethod, stepsize)), index = None)
#s1_stats_rep = pd.read_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_sess1.csv'%modelname))
#s2_stats_rep = pd.read_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'rep_%s_sess2.csv'%modelname))
# plot replication 
sns.set(font_scale = 2)
sns.set_style("white")
figFxs.plot_group_emp_rep_wtw(modelname, s1_WTW_rep, s2_WTW_rep, s1_WTW_emp, s2_WTW_emp, hdrdata_sess1, hdrdata_sess2, s1_paradf, s2_paradf)
plt.tight_layout()
plt.gcf().set_size_inches(12, 6)
plt.savefig(os.path.join("..", "figures", expname, "emp_rep_%s_wtw_%s_stepsize%.2f.pdf"%(modelname, fitMethod, stepsize)))
figFxs.plot_group_emp_rep(modelname, s1_stats_rep, s2_stats_rep, s1_stats, s2_stats)
plt.gcf().set_size_inches(10, 6)
plt.savefig(os.path.join("..", "figures", expname, "emp_rep_%s_%s_stepsize%.2f.pdf"%(modelname, fitMethod, stepsize)))
# plot dist distributions 
s1_dist_vals = s1_dist_vals_.mean(axis = 1)
s1_dist_grand_median = np.median(s1_dist_vals_)
# median of median; 805.9735405000001
# mean of median: 810.8283439999999
s1_dist_grand_median_ = np.median(s1_dist_vals_, axis = 0)
# 
s1_ntrial_vals = [y.shape[0] for x, y in zip(hdrdata_sess1.id, trialdata_sess1_.values()) if np.isin(x, s1_paradf['id'])]
plt.scatter(s1_ntrial_vals, s1_dist_vals) ... # hmmm I am not sure ... let's use this temperally 

# 
s2_dist_vals = s2_dist_vals_.mean(axis = 1)
s2_dist_grand_median = np.median(s2_dist_vals_)
s2_dist_grand_median_ = np.median(s2_dist_vals_, axis = 0)



# methods = ['QL2', 'QL2_reset']
# figFxs.plot_group_emp_rep_wtw_multi(modelname, s1_WTW_rep_, s2_WTW_rep_, s1_WTW_emp, s2_WTW_emp, hdrdata_sess1, hdrdata_sess2, s1_paradf_, s2_paradf_, methods, estimation = "mean")
# plt.gcf().set_size_inches(10, 6)
# plt.savefig(os.path.join("..", "figures", expname, "emp_rep_multi.pdf"))


############### replicate with group-average paras
s1_median_paradf = copy.copy(s1_paradf)
s1_median_paradf.iloc[:, :5] = np.tile(np.median(s1_paradf.iloc[:,:5], axis = 0), s1_paradf.shape[0]).reshape(s1_paradf.shape[0],5)
s1_stats_rep_median, s1_WTW_rep_median, s1_dist_vals_median = modelFxs.group_model_rep(trialdata_sess1_, s1_median_paradf, modelname, 'whole', stepsize, isTrct = True, plot_each = False)
np.median(s1_dist_vals_median)
# 1371.892139

################ replicate without tau
s1_median_paradf = copy.copy(s1_paradf)
s1_median_paradf['tau'] = np.median(s1_paradf.iloc[:,2], axis = 0)
s1_stats_rep_median, s1_WTW_rep_median, s1_dist_vals_median = modelFxs.group_model_rep(trialdata_sess1_, s1_median_paradf, modelname, 'whole', stepsize, isTrct = True, plot_each = False)
np.median(s1_dist_vals_median)
# 935.7524564999999, 925.9020255 median of median
# np.median(s1_dist_vals_median, axis = 0).mean(), 925.4333684999999

################ replicate without gamma
s1_median_paradf = copy.copy(s1_paradf)
s1_median_paradf['gamma'] = np.median(s1_paradf.iloc[:,3], axis = 0)
s1_stats_rep_median, s1_WTW_rep_median, s1_dist_vals_median = modelFxs.group_model_rep(trialdata_sess1_, s1_median_paradf, modelname, 'whole', stepsize, isTrct = True, plot_each = False)
np.median(s1_dist_vals_median)
# 846.9297135, median of median
# mean of median 847.4307992000001

################ replicate without priors
s1_median_paradf = copy.copy(s1_paradf)
s1_median_paradf['prior'] = np.median(s1_paradf.iloc[:,4], axis = 0)
s1_stats_rep_median, s1_WTW_rep_median, s1_dist_vals_median = modelFxs.group_model_rep(trialdata_sess1_, s1_median_paradf, modelname, 'whole', stepsize, isTrct = True, plot_each = False)
np.median(s1_dist_vals_median)
# median of median: 812.7610375
# mean of median 819.5614511
# this makes sense, especially in the first case


################ replicate without alphas
s1_median_paradf = copy.copy(s1_paradf)
s1_median_paradf['alpha'] = np.median(s1_paradf.iloc[:,0], axis = 0)
s1_stats_rep_median, s1_WTW_rep_median, s1_dist_vals_median = modelFxs.group_model_rep(trialdata_sess1_, s1_median_paradf, modelname, 'whole', stepsize, isTrct = True, plot_each = False)
np.median(s1_dist_vals_median)
# 869.8801535000001... hmmm also little impacts 


################ replicate without nus
s1_median_paradf = copy.copy(s1_paradf)
s1_median_paradf['nu'] = np.median(s1_paradf.iloc[:,1], axis = 0)
s1_stats_rep_median, s1_WTW_rep_median, s1_dist_vals_median = modelFxs.group_model_rep(trialdata_sess1_, s1_median_paradf, modelname, 'whole', stepsize, isTrct = True, plot_each = False)
np.median(s1_dist_vals_median)
# np.median(s1_dist_vals_median) 910.9085140000001
# np.median(s1_dist_vals_median, axis = 0).mean() 912.0039198000001


################ replicate with only tau
s1_median_paradf = copy.copy(s1_paradf)
for para in paranames:
    if para != 'tau':
        s1_median_paradf[para] = np.median(s1_paradf[para], axis = 0)
s1_stats_rep_median, s1_WTW_rep_median, s1_dist_vals_median = modelFxs.group_model_rep(trialdata_sess1_, s1_median_paradf, modelname, 'whole', stepsize, isTrct = True, plot_each = False)
# np.median(s1_dist_vals_median), 1705.9849769999998, 1714.100638
# np.median(s1_dist_vals_median, axis = 0).mean(), 1710.1806980000001



################ replicate with only alpha
s1_median_paradf = copy.copy(s1_paradf)
for para in paranames:
    if para != 'alpha':
        s1_median_paradf[para] = np.median(s1_paradf[para], axis = 0)
s1_stats_rep_median, s1_WTW_rep_median, s1_dist_vals_median = modelFxs.group_model_rep(trialdata_sess1_, s1_median_paradf, modelname, 'whole', stepsize, isTrct = True, plot_each = False)
# np.median(s1_dist_vals_median), 1354.3742914999998
# np.median(s1_dist_vals_median, axis = 0).mean(), 1356.4666567



################ replicate with only nu
s1_median_paradf = copy.copy(s1_paradf)
for para in paranames:
    if para != 'nu':
        s1_median_paradf[para] = np.median(s1_paradf[para], axis = 0)
s1_stats_rep_median, s1_WTW_rep_median, s1_dist_vals_median = modelFxs.group_model_rep(trialdata_sess1_, s1_median_paradf, modelname, 'whole', stepsize, isTrct = True, plot_each = False)
# np.median(s1_dist_vals_median), 1395.475888
# np.median(s1_dist_vals_median, axis = 0).mean(), 1397.1704942000001


################ replicate with only gamma
s1_median_paradf = copy.copy(s1_paradf)
for para in paranames:
    if para != 'gamma':
        s1_median_paradf[para] = np.median(s1_paradf[para], axis = 0)
s1_stats_rep_median, s1_WTW_rep_median, s1_dist_vals_median = modelFxs.group_model_rep(trialdata_sess1_, s1_median_paradf, modelname, 'whole', stepsize, isTrct = True, plot_each = False)
# np.median(s1_dist_vals_median), 1294.1244175
# np.median(s1_dist_vals_median, axis = 0).mean(), 1294.4569675000002




######### using different distance calculations 
######### using with methods not the without method

################ ok this dosen't seem to work 
################ let me try the distance calculation 
# compare parameter reliabiliy
subtitles = [r'$\mathbf{log(\alpha)}$', r'$\mathbf{log(\nu)}$', r'$\mathbf{\tau}$', r'$\mathbf{\gamma}$', r'$\mathbf{log(\eta)}$']
paranames = modelFxs.getModelParas(modelname)
npara = len(paranames)
# plot parameter distributions
figFxs.plot_parameter_distribution(modelname, s1_paradf.iloc[:,:-1], s2_paradf.iloc[:,:-1], color = "grey", edgecolor = "black")
plt.gcf().set_size_inches(5 * npara, 5 * 2)
plt.savefig(os.path.join("..", 'figures', expname, "%s_%s_stepsize%.2f_para_dist.pdf"%(modelname, fitMethod, stepsize)))
# plot parameter correlations
figFxs.plot_parameter_reliability(modelname, s1_paradf.iloc[:,:-1], s2_paradf.iloc[:,:-1], subtitles)
plt.gcf().set_size_inches(5 * npara, 5)
plt.savefig(os.path.join("..", 'figures', expname, "%s_%s_stepsize%.2f_para_reliability.pdf"%(modelname, fitMethod, stepsize)))


#####################################################
##################### split half reliability ########
modelname = 'QL2reset_FL2'
stepsize = 0.5
s1_even_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, 'even', stepsize)
s1_odd_paradf = loadFxs.load_parameter_estimates(expname, 1, hdrdata_sess1, modelname, 'odd', stepsize)
figFxs.plot_parameter_reliability(modelname, s1_even_paradf.iloc[:,:-1], s1_odd_paradf.iloc[:,:-1], subtitles)
plt.savefig(os.path.join("..", 'figures', expname, "%s_stepsize%.2f_para_split_hald.pdf"%(modelname, stepsize)))

# is the reliability superficial? 
a = pd.merge(s1_paradf, s1_stats, how = 'inner', on = 'id')
spearmanr(a.loc[a.block == 1, 'tau'], a.loc[a.block == 1, 'auc'])
spearmanr(a.loc[a.block == 2, 'tau'], a.loc[a.block == 2, 'auc'])
# .... hmmm
spearmanr(a.loc[a.block == 1, 'std_wtw'], a.loc[a.block == 1, 'tau'])
spearmanr(a.loc[a.block == 2, 'std_wtw'], a.loc[a.block == 2, 'tau'])

# prior has high correlation with AUC





