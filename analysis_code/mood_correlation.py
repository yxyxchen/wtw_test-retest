
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



s1_selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
# s1_selfdf.to_csv(os.path.join("..", "analysis_results", expname, "selfreport", "selfreport_sess1.csv"), index = None)
s2_selfdf = loadFxs.parse_group_selfreport(expname, 2, isplot = False)
# s2_selfdf.to_csv(os.path.join("..", "analysis_results", expname, "selfreport", "selfreport_sess2.csv"), index = None)

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
to_be_tested_vars = list(zip([x + "_sess1" for x in expParas.selfreport_vars], [x + "_sess2" for x in expParas.selfreport_vars]))
spearman_rho_, pearson_rho_, abs_icc_, con_icc_, n_, report = analysisFxs.calc_zip_reliability(selfdf, to_be_tested_vars)
report.sort_values(by = "spearman_rho")



# I would like to know autocorrelations though ...




###### hmmm let me calcualte changes in PNA ..., changes in the two dimensions are not correlated 
selfdf = analysisFxs.sub_across_sessions(s1_selfdf, s2_selfdf, vars = ["PAS", "NAS", "UPPS", "BIS", "GMK"])
selfdf[["PAS_diff", "NAS_diff"]].plot.hist()
selfdf[["PAS_diff", "NAS_diff"]].plot.scatter(x = "NAS_diff", y = "PAS_diff") # is this really meaningful??? or is it just ramdom???

statsdf = analysisFxs.sub_across_sessions(analysisFxs.pivot_by_condition(s1_stats), analysisFxs.pivot_by_condition(s2_stats))

df = selfdf[["id", "PAS_diff", "NAS_diff", "UPPS_diff", "BIS_diff", "GMK_diff"]].merge(statsdf[["id", "auc_diff", "std_wtw_diff", "auc_delta_diff"]], on = "id")
r_, p_ = analysisFxs.calc_prod_correlations(df, ["PAS_diff", "NAS_diff"], ["auc_diff", "std_wtw_diff", "auc_delta_diff"])

r_, p_ = analysisFxs.calc_prod_correlations(df, ["PAS_diff", "NAS_diff"], ["UPPS_diff", "BIS_diff", "GMK_diff"])

r_, p_ = analysisFxs.calc_prod_correlations(df, ["auc_diff", "std_wtw_diff", "auc_delta_diff"], ["UPPS_diff", "BIS_diff", "GMK_diff"])
#######


