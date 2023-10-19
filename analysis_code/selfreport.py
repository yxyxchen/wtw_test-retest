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

# plot constants
selfreport_vars = ["BIS", "UPPS", "discount_logk", "Motor", "Nonplanning", "Attentional", "motor", "perseverance", "selfcontrol", "cogcomplex", "cogstable", "attention", "NU", "PU", "PM", "PS", "SS"]
selfreport_totalscores = ["BIS", "UPPS", "discount_logk"]

############# plot reliability ########
# load hdrdata 
expname = "passive"
hdrdata, _ = loadFxs.group_quality_check(expname, 2, plot_quality_check = True)
s1_selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
s2_selfdf = loadFxs.parse_group_selfreport(expname, 2, isplot = False)
s1_selfdf = s1_selfdf[np.isin(s1_selfdf["id"], hdrdata["id"])]
s2_selfdf = s2_selfdf[np.isin(s2_selfdf["id"], hdrdata["id"])] 
selfdf = analysisFxs.agg_across_sessions(s1_selfdf, s2_selfdf)

# plot reliability 
# plt.style.use('classic')
# sns.set(font_scale = 1)
# sns.set_style("white")
# df = s1_selfdf.melt(id_vars = "id", value_vars = selfreport_totalscores).merge(s2_selfdf.melt(id_vars = "id", value_vars = selfreport_totalscores), on = ["id", "variable"], suffixes = ["_sess1", "_sess2"])
# g = sns.FacetGrid(data  = df, col = "variable", sharex = False, sharey = False)
# g.map(figFxs.my_regplot, "value_sess1", "value_sess2")
# g.set_titles(col_template="{col_name}")
# g.set(xlabel='Session 1', 
#       ylabel='Session 2')
# g.savefig(os.path.join("..", "figures", expname, "selfreport_totalscore_reliability.pdf"))
# df = s1_selfdf.melt(id_vars = "id", value_vars = selfreport_vars).merge(s2_selfdf.melt(id_vars = "id", value_vars = selfreport_vars), on = ["id", "variable"], suffixes = ["_sess1", "_sess2"])
# g = sns.FacetGrid(data  = df, col = "variable", sharex = False, sharey = False)
# g.map(figFxs.my_regplot, "value_sess1", "value_sess2")
# g.set_titles(col_template="{col_name}")
# g.set(xlabel='Session 1', 
#       ylabel='Session 2')
# g.savefig(os.path.join("..", "figures", expname, "selfreport_reliability.pdf"))

# calculate reliability table
df = [s1_selfdf, s2_selfdf]
UPPS_subscales = ["NU", "PU", "PM", "PS", "SS"]
BIS_l1_subscales = ["Attentional", "Motor", "Nonplanning"]
BIS_l2_subscales = ["attention", "cogstable", "motor", "perseverance", "selfcontrol", "cogcomplex"]

_, _, _, _, _, report = analysisFxs.calc_zip_reliability(analysisFxs.hstack_sessions(s1_selfdf, s2_selfdf), [(x + '_sess1', x + '_sess2') for x in selfreport_totalscores + UPPS_subscales + BIS_l2_subscales])
report.round(3).to_csv(os.path.join("..", "figures", expname, "selfreport_reliability.csv"))

############## plot practice effect ####################
# df = analysisFxs.vstack_sessions(s1_selfdf.melt(id_vars = "id", value_vars = selfreport_totalscores), s2_selfdf.melt(id_vars = "id", value_vars = selfreport_totalscores))
# g = sns.FacetGrid(data = df, col = "variable", sharex = False, sharey = False)
# g.map(sns.swarmplot, "sess", "value", color = "grey", edgecolor = "black", alpha = 0.4, linewidth=1,  size = 3)
# g.map(sns.boxplot, "sess", "value", boxprops={'facecolor':'None'}, medianprops={"linestyle":"--", "color": "red"})
# g.set_titles(col_template="{col_name}")
# g.set(xlabel='')
# g.savefig(os.path.join("..", "figures", expname, "selfreport_totalscore_practice.pdf"))

# df = analysisFxs.vstack_sessions(s1_selfdf.melt(id_vars = "id", value_vars = selfreport_vars), s2_selfdf.melt(id_vars = "id", value_vars = selfreport_vars))
# g = sns.FacetGrid(data = df, col = "variable", sharex = False, sharey = False)
# g.map(sns.swarmplot, "sess", "value", color = "grey", edgecolor = "black", alpha = 0.4, linewidth=1,  size = 3)
# g.map(sns.boxplot, "sess", "value", boxprops={'facecolor':'None'}, medianprops={"linestyle":"--", "color": "red"})
# g.set_titles(col_template="{col_name}")
# g.set(xlabel='')
# g.savefig(os.path.join("..", "figures", expname, "selfreport_practice.pdf"))


#################### plot correlations among selfreport datas
# load data 
selfdf_ = []
for expname in ["passive", "active"]:	
	####################### analyze only selfreport data ####################
	if expname == "passive":
		hdrdata, _ = loadFxs.group_quality_check(expname, 2, plot_quality_check = True)
		s1_selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
		s2_selfdf = loadFxs.parse_group_selfreport(expname, 2, isplot = False)
		s1_selfdf = s1_selfdf[np.isin(s1_selfdf["id"], hdrdata["id"])]
		s2_selfdf = s2_selfdf[np.isin(s2_selfdf["id"], hdrdata["id"])]
		selfdf = analysisFxs.agg_across_sessions(s1_selfdf, s2_selfdf)
	else:
		hdrdata, _ = loadFxs.group_quality_check(expname, 2, plot_quality_check = True)
		selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
		selfdf = selfdf[np.isin(selfdf["id"], hdrdata["id"])]
	selfdf_.append(selfdf)

selfdf = pd.concat(selfdf_, axis = 0).reset_index()

plt.style.use('classic')
sns.set(font_scale = 2)
sns.set_style("white")
g = sns.pairplot(selfdf[["UPPS", "BIS", "discount_logk"]], kind = "reg", diag_kws = {"color": "grey", "edgecolor": "black"},\
	plot_kws ={'line_kws':{'color':'red'}, "scatter_kws": {"color": "grey", "edgecolor": "black"}})
g.map_lower(figFxs.annotate_reg)
plt.savefig(os.path.join("..", "figures", "combined", "impulsivity_corr.pdf"))



