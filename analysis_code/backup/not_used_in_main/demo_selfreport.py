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
import statsmodels.formula.api as smf

# plot styles
plt.style.use('classic')
sns.set(font_scale = 2)
sns.set_style("white")
condition_palette = ["#762a83", "#1b7837"]
UPPS_subscales = ["NU", "PU", "PM", "PS", "SS"]
BIS_l1_subscales = ["Attentional", "Motor", "Nonplanning"]
BIS_l2_subscales = ["attention", "cogstable", "motor", "perseverance", "selfcontrol", "cogcomplex"]



def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model

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



# load selfreport and demographic data
hdrdata_ = []
selfdf_ = []
for expname in ["passive", "active"]:	
	hdrdata_sess2, trialdata_sess2_ = loadFxs.group_quality_check(expname, 2, plot_quality_check = True)
	hdrdata_.append(hdrdata_sess2)
	####################### analyze only selfreport data ####################
	if expname == "passive":
		s1_selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
		s2_selfdf = loadFxs.parse_group_selfreport(expname, 2, isplot = False)
		s1_selfdf = s1_selfdf[np.isin(s1_selfdf["id"], s2_selfdf["id"])]
		selfdf = analysisFxs.agg_across_sessions(s1_selfdf, s2_selfdf)
	else:
		selfdf = loadFxs.parse_group_selfreport(expname, 1, isplot = False)
	selfdf_.append(selfdf)

selfdf = pd.concat(selfdf_, axis = 0).reset_index()
hdrdata = pd.concat(hdrdata_, axis = 0).reset_index()


################ simple effects ###########
# plot 
self_vars = [ "BIS", "UPPS", "discount_logk"] + UPPS_subscales + BIS_l1_subscales + BIS_l2_subscales
demo_vars = ["age", "education"]

for var in demo_vars:
	plt.style.use('classic')
	sns.set(font_scale = 1)
	sns.set_style("white")
	df = selfdf.merge(hdrdata, on = "id")
	tmp = df.melt(id_vars = ["id", var, "gender"], value_vars = self_vars )
	g = sns.FacetGrid(data = tmp, col = "variable", sharey = False, col_order = self_vars )
	g.map(sns.regplot, var, "value", line_kws = {'color':'red'}, scatter_kws = {"color": "grey", "edgecolor": "black"})
	g.map(figFxs.annotate_reg, var, "value")
	g.savefig(os.path.join("..", "figures", "combined", var + "_selfreport_corr.pdf"))

# get p value and r value map
df = selfdf.merge(hdrdata, on = "id")
r_df = pd.DataFrame(columns = self_vars, index = demo_vars)
p_df = pd.DataFrame(columns = self_vars, index = demo_vars)
for self_var, demo_var in itertools.product(self_vars, demo_vars):
	res = spearmanr(df[self_var], df[demo_var], nan_policy = "omit")
	r_df.loc[demo_var, self_var] = res[0]
	p_df.loc[demo_var, self_var] = res[1]



####### earlier results 
# with gender 
df = selfdf.merge(hdrdata, on = "id")
df = df[np.isin(df["gender"], ["Female", "Male"])]
tmp = df.melt(id_vars = ["id", "age", "gender"], value_vars = [ "BIS", "UPPS", "survey_impulsivity", "discount_logk"])
g = sns.FacetGrid(data = tmp, col = "variable", row = "gender", sharey = False, col_order = [ "BIS", "UPPS", "survey_impulsivity", "discount_logk"], margin_titles=True )
g.map(sns.regplot, "age", "value", line_kws = {'color':'red'}, scatter_kws = {"color": "grey", "edgecolor": "black"})
g.map(figFxs.annotate_reg, "age", "value")
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.savefig(os.path.join("..", "figures", expname, "age_gender_selfreport_corr_limited.pdf"))


########
df = selfdf.merge(hdrdata, on = "id")
df = df[np.isin(df["gender"], ["Female", "Male"])]
df["age"] = df["age"] - np.mean(df["age"])
df["education"] = df["education"] - np.mean(df["education"])
res_ = []
for self_var in self_vars:
	res_.append(forward_selected(df[[self_var, "gender", "age", "education"]], self_var).summary())


full_res_ = []
for self_var in self_vars:
	full_res = smf.ols(self_var + " ~ age * gender * education", data = df[[self_var, "gender", "age", "education"]]).fit()
	full_res_.append(full_res)

score_, model_ = full_selected(df[[self_var, "gender", "age", "education"]], self_var)


best_score_ = []
best_model_ = []
all_scores_ = []
for self_var in self_vars:
	best_score, best_model, all_scores = full_selected(df[[self_var, "gender", "age", "education"]], self_var)
	best_score_.append(best_score)
	best_model_.append(best_model)
	all_scores_.append(all_scores)





