import pandas as pd
import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
# plot styles
plt.style.use('classic')
sns.set(font_scale = 1.5)
sns.set_style("white")
import itertools
import copy # pay attention to copy 
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sksurv.nonparametric import kaplan_meier_estimator as km
from scipy.interpolate import interp1d
from subFxs import expParas
from scipy import stats
import code
import importlib
import random


####### functions to manipulate data 

def split_odd_even(trialdata_):
    even_trialdata_ = dict()
    odd_trialdata_ = dict()
    for key in trialdata_.keys():
        trialdata = trialdata_[key]
        even_trialdata = trialdata[trialdata.trialIdx % 2 == 0].reset_index()
        odd_trialdata = trialdata[trialdata.trialIdx % 2 == 1].reset_index()
        odd_trialdata_[key] = odd_trialdata
        even_trialdata_[key] = even_trialdata
    return odd_trialdata_, even_trialdata_

def vstack_sessions(s1_df, s2_df):
    s1_df['sess'] = "Session 1"
    s2_df['sess'] = "Session 2"
    df = pd.concat([s1_df[np.isin(s1_df.id, s2_df.id)], s2_df], axis = 0)
    return df


def hstack_sessions(s1_df, s2_df, on_var = "id", suffixes = ["_sess1", "_sess2"]):
    df = s1_df.merge(s2_df, on = on_var, suffixes = suffixes)
    return df


def pivot_by_condition(df):
    """ pivot a table of summary statistics based on condition 
    """
    # code.interact(local = dict(locals(), **globals()))
    if "ipi" in df:
        columns = ['auc', 'std_wtw', "ipi", "diff_auc", "diff_wtw"]
    else: 
        columns = ['auc', 'std_wtw', "diff_auc", "diff_wtw"]
    index = ['id']
    HP_df = df.loc[df['condition'] == 'HP', columns + index]
    LP_df = df.loc[df['condition'] == 'LP', columns + index]
    out_df = HP_df.merge(LP_df, left_on = index, right_on = index, suffixes = ['_HP', "_LP"])
    out_df['auc_delta'] = out_df['auc_HP'] - out_df['auc_LP']
    out_df['auc'] = (out_df['auc_HP'] + out_df['auc_LP']) / 2
    out_df["init_wtw"] = df.loc[df['condition'] == 'LP', "init_wtw"].values
    if "ipi" in df:
        out_df['ipi'] = (out_df['ipi_HP'] + out_df['ipi_LP']) / 2
    out_df['std_wtw'] = (out_df['std_wtw_HP']**2 / 2 + out_df['std_wtw_LP']**2 / 2)**0.5
    return out_df



def agg_across_sessions(s1_df, s2_df):
    """ average data across sessions, used for correlation analyses
    """
    df = vstack_sessions(s1_df, s2_df)
    # make it better later
    outdf = df.groupby("id").mean().reset_index()
    return outdf


def sub_across_sessions(s1_df, s2_df, vars = ["auc", "std_wtw", "auc_delta"]):
    """ calc the difference between the two sessions
    """
    df = hstack_sessions(s1_df, s2_df)
    for var in vars:
        df[var + "_diff"] = df[var + "_sess2"] - df[var + "_sess1"]
    return df



def calc_se(x):
    """calculate standard error after removing na values 
    """
    # if not isinstance(x, pd.Series):
    #     x = pd.Series(x)

    if not isinstance(x, np.ndarray):
        x = np.array(x)
    size = len(x)
    x = x[~np.isnan(x)]
    ndrop = size - len(x) 
    if ndrop > 0:
        print("Remove NaN values in calculating standard error"%ndrop)
    return np.nanstd(x) / np.sqrt(len(x))


################ I want to write some correlation functions that I can use repeatedly ##########
# def calc_cross_correlations(df, row_vars, col_vars):
def calc_zip_reliability(df, vars):
    n_var = len(vars)
    spearman_rho_ = np.zeros(n_var)
    # spearman_p_ = np.zeros(n_var)
    pearson_rho_ = np.zeros(n_var)
    # pearson_p_ = np.zeros(n_var)
    abs_icc_ = np.zeros(n_var)
    con_icc_ = np.zeros(n_var)
    n_ = np.zeros(n_var)

    for i, var in enumerate(vars):
        spearman_rho, pearson_rho, abs_icc, con_icc, ssbs, ssbm, sse, msbs, msbm, mse = calc_reliability(df[var[0]].values, df[var[1]].values)
        spearman_rho_[i] = spearman_rho
        pearson_rho_[i] = pearson_rho
        abs_icc_[i] = abs_icc
        con_icc_[i] = con_icc
        n_[i] = min(np.sum(~np.isnan(df[var[0]])), np.sum(~np.isnan(df[var[1]])))
    
    report = pd.DataFrame({
        "spearman_rho": spearman_rho_,
        "pearson_rho": pearson_rho_,
        "abs_icc": abs_icc_,
        "con_icc": con_icc_,
        "sample size" : n_
        }, index = vars)
    return spearman_rho_, pearson_rho_, abs_icc_, con_icc_, n_, report

def calc_zip_correlations(df, vars):
    n_var = len(vars)
    rs = np.zeros(n_var)
    ps = np.zeros(n_var)
    ns = np.zeros(n_var)
    for i, var in enumerate(vars):
        res = stats.spearmanr(df[var[0]], df[var[1]], nan_policy = 'omit')
        rs[i] = res[0]
        ps[i] = res[1]
        ns[i] = min(np.sum(~np.isnan(df[var[0]])), np.sum(~np.isnan(df[var[1]])))
    report = pd.DataFrame({
        "Spearman's r": rs,
        "p": ps,
        "sample size" : ns
        }, index = vars)
    return rs, ps, ns, report

def calc_prod_correlations(df, row_vars, col_vars):
    nrow = len(row_vars)
    ncol = len(col_vars)
    r_ = pd.DataFrame(np.full((nrow, ncol), np.nan), index = row_vars, columns = col_vars)
    p_ = pd.DataFrame(np.full((nrow, ncol), np.nan), index = row_vars, columns = col_vars)
    for row_var, col_var in itertools.product(row_vars, col_vars):
        res = stats.spearmanr(df[row_var], df[col_var], nan_policy = 'omit')
        # res = stats.pearsonr(df[row_var], df[col_var])
        r_.loc[row_var, col_var] = res[0]
        p_.loc[row_var, col_var] = res[1]
    return r_, p_

#########
def calc_icc(x, y):
    """ a function to calc icc
        References: Liljequist, D., Elfving, B., & Roaldsen, K. S. (2019). https://doi.org/10.1371/journal.pone.0219854
    """
    k = 2
    n = len(x) # number of measures 
    gm = np.mean(np.concatenate([x, y])) # grand mean 
    sm = (x + y) / k # subject-wise mean
    mm = np.array([x.mean(), y.mean()]) # measure-wise mean
    ssbs = np.sum((sm - gm)**2 * k) # sum of square between subjects
    ssbm = np.sum((mm - gm) **2 * n) # sum of square between measures
    sst = np.sum(np.concatenate([(x - gm)**2, (y - gm)**2]))
    sse = sst - ssbs - ssbm
    msbs = ssbs / (n-1)
    msbm = ssbm / (k-1)
    mse = sse / (n-1) * (k-1)
    abs_icc = (msbs - mse)  / (msbs + (k-1) * mse + k / n * (msbm - mse))
    con_icc = (msbs - mse)  / (msbs + (k-1) * mse)

    return abs_icc, con_icc, ssbs, ssbm, sse, msbs, msbm, mse

def calc_reliability(x, y):
    x, y = x[np.logical_and(~np.isnan(x), ~np.isnan(y))], y[np.logical_and(~np.isnan(x), ~np.isnan(y))]
    spearman_rho = stats.spearmanr(x, y, nan_policy = "omit")[0]
    pearson_rho = stats.pearsonr(x, y)[0]
    abs_icc, con_icc, ssbs, ssbm, sse, msbs, msbm, mse = calc_icc(x, y)
    return spearman_rho, pearson_rho, abs_icc, con_icc, ssbs, ssbm, sse, msbs, msbm, mse

def calc_bootstrap_reliability(x, y, n_perm = 1000, alpha = 0.95, n = None):
    """ calc reliability measure, bootstrapped CI and distribution. 
        Notice here using the bootstrap we are constructing the distribution of r, rather than the NULL distribution 
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    if not isinstance(y, np.ndarray):
        y = np.array(y)
    # calc the observed r
    r = stats.spearmanr(x, y, nan_policy = 'omit')[0]

    # determine the sample size
    if n is None:
        n = len(x)

    # bootstrap 
    random.seed(123)
    # spearman_rho_ = np.zeros(n_perm)
    # pearson_rho_ = np.zeros(n_perm)
    abs_icc_ = np.zeros(n_perm)
    # con_icc_ = np.zeros(n_perm)
    for i in range(n_perm):
        idxs = np.random.choice(len(x), n, replace=True)
        # spearman_rho_[i] = stats.spearmanr(x[idxs], y[idxs], nan_policy = 'omit')[0]
        # pearson_rho_[i] = stats.pearsonr(x[idxs], y[idxs])[0]
        tmp = calc_icc(x[idxs], y[idxs])
        abs_icc_[i] = tmp[0]
        # con_icc_[i] = tmp[1]

    # calc confidence intervals
    ci_lower = np.percentile(abs_icc_, (1 - alpha) / 2 * 100)
    ci_upper = np.percentile(abs_icc_, 100 - (1 - alpha) / 2 * 100)
    return abs_icc_, ci_lower, ci_upper


# def my_bootstrap_corr(x, y, n_perm = 1000, alpha = 0.95, n = None):
#     """ calc Spearman r, bootstrapped CI and distribution. 
#         Notice here using the bootstrap we are constructing the distribution of r, rather than the NULL distribution 
#     """
#     if not isinstance(x, np.ndarray):
#         x = np.array(x)

#     if not isinstance(y, np.ndarray):
#         y = np.array(y)
#     # calc the observed r
#     r = stats.spearmanr(x, y, nan_policy = 'omit')[0]

#     # determine the sample size
#     if n is None:
#         n = len(x)

#     # bootstrap 
#     random.seed(123)
#     r_ = np.zeros(n_perm)
#     for i in range(n_perm):
#         idxs = np.random.choice(len(x), n, replace=True)
#         r_[i] = stats.spearmanr(x[idxs], y[idxs], nan_policy = 'omit')[0]

#     # calc confidence intervals
#     ci_lower = np.percentile(r_, (1 - alpha) / 2 * 100)
#     ci_upper = np.percentile(r_, 100 - (1 - alpha) / 2 * 100)

#     return r, (ci_lower, ci_upper), r_

def my_compare_correlations(x1, x2, y1, y2, n_perm = 1000):
    """
        assume there are two samples, x:(x1,x2) and y(y1, y2)

        I probably will lost my references here

    """
    # constants 
    nx = len(x1)
    ny = len(y1)

    # create the standardized pooled samples
    norm_x1 = (x1 - np.mean(x1)) / np.std(x1)
    norm_y1 = (y1 - np.mean(y1)) / np.std(y1)
    norm_x2 = (x2 - np.mean(x2)) / np.std(x2)
    norm_y2 = (y2 - np.mean(y2)) / np.std(y2)

    def calc_T_tilda(s1_x, s2_x, s1_y, s2_y, nx, ny):  
        """ 
            nx is the sample size of the first group
            ny is the sample size of the second group
            s1 is the first session of the standardized pooled sample
            s2 is the second session of the standardized pooled sample
        """ 
        # 
        n = nx + ny

        # # spilt the standardized pooled sample
        # s1_x = s1[:nx]
        # s1_y = s1[nx:]

        # s2_x = s2[:nx]
        # s2_y = s2[nx:]

            # calc the testing statistics 
        r_x = stats.spearmanr(s1_x, s2_x, nan_policy = 'omit')[0]
        #r_x = stats.pearsonr(s1_x, s2_x, nan_policy = 'omit')[0]
        r_y = stats.spearmanr(s1_y, s2_y, nan_policy = 'omit')[0]
        #r_y = stats.pearsonr(s1_y, s2_y, nan_policy = 'omit')[0]

        # 
        r_avg = (nx / n * r_x + ny / n * r_y)
        z_x = s1_x * s2_x - 0.5 * r_avg * (s1_x**2 + s2_x**2)
        z_y = s1_y * s2_y - 0.5 * r_avg * (s1_y**2 + s2_y**2)

        z_bar_x = np.mean(z_x)
        z_bar_y = np.mean(z_y)

        var_x = 1 / (nx - 1) * np.sum((z_x - z_bar_x)**2)
        var_y = 1 / (ny - 1) * np.sum((z_y - z_bar_y)**2)


        var_T = var_x / nx + var_y / ny

        T = np.sqrt(nx * ny / n) * (r_x - r_y) 

        T_tilda = T / np.sqrt(var_T)

        return r_x, r_y, T_tilda

    # calc the observed test statistics
    # s1 = np.concatenate((norm_x1, norm_y1))
    # s2 = np.concatenate((norm_x2, norm_y2))

    # 
    r_x_observed, r_y_observed, T_tilda_observed = calc_T_tilda(norm_x1, norm_x2, norm_y1, norm_y2, nx, ny)

    # calc test statistics for permuated samples 
    T_tilda_sampled = np.zeros(n_perm)
    r_x_sampled= np.zeros(n_perm)
    r_y_sampled = np.zeros(n_perm)
    for i in range(n_perm):
        # idxs = np.random.permutation(nx + ny)
        s1_x = norm_x1
        s2_x = norm_x2
        s1_y = norm_y1
        s2_y = norm_y2
        idxs = np.random.choice([True, False], nx)
        s1_x[idxs], s1_y[idxs] = s1_y[idxs], s1_x[idxs] # switch x and y labels
        s2_x[idxs], s2_y[idxs] = s2_y[idxs], s2_x[idxs]
        r_x_sampled[i], r_y_sampled[i], T_tilda_sampled[i] = calc_T_tilda(s1_x, s2_x, s1_y, s2_y, nx, ny)

    # calc p values
    # r_diff_sampled = r_x_sampled - r_y_sampled
    p_val = np.mean(np.abs(T_tilda_sampled) > np.abs(T_tilda_observed))
    # code.interact(local = dict(locals(), **globals()))
    return r_x_observed, r_y_observed, p_val

# I think I can first focus on paired permustations first
def my_paired_permutation(x, y, func, n_perm = 1000):
    # calculate the observed critical stats
    observed_t = func(x, y)

    # save the distribution of the critical statistics 
    permutated_t_ = np.zeros(n_perm)

    for i in range(n_perm):
        # flip random data points
        flipped_idxs = np.random.choice([True, False], len(x))
        x_prime = x
        y_prime = y
        x_prime[flipped_idxs], y_prime[flipped_idxs] = y_prime[flipped_idxs], x_prime[flipped_idxs]

        permutated_t_[i] = func(x_prime, y_prime)

    # calculate p value 
    p = np.mean(np.absolute(permutated_t_) > abs(observed_t))

    # return outputs
    return observed_t, permutated_t_, p

def my_paired_multiple_permuation(X, Y, func, n_perm = 1000):
    # constants 
    n_obs = X.shape[0]
    n_var = X.shape[1]

    # save the distribution of the critical statistics 
    permutated_abs_t_max_ = np.zeros(n_perm)
    for i in range(n_perm):
        print(i)
        flipped_idxs = np.random.choice([True, False], n_obs)
        permutated_abs_t_max = -10; # I think I should focus on absolute values here
        for j in range(n_var):
            x_prime = X[:, j]
            y_prime = Y[:, j]
            x_prime[flipped_idxs], y_prime[flipped_idxs] = y_prime[flipped_idxs], x_prime[flipped_idxs]
            t = func(x_prime, y_prime)
            if abs(t) > permutated_abs_t_max:
                permutated_abs_t_max = abs(t)
        permutated_abs_t_max_[i] = permutated_abs_t_max

    # calc the observed critical statistics and pvals
    observed_t_ = np.zeros(n_var)
    p_ = np.zeros(n_var)
    for i in range(n_var):
        observed_t_[i] = func(X[:,i], Y[:, i])
        p_[i] = np.mean(permutated_abs_t_max_ > abs(observed_t_[i]))

    return observed_t_, permutated_abs_t_max_, p_

####################################################
def score_PANAS(choices, isplot = True):
    """score PANAS questionaire answers

    Inputs:
        choices: choice data, from 1 to 4
        isplot: whether to plot figures or not
    """
    # questionaire inputs: items and reversed items for each component 
    PAS_items = [1, 3, 5, 9, 10, 12, 14, 16, 17, 19] # positive affect items. Scores can range from 10-50, higher scores -> higher lvels of positive affect
    NAS_items  = [2, 4, 6, 7, 8, 11, 13, 15, 18, 20] # negative affect items. higher scores -> higher lvels of positive affect

    # calculate PAS and NAS scores
    PAS = choices.loc[['PA-' + str(x) for x in PAS_items]].sum()
    NAS = choices.loc[['PA-' + str(x) for x in NAS_items]].sum()

    # plot
    if isplot:
        fig, ax = plt.subplots()
        ax.acorr(choices, maxlags = 10)
        ax.set_ylabel("PANAS Autocorrelation")
        ax.set_xlabel("Lags")

    return PAS, NAS



def score_BIS(choices, isplot = True):
    """score BIS questionaire answers

    Inputs:
        choices: choice data, from 1 to 4
        isplot: whether to plot figures or not

    Outputs:

    """
    # questionaire inputs: items and reversed items for each component
    attention_items = [[5, 11, 28], [9, 20]]
    cogstable_items = [[6, 24, 26], []]
    motor_items = [[2, 3, 4, 17, 19, 22, 25], []]
    perseverance_items = [[16, 21, 23], [30]]
    selfcontrol_items = [[], [1, 7, 8, 12, 13,  14]]
    cogcomplex_items = [[18, 27], [10, 15, 29]]

    def items2score(items):
        score = np.sum([choices.loc['BIS-'+str(item)] for item in items[0]]) +\
        np.sum([5 - choices.loc['BIS-'+str(item)] for item in items[1]])
        return score

    # calculate scores 
    attention = items2score(attention_items)
    cogstable = items2score(cogstable_items)
    motor = items2score(motor_items)
    perseverance = items2score(perseverance_items)
    selfcontrol = items2score(selfcontrol_items)
    cogcomplex = items2score(cogcomplex_items)
    
    Attentional = attention + cogstable
    Motor = motor + perseverance
    Nonplanning  = selfcontrol + cogcomplex
    BIS = Attentional + Motor + Nonplanning

    # plot 
    if isplot:
        fig, ax = plt.subplots()
        ax.acorr(choices, maxlags = 10)
        ax.set_ylabel("BIS Autocorrelation")
        ax.set_xlabel("Lags")
    return (attention, cogstable, motor, perseverance, selfcontrol, cogcomplex), Attentional, Motor, Nonplanning, BIS

def score_upps(choices, isplot = True):
    """score upps questionaire answers 
    Inputs:
        choices: choice data, from 1 to 4
        isplot: whether to plot figures or not 
    
    Outputs:
        NU: negative urgency, 
        PU: postitive urgency 
        PM: lack of premeditation
        PS: lack of perseverance
        SS: sensation seeking
        UPPS: overall score. Higher values mean more impulsive 
        
    """
    # questionaire inputs: items and reversed items for each component
    NU_items =  [[53], [2, 7, 12, 17, 22, 29, 34, 39, 44, 50, 58]] # negative urgency 
    PU_items = [[], [5, 10,  15, 20, 25, 30, 35, 40, 45, 49, 52, 54, 57, 59]] # positive urgency
    PM_items = [[1, 6, 11, 16, 21, 28, 33, 38, 43, 48, 55], []] # lack of premeditation
    PS_items = [[4, 14, 19, 24, 27, 32, 37,  42], [9, 47]] # lack of perseverance
    SS_items = [[], [3, 8, 13, 18, 23, 26, 31, 36, 41, 46, 51, 56]] # sensation seeking
    
    def items2score(items):
        score = np.sum([choices.loc['UP-'+str(item)] for item in items[0]]) +\
        np.sum([5 - choices.loc['UP-'+str(item)] for item in items[1]])
        return score

    # calculate scores 
    NU = items2score(NU_items)
    PU = items2score(PU_items)
    PM = items2score(PM_items)
    PS = items2score(PS_items)
    SS = items2score(SS_items)
    UPPS = NU + PU + PM + PS + SS
    
    # determine data quality 
    if isplot:
        fig, ax = plt.subplots()
        ax.acorr(choices, maxlags = 10)
        ax.set_ylabel("UPPS Autocorrelation")
        ax.set_xlabel("Lags")
    # c = collections.Counter(choices.values)
    return NU, PU, PM, PS, SS, UPPS
        
    
def calc_k(ddchoices, isplot = True):
    """Estimate k, log(k) and the standard error of log(k) for the questionaire
    
    Inputs:
        ddchoices: a pandas series of choice data. 1 -> immediate reward, 2-> delayed reward
        
    Outputs:
        k: discounting parameter k
        logk: log(k)
        se_logk: standard error of log(k)
        
    Comments:
        Here we use the logistic regression method described in Wileyto et al. (2004).
        It enables us to gauge the uncertianty in parameter estimation.
        The alternative method, one that is based on indifference points,is ideal for determining the shape of the function. 
    """
    #print(ddchoices)
    # questionaire inputs
    Vi = np.array([54, 55, 19, 31, 14, 47, 15, 25, 78, 40, 11, 67, 34, 27, 69, 49, 80, 24 ,33, 28, 34, 25, 41, 54, 54, 22, 20]) # immediate reward
    Vd = np.array([55, 75, 25, 85, 25, 50, 35, 60, 80, 55, 30, 75, 35, 50, 85, 60, 85, 35, 80, 30, 50, 30, 75, 60, 80, 25, 55]) # delayed reward
    T = np.array([117, 61, 53, 7, 19, 160, 13, 14, 162, 62, 7, 119, 186, 21, 91, 89, 157, 29, 14, 179, 30, 80, 20, 111, 30, 136, 7]) # delay in days
    
    # transformed variables 
    R = Vi / Vd #reward ratio
    TR = 1 - 1/R # transformed reward ratio
    pD = ddchoices.values.astype("float") - 1 # prob of choosing delayed rewards
    percentD = pD.sum() / len(pD)
    
    # logistic regression
    np.column_stack((pD,TR, T))
    regdf = pd.DataFrame({
        "Vi" : Vi,
        "Vd" : Vd, 
        'R': R,
        "T": T,
        "TR": TR,
        "pD": pD,
        
    })
    if all(pD == 1) or all(pD == 0) or sum(abs(np.diff(regdf.sort_values(by = 'TR').pD))) <= 1 \
    or sum(abs(np.diff(regdf.sort_values(by = 'T').pD))) <= 1: 
        print("All the same intertemporal choices.")
        k = np.nan
        logk = np.nan
        var_logk = np.nan
        se_logk = np.nan
    else:
        try:
            results = smf.glm("pD ~ -1 + TR + T",data = regdf, family=sm.families.Binomial()).fit()
        except:
            print("Use Bayesian Methods here")
            code.interact(local = dict(locals(), **globals()))

        # calculate k and related stats
        k = results.params[1]/results.params[0]
        logk = np.log(k)
        g = np.array([-1 / results.params[0], 1 / results.params[1]]) # first order derivative of logk on betas 
        var_logk = g.dot(results.cov_params()).dot(g.T)
        se_logk = np.sqrt(var_logk)
    
        # calculate SV
        regdf['SV'] = regdf.eval("Vd / (1 + @k * T) / Vi")
        
        # bin the SV variable
        regdf['SV_bin'] = pd.qcut(regdf['SV'], 5)

        # if plot figures 
        if isplot:
            fig, ax = plt.subplots(1,2)
            # check data quality
            ax[0].plot(pD)
            
            # check model fit 
            plotdf = regdf.groupby("SV_bin").mean()
            plotdf.plot("SV", "pD", ax = ax[1])
            ax[1].vlines(1,0,1, color = "r", linestyles ="dotted")
            ax[1].hlines(0.5,0,3, color = "r", linestyles ="dotted")
            ax[1].set_xlabel("SV (fraction of the immediate reward)")
            ax[1].set_ylabel("P(delayed)")
            ax[1].set_title("k = %.3f, logk_se = %.3f"%(k, se_logk))
            ax[1].set_ylim([-0.1, 1.1])
        
    return k, logk, se_logk, percentD


def resample(ys, xs, Xs):
    """ Resample pair-wise sequences, to the closet right point 
    
    Inputs:
        ys: y in the original sequence
        xs: x in the original sequence
        Xs: x in the new sequence
    
    Outputs: 
    Ys : y in the new sequence 
    """
    Ys = [ys[xs >= X][0] if X <= max(xs) else ys[-1] for X in Xs]
    return Ys

def kmsc(data, tMax, Time, plot_KMSC = False):
    """Survival analysis of willingness to wait 
    Inputs:
        data: task data
        tMax: duration of the analysis time window
        plotKMSC: whether to plot 
        Time: upsampled time data
        
    Outputs:
        time:time data
        psurv: prob of survival data
        Time: upsampled time data
        Psurv: upsampled prob of survival data
        auc: area under the survival curve
        std_wtw: std of willingness to wait across trials. 
    """
    durations = data.timeWaited
    event_observed = np.equal(data.trialEarnings, 0) # 1 if the participant quits and 0 otherwise 
    time, psurv = km(event_observed, durations)
    
    # add the first and the last datapoints
    # ok here is the wierd point
    psurv = psurv[time < tMax]
    time = time[time < tMax]
    time = np.insert(time, 0, 0);  time = np.append(time, tMax)
    psurv = np.insert(psurv, 0, 1);  psurv = np.append(psurv, np.max(psurv[-1],0))
    
    # upsample to a high resolution 
    Psurv = resample(psurv, time, Time) 
    
    # plot 
    if plot_KMSC:
        plt.plot(time, psurv)
        plt.legend()
    
    # calculate AUC
    auc = np.sum(np.diff(time) * psurv[:-1])
    
    # calculate std_wtw
    cdf_wtw = 1 - psurv
    cdf_wtw[-1] = 1  # assume everyone quits at tMax
    pdf_wtw = np.diff(cdf_wtw)
    
    # np.sum(pdf_wtw * time[1:]) #check, right-aligned rule
    var_wtw = np.sum(pdf_wtw * (time[1:] - auc)**2)
    std_wtw = np.sqrt(var_wtw)
    
    return time, psurv, Time, Psurv, auc, std_wtw


def rtplot_multiblock(trialdata):
    """ Plot figures to visually check RT, for multiple blocks
    """ 
    # calc ready_RT
    trialdata.eval("ready_RT = trialStartTime - trialReadyTime", inplace = True)
    blockbounds = [max(trialdata.totalTrialIdx[trialdata.blockIdx == i]) for i in np.unique(trialdata.blockIdx)]

    # plot
    fig, ax = plt.subplots(1,6)
    # ready RT timecourse
    trialdata.plot("totalTrialIdx",  "ready_RT", ax = ax[0])
    ax[0].set_ylabel("Ready RT (s)")
    ax[0].set_xlabel("Trial")
    ax[0].get_legend().remove()
    ax[0].vlines(blockbounds, 0, max(trialdata.ready_RT), color = "grey", linestyles = "dotted", linewidth=2)
    #  ready RT histogram
    trialdata['ready_RT'].plot.hist(ax = ax[1])
    ax[1].set_xlabel("Ready RT (s)")
    # sell RT timecourse
    trialdata.loc[trialdata.trialEarnings!=0,:].plot("totalTrialIdx", "RT", ax = ax[2])
    ax[2].set_ylabel("Sell RT(s)")
    ax[2].set_xlabel("Trial")
    ax[2].get_legend().remove()
    ax[2].vlines(blockbounds, 0, max(trialdata.RT), color = "grey", linestyles = "dotted", linewidth=2)
    # sell RT histogram
    trialdata.loc[trialdata.trialEarnings!=0,'RT'].plot.hist(ax = ax[3])
    ax[3].set_xlabel("Sell RT (s)")
    # 
    # code.interact(local = dict(globals(), **locals()))
    trialdata[np.logical_and(trialdata.trialEarnings != 0, trialdata.blockIdx == 1)].plot.scatter("timeWaited",  "RT", color = expParas.conditionColors[0], ax = ax[4])
    trialdata[np.logical_and(trialdata.trialEarnings != 0, trialdata.blockIdx == 2)].plot.scatter("timeWaited",  "RT", color = expParas.conditionColors[1], ax = ax[5])

def trialplot_multiblock(trialdata):
    """Plot figures to visually check trial-by-trial behavior, for multiple blocks

    """
    fig, ax = plt.subplots()
    trialdata[trialdata.trialEarnings != 0].plot('totalTrialIdx', 'timeWaited', ax = ax, color = "blue", label = 'rwd')
    trialdata[trialdata.trialEarnings != 0].plot.scatter('totalTrialIdx', 'timeWaited', ax = ax, color = "blue", label='_nolegend_', s = 100)

    trialdata[trialdata.trialEarnings == 0].plot('totalTrialIdx', 'timeWaited', ax = ax, color = "red", label = 'unrwd')
    trialdata[trialdata.trialEarnings == 0].plot.scatter('totalTrialIdx', 'timeWaited', ax = ax, color = "red", label='_nolegend_', s = 100)

    trialdata[trialdata.trialEarnings == 0].plot.scatter('totalTrialIdx', 'scheduledDelay', ax = ax, color = "black", label = "scheduled", s = 100)
    blockbounds = [max(trialdata.totalTrialIdx[trialdata.blockIdx == i]) for i in np.unique(trialdata.blockIdx)]
    ax.vlines(blockbounds, 0, max(expParas.tMaxs), color = "grey", linestyles = "dotted")
    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Trial")
    ax.set_ylim([-2, max(expParas.tMaxs) + 2])
    ax.set_xlim([-2, trialdata.shape[0] + 2])
    ax.legend(loc='upper right', frameon=False)

def wtwTS(trialEarnings_, timeWaited_, sellTime_, tMax, TaskTime, plot_WTW = False):
    """
    sellTime_ here is a continous time.
    I uppack data here since the required inputs are different sometimes
    """
    # check whether they are values 
    # For trials before the first quit trial, wtw = the largest timeWaited value 
    if any(trialEarnings_ == 0):
        first_quit_idx = np.where(trialEarnings_ == 0)[0][0] # in case there is no quitting 
    else:
        first_quit_idx = len(trialEarnings_) - 1
    wtw_now = max(timeWaited_[:first_quit_idx+1])
    wtw = [wtw_now for i in range(first_quit_idx)] # make the change here...

    # For trials after the first quit trial, quitting indicates a new wtw value 
    # Otherwise, only update wtw if the current timeWaited is larger 
    for i in range(first_quit_idx, len(trialEarnings_)):
        if trialEarnings_[i] == 0:
            wtw_now = timeWaited_[i]
        else:
            wtw_now = max(timeWaited_[i], wtw_now)
        wtw.append(wtw_now)

    # code.interact(local = dict(locals(), **globals()))
    # cut off
    wtw = np.array([min(x, tMax) for x in wtw])

    # upsample 
    WTW = resample(wtw, sellTime_, TaskTime)

    # ok so this is problematics....previously...
    if plot_WTW:
        fig, ax = plt.subplots()
        trialIdx_ = np.arange(len(trialEarnings_))
        ax.plot(trialIdx_, wtw)
        # blockbounds = [max(trialIdx_[blockIdx_ == i]) for i in np.unique(blockIdx_)]
        # ax.vlines(blockbounds, 0, tMax, color = "grey", linestyles = "dotted", linewidth = 3)
        ax.set_ylabel("WTW (s)")
        ax.set_xlabel("Trial")
        ax.set_ylim([-0.5, tMax + 0.5]) 

    return wtw, WTW, TaskTime

def desc_RT(trialdata):
    """Return descriptive stats of sell_RT and ready_RT, pooling all data together
    """
    # calc 
    # if trialReadyTime in trialdata:
    #     trialdata.eval("ready_RT = trialStartTime - trialReadyTime", inplace = True)
    #     # calc summary stats
    #     out = trialdata.agg({
    #             "ready_RT": ["median", "mean", calc_se]
    #         })
    #     ready_RT_median, ready_RT_mean, ready_RT_se = out.ready_RT

    out = trialdata.loc[trialdata.trialEarnings != 0, :].agg({
            "RT": ["median", "mean"]
        })
    sell_RT_median,  sell_RT_mean = out.RT
    sell_RT_se  = calc_se(trialdata.loc[trialdata.trialEarnings != 0, :].RT)
    return sell_RT_median, sell_RT_mean, sell_RT_se
############################ individual level analysis functions ###############
def ind_MF(trialdata, key, isTrct = True, plot_RT = False, plot_trial = False, plot_KMSC = False, plot_WTW = False, n_subblock = 4):
    """Conduct model-free (MF) analysis for a single participant 
    Inputs:
        trialdata: a pd dataframe that contains task data
        key: in the format of (id, sessIdx). For example, ("s0001", 1)
        plot_RT: whether to plot 
        plot_trial:
    """
    # initialize the output
    stats = [] # for scalar outputs
    objs = {} # for others

    # RT visual check
    if plot_RT:
        rtplot_multiblock(trialdata)

    # trial-by-trial behavior visual check
    if plot_trial:
        trialplot_multiblock(trialdata)

    # calc without truncating 
    nBlock = len(np.unique(trialdata.blockIdx))
    wtw = []
    WTW = []
    for i in range(nBlock):
        blockdata = trialdata[trialdata.blockIdx == i + 1]
        condition = blockdata.condition.values[0]
        conditionColor = expParas.conditionColors[condition]
        # WTW timecourse
        block_wtw, block_WTW, _ = wtwTS(blockdata['trialEarnings'].values, blockdata['timeWaited'].values, blockdata['sellTime'].values, expParas.tMax, expParas.BlockTime, False)
        wtw.append(block_wtw)
        WTW.append(block_WTW)

    ################## calculate summary stats for each block ###############
    if isTrct:
        trialdata = trialdata[trialdata.sellTime <= expParas.blocksec - np.max(expParas.tMaxs)]

    ################## this part of code  can be modified for different experiments ##########
    # initialize the figure 
    if plot_KMSC:
        fig, ax = plt.subplots()
        ax.set_xlim([0, expParas.tMax])
        ax.set_ylim([-0.1, 1.1])
        ax.set_xlabel("Elapsed time (s)")
        ax.set_ylabel("Survival rate")
    
    # loop over blocks 
    for i in range(nBlock):
        blockdata = trialdata[trialdata.blockIdx == i + 1]
        condition = blockdata.condition.values[0]
        conditionColor = expParas.conditionColors[condition]

        # keypress stats
        if 'mean_ipi' in blockdata:
            ipi = np.mean(blockdata["mean_ipi"])

        # Survival analysis
        time, psurv, Time, Psurv, block_auc, block_std_wtw = kmsc(blockdata, expParas.tMax, expParas.Time, False)
        if plot_KMSC:
            ax.plot(time, psurv, color = conditionColor, label = condition)

        # Survival analysis into subblocks 
        sub_aucs = []
        sub_std_wtws = []
        subblocksec = (expParas.blocksec - np.max(expParas.tMaxs)) / n_subblock
        for k in range(n_subblock):
            # code.interact(local = dict(locals(), **globals()))
            filter = np.logical_and(blockdata['sellTime'] >= k * subblocksec, blockdata['sellTime'] < (k + 1) * subblocksec)
            try:
                time, psurv, Time, Psurv, auc, std_wtw = kmsc(blockdata.loc[filter, :], expParas.tMax, Time, False)
            except:
                if sum(filter) == 0:
                    auc = np.nan
                    std_wtw = np.nan
                    # code.interact(local = dict(locals(), **globals()))
            sub_aucs.append(auc) 
            sub_std_wtws.append(std_wtw)

        # remove the data near the beginning of the block
        # this is not necessary for the HP block
        _, _, _, _, block_auc_rh, block_std_wtw_rh = kmsc(blockdata.loc[blockdata['trialStartTime'] >= 90], expParas.tMax, expParas.Time, False)

        # RT stats 
        # ready_RT_median, ready_RT_mean, ready_RT_se
        sell_RT_median, sell_RT_mean, sell_RT_se = desc_RT(blockdata)
        
        # get init_wtw and end_wtw
        junk, _, _ = wtwTS(blockdata['trialEarnings'].values, blockdata['timeWaited'].values, blockdata['sellTime'].values, expParas.tMax, expParas.BlockTime, False)
        init_wtw = junk[0]
        end_wtw = junk[-1]

        if 'mean_ipi' in blockdata:
            tmp = {"id": key[0], "sess": key[1], "key": str(key), "block": i + 1, "auc": block_auc,  "std_wtw": block_std_wtw, \
            "auc_rh": block_auc_rh, "std_wtw_rh": block_std_wtw_rh, "ipi": ipi, \
            "diff_auc": sub_aucs[3] - sub_aucs[0], "diff_wtw": end_wtw - init_wtw, \
            "init_wtw": init_wtw, "end_wtw": end_wtw, "sell_RT_median": sell_RT_median, "sell_RT_mean": sell_RT_mean, "sell_RT_se": sell_RT_se, "condition": condition}
        else:
            tmp = {"id": key[0], "sess": key[1], "key": str(key), "block": i + 1, "auc": block_auc,  "std_wtw": block_std_wtw, \
            "auc_rh": block_auc_rh, "std_wtw_rh": block_std_wtw_rh,\
            "diff_auc": sub_aucs[3] - sub_aucs[0], "diff_wtw": end_wtw - init_wtw, \
            "init_wtw": init_wtw, "end_wtw": end_wtw, "sell_RT_median": sell_RT_median, "sell_RT_mean": sell_RT_mean, "sell_RT_se": sell_RT_se, "condition": condition}
        tmp.update(dict(zip(['auc' + str((i + 1)) for i in range(n_subblock)], sub_aucs)))
        tmp.update(dict(zip(['std_wtw' + str((i + 1)) for i in range(n_subblock)], sub_std_wtws)))
        stats.append(pd.DataFrame(tmp, index = [i]))
        objs['Psurv_block'+str(i+1)] = Psurv

    WTW = np.concatenate(WTW)
    wtw =  np.concatenate(wtw)
    stats = pd.concat(stats, ignore_index = True)
    objs['WTW'] = WTW
    objs['wtw'] = wtw
    # plot 
    if plot_WTW:
        fig_wtw, ax_wtw = plt.subplots()
        ax_wtw.plot(np.arange(len(wtw)) + 1, wtw)
        ax_wtw.vlines(sum(trialdata.blockIdx == 1) + 0.5, 0, expParas.tMax, linestyle = "dashed")
        ax_wtw.set_xlabel('Trial')
        ax_wtw.set_ylabel('WTW (s)')

    ############ return  ############# y
    return stats, objs


def ind_sim_dist(simdata, empdata):
    distance = np.sum((simdata['timeWaited'].values - empdata['timeWaited'].values) ** 2)
    return distance



def ind_sim_MF(simdata, empdata, key, plot_trial = False, plot_KMSC = False, plot_WTW = False):
    """ 
    """
    # initialize the output
    stats = [] # for scalar outputs
    objs = {} # for others
    wtw = [] # wtw for each trial
    WTW = [] # wtw resampled, trials beyond 600s are included

    # code.interact(local = dict(locals(), **globals()))
    # trial-by-trial behavior visual check
    if plot_trial:
        trialplot_multiblock(simdata)

    if plot_KMSC:
        fig, ax = plt.subplots()

    # calc AUC values and WTW for each block
    nBlock = len(np.unique(simdata.blockIdx))
    for i in range(nBlock):
        blockdata = simdata[simdata.blockIdx == i + 1]
        emp_blockdata = empdata[empdata.blockIdx == i + 1]
        condition = blockdata.condition.values[0]
        conditionColor = expParas.conditionColors[condition]
        # code.interact(local = dict(locals(), **globals()))

        # Survival analysis
        time, psurv, Time, Psurv, block_auc, block_std_wtw = kmsc(blockdata, expParas.tMax, expParas.Time, False)
        if plot_KMSC:
            ax.plot(time, psurv, color = conditionColor, label = condition)

        # save results
        tmp = pd.DataFrame({"key": str(key), "block": i + 1, "auc": block_auc, "std_wtw": block_std_wtw,\
            "condition": condition}, index = [i])
        stats.append(tmp) 
        objs['Psurv_block'+str(i+1)] = Psurv

        # WTW analysis
        block_wtw, block_WTW, block_TaskTime = \
        wtwTS(
            blockdata['trialEarnings'].values,
            blockdata['timeWaited'].values,
            emp_blockdata['sellTime'].values,
            expParas.tMax, 
            np.linspace(0, expParas.blocksec, int(len(expParas.TaskTime) / 2)),
            False
        )
        wtw.append(block_wtw)
        WTW.append(block_WTW)

    wtw = np.concatenate(wtw)
    if plot_WTW:
        fig, ax = plt.subplots()
        ax.plot(simdata.totalTrialIdx, wtw)

    # combine results from blocks
    WTW = np.concatenate(WTW)
    objs['WTW'] = WTW
    stats = pd.concat(stats, ignore_index = True)
        
    ############ return  ############# 
    return stats, objs

########################## group-level analysis functions ##############
def group_MF(trialdata_, plot_each = False, n_subblock = 4):
    # check sample sizes 
    nsub = len(trialdata_)
    print("Analyze %d valid participants"%nsub)
    print("\n")
    # analysis constants
    Time = expParas.Time
    TaskTime = expParas.TaskTime
    stats_ = []
    # initialize outputs 
    Psurv_block1_ = np.empty([nsub, len(Time)])
    Psurv_block2_ = np.empty([nsub, len(Time)])
    WTW_ = np.empty([nsub, len(TaskTime)])

    # run MF for each participant 
    idx = 0
    for key, trialdata in trialdata_.items():
        if plot_each:
            stats, objs  = ind_MF(trialdata, key, plot_RT = False, plot_trial = True, plot_KMSC = False, plot_WTW = True, n_subblock = n_subblock)
            plt.show()
            input("Press Enter to continue...")
            plt.clf()
        else:
            stats, objs  = ind_MF(trialdata, key, n_subblock = n_subblock)
            stats_.append(stats)
        Psurv_block1_[idx, :] = objs['Psurv_block1']
        Psurv_block2_[idx, :] = objs['Psurv_block2']
        WTW_[idx, :] = objs['WTW']
        idx += 1

    # plot the group-level results
    # if plot_group:
    #   fig1, ax1 = plot_group_WTW(WTW_, TaskTime)
    #   fig2, ax2 = plot_group_KMSC(Psurv_block1_, Psurv_block2_, Time)

    stats_ = pd.concat(stats_)
    # stats_.to_csv(os.path.join('..', 'analysis_results', expname, 'taskstats', 'emp_sess%d.csv'%key[1]), index = None)
    # save some data 
    # code.interact(local = dict(globals(), **locals()))
    # stats_.to_csv(os.path.join(logdir, "stats_sess%d.csv"%sess), index = False)

    return stats_, Psurv_block1_, Psurv_block2_, WTW_

def group_sim_MF(simdata_, empdata, plot_each = False):
    """
        conduct MF analysis for repeated simulated datasets
    """
    # tGrid constants
    Time = expParas.Time
    TaskTime = expParas.TaskTime

    # initialize outputs
    # code.interact(local = dict(locals(), **globals()))
    stats_ = []
    Psurv_block1_ = np.empty((len(simdata_), len(Time)))
    Psurv_block2_ = np.empty((len(simdata_), len(Time)))
    WTW_ = np.empty((len(simdata_), len(TaskTime)))


    # loop over participants 
    for i, simdata in enumerate(simdata_):
        if plot_each:
            stats, objs  = ind_sim_MF(simdata, empdata, 'sim_%d'%i, plot_trial = True, plot_KMSC = False, plot_WTW = True)
            plt.show()
            input("Press Enter to continue...")
            plt.clf()
        else:
            stats, objs  = ind_sim_MF(simdata, empdata, 'sim_%d'%i)
        
        # append data
        stats_.append(stats)
        Psurv_block1_[i, :] = objs['Psurv_block1']
        Psurv_block2_[i, :] = objs['Psurv_block2']
        WTW_[i, :] = objs['WTW']

    stats_ = pd.concat(stats_)
    # plot for the group level 
    # if plot_group:
    #   fig1, ax1 = plot_group_WTW(WTW_, TaskTime)
    #   fig2, ax2 = plot_group_KMSC(Psurv_block1_, Psurv_block2_, Time)

    return stats_, Psurv_block1_, Psurv_block2_, WTW_


def group_sim_dist(simdata_, empdata):
    """
        conduct MF analysis for repeated simulated datasets
    """

    # loop over participants 
    dist_vals = []
    for i, simdata in enumerate(simdata_):
        dist  = ind_sim_dist(simdata, empdata)
        dist_vals.append(dist)
    return dist_vals


def plot_group_AUC(stats, ax):
    ax.scatter(stats.loc[stats['condition'] == 'LP', 'auc'], stats.loc[stats['condition'] == 'HP', 'auc'], color = 'grey')
    ax.plot([-0.5, expParas.tMax + 0.5], [-0.5, expParas.tMax + 0.5], color = 'red', ls = "--")
    ax.set_ylim([-0.5, expParas.tMax + 0.5])
    ax.set_xlim([-0.5, expParas.tMax + 0.5])
    ax.set_xlabel("LP AUC (s)")
    ax.set_ylabel("HP AUC (s)")


####### hazard plot
def psurv2hazard(t, s):
    # first 

    time = t[1:]
    f = -np.diff(s)
    hazard = f / s[:-1]
    hazard[f == 0] = 0

    # exclude data points after hazard reaches 1
    time = time[:sum(s > 0)]
    hazard = hazard[:sum(s > 0)]

    return time, hazard

def plot_ind_both_kmsc(trialdata_s1, trialdata_s2, axs, isTrct = True):
    if isTrct:
        trialdata_s1 = trialdata_s1[trialdata_s1.sellTime <= expParas.blocksec - np.max(expParas.tMaxs)]
        trialdata_s2 = trialdata_s2[trialdata_s2.sellTime <= expParas.blocksec - np.max(expParas.tMaxs)]
    nBlock = len(np.unique(trialdata_s1.blockIdx))
    for i in range(nBlock):
        blockdata = trialdata_s1[trialdata_s1.blockIdx == i + 1]
        condition = blockdata.condition.values[0]
        conditionColor = expParas.conditionColors[condition]
        # Survival analysis
        t, s, _, _, auc, _= kmsc(blockdata, expParas.tMax, expParas.Time, False)
        time, hazard = psurv2hazard(t, s)
        axs[0].plot(t, s, color = conditionColor)
        axs[0].text(0.8, 0.5 - i * 0.1,  condition + "1" + " : " + str(round(auc,2)), transform=axs[0].transAxes)
        axs[1].plot(time, hazard, color = conditionColor)

    nBlock = len(np.unique(trialdata_s2.blockIdx))
    for i in range(nBlock):
       blockdata = trialdata_s2[trialdata_s2.blockIdx == i + 1]
       condition = blockdata.condition.values[0]
       conditionColor = expParas.conditionColors[condition]
       # Survival analysis
       t, s, _, _, auc, _= kmsc(blockdata, expParas.tMax, expParas.Time, False)
       time, hazard = psurv2hazard(t, s)
       axs[0].plot(t, s, color = conditionColor, linestyle='dashed')
       axs[0].text(0.8, 0.3 - i * 0.1, condition + "2" + " : " + str(round(auc,2)), transform=axs[0].transAxes)
       axs[1].plot(time, hazard, color = conditionColor, linestyle='dashed')

    axs[0].set_xlabel("Elapsed time (s)")
    axs[1].set_xlabel("Elapsed time (s)")
    axs[0].set_ylabel("Survival rate")
    axs[1].set_ylabel("Hazard rate")

def plot_ind_both_wtw(trialdata_s1, trialdata_s2, ax, isTrct = True):
    if isTrct:
        trialdata_s1 = trialdata_s1[trialdata_s1.sellTime <= expParas.blocksec - np.max(expParas.tMaxs)]
        trialdata_s2 = trialdata_s2[trialdata_s2.sellTime <= expParas.blocksec - np.max(expParas.tMaxs)]
    
    # plot wtw for the first session
    nBlock = len(np.unique(trialdata_s1.blockIdx))
    WTW_sess1 = []
    for i in range(nBlock):
        blockdata = trialdata_s1[trialdata_s1.blockIdx == i + 1]
        condition = blockdata.condition.values[0]
        conditionColor = expParas.conditionColors[condition]
        # wtw analysis 
        _, WTW, TaskTime = wtwTS(blockdata['trialEarnings'].values, blockdata['timeWaited'].values, blockdata['sellTime'].values, expParas.tMax, expParas.BlockTime, plot_WTW = False)
        WTW_sess1 = WTW_sess1 + WTW

    ax.plot(expParas.TaskTime, WTW_sess1, color = conditionColor)

    # plot for block 2
    nBlock = len(np.unique(trialdata_s2.blockIdx))
    WTW_sess2 = []
    for i in range(nBlock):
        blockdata = trialdata_s2[trialdata_s2.blockIdx == i + 1]
        condition = blockdata.condition.values[0]
        conditionColor = expParas.conditionColors[condition]
        # wtw analysis 
        _, WTW, TaskTime = wtwTS(blockdata['trialEarnings'].values, blockdata['timeWaited'].values, blockdata['sellTime'].values, expParas.tMax, expParas.BlockTime, plot_WTW = False)
        WTW_sess2 = WTW_sess2 + WTW

    ax.plot(expParas.TaskTime, WTW_sess2, color = conditionColor, linestyle='dashed')
    
    # add block boundary 
    ax.vlines(expParas.blocksec, -1, 14, color = "red")
    ax.set_ylim([-1, 14])
    # Add labels 
    ax.set_xlabel("Task time (s)")
    ax.set_ylabel("WTW (s)")


