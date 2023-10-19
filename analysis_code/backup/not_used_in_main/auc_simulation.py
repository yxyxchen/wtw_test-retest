import numpy as np 
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
from sksurv.nonparametric import kaplan_meier_estimator as km
from scipy.interpolate import interp1d
import code
import seaborn as sns
import scipy.stats as stats
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import os
import importlib
from datetime import datetime as dt
import importlib
# my customized modules
import subFxs
from subFxs import analysisFxs
from subFxs import expParas
from subFxs import modelFxs
from subFxs import simFxs 
from subFxs import normFxs
from subFxs import loadFxs
from subFxs import figFxs
from subFxs import analysisFxs

plt.style.use('classic')
sns.set(font_scale = 1)
sns.set_style("white")

HP_delays = [9000, 10500, 3000, 4500, 3000, 12000, 10500, 10500, 12000, 6000, 12000,
9000, 1500, 6000, 6000, 10500, 7500, 1500, 4500, 4500, 6000, 7500, 7500, 9000, 12000,
12000, 3000, 7500, 6000, 4500, 9000, 7500, 3000, 3000, 9000, 9000, 3000, 10500, 9000,
4500, 10500, 1500, 9000, 6000, 1500, 1500, 12000, 1500, 3000, 1500, 7500, 4500, 7500,
10500, 4500, 1500, 10500, 6000, 3000, 6000, 9000, 1500, 3000, 4500, 12000, 7500, 12000,
4500, 12000, 3000, 6000, 12000, 1500, 6000, 3000, 10500, 7500, 12000, 7500, 7500, 6000,
4500, 4500, 7500, 3000, 12000, 10500, 9000, 7500, 9000, 10500, 6000, 1500, 9000, 3000,
9000, 12000, 4500, 10500, 4500, 6000, 6000, 10500, 1500, 7500, 1500, 4500, 9000, 4500,
1500, 12000, 9000, 6000, 7500, 4500, 3000, 1500, 1500, 10500, 3000, 7500, 10500, 12000,
12000, 6000, 9000, 9000, 4500, 4500, 9000, 7500, 10500, 10500, 6000, 3000, 3000, 9000,
9000, 3000, 10500, 3000, 1500, 4500, 1500, 1500, 12000, 7500, 4500, 10500, 9000, 10500,
10500, 1500, 9000, 12000, 3000, 7500, 7500, 1500, 7500, 3000, 4500, 6000, 1500, 6000,
10500, 7500, 9000, 1500, 3000, 6000, 6000, 9000, 6000, 7500, 12000, 9000, 4500, 7500,
6000, 4500, 3000, 3000, 12000, 1500, 10500, 12000, 4500, 12000, 10500, 4500, 7500, 6000,
12000, 12000, 6000, 4500, 6000, 12000, 6000, 6000, 9000, 10500, 6000, 10500, 10500, 1500,
7500, 9000, 6000, 3000, 3000, 7500, 7500, 3000, 9000, 3000, 1500, 9000, 7500, 10500, 7500,
4500, 12000, 10500, 9000, 9000, 12000, 7500, 1500, 6000, 7500, 12000, 4500, 4500, 3000,
12000, 1500, 3000, 4500, 9000, 1500, 12000, 9000, 7500, 4500, 10500, 4500, 1500, 10500,
12000, 3000, 10500, 3000, 6000, 1500, 4500, 7500, 6000, 6000, 1500, 1500, 3000, 4500, 6000,
10500, 7500, 10500, 10500, 4500, 1500, 4500, 12000, 12000, 3000, 7500, 7500, 12000, 4500,
3000, 12000, 7500, 9000, 1500, 6000, 4500, 10500, 12000, 10500, 6000, 12000, 6000, 9000,
10500, 1500, 12000, 9000, 4500, 4500, 9000]

LP_delays = [23999, 4253, 1203, 23999, 7650, 4253, 569, 569, 23999, 23999, 13596, 2312,
7650, 1203, 4253, 13596, 1203, 7650, 7650, 13596, 4253, 23999, 206, 23999, 1203, 1203,
2312, 206, 2312, 13596, 23999, 2312, 4253, 206, 4253, 7650, 206, 13596, 13596, 206, 7650,
2312, 569, 206, 1203, 13596, 7650, 23999, 569, 1203, 569, 7650, 569, 2312, 23999, 4253, 2312,
1203, 206, 206, 569, 13596, 569, 4253, 4253, 569, 206, 4253, 23999, 23999, 206, 13596, 206,
7650, 7650, 206, 1203, 1203, 4253, 1203, 206, 206, 2312, 2312, 569, 1203, 569, 4253, 13596,
4253, 4253, 206, 23999, 1203, 23999, 569, 569, 23999, 13596, 569, 2312, 13596, 7650, 4253,
7650, 2312, 23999, 2312, 4253, 2312, 1203, 7650, 1203, 2312, 7650, 23999, 7650, 569, 7650,
13596, 2312, 206, 569, 13596, 1203, 13596, 13596, 23999, 4253, 23999, 13596, 4253, 4253,
569, 2312, 2312, 1203, 7650, 2312, 569, 23999, 1203, 569, 1203, 2312, 4253, 7650, 569,
13596, 2312, 206, 1203, 4253, 2312, 23999, 7650, 23999, 206, 7650, 7650, 206, 206, 23999,
569, 206, 4253, 13596, 206, 13596, 13596, 7650, 13596, 1203, 206, 2312, 2312, 13596, 569,
7650, 1203, 1203, 13596, 23999, 2312, 7650, 4253, 1203, 23999, 23999, 4253, 206, 569, 4253,
206, 1203, 13596, 4253, 2312, 569, 569, 23999, 206, 7650, 13596, 1203, 4253, 4253, 1203, 23999,
2312, 2312, 1203, 206, 13596, 2312, 23999, 7650, 4253, 569, 7650, 7650, 1203, 7650, 2312,
13596, 206, 569, 2312, 7650, 206, 2312, 4253, 13596, 569, 569, 4253, 7650, 23999, 569, 206,
4253, 23999, 1203, 2312, 206, 23999, 13596, 23999, 23999, 13596, 13596, 7650, 569, 1203, 569,
13596, 206, 206, 4253, 13596, 1203, 1203, 569, 206, 13596, 2312, 1203, 4253, 206, 23999, 1203,
1203, 206, 569, 13596, 23999, 2312, 569, 7650, 7650, 569, 23999, 569, 4253, 1203, 13596, 569,
1203, 23999, 206, 7650, 13596, 4253, 4253, 2312, 2312, 23999, 4253, 569, 2312]

# HP_delays = [7500, 10500, 4500, 9000, 12000, 9000, 9000, 3000, 12000, 6000,
# 10500, 7500, 3000, 1500, 7500, 7500, 1500, 6000, 7500, 12000, 3000, 3000, 6000,
# 3000, 9000, 7500, 9000, 6000, 1500, 4500, 6000, 9000, 10500, 3000, 10500, 12000,
# 10500, 6000, 12000, 1500, 9000, 1500, 12000, 7500, 4500, 4500, 7500, 6000, 6000,
# 4500, 3000, 4500, 1500, 3000, 7500, 3000, 9000, 4500, 10500, 10500, 1500, 1500,
# 10500, 9000, 12000, 12000, 4500, 12000, 9000, 6000, 6000, 4500, 12000, 1500, 4500,
# 4500, 7500, 6000, 1500, 3000, 10500, 6000, 12000, 4500, 10500, 9000, 3000, 7500,
# 10500, 4500, 1500, 12000, 12000, 7500, 4500, 3000, 4500, 9000, 7500, 9000, 1500,
# 6000, 10500, 1500, 9000, 10500, 10500, 7500, 7500, 1500, 10500, 3000, 3000, 12000,
# 3000, 1500, 7500, 12000, 10500, 12000, 6000, 9000, 9000, 4500, 6000, 3000, 6000, 7500,
# 4500, 7500, 3000, 3000, 1500, 1500, 7500, 1500, 12000, 4500, 6000, 6000, 3000, 7500,
# 12000, 3000, 6000, 10500, 10500, 4500, 4500, 9000, 1500, 6000, 4500, 10500, 9000, 6000,
# 1500, 4500, 12000, 6000, 12000, 7500, 7500, 10500, 6000, 9000, 10500, 1500, 9000, 9000,
# 7500, 6000, 7500, 9000, 3000, 9000, 4500, 1500, 10500, 3000, 12000, 1500, 3000, 10500,
# 7500, 9000, 12000, 9000, 9000, 3000, 4500, 3000, 1500, 1500, 4500, 12000, 12000, 10500,
# 12000, 9000, 7500, 1500, 9000, 10500, 4500, 1500, 6000, 12000, 4500, 9000, 4500, 7500,
# 10500, 12000, 10500, 6000, 1500, 7500, 6000, 7500, 3000, 4500, 4500, 3000, 9000, 6000, 
# 10500, 3000, 6000, 4500, 10500, 1500, 1500, 3000, 7500, 12000, 6000, 9000, 12000, 12000,
# 3000, 3000, 12000, 1500, 10500, 7500, 4500, 6000, 9000, 10500, 10500, 9000, 1500, 12000, 7500,
# 7500, 7500, 12000, 9000, 7500, 1500, 6000, 6000, 7500, 10500, 3000, 4500, 3000, 10500, 1500,
# 4500, 2000, 10500, 12000, 12000, 3000, 12000, 7500, 3000, 3000, 7500, 4500, 9000, 10500, 4500,
# 7500, 9000, 9000, 6000, 6000, 10500, 7500, 6000, 3000, 6000, 4500, 6000, 12000, 1500, 3000]

# LP_delays = [13596, 2312, 1203, 7650, 4253, 206, 13596, 23999, 206, 23999, 1203, 206, 4253,
# 7650, 569, 1203, 13596, 7650, 2312, 2312, 206, 206, 7650, 23999, 13596, 1203, 1203, 2312,
# 23999, 2312, 7650, 7650, 1203, 4253, 1203, 23999, 569, 7650, 13596, 13596, 569, 23999, 4253,
# 2312, 569, 206, 1203, 569, 4253, 569, 13596, 4253, 4253, 13596, 206, 569, 569, 2312, 4253, 23999,
# 7650, 206, 2312, 13596, 1203, 206, 23999, 23999, 2312, 4253, 1203, 13596, 569, 2312, 23999, 23999,
# 13596, 23999, 206, 4253, 569, 23999, 7650, 23999, 1203, 2312, 569, 7650, 2312, 13596, 4253, 2312,
# 206, 569, 206, 2312, 2312, 7650, 13596, 7650, 569, 1203, 1203, 23999, 4253, 7650, 1203, 569, 13596,
# 206, 13596, 2312, 1203, 7650, 4253, 4253, 13596, 13596, 7650, 7650, 206, 1203, 4253, 206, 7650, 2312,
# 1203, 7650, 7650, 13596, 1203, 23999, 569, 4253, 23999, 7650, 4253, 569, 569, 4253, 1203, 569, 1203,
# 4253, 7650, 569, 13596, 23999, 13596, 13596, 569, 569, 23999, 1203, 206, 206, 7650, 23999, 4253, 2312,
# 2312, 4253, 13596, 4253, 23999, 569, 206, 206, 23999, 206, 2312, 206, 4253, 4253, 206, 13596, 206, 569,
# 2312, 13596, 2312, 569, 7650, 206, 1203, 13596, 23999, 2312, 23999, 23999, 4253, 1203, 1203, 2312, 7650,
# 1203, 7650, 7650, 13596, 569, 1203, 13596, 7650, 2312, 569, 569, 23999, 206, 7650, 569, 206, 4253, 569,
# 7650, 1203, 4253, 7650, 4253, 4253, 13596, 4253, 23999, 7650, 206, 569, 4253, 206, 1203, 1203, 206, 2312,
# 206, 206, 13596, 1203, 23999, 23999, 2312, 2312, 4253, 2312, 7650, 23999, 13596, 13596, 206, 23999, 1203,
# 569, 13596, 2312, 13596, 4253, 4253, 23999, 569, 2312, 1203, 2312, 23999, 23999, 569, 4253, 569, 206, 569,
# 13596, 1203, 13596, 13596, 206, 4253, 13596, 7650, 13596, 569, 2312, 206, 7650, 1203, 206, 206, 1203, 2312,
# 23999, 1203, 1203, 7650, 23999, 7650, 7650, 206, 2312, 4253, 2312, 1203, 23999, 2312, 13596, 23999]

########## threshold model
def ind_threshold_sim(threshold, sigma, condition, duration):
    if condition == "HP":
        delays = HP_delays
    elif condition == "LP":
        delays = LP_delays

    # shuffle delay in a certain way
    tmp = np.random.choice(np.arange(len(delays)))
    delays = delays[tmp:] + delays[:tmp]

    # track elapsed time 
    elapsedTime = 0
    tIdx = 0
    trialEarnings_ = []
    timeWaited_ = []
    scheduledDelay_ = []
    sellTime_ = []
    # simulate 
    while elapsedTime < duration:
        if tIdx >= len(delays):
            tIdx = 0
        delay = delays[tIdx] / 1000 # seconds
        this_threshold = threshold + np.random.normal(0, sigma) 
        trialEarnings = 10 if this_threshold > delay else 0
        timeWaited = np.min([delay, this_threshold])

        # if sellTime < ...
        if elapsedTime + timeWaited < duration: # < not <=, otherwise i will get wierd results with strict thresholds 
            sellTime_.append(elapsedTime + timeWaited)
            trialEarnings_.append(trialEarnings)
            timeWaited_.append(timeWaited)
            scheduledDelay_.append(delay)
        tIdx += 1
        elapsedTime  = elapsedTime + timeWaited + expParas.iti

    taskdata = pd.DataFrame({
        "scheduledDelay": scheduledDelay_,
        "sellTime": sellTime_,
        "timeWaited": timeWaited_,
        "trialEarnings": trialEarnings_
        })
    return taskdata
######## main #########
if __name__ == "__main__":
    code.interact(local = dict(locals(), **globals()))
thresholds = np.linspace(1.5, 11.5, 5) #if threshold happen to be exactly the same as the numbers then when >= ... wierd results
# thresholds = [1.5]
durations = np.linspace(1, 5, 5) * 60

n_sim = 20
condition = "LP"
sigma = 0



# initialize outputs 
auc_ = []
std_wtw_ = []
condition_ = []
duration_ = []
threshold_ = []
for threshold in thresholds:
    for duration in durations:
        for i in range(n_sim):
            data = ind_threshold_sim(threshold, sigma, condition, duration)
            time, psurv, Time, Psurv, auc, std_wtw = analysisFxs.kmsc(data, expParas.tMax, expParas.Time, plot_KMSC = False)
            auc_.append(auc)
            std_wtw_.append(std_wtw)
            condition_.append(condition)
            threshold_.append(threshold)
            duration_.append(duration)

plotdf = pd.DataFrame({
    "auc": auc_,
    "duration": duration_,
    "threshold": threshold_,
    })
plotdf.groupby(["duration", "threshold"]).agg({"auc":["mean"]})
g = sns.FacetGrid(plotdf, col = "duration")
g.map(sns.boxplot, "threshold", "auc")
plt.show()


# I can update them later 
