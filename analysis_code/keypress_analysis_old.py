########################### import modules ############################
import pandas as pd
import numpy as np
import os
import glob
import re
import itertools
import copy # pay attention to copy 
import code
import math
from datetime import datetime
import matplotlib.pyplot as plt

data = pd.read_csv("../keypress_data/keypress-s0001-sess1.csv")
taskdata = pd.read_csv("./data/active/task-s0001-sess1.csv")
data['totalKeypress'] = np.arange(data.shape[0]) + 1

# I want to know what is the final keypress here
newdata = pd.concat([taskdata, pd.DataFrame(data.groupby(["bkIdx", "trialIdx"]).aggregate({"totalKeypress": ["max"], "blocktime": ["max"]}).values, columns = ["n_keypress_end", "last_keypress_time"])], axis = 1)
newdata = pd.concat([newdata, pd.DataFrame(data.groupby(["bkIdx", "trialIdx"]).aggregate({"totalKeypress": [lambda x: min(x) - 1]}).values, columns = ["n_keypress_before"])], axis = 1)

plt.plot(data.loc[data.bkIdx == 1  & (data.blocktime < 60),"blocktime"], data.loc[data.bkIdx == 1 & (data.blocktime < 60), "totalKeypress"])
# add when a trial starts
selection = np.logical_and(newdata.condition == "LP", newdata.trialStartTime < 60)
# trial_start_data = [[(x,y - 2),(x, y + 2)] for x, y in zip(newdata.loc[selection, "trialStartTime"], newdata.loc[selection, "n_keypress"]) ]
for x, y in zip(newdata.loc[selection, "trialStartTime"], newdata.loc[selection, "n_keypress_before"]):
    # plt.plot((x,x),(y-100, y + 100), color = "red")
    plt.axvline(x = x, color = "red")

for x in newdata.loc[selection, "last_keypress_time"]:
    # plt.plot((x,x),(y-100, y + 100), color = "red")
    plt.axvline(x = x, color = "black")

plt.xlabel("Task time (s)")
plt.ylabel("Total keypress")
code.interact(local = dict(locals(), **globals()))
# add when a trial stops 



