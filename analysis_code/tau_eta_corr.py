

import numpy as np
import pandas as pd 
import itertools


eta_vals = np.linspace(0.5, 5, 10)
tau_vals = np.linspace(1, 10, 10)

ts = np.linspace(1, 20, 20)
auc_ = np.zeros(shape = (len(eta_vals), len(tau_vals)))
for i, j in itertools.product(np.arange(len(tau_vals)), np.arange(len(eta_vals))):
    tau = tau_vals[i]
    eta = eta_vals[j]
    pwaits = 1 / (1 + np.exp((0.1*ts - eta) * tau))
    psurvivals = np.cumprod(pwaits)
    auc_[i,j] = np.sum(np.concatenate([[1], psurvivals[:-1]]) + psurvivals) / 2


pd.DataFrame(auc_, columns = ["eta = %.1f"%x for x in eta_vals], index = ["tau = %.1f"%x for x in tau_vals])


# 


