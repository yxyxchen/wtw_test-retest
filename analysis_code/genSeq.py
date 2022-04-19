

import numpy as np
import pandas as pd
import subFxs
from subFxs import taskFxs
import random 

random.seed(10)

vals = []
seq = ''
for i in range(300):
	out = taskFxs.discreteLogSpace32_1p75(seq)
	vals.append(out['nextDelay'])
	seq = out['seq']
pd.DataFrame(vals).to_csv("LP.csv", header = False, index = None)


vals = []
seq = ''
for i in range(300):
	out = taskFxs.discreteUnif16(seq)
	vals.append(out['nextDelay'])
	seq = out['seq']
pd.DataFrame(vals).to_csv("HP.csv", header = False, index = None)