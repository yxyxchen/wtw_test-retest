import numpy as np
from subFxs import expParas

iti = 2
tMax = 16 # maximal analysis time frame
nBlock = 2
blocksec = 600
TaskTime = np.linspace(0, blocksec * nBlock, 600)
BlockTime = np.linspace(0, blocksec, 300)
Time = np.linspace(0, tMax,  int(tMax / 0.1))
tokenValue = 10


conditions = ["HP", "LP"]
conditionColors = {'HP': "#1b7837", 'LP': "#762a83"}
tMaxs = [16, 32] # maximal trial duration in each condition 


# optimal behaviors 
optimWaitThresholds = [16, 3.0831765] 
optimRewardRates = [0.9090909, 1.169591]



