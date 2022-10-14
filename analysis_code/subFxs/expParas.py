import numpy as np
from subFxs import expParas

iti = 1.5
tMax = 12 # maximal analysis time frame
nBlock = 2
blocksec = 600
TaskTime = np.linspace(0, blocksec * nBlock, 600)
BlockTime = np.linspace(0, blocksec, 300)
Time = np.linspace(0, tMax,  int(tMax / 0.1))
tokenValue = 2

# generate possible delay values in LP
# will this influence my results in the other paper? Maybe but I don't care
def tmp():
	""" Reference: email with Joe on Jun 7, 2021
	"""
	maxValue = 24
	fac = 1.75
	n = 8
	d1 = np.log(maxValue/(fac**n - 1))
	d2 = np.log(maxValue + maxValue/(fac**n - 1))
	support = np.exp(np.linspace(d1,d2,n+1))
	support = support[1:n+1] - support[0]

	# or alternatively 
	# maxDelay = 24
	# stepSize = 1.75
	# n = 8
	# c = maxDelay / (stepSize**n - 1)
	# support = np.exp(np.linspace(np.log(c*stepSize), np.log(maxDelay+c), n)) - c
	return support

conditions = ["HP", "LP"]
conditionColors = {'HP': "#1b7837", 'LP': "#762a83"}
tMaxs = [12, 24] # maximal trial duration in each condition 
delayPossVals = [np.linspace(1.5, 12, 8), tmp()]


# optimal behaviors 
optimWaitThresholds = [12, delayPossVals[1][3]] 
optimRewardRates = [2 / (np.mean(expParas.delayPossVals[0]) + expParas.iti),\
					1 / (0.5 * delayPossVals[1][3] + 0.5 * np.mean(delayPossVals[1][:4]) + 1.5)]

# 
selfreport_vars = ['NU', 'PU', 'PM', 'PS', 'SS', 'attention', 'cogstable', 'motor', 'perseverance', 'selfcontrol', 'cogcomplex', 'UPPS', 'BIS', 'GMK'] 