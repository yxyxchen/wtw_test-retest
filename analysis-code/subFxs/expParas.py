import numpy as np

iti = 1.5
nBlock = 2
blocksec = 600
conditions = ["LP", "HP"]
conditionColors = ["#762a83", "#1b7837"]
tMaxs = [12, 24]
tMax = 12
TaskTime = np.linspace(0, blocksec * nBlock, 600 * nBlock)
Time = np.linspace(0, tMax,  int(tMax / 0.1))




