import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn 
import code
from subFxs import expParas


def sample2dist_discrete(delays):
    """ convert discrete delay samples into a delay CDF and PDF
    """
    vals = np.unique(delays)
    time = np.array(sorted(vals))
    pdf = np.zeros_like(time)
    cdf = np.zeros_like(time)
    for i, val in enumerate(vals):
        pdf[i] = np.mean(vals == val)
        cdf[i] = np.mean(vals <= val)

    dist = {
        'time': time,
        'pdf': pdf,
        "cdf": cdf
    }
    return dist
    
def findOptim_discrete(dist, tokenValue, iti):
    giveupTimes = dist['time']
    pdf = dist['pdf']
    cdf = dist['cdf']

    # average delay durations for all policies 
    meanRewardDelays = np.cumsum(giveupTimes * pdf) / np.cumsum(pdf)
    rewardRates = (tokenValue * cdf) / \
    ((meanRewardDelays * cdf + giveupTimes * (1 - cdf)) + iti) 

    optimWaitThreshold = giveupTimes[np.argmax(rewardRates)]
    optimRewardRate = max(rewardRates)
    return optimWaitThreshold, optimRewardRate, giveupTimes, rewardRates

def sample2dist_continous(delays, delayMax, stepsize):
    '''convert continous delay samples into a delay CDF and PDF, with a specific temporal resolution 
    
    key arguments:
    
    delays: delay samples
    
    delayMax: upper limit of delay durations
    
    stepsize: sampling resolution of the output dist
    
    '''
    nBin = delayMax / stepsize
    if not nBin.is_integer():
        print("delayMax should be divisble by stepsize")
        return
    
    nBin = int(delayMax / stepsize)
    time = np.linspace(stepsize, stepsize + nBin * stepsize, num = nBin, endpoint = False) # e.g., [0.1, 16.0]
    cdf = np.zeros_like(time)
    for i in range(nBin):
        cdf[i] = np.mean(delays <= time[i])  # F(x) = Pr(X <= x)
      
    # sanity check 
    if cdf[-1] < 0.9999999:
        print("ERROR in CDF!!!!")

    dist = {
        "stepsize": stepsize,
        "time": time,
        "cdf": cdf
    }
    return dist


def findOptim_continous(dist, tokenValue, iti):
    '''find the optimal give-up time based on the delay distribution
    
    key arguments:
        
        dist -- dist of delay, encoded at 1-s resolution     
    '''
    giveupTimes = dist['time']
    bin = giveupTimes[1] - giveupTimes[0]
    # code.interact(local = dict(locals(), **globals()))
    # I might want to distinguish discrete and continous distributions
    cdf = dist['cdf']
    pdf = np.diff(np.append(0, cdf))

    # average delay durations for all policies 
    meanRewardDelays = np.cumsum((giveupTimes - 0.5 * bin) * pdf) / np.cumsum(pdf)
    rewardRates = (tokenValue * cdf) / \
    ((meanRewardDelays * cdf + giveupTimes * (1 - cdf)) + iti) 
    rewardRates = np.nan_to_num(rewardRates, 0)

    # optimize 
    optimWaitThreshold = giveupTimes[np.argmax(rewardRates)]
    optimRewardRate = max(rewardRates)
    return optimWaitThreshold, optimRewardRate, giveupTimes, rewardRates

    
############## 
if __name__ == "__main__":
    # code.interact(local = dict(locals(), **globals()))
    dist = sample2dist(expParas.delayPossVals[0], expParas.tMaxs[0], 0.05)
    optimWaitThreshold, optimRewardRate, giveupTimes, rewardRates = findOptim(dist, expParas.tokenValue, expParas.iti)
    plt.plot(giveupTimes, rewardRates)
    plt.show()
    code.interact(local = dict(locals(), **globals()))
    # fig, ax = plt.subplots()
        # ax.plot(dist["time"], dist["cdf"])
        # plt.show()












