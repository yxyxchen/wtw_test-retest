# functions related to sampling delay durations
# regular packages 
import re
import numpy as np
import random 
import math
import pandas as pd 
from itertools import chain

##################################
def nextQuantile(nQuantiles, seq):
    # sample a sequence of quantiles balancing first-order transition frequencies
    # inputs: the number of unique quantiles (int), and the sequence so far (string)
    # returns the next quantile in the sequence (zero-based)
    
    # if seq is empty, choose randomly
    if len(seq) < 1: 
        nextQ = random.randrange(nQuantiles)
        
    # otherwise, see which transitions are less frequent in the sequence so far
    else:
        candidates = range(nQuantiles)
        cost = [0] * nQuantiles # initialize the cost at zero for each quantile
        for c in candidates:
            newTransition = seq[-1] + str(c) # what quantile transition would be created?
            cost[c] = len(re.findall('(?=({}))'.format(newTransition), seq)) # how many times does it already occur?
            # n.b. seq.count would not work here, it counts only 1 instance of '11' in '111'
        leastCost = min(cost)
        goodCands = [i for i,x in enumerate(cost) if x==leastCost]
        nextQ = random.choice(goodCands)
        
    # return both the next quantile and the full updated sequence
    seqUpdated = seq + str(nextQ)
    return {'nextQuantile':nextQ, 'seq':seqUpdated}

#################
def fixed10(seq):
    return {'nextDelay':10, 'seq':seq}


########################
def discreteUnif12(seq):
    # discrete uniform distribution, every 2 s from 2 to 16
    support = np.linspace(1.5, 12, 8)
    nQuantiles = len(support)
    out = nextQuantile(nQuantiles, seq)
    nextDelay = support[out['nextQuantile']]
    return {'nextDelay':nextDelay, 'seq':out['seq']}

########################
def discreteUnif16(seq):
    # discrete uniform distribution, every 2 s from 2 to 16
    support = range(2,17,2)
    nQuantiles = len(support)
    out = nextQuantile(nQuantiles, seq)
    nextDelay = support[out['nextQuantile']]
    return {'nextDelay':nextDelay, 'seq':out['seq']}


#################################
def discreteLogSpace24_1p75(seq):
    # discrete distribution of log-spaced values
    # max is 32, successive intervals scale by 1.75
    ## define the support
    maxValue = 24
    fac = 1.75
    n = 8
    d1 = np.log(maxValue/(fac**n - 1))
    d2 = np.log(maxValue + maxValue/(fac**n - 1))
    support = np.exp(np.linspace(d1,d2,n+1))
    support = support[1:n+1] - support[0] # subtract first element
    ## draw the sample
    nQuantiles = len(support)
    out = nextQuantile(nQuantiles, seq)
    nextDelay = support[out['nextQuantile']]
    return {'nextDelay':nextDelay, 'seq':out['seq']}

#################################
def discreteLogSpace32_1p75(seq):
    # discrete distribution of log-spaced values
    # max is 32, successive intervals scale by 1.75
    ## define the support
    maxValue = 32
    fac = 1.75
    n = 8
    d1 = np.log(maxValue/(fac**n - 1))
    d2 = np.log(maxValue + maxValue/(fac**n - 1))
    support = np.exp(np.linspace(d1,d2,n+1))
    support = support[1:n+1] - support[0] # subtract first element
    ## draw the sample
    nQuantiles = len(support)
    out = nextQuantile(nQuantiles, seq)
    nextDelay = support[out['nextQuantile']]
    return {'nextDelay':nextDelay, 'seq':out['seq']}

#####################################
def stressPareto_4_0_2():
    # parameters 
    k = 4
    mu = 0
    sigma = 2
    delayMax = 40
    # generate one sample
    sample = min(mu + sigma * (rd.uniform(0, 1) ** (-k) - 1) / k, delayMax)
    return sample
    
def stressUnif20():
    # parameters 
    delayMax = 20
    
    # generate one sample
    sample = rd.uniform(0, delayMax)
    return sample
    
    