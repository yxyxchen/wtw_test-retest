# our reinfocement learning generative models simulate adapative persistence behavior as wait-or-quit choices for a fixed period of time
# QL1: Q-learning model with a single learning rate
# QL2: Q-learning model with separate learning rates for rewards and non-rewards
# RL1: R-learning model with a single learning rate 
# RL2: R-learning model with separate learning rates for rewards and non-rewards

# inputs:
# paras: learning parameters
# condition: HP or LP
# duration : duration of the task 

# outputs
# trialNum : [nTrialx1 int] 1 : nTrial
# condition : [nTrialx1 factor] from inputs
# scheduledWait : [nTrialx1 num] from inputs 
# trialEarnings : [nTrialx1 int] payment for each trial, either 10 or 0
# timeWaited : [nTrialx1 num] how long the agent waits after the iti in each trial 
# sellTime : [nTrialx1 num]  when the agent sells the token on each trial 
# Qwaits_ : [20/40 x nTrial num] value of waiting at each second in each trial
# V0_ : [nTrialx1 num] value of entering a pre-trial iti, namely t = 0

QL2 = function(paras, condition, duration, normResults){
  source("subFxs/taskFxs.R")
  
  # default settings 
  iti = 2
  smallReward = 0

  # normative analysis 
  optimRewardRates = normResults$optimRewardRates
  optimWaitThresholds = normResults$optimWaitThresholds
  
  # learning parameters
  alphaR = paras[1]; alphaU = paras[2]; tau = paras[3]; gamma = paras[4]; prior = paras[5]
  # num of trials
  nTrialMax = ceiling(duration / iti)
  # duration of a sampling interval 
  stepSec = 1
  # max delay duration 
  delayMax = ifelse(condition == "HP", delayMaxs[1], delayMaxs[2])
    
  # initialize action values 
  V0 = mean(unlist(optimRewardRates)) / (1 - 0.85) # state value for t = 0
  tWaits = seq(iti, delayMax + iti, by = stepSec) # decision points 
  tMax = max(tWaits) #  time point for the last decision point
  Qwaits = -0.1 * (tWaits) + prior + V0 # value of waiting at each decision points
  
  # initialize output variables
  Qwaits_ = matrix(NA, length(tWaits), nTrialMax); Qwaits_[,1] = Qwaits 
  V0_ = vector(length = nTrialMax); V0_[1] = V0
  scheduledWait_ =  rep(0, nTrialMax)
  trialEarnings_ = rep(0, nTrialMax)
  timeWaited_ = rep(0, nTrialMax)
  sellTime_ = rep(0, nTrialMax)
  
  # track elpased time from the beginning of the task 
  elapsedTime = 0 
  
  # loop over trials
  tIdx = 1
  while(elapsedTime < duration) {
    # current scheduledWait 
    scheduledWait = drawSample(condition)
    scheduledWait_[tIdx] = scheduledWait
    
    # sample at a temporal resolution of 1 sec until a trial ends
    t = 0
    while(t <= tMax){
      # take actions after the iti
      if(t >= iti){
        # decide whether to wait or quit
        pWait =  1 / sum(1  + exp((V0 + smallReward - Qwaits[tWaits == t])* tau))
        action = ifelse(runif(1) < pWait, 'wait', 'quit')
        
        # if a reward occurs and the agent is still waiting, the agent gets the reward
        alreadyWait = t - iti  # how long the agent has waited since the token appears
        tokenMature = (scheduledWait >= alreadyWait ) & (scheduledWait < (alreadyWait + stepSec)) # whether the token matures before the next decision point
        getToken = (action == 'wait' && tokenMature) # whether the agent obtains the matured token
        
        # a trial ends if the agent obtains the matured token or quits. 
        # if the trial ends,return to t = 0. Otherwise, proceed to t + 1.
        isTerminal = (getToken || action == "quit")
        if(isTerminal){
          # update trial-wise variables 
          T =  ifelse(getToken, scheduledWait + iti, t) # when the trial ends
          timeWaited =  T - iti # how long the agent waits since the token appears
          trialEarnings = ifelse(getToken, tokenValue, smallReward) 
          sellTime = elapsedTime + timeWaited # elapsed task time when the agent sells the token
          elapsedTime = elapsedTime + T  # elapsed task time before the next token appears
          # record trial-wise variables
          trialEarnings_[tIdx] = trialEarnings
          timeWaited_[tIdx] = timeWaited
          sellTime_[tIdx] = sellTime
          break
        }
      }
      t = t + stepSec
    }
    
    # when the trial endes, update value functions for all time points before T in the trial
    
    # determine the learning rate depending on the payoff
    if(trialEarnings > 0){
      alpha = alphaR
    }else{
      alpha = alphaU
    }
    
    # calculate expected returns for t >= 2
    Gts =  gamma ^ (T - tWaits) * (trialEarnings + V0)
    # only update value functions before time t = T
    updateFilter = tWaits <= T 
    # update Qwaits
    Qwaits[updateFilter] = Qwaits[updateFilter] + alpha * (Gts[updateFilter] - Qwaits[updateFilter])
    
    # calculate expected returns for t == 0 and update V0
    Gt =  gamma ^ T * (trialEarnings + V0)
    V0 = V0 + alpha * (Gt - V0)
    
    # record updated values
    Qwaits_[,tIdx + 1] = Qwaits
    V0_[tIdx + 1] = V0

    # proceed to the next trial
    tIdx = tIdx + 1
  } # end of the loop over trials
  
  # return outputs
  nTrial = tIdx
  outputs = list( 
    "trialNum" = 1 : nTrial, 
    "condition" = rep(condition, nTrial),
    "trialEarnings" = trialEarnings_[1 : nTrial], 
    "timeWaited" = timeWaited_[1 : nTrial],
    "sellTime" = sellTime_[1 : nTrial],
    "scheduledWait" = scheduledWait_[1 : nTrial],
    "Qwaits_" = Qwaits_[, 1 : nTrial], 
    "V0_" = V0_[1 : nTrial]
  )
  return(outputs)
}