# our reinfocement learning generative models simulate adapative persistence behavior as wait-or-quit choices 
# QL1: Q-learning model with a single learning rate
# QL2: Q-learning model with separate learning rates for rewards and non-rewards
# RL1: R-learning model with a single learning rate 
# RL2: R-learning model with separate learning rates for rewards and non-rewards

# inputs:
# paras: learning parameters
# condition_: HP or LP
# scheduledWait_: trial-wise delay

# outputs
# trialNum : [nTrialx1 int] 1 : nTrial
# condition : [nTrialx1 factor] from inputs
# scheduledWait : [nTrialx1 num] from inputs 
# trialEarnings : [nTrialx1 int] payment for each trial, either 10 or 0
# timeWaited : [nTrialx1 num] how long the agent waits after the iti in each trial 
# sellTime : [nTrialx1 num]  when the agent sells the token on each trial 
# Qwaits_ : [20/40 x nTria num] value of waiting at each second in each trial
# V_ : [nTrialx1 num] value of entering a pre-trial iti, namely t = 0

RL1 = function(paras, condition_, scheduledWait_, normResults){
  # default settings 
  iti = 2
  load("expParas.RData")
  
  # normative analysis 
  optimRewardRates = normResults$optimRewardRates
  
  # learning parameters
  alpha = paras[1]; tau = paras[2]; eta = paras[3]; beta = paras[4];
  
  # num of trials
  nTrial = length(scheduledWait_) 
  # duration of a sampling interval 
  stepSec = 1
  # max delay duration 
  delayMax = ifelse(unique(condition_) == "HP", delayMaxs[1], delayMaxs[2])
    
  # initialize action values 
  V = 0 # state value for t = 0
  rewardRate = mean(unlist(optimRewardRates))
  tWaits = seq(iti, delayMax + iti, by = stepSec) # decision points 
  tMax = max(tWaits) #  time point for the last decision point
  Qwaits = -0.1 * (tWaits) + eta + V # value of waiting at each decision points
  
  # initialize output variables
  Qwaits_ = matrix(NA, length(tWaits), nTrial); Qwaits_[,1] = Qwaits 
  V_ = vector(length = nTrial); V_[1] = V
  trialEarnings_ = rep(0, nTrial)
  timeWaited_ = rep(0, nTrial)
  sellTime_ = rep(0, nTrial)
  
  # track elpased time from the beginning of the task 
  elapsedTime = -iti # the first trial doesn't have a pre-trial ITI 
  
  # loop over trials
  for(tIdx in 1 : nTrial) {
    # current scheduledWait 
    scheduledWait = scheduledWait_[tIdx]
    # sample at a temporal resolution of 1 sec until a trial ends
    t = -2
    while(t <= tMax){
      # take actions after the iti
      if(t >= 0){
        # decide whether to wait or quit
        pWait =  1 / sum(1  + exp((V - Qwaits[tWaits == t])* tau))
        action = ifelse(runif(1) < pWait, 'wait', 'quit')
        
        # if a reward occurs and the agent is still waiting, the agent gets the reward
        tokenMature = (scheduledWait >= t) & (scheduledWait < (t + stepSec)) # whether the token matures before the next decision point
        getToken = (action == 'wait' && tokenMature) # whether the agent obtains the matured token
        
        # a trial ends if the agent obtains the matured token or quits. 
        # if the trial ends,return to t = 0. Otherwise, proceed to t + 1.
        isTerminal = (getToken || action == "quit")
        if(isTerminal){
          # update trial-wise variables 
          T =  ifelse(getToken, scheduledWait, t) # when the trial ends
          timeWaited =  T # how long the agent waits since the token appears
          trialEarnings = ifelse(getToken, tokenValue, 0) 
          elapsedTime = elapsedTime + timeWaited + iti
          sellTime = elapsedTime # elapsed task time when the agent sells the token
          
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
    if(tIdx < nTrial){
      
      # calculate expected returns for t >= 2
      Gts = trialEarnings - rewardRate * (T - tWaits) + V
      # only update value functions before time t = T
      updateFilter = tWaits <= T 
      # update Qwaits
      Qwaits[updateFilter] = Qwaits[updateFilter] + alpha * (Gts[updateFilter] - Qwaits[updateFilter])
      
      # calculate expected returns for t == 0 and update V
      Gt =  trialEarnings - rewardRate * T + V
      delta = Gt - V
      V = V + alpha * delta
      rewardRate = rewardRate + beta * delta
      
      # record updated values
      Qwaits_[,tIdx + 1] = Qwaits
      V_[tIdx + 1] = V
    }# end of the loop within a trial 
    if(tIdx < nTrial & (condition_[tIdx] != condition_[tIdx + 1])){
      elapsedTime = -iti
    }
  } # end of the loop over trials
  
  # return outputs
  outputs = list( 
    "trialNum" = 1 : nTrial, 
    "condition" = condition_,
    "trialEarnings" = trialEarnings_, 
    "timeWaited" = timeWaited_,
    "sellTime" = sellTime_,
    "scheduledWait" = scheduledWait_,
    "Qwaits_" = Qwaits_, 
    "V_" = V_
  )
  return(outputs)
}