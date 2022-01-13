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
# Qwaits_ : [20/40 x nTrial num] value of waiting at each second in each trial
# V0_ : [nTrialx1 num] value of entering a pre-trial iti, namely t = 0

omni = function(paras, condition_, scheduledWait_, normResults){
  # default settings 
  iti = 2
  load("expParas.RData")
  
  # normative analysis 
  subjectValues_ = normResults$subjectValues 
  
  # learning parameters
  tau = paras[1]
  
  # num of trials
  nTrial = length(scheduledWait_) 
  # duration of a sampling interval 
  stepSec = 1
  # max delay duration 
  delayMax = ifelse(unique(condition_) == "HP", delayMaxs[1], delayMaxs[2])
    
  # initialize action values 
  tWaits = seq(0, delayMax, by = stepSec) # decision points 
  tMax = max(tWaits) #  time point for the last decision point
  
  # initialize output variables
  trialEarnings_ = rep(0, nTrial)
  timeWaited_ = rep(0, nTrial)
  sellTime_ = rep(0, nTrial)
  
  # track elpased time from the beginning of the task 
  elapsedTime = -iti # the first trial doesn't have a pre-trial ITI 
  
  # loop over trials
  for(tIdx in 1 : nTrial) {
    # current scheduledWait 
    scheduledWait = scheduledWait_[tIdx]
    subjectValues = subjectValues_[[condition_[tIdx]]]
    subjectValues = subjectValues[seq(0, delayMax, by = 0.1) %in% seq(0, delayMax, by = stepSec)]
    
    # sample at a temporal resolution of 1 sec until a trial ends
    t = -2
    while(t <= tMax){
      # take actions after the iti
      if(t >= 0){
        # decide whether to wait or quit
        pWait =  1 / sum(1  + exp((0 - subjectValues[tWaits == t])* tau))
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
    "scheduledWait" = scheduledWait_
  )
  return(outputs)
}