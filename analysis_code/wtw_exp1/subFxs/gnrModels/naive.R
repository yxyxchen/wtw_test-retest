# our BL generative models simulate adapative persistence behavior as wait-or-quit choices 

# inputs:
# paras: pWait
# condition_: HP or LP
# scheduledWait_: trial-wise delay

# outputs
# trialNum : [nTrialx1 int] 1 : nTrial
# condition : [nTrialx1 factor] from inputs
# scheduledWait : [nTrialx1 num] from inputs 
# trialEarnings : [nTrialx1 int] payment for each trial, either 10 or 0
# timeWaited : [nTrialx1 num] how long the agent waits after the iti in each trial 
# sellTime : [nTrialx1 num]  when the agent sells the token on each trial 

naive = function(paras, condition_, scheduledWait_, normResults){
  # default settings 
  iti = 2
  load("expParas.RData")
  
  # learning parameters
  theta = paras[1]; 
  
  # num of trials
  nTrial = length(scheduledWait_) 
  # duration of a sampling interval 
  stepSec = 1
  # max delay duration 
  delayMax = max(delayMaxs)
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
    
    # sample at a temporal resolution of 1 sec until a trial ends
    t = -2
    while(t <= tMax){
      # take actions after the iti
      if(t >= 0){
        # decide whether to wait or quit
        action = ifelse(runif(1) < theta, 'wait', 'quit')
        
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
    if(tIdx < nTrial & (condition_[tIdx] != condition_[tIdx + 1])){
      elapsedTime = -iti
    }
  }
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