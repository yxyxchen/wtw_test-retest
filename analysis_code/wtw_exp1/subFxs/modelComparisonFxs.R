getLikFun = function(modelName){
  if(modelName == "QL1") likFun = QL1
  else if(modelName == "QL2") likFun = QL2
  else if(modelName == "RL1") likFun = RL1
  else if(modelName == "RL2") likFun = RL2
  else if(modelName == "BL") likFun = BL
  else{
    return("wrong model name!")
  }
  return(likFun)
}

QL1 = function(paras, cond, trialEarnings, timeWaited){
  # parse paras
  phi = paras[1]; tau = paras[2]; gamma = paras[3]; prior = paras[4]
  
  # prepare inputs
  nTrial = length(trialEarnings)
  tMax= ifelse(cond == "HP", tMaxs[1], tMaxs[2])
  nTimeStep = tMax / stepDuration
  Ts = round(ceiling(timeWaited / stepDuration) + 1)
  
  # initialize action values
  subOptimalRatio = 0.9 
  wIni = (5/6 + 0.93) / 2 * stepDuration / (1 - 0.9) * subOptimalRatio
  Viti = wIni 
  Qwait = prior*0.1 - 0.1*(0 : (nTimeStep - 1)) + Viti
  
  # initialize varibles for recording action values
  Qwaits = matrix(NA, nTimeStep, nTrial); Qwaits[,1] = Qwait
  Vitis = vector(length = nTrial); Vitis[1] = Viti
  
  # initialize variables for recording targets and deltas in updating Qwait
  deltas = matrix(NA, nTimeStep, nTrial) 
  targets = matrix(NA, nTimeStep, nTrial)
  
  # initialize outputs 
  lik_ = matrix(nrow = nTimeStep, ncol = nTrial)
  
  # loop over trials
  for(tIdx in 1 : nTrial) {
    # calculate likelyhood
    nextReward = trialEarnings[tIdx]
    getReward = ifelse(nextReward == tokenValue, T, F)
    lik_[,tIdx] =  sapply(1 : nTimeStep, function(i) 1 / sum(1  + exp((gamma * Viti - Qwait[i])* tau)))
    # update action values 
    T = Ts[tIdx]
    if(tIdx < nTrial){
      returns = sapply(1 : (T-1), function(t) gamma^(T-t-1) *nextReward + gamma^(T-t) * Viti)
      if(getReward){
        targets[1 : (T-1), tIdx] = returns[1 : (T-1)];
        deltas[1 : (T-1), tIdx] = returns[1 : (T-1)] - Qwait[1 : (T-1)]
        Qwait[1 : (T-1)] = Qwait[1 : (T-1)] + phi*(returns[1 : (T-1)] - Qwait[1 : (T-1)])
      }else{
        if(T > 2){
          targets[1 : (T-2), tIdx] = returns[1 : (T-2)]
          deltas[1 : (T-2), tIdx] = returns[1 : (T-2)] - Qwait[1 : (T-2)]
          Qwait[1 : (T-2)] = Qwait[1 : (T-2)] + phi*(returns[1 : (T-2)] - Qwait[1 : (T-2)])
        }
      }
      # update Viti
      delta = gamma^(iti / stepDuration) * returns[1] - Viti
      Viti = Viti + phi* delta 
      
      # record action values
      Qwaits[,tIdx + 1] = Qwait
      Vitis[tIdx + 1] = Viti
    }# end of the value update procedure
  } # end of the loop over trials
  
  # return outputs
  outputs = list( 
    "lik_" = lik_,
    "Qwaits" = Qwaits, "targets" = targets, "deltas" = deltas, "Vitis" = Vitis
  )
  return(outputs)
}

QL2 = function(paras, cond, trialEarnings, timeWaited){
  # parse paras
  phi = paras[1]; phiP = paras[2]; tau = paras[3]; gamma = paras[4]; prior = paras[5]
  
  # prepare inputs
  nTrial = length(trialEarnings)
  tMax= ifelse(cond == "HP", tMaxs[1], tMaxs[2])
  nTimeStep = tMax / stepDuration
  Ts = round(ceiling(timeWaited / stepDuration) + 1)
  
  # initialize action values
  subOptimalRatio = 0.9 
  wIni = (5/6 + 0.93) / 2 * stepDuration / (1 - 0.9) * subOptimalRatio
  Viti = wIni 
  Qwait = prior*0.1 - 0.1*(0 : (nTimeStep - 1)) + Viti
  
  # initialize varibles for recording action values
  Qwaits = matrix(NA, nTimeStep, nTrial); Qwaits[,1] = Qwait
  Vitis = vector(length = nTrial); Vitis[1] = Viti
  
  # initialize variables for recording targets and deltas in updating Qwait
  deltas = matrix(NA, nTimeStep, nTrial) 
  targets = matrix(NA, nTimeStep, nTrial)
  
  # initialize outputs 
  lik_ = matrix(nrow = nTimeStep, ncol = nTrial)
  
  # loop over trials
  for(tIdx in 1 : nTrial) {
    # calculate likelyhood
    nextReward = trialEarnings[tIdx]
    getReward = ifelse(nextReward == tokenValue, T, F)
    lik_[,tIdx] =  sapply(1 : nTimeStep, function(i) 1 / sum(1  + exp((gamma * Viti - Qwait[i])* tau)))
    # update action values 
    T = Ts[tIdx]
    if(tIdx < nTrial){
      returns = sapply(1 : (T-1), function(t) gamma^(T-t-1) *nextReward + gamma^(T-t) * Viti)
      if(getReward){
        targets[1 : (T-1), tIdx] = returns[1 : (T-1)];
        deltas[1 : (T-1), tIdx] = returns[1 : (T-1)] - Qwait[1 : (T-1)]
        Qwait[1 : (T-1)] = Qwait[1 : (T-1)] + phi*(returns[1 : (T-1)] - Qwait[1 : (T-1)])
      }else{
        if(T > 2){
          targets[1 : (T-2), tIdx] = returns[1 : (T-2)]
          deltas[1 : (T-2), tIdx] = returns[1 : (T-2)] - Qwait[1 : (T-2)]
          Qwait[1 : (T-2)] = Qwait[1 : (T-2)] + phiP*(returns[1 : (T-2)] - Qwait[1 : (T-2)])
        }
      }
      # update Viti
      delta = gamma^(iti / stepDuration) * returns[1] - Viti
      if(nextReward > 0) Viti = Viti + phi* delta else Viti = Viti + phiP* delta
 
      # record action values
      Qwaits[,tIdx + 1] = Qwait
      Vitis[tIdx + 1] = Viti
    }# end of the value update procedure
  } # end of the loop over trials
  
  # return outputs
  outputs = list( 
    "lik_" = lik_,
    "Qwaits" = Qwaits, "targets" = targets, "deltas" = deltas, "Vitis" = Vitis
  )
  return(outputs)
}

RL1 = function(paras, cond, trialEarnings, timeWaited){
  # parse para
  phi = paras[1]; tau = paras[2]; prior = paras[3]; beta = paras[4];
  
  # prepare inputs
  nTrial = length(trialEarnings)
  tMax= ifelse(cond == "HP", tMaxs[1], tMaxs[2])
  nTimeStep = tMax / stepDuration
  Ts = round(ceiling(timeWaited / stepDuration) + 1)
  
  # initialize actionValues
  subOptimalRatio = 0.9 
  wIni = (5/6 + 0.93) / 2 * stepDuration * subOptimalRatio
  reRate = wIni
  Viti = 0 
  Qwait = prior*0.1 - 0.1*(0 : (nTimeStep - 1)) + Viti
  
  # initialize varibles for recording action values
  Qwaits = matrix(NA, nTimeStep, nTrial); Qwaits[,1] = Qwait
  Vitis = vector(length = nTrial); Vitis[1] = Viti
  reRates = vector(length = nTrial); reRates[1] = reRate
  
  # initialize variables for recording targets and deltas in updating Qwait
  deltas = matrix(NA, nTimeStep, nTrial) 
  targets = matrix(NA, nTimeStep, nTrial)
  
  # initialize outputs 
  lik_ = matrix(nrow = nTimeStep, ncol = nTrial)
  
  # loop over trials
  for(tIdx in 1 : nTrial) {
    # calculate likelyhood
    nextReward = trialEarnings[tIdx]
    getReward = ifelse(nextReward == tokenValue, T, F)
    lik_[,tIdx] =  sapply(1 : nTimeStep, function(i) 1 / sum(1  + exp((Viti - reRate- Qwait[i])* tau)))
    
    # update values 
    T = Ts[tIdx]
    if(tIdx < nTrial){
      returns = sapply(1 : (T-1), function(t) nextReward - reRate * (T-t) + Viti)
      if(getReward){
        targets[1 : (T-1), tIdx] = returns[1 : (T-1)];
        deltas[1 : (T-1), tIdx] = returns[1 : (T-1)] - Qwait[1 : (T-1)]
        Qwait[1 : (T-1)] = Qwait[1 : (T-1)] + phi*(returns[1 : (T-1)] - Qwait[1 : (T-1)])
      }else{
        if(T > 2){
          targets[1 : (T-2), tIdx] = returns[1 : (T-2)]
          deltas[1 : (T-2), tIdx] = returns[1 : (T-2)] - Qwait[1 : (T-2)]
          Qwait[1 : (T-2)] = Qwait[1 : (T-2)] + phi * (returns[1 : (T-2)] - Qwait[1 : (T-2)])
        }
      }
      # update Viti
      delta = (returns[1] - reRate * (iti / stepDuration) - Viti)
      Viti = Viti + phi * delta
      # update reRate 
      reRate = reRate + beta * delta
      # record action values
      Qwaits[,tIdx + 1] = Qwait
      Vitis[tIdx + 1] = Viti
      reRates[tIdx + 1] = reRate
    }# end of the value update procedure
  } # end of the loop over trials
  
  # return outputs
  outputs = list( 
    "lik_" = lik_,
    "Qwaits" = Qwaits, "targets" = targets, "deltas" = deltas,
    "Vitis" = Vitis, "reRates" = reRates
  )
  return(outputs)
}


RL2 = function(paras, cond, trialEarnings, timeWaited){
  # parse para
  phi = paras[1]; phiP = paras[2]; tau = paras[3]; prior = paras[4]
  beta = paras[5]; betaP = paras[6]
  
  # prepare inputs
  nTrial = length(trialEarnings)
  tMax= ifelse(cond == "HP", tMaxs[1], tMaxs[2])
  nTimeStep = tMax / stepDuration
  Ts = round(ceiling(timeWaited / stepDuration) + 1)
  
  # initialize actionValues
  subOptimalRatio = 0.9 
  wIni = (5/6 + 0.93) / 2 * stepDuration * subOptimalRatio
  reRate = wIni
  Viti = 0 
  Qwait = prior*0.1 - 0.1*(0 : (nTimeStep - 1)) + Viti
  
  # initialize varibles for recording action values
  Qwaits = matrix(NA, nTimeStep, nTrial); Qwaits[,1] = Qwait
  Vitis = vector(length = nTrial); Vitis[1] = Viti
  reRates = vector(length = nTrial); reRates[1] = reRate
  
  # initialize variables for recording targets and deltas in updating Qwait
  deltas = matrix(NA, nTimeStep, nTrial) 
  targets = matrix(NA, nTimeStep, nTrial)
  
  # initialize outputs 
  lik_ = matrix(nrow = nTimeStep, ncol = nTrial)
  
  # loop over trials
  for(tIdx in 1 : nTrial) {
    # calculate likelyhood
    nextReward = trialEarnings[tIdx]
    getReward = ifelse(nextReward == tokenValue, T, F)
    lik_[,tIdx] =  sapply(1 : nTimeStep, function(i) 1 / sum(1  + exp((Viti - reRate- Qwait[i])* tau)))
    
    # update values 
    T = Ts[tIdx]
    if(tIdx < nTrial){
      returns = sapply(1 : (T-1), function(t) nextReward - reRate * (T-t) + Viti)
      if(getReward){
        targets[1 : (T-1), tIdx] = returns[1 : (T-1)];
        deltas[1 : (T-1), tIdx] = returns[1 : (T-1)] - Qwait[1 : (T-1)]
        Qwait[1 : (T-1)] = Qwait[1 : (T-1)] + phi*(returns[1 : (T-1)] - Qwait[1 : (T-1)])
      }else{
        if(T > 2){
          targets[1 : (T-2), tIdx] = returns[1 : (T-2)]
          deltas[1 : (T-2), tIdx] = returns[1 : (T-2)] - Qwait[1 : (T-2)]
          Qwait[1 : (T-2)] = Qwait[1 : (T-2)] + phiP*(returns[1 : (T-2)] - Qwait[1 : (T-2)])
        }
      }
      # update Viti
      delta = (returns[1] - reRate * (iti / stepDuration) - Viti)
      Viti = ifelse(nextReward > 0, Viti + phi * delta, Viti + phiP * delta)
      # update reRate 
      reRate = ifelse(nextReward > 0, reRate + beta * delta, reRate + betaP * delta) 
      # record action values
      Qwaits[,tIdx + 1] = Qwait
      Vitis[tIdx + 1] = Viti
      reRates[tIdx + 1] = reRate
    }# end of the value update procedure
  } # end of the loop over trials
  
  # return outputs
  outputs = list( 
    "lik_" = lik_,
    "Qwaits" = Qwaits, "targets" = targets, "deltas" = deltas,
    "Vitis" = Vitis, "reRates" = reRates
  )
  return(outputs)
}


BL = function(paras, cond, trialEarnings, timeWaited){
  # parse para
  pWait = paras[1];
  
  # prepare inputs
  nTrial = length(trialEarnings)
  tMax= max(tMaxs)
  nTimeStep = tMax / stepDuration
  Ts = round(ceiling(timeWaited / stepDuration) + 1)
  # calculate likelyhood
  lik_ = matrix(pWait, nrow = nTimeStep, ncol = nTrial)
  # return outputs
  outputs = list( 
    "lik_" = lik_
  )
  return(outputs)
}


