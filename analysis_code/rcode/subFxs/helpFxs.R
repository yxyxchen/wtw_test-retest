library("stringr")
# return learning parameters for each model 
getParaNames = function(modelName){
  # if(modelName == "QL1reset_FL3") paraNames = c("alpha", "tau", "gamma", "eta1", "eta2")
  if(substr(modelName, 1, 3) == "QL1") paraNames = c("alpha", "tau", "gamma", "eta")
  # else if(modelName == "QL2reset_FL3") paraNames = c("alpha", "nu", "tau", "gamma", "eta1", "eta2")
  else if(substr(modelName, 1, 3) == "QL2") paraNames = c("alpha", "nu", "tau", "gamma", "eta")
  else if(substr(modelName, 1, 3) == "RL1") paraNames = c("alpha", "tau", "eta", "beta")
  else if(substr(modelName, 1, 3) == "RL2") paraNames = c("alpha", "nu", "tau", "eta", "beta")
  else if(modelName == "naive") paraNames = c("theta")
  else if(modelName == "omni") paraNames = c("tau")
  return(paraNames)
}

getGroupParaNames = function(modelName){
  if(modelName == "QL2reset_HM_simple"){
    groupParaNames = c("alpha", "tau", "eta")
  }else{
    groupParaNamesc("alpha", "tau", "gamma", "eta")
  }
}
# check MCMC fitting results 
checkFit = function(paraNames, expPara){
  ids = expPara$id
  # detect participants with high Rhats 
  RhatCols = which(str_detect(colnames(expPara), "hat"))[1 : length(paraNames)] # columns recording Rhats
  if(length(RhatCols) > 1){
    high_Rhat_ids = ids[apply(expPara[,RhatCols] >= 1.01, MARGIN = 1, sum) > 0]
  }else{
    high_Rhat_ids = ids[expPara[,RhatCols] >= 1.01 ]
  }
  
  # detect participants with low ESSs
  ESSCols = which(str_detect(colnames(expPara), "Effe"))[1 : length(paraNames)]# columns recording ESSs
  if(length(ESSCols) > 1){
    low_ESS_ids = ids[apply(expPara[,ESSCols] < (4 * 100), MARGIN = 1, sum) > 0]
  }else{
    low_ESS_ids = ids[expPara[,ESSCols] < (4 * 100)]
  }

  # detect divergent transitions
  dt_ids = ids[expPara$nDt > 0]
  
  # identify participants satisifying all three criteria:
  passCheck = !ids %in% unique(c(dt_ids, high_Rhat_ids, low_ESS_ids))
  
  return(passCheck)
}


