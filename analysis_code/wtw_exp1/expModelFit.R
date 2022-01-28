expModelFit = function(expname, sess, modelName, isFirstFit, batchIdx = NULL, fit_method = NULL, parallel = F){
  # I might want to merge expModelFit and expModelFit_trct later
  # generate output directories
  # dir.create("stanWarnings")
  
  # load experiment parameters
  load("expParas.RData")
  
  # load sub-functions and packages
  library("dplyr"); library("tidyr")
  source("subFxs/loadFxs.R")
  source("subFxs/helpFxs.R")
  source('subFxs/modelFitGroup.R')
  
  # I need to prepare my outputs here; it sucks 
  allData = loadAllData(expname, sess)
  hdrData = allData$hdrData
  trialData = allData$trialData
  
  # I want to make the output dir specific
  if(fit_method == 'trct'){
    outputDir = sprintf("../../analysis_results/%s/modelfit/%s_trct", expname, modelName)
    # truncate the first half block
    ids = names(trialData)
    nSub = length(ids)
    for(i in length(ids)){
      id = ids[i]
      thisTrialData = trialData[[id]]
      thisTrialData = thisTrialData[thisTrialData$trialStartTime >= 300 | thisTrialData$blockNum > 1,]
      trialData[[id]] = thisTrialData
    }
  }else{
    outputDir = sprintf("../../analysis_results/%s/modelfit/%s", expname, modelName)
  }
  
  
  # I also want to add a 
  if(isFirstFit){
    config = list(
      nChain = 4,
      nIter = 100,
      adapt_delta = 0.99,
      max_treedepth = 11,
      warningFile = sprintf("stanWarnings/exp_%s.txt", modelName)
    )
    # divide data into small batches if batchIdx exists 
    # work onn this one later
    if(!is.null(batchIdx)){
      if(batchIdx == 1){
        trialData = trialData[1 : 57]
      }else if(batchIdx == 2){
        trialData = trialData[58 : 114]
      }else if(batchIdx == 3){
        trialData = trialData[115 : length(trialData)]
      }
    }
  }
  # if it is the first time to fit the model, fit all participants
  # otherwise, check model fitting results and refit those that fail any of the following criteria
  ## no divergent transitions 
  ## Rhat < 1.01 
  ## Effective Sample Size (ESS) > nChain * 100
  if(!isFirstFit){
    ids = names(trialData)
    paraNames = getParaNames(modelName)
    expPara = loadExpPara(paraNames, outputDir)
    passCheck = checkFit(paraNames, expPara)
    trialData = trialData[!passCheck]
    
    # increase the num of Iterations 
    config = list(
      nChain = 4,
      nIter = 1200,
      adapt_delta = 0.99,
      max_treedepth = 11,
      warningFile = sprintf("stanWarnings/exp_refit_%s.txt", modelName)
    )
  }

  # fit the model for all participants
  modelFitGroup(sess, modelName, trialData, config, outputDir, parallel = parallel, isTrct = T)
}
if (sys.nframe() == 0){
  # use this command to test: Rscript expModelFit_trct.R 'active' 1 'QL2' T 1
  args = commandArgs(trailingOnly = T)
  if(length(args) == 6){
    expModelFit(args[1], as.numeric(args[2]), args[3], as.logical(args[4]), as.numeric(args[5]))
  }else{
    expModelFit(args[1], as.numeric(args[2]), args[3], as.logical(args[4]), as.numeric(args[5]), args[6])
  }
}

