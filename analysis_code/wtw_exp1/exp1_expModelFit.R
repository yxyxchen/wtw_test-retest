expModelFit = function(sess, modelName, isFirstFit, batchIdx = NULL, parallel = F){
  # generate output directories
  # dir.create("stanWarnings")
  
  # load experiment parameters
  load("expParas.RData")
  
  # load sub-functions and packages
  library("dplyr"); library("tidyr")
  source("subFxs/loadFxs.R")
  source("subFxs/helpFxs.R")
  source('subFxs/modelFitGroup.R')
  
  # prepare inputs
  # I need to prepare my outputs here; it sucks 
  allData = loadAllData(sess)
  hdrData = allData$hdrData
  trialData = allData$trialData
  # names(trialData) = hdrData$id # this is probably wrong
  outputDir = sprintf("../../analysis_results/modelfit/%s", modelName)

  if(isFirstFit){
    config = list(
      nChain = 4,
      nIter = 4000,
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
        trialData[115 : length(trialData)]
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

