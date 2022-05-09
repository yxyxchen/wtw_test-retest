expModelFit = function(expname, sess, modelName, isFirstFit, fit_method, batchIdx = NULL, parallel = F){
  # load experiment parameters
  load("expParas.RData")
  
  # load sub-functions and packages
  library("dplyr"); library("tidyr")
  source("subFxs/loadFxs.R")
  source("subFxs/helpFxs.R")
  source('subFxs/modelFitGroup.R')
  
  # load taskdata 
  allData = loadAllData(expname, sess)
  hdrData = allData$hdrData
  trialData = allData$trialData
  
  # set output directory 
  if(fit_method == "whole"){
    outputDir = sprintf("../../analysis_results/%s/modelfit/%s", expname, modelName)
  }else if(fit_method == 'trct'){
    outputDir = sprintf("../../analysis_results/%s/modelfit/%s_trct", expname, modelName)
    # truncate the first half block
    ids = names(trialData)
    nSub = length(ids)
    for(i in 1 : length(ids)){
      id = ids[i]
      thisTrialData = trialData[[id]]
      thisTrialData = thisTrialData[thisTrialData$trialStartTime >= 300 | thisTrialData$blockNum > 1,]
      trialData[[id]] = thisTrialData
    }
  }else if(fit_method == 'onlyLP'){
    outputDir = sprintf("../../analysis_results/%s/modelfit/%s_%s", expname, modelName, fit_method)
    # only include the first block 
    ids = names(trialData)
    nSub = length(ids)
    for(i in 1 : length(ids)){
      id = ids[i]
      thisTrialData = trialData[[id]]
      thisTrialData = thisTrialData[thisTrialData$condition == "LP",]
      trialData[[id]] = thisTrialData
    }
  }else if(fit_method == 'onlyHalfLP'){
    outputDir = sprintf("../../analysis_results/%s/modelfit/%s_%s", expname, modelName, fit_method)
    ids = names(trialData)
    nSub = length(ids)
    for(i in 1 : length(ids)){
      id = ids[i]
      thisTrialData = trialData[[id]]
      thisTrialData = thisTrialData[thisTrialData$condition == "LP" & thisTrialData$trialStartTime <300,]
      trialData[[id]] = thisTrialData
    }
  }
  dir.create(sprintf("../../analysis_results/%s/modelfit", expname), showWarnings = FALSE)
  dir.create(outputDir, showWarnings = FALSE)
  
  # set model fit configurations
  if(isFirstFit){
    config = list(
      nChain = 4,
      nIter = 1000 + 1000,
      warmup = 1000,
      # nIter = 200 + 200,
      # warmup = 200,
      adapt_delta = 0.99,
      max_treedepth = 11,
      warningFile = sprintf("stanWarnings/exp_%s.txt", modelName)
    )
    # divide data into small batches if batchIdx exists 
    if(!is.null(batchIdx)){
      nSub = length(trialData)
      batchsize = floor(nSub / 3)
      if(batchIdx == 1){
        trialData = trialData[1 : batchsize]
      }else if(batchIdx == 2){
        trialData = trialData[(batchsize + 1) : (batchsize * 2)]
      }else if(batchIdx == 3){
        trialData = trialData[(batchsize * 2 + 1) : nSub]
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
    expPara = loadExpPara(paraNames, outputDir, sess)
    passCheck = checkFit(paraNames, expPara)
    trialData = trialData[!passCheck]
    
    # increase the num of Iterations 
    config = list(
      nChain = 4,
      nIter = 1000 + 5000,
      warmup = 5000,
      adapt_delta = 0.99,
      max_treedepth = 11,
      warningFile = sprintf("stanWarnings/exp_refit_%s.txt", modelName)
    )
  }

  # fit the model for all participants
  modelFitGroup(sess, modelName, trialData, config, outputDir, parallel = parallel, isTrct = T)
}

############## main script ##############
if (sys.nframe() == 0){
  args = commandArgs(trailingOnly = T)
  print(args)
  # print(length(args))
  # print(args[1])
  if(length(args) == 6){
    expModelFit(args[1], as.numeric(args[2]), args[3], as.logical(args[4]), args[5], as.numeric(args[6]))
  }
}


