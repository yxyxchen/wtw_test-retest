expModelFit = function(expname, sess, modelName, isFirstFit, fit_method, stepSec = 0.5, batchIdx = NULL, parallel = F){
  # load experiment parameters
  if(expname == "timing"){
    load("expParas_timing.RData")
  }else{
    load("expParas.RData")
  }

  # load sub-functions and packages
  library("dplyr"); library("tidyr")
  source("subFxs/loadFxs.R")
  source("subFxs/helpFxs.R")
  source('subFxs/modelFitGroup.R')

  # load taskdata 
  allData = loadAllData(expname, sess)
  hdrData = allData$hdrData
  trialData = allData$trialData
  

  # make directions 
  outputDir = sprintf("../../analysis_results/%s/modelfit/%s/stepsize%.2f/%s", expname,  fit_method, stepSec, modelName)
  dir.create(sprintf("../../analysis_results/%s", expname))
  dir.create(sprintf("../../analysis_results/%s/modelfit", expname))
  dir.create(sprintf("../../analysis_results/%s/modelfit/%s", expname, fit_method))
  dir.create(sprintf("../../analysis_results/%s/modelfit/%s/stepsize%.2f", expname, fit_method, stepSec))
  dir.create(outputDir)
  # set output directory 

  # prepare data
  ids = names(trialData)
  nSub = length(ids)
  if(fit_method == 'trct'){
    # truncate the first half block
    for(i in 1 : length(ids)){
      id = ids[i]
      thisTrialData = trialData[[id]]
      thisTrialData = thisTrialData[thisTrialData$trialStartTime > 30,]
      trialData[[id]] = thisTrialData
    }
  }else if(fit_method == 'onlyLP'){
    for(i in 1 : length(ids)){
      id = ids[i]
      thisTrialData = trialData[[id]]
      thisTrialData = thisTrialData[thisTrialData$condition == "LP",]
      trialData[[id]] = thisTrialData
    }
  }else if(fit_method == 'onlyHP'){
    for(i in 1 : length(ids)){
      id = ids[i]
      thisTrialData = trialData[[id]]
      thisTrialData = thisTrialData[thisTrialData$condition == "HP",]
      trialData[[id]] = thisTrialData
    }
  }else if(fit_method == "even"){
      for(i in 1 : length(ids)){
        id = ids[i]
        thisTrialData = trialData[[id]]
        thisTrialData = thisTrialData[thisTrialData$trialNum %% 2 == 0,]
        trialData[[id]] = thisTrialData
      }
  }else if(fit_method == "odd"){
    for(i in 1 : length(ids)){
      id = ids[i]
      thisTrialData = trialData[[id]]
      thisTrialData = thisTrialData[thisTrialData$trialNum %% 2 == 1,]
      trialData[[id]] = thisTrialData
    }
  }
  
  
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
    # load
    existing_files = list.files(sprintf("../../analysis_results/%s/modelfit/%s/stepsize%.2f/%s", expname, fit_method, stepSec, modelName),
                        pattern = sprintf("sess%d_summary.txt", sess))
    existing_ids = substr(existing_files, 1, 5)
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
      # trialData = trialData[!(names(trialData) %in%  existing_ids)]
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
  modelFitGroup(expname, sess, modelName, trialData, stepSec, config, outputDir, parallel = parallel, isTrct = T)
}

############## main script ##############
if (sys.nframe() == 0){
  args = commandArgs(trailingOnly = T)
  print(args)
  # print(length(args))
  # print(args[1])
  if(length(args) == 7){
    expModelFit(args[1], as.numeric(args[2]), args[3], as.logical(args[4]), args[5], as.numeric(args[6]), as.numeric(args[7]))
  }
}
expname = "timing"
sess = 1
modelName = "QL2reset_slope_two"
isFirstFit = TRUE
fit_method = "whole"
batchIdx = NULL
parallel = FALSE
stepSec = 0.50

