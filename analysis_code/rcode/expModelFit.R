expModelFit = function(expname, sess, modelName, isFirstFit, fit_method, stepSec = 0.5, batchIdx = NULL, parallel = F){
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
    outputDir = sprintf("../../analysis_results/%s/modelfit/whole/stepsize%.2f/%s", expname, stepSec, modelName)
    dir.create(sprintf("../../analysis_results/%s/modelfit/whole", expname))
    dir.create(sprintf("../../analysis_results/%s/modelfit/whole/stepsize%.2f", expname, stepSec))
    dir.create(outputDir)
  }else if(fit_method == 'trct'){
    outputDir = sprintf("../../analysis_results/%s/modelfit/trct/stepsize%.2f/%s", expname,  stepSec, modelName)
    dir.create(sprintf("../../analysis_results/%s/modelfit/trct", expname))
    dir.create(sprintf("../../analysis_results/%s/modelfit/trct/stepsize%.2f", expname, stepSec))
    dir.create(outputDir)
    # truncate the first half block
    ids = names(trialData)
    nSub = length(ids)
    for(i in 1 : length(ids)){
      id = ids[i]
      thisTrialData = trialData[[id]]
      thisTrialData = thisTrialData[thisTrialData$trialStartTime > 30,]
      trialData[[id]] = thisTrialData
    }
  }else if(fit_method == 'onlyLP'){
    outputDir = sprintf("../../analysis_results/%s/modelfit/onlyLP/stepsize%.2f/%s", expname,  stepSec, modelName)
    dir.create(sprintf("../../analysis_results/%s/modelfit/onlyLP", expname))
    dir.create(sprintf("../../analysis_results/%s/modelfit/onlyLP/stepsize%.2f", expname, stepSec))
    dir.create(outputDir)
    # only include the first block 
    ids = names(trialData)
    nSub = length(ids)
    for(i in 1 : length(ids)){
      id = ids[i]
      thisTrialData = trialData[[id]]
      thisTrialData = thisTrialData[thisTrialData$condition == "LP",]
      trialData[[id]] = thisTrialData
    }
  }else if(fit_method == 'onlyHP'){
    outputDir = sprintf("../../analysis_results/%s/modelfit/onlyHP/stepsize%.2f/%s", expname,  stepSec, modelName)
    dir.create(sprintf("../../analysis_results/%s/modelfit/onlyHP", expname))
    dir.create(sprintf("../../analysis_results/%s/modelfit/onlyHP/stepsize%.2f", expname, stepSec))
    dir.create(outputDir)
    # only include the first block 
    ids = names(trialData)
    nSub = length(ids)
    for(i in 1 : length(ids)){
      id = ids[i]
      thisTrialData = trialData[[id]]
      thisTrialData = thisTrialData[thisTrialData$condition == "HP",]
      trialData[[id]] = thisTrialData
    }
  }else if(fit_method == "even"){
      outputDir = sprintf("../../analysis_results/%s/modelfit/even/stepsize%.2f/%s", expname,  stepSec, modelName)
      dir.create(sprintf("../../analysis_results/%s/modelfit/even", expname))
      dir.create(sprintf("../../analysis_results/%s/modelfit/even/stepsize%.2f", expname, stepSec))
      dir.create(outputDir)
      # only include even trials
      ids = names(trialData)
      nSub = length(ids)
      for(i in 1 : length(ids)){
        id = ids[i]
        thisTrialData = trialData[[id]]
        thisTrialData = thisTrialData[thisTrialData$trialNum %% 2 == 0,]
        trialData[[id]] = thisTrialData
      }
  }else if(fit_method == "odd"){
    outputDir = sprintf("../../analysis_results/%s/modelfit/even/stepsize%.2f/%s", expname,  stepSec, modelName)
    dir.create(sprintf("../../analysis_results/%s/modelfit/even", expname))
    dir.create(sprintf("../../analysis_results/%s/modelfit/even/stepsize%.2f", expname, stepSec))
    dir.create(outputDir)
    # only include even trials
    ids = names(trialData)
    nSub = length(ids)
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
      trialData = trialData[!(names(trialData) %in%  existing_ids)]
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
  modelFitGroup(sess, modelName, trialData, stepSec, config, outputDir, parallel = parallel, isTrct = T)
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
expname = "passive"
sess = 2
modelName = "QL2reset_FL2"
isFirstFit = TRUE
fit_method = "whole"
batchIdx = NULL
parallel = FALSE
stepSec = 0.50

