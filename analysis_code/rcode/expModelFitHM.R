expModelFitHM = function(expname, sess, modelName, fit_method, stepSec = 0.5, parallel = F, chainIdx = 1){
  # load experiment parameters
  load("expParas.RData")

  # load sub-functions and packages
  library("dplyr"); library("tidyr")
  source("subFxs/loadFxs.R")
  source("subFxs/helpFxs.R")
  source('subFxs/modelFitHM.R')
  # load taskdata 
  allData = loadAllData(expname, sess)
  hdrData = allData$hdrData
  trialData = allData$trialData
  
  # make directions 
  outputDir = sprintf("../../analysis_results/%s/modelfit_hm/%s/stepsize%.2f/%s/chain%d", expname, fit_method, stepSec, modelName, chainIdx)
  dir.create(sprintf("../../analysis_results/%s/modelfit_hm", expname))
  dir.create(sprintf("../../analysis_results/%s/modelfit_hm/%s", expname, fit_method))
  dir.create(sprintf("../../analysis_results/%s/modelfit_hm/%s/stepsize%.2f", expname,  fit_method, stepSec))
  dir.create(sprintf("../../analysis_results/%s/modelfit_hm/%s/stepsize%.2f/%s/", expname,  fit_method, stepSec, modelName))
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
  config = list(
    nChain = 1,
    # nIter = 1000 + 1000,
    # warmup = 1000,
    nIter =  50 + 50,
    warmup = 50,
    adapt_delta = 0.99,
    max_treedepth = 11,
    warningFile = sprintf("stanWarnings/exp_%s.txt", modelName)
  )
    # load
  existing_files = list.files(sprintf("../../analysis_results/%s/modelfit/%s/stepsize%.2f/%s", expname, fit_method, stepSec, modelName),
                      pattern = sprintf("sess%d_summary.txt", sess))
  existing_ids = substr(existing_files, 1, 5)
  # divide data into small batches if batchIdx exists 
  trialData = trialData[1:50]

  # fit the model for all participants
  modelFitHM(sess, modelName, trialData, stepSec, config, outputDir, parallel = parallel, isTrct = T)
}

############## main script ##############
if (sys.nframe() == 0){
  args = commandArgs(trailingOnly = T)
  print(args)
  # print(length(args))
  # print(args[1])
  if(length(args) == 7){
    expModelFitHM(args[1], as.numeric(args[2]), args[3],  args[4], as.numeric(args[5]),  as.numeric(args[6]), as.numeric(args[7]))
  }
}
expname = "passive"
sess = 2
modelName = "QL2reset_HM_simple"
isFirstFit = TRUE
fit_method = "whole"
batchIdx = NULL
parallel = FALSE
stepSec = 0.50

