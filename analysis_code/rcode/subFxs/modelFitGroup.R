# fit a model for multiple participants in Rstan 

# inputs:
# modelName : the model name
# trialData : a nSubx1 list, each element containing behavioral data for one participant
# config : a list containing the Rstab configuration 
# outputDir: the directory to save parameter estimations

# the config variable contains:
# nChain : number of chains 
# nIter : number of interations on each chain 
# adapt_delta: real number from 0 to 1, and increaseing it forces stan to explore the target distribution in a smaller step
# max_treedepth: maximal depth of the trees that stan evaluates during each iteration
# warningFile : file for saving warnings generated Rstan

modelFitGroup = function(sess, modelName, trialData, config, outputDir, parallel, isTrct = T){
  # create the output directory 
  dir.create(outputDir)
 
   # create the file for Rstan warnings and erros
  writeLines("", config[['warningFile']])
  
  #load libraries
  library('plyr'); 
  library('rstan');library("loo");library("coda") 
  source('subFxs/modelFitSingle.R') # for fitting each single participant
  load("expParas.RData")
  
  # compile the Rstan model 
  options(warn= 1) 
  #Sys.setenv(USE_CXX14=1) # settings for the local PC
  rstan_options(auto_write = TRUE) 
  model = stan_model(file = sprintf("stanModels/%s.stan", modelName))

  # determine parameters 
  paraNames = getParaNames(modelName)
  
  # load expData
  ids = names(trialData)
  nSub = length(ids)                    
  
  # parallel compuation settings
  nCore = as.numeric(Sys.getenv("NSLOTS")) # settings for SCC
  if(is.na(nCore)) nCore = 1 # settings for SCC
  if(parallel){
    nCore = parallel::detectCores() -1 # settings for the local PC
    registerDoMC(nCore) # settings for the local PC
  }
  print(sprintf("Model fitting using %d cores", nCore))
  
  for(i in 1 : nSub){
      id = ids[i]
      print(id)
      thisTrialData = trialData[[id]]
      # truncate the last portion in each block 
      if(isTrct){
        excludedTrials = which(thisTrialData$trialStartTime > (blockSec - max(delayMaxs)))
        thisTrialData = thisTrialData[!(1 : nrow(thisTrialData)) %in% excludedTrials,]
      }
      outputFile = sprintf("%s/%s_sess%d", outputDir, id, sess)
      modelFitSingle(id, thisTrialData, modelName, paraNames, model, config, outputFile)
  }
}
