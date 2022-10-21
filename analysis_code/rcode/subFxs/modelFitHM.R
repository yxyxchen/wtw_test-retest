# fit a reinforcement learning model for a single participant in Rstan 
# inputs:
  # thisTrialData: behavioral data for this participant
  # modelName: the name of   the model 
  # paraNames: parameters for the model
  # model: the Bayesian model 
  # config: a list containing the Rstab configuration 
  # outputFile: filename to save the data

modelFitHM = function(sess, modelName, trialData, stepSec, config, outputDir, parallel, isTrct = T){
    # load experiment constants 
    load('expParas.RData')
    #load libraries
    library('plyr'); 
    library('rstan');library("loo");library("coda") 
    source('subFxs/modelFitSingle.R') # for fitting each single participant
    # compile the Rstan model 
    options(warn= 1) 
    Sys.setenv(USE_CXX14=1) 
    
    # generate output file 
    outputFile = sprintf("%s/sess%d", outputDir, sess)
    # create the output directory 
    dir.create(outputDir)
    # create the file for Rstan warnings and erros
    writeLines("", config[['warningFile']])
    # rstan_options(auto_write = TRUE) 
    model = stan_model(file = sprintf("stanModels/%s.stan", modelName))
    
    # determine parameters 
    paraNames = getParaNames(modelName)
    # parse the stan configuration
    nChain = config[['nChain']] # number of MCMC chains
    nIter = config[['nIter']] # number of total iterations on each chain
    warmup = config[['warmup']] # number of warm-up iterations on each chain
    controlList = list(adapt_delta = config[['adapt_delta']],
                       max_treedepth = config[['max_treedepth']] )
    warningFile = config[['warningFile']] # output file for stan warnings and errors
    
    ####### prepare inputs ######
    # analysis constants 
    iti = 1.5 
    delayMax = max(delayMaxs)
    tWaits = seq(0, delayMax - stepSec, by = stepSec)
    nWaitOrQuit = length(tWaits) 
    
    # prepare inputs
    ids = names(trialData)
    S = length(ids)
    N_subj = vector(length = S)
    block1_N_subj = vector(length = S)
    for(sIdx in 1 : S){
      id = ids[sIdx]
      thisTrialData = trialData[[id]]
      if(isTrct){
        excludedTrials = which(thisTrialData$trialStartTime > (blockSec - max(delayMaxs)))
        thisTrialData = thisTrialData[!(1 : nrow(thisTrialData)) %in% excludedTrials,]
      }
      thisTrialData = within(thisTrialData, {timeWaited[trialEarnings!= 0] = scheduledWait[trialEarnings!= 0]})
      thisTrialData$timeWaited = pmin(thisTrialData$timeWaited, max(delayMaxs)) 
      trialData[[id]] = thisTrialData
      
      N_subj[sIdx] = nrow(thisTrialData)
      block1_N_subj[sIdx] = sum(thisTrialData$condition == "LP")
    }
    N = max(N_subj)
    R_ = matrix(0, S, N)
    T_ = matrix(0, S, N)
    nMadeActions_ = matrix(0, S, N)
    for(sIdx in 1 : S){
      id = ids[sIdx]
      thisTrialData = trialData[[id]]
      R_[sIdx, 1 : N_subj[sIdx]] = thisTrialData$trialEarnings
      T_[sIdx, 1 : N_subj[sIdx]] = thisTrialData$timeWaited
      nMadeActions_[sIdx, 1 : N_subj[sIdx]] = ceiling(thisTrialData$timeWaited / stepSec)
    }
    nTotalAction = sum(nMadeActions_)
  
    ## orgianze inputs into a list
    inputs <- list(
      iti = iti,
      stepSec = stepSec,
      nWaitOrQuit = nWaitOrQuit, # number of all possible decision time points
      tWaits = tWaits, # decision time points
      S = S,
      N = N,
      N_subj = N_subj,
      block1_N_subj = block1_N_subj,
      R_ = R_,
      T_ = T_,
      nMadeActions_ = nMadeActions_,
      nTotalAction = nTotalAction
    )
    if(substr(modelName, 1, 2) == 'QL'){
      V0_ini = 0.27782194519542547  / (1 - 0.85) # unit: cents
      inputs$V0_ini = V0_ini
    }else{
      rewardRate_ini = 0.27782194519542547 # unit: cents per second 
      inputs$rewardRate_ini = rewardRate_ini
    }
    
   # fit the model
   withCallingHandlers({
      fit = sampling(object = model, data = inputs, cores = 1, chains = nChain, warmup = warmup,
                     iter = nIter, control = controlList) 
      print(sprintf("Finish %s!", modelName))
      write(sprintf("Finish %s!", modelName), warningFile, append = T, sep = "\n")
    }, warning = function(w){
      write(paste(modelName, w), warningFile, append = T, sep = "\n")
    })
  
  # extract posterior samples
  samples = fit %>% rstan::extract(permuted = F, pars = c(paraNames, "totalLL")) %>%
    adply(2, function(x) x) %>% dplyr::select(-chains) 
  write.table(samples, file = sprintf("%s_sample.txt", outputFile), 
              sep = ",", col.names = F, row.names=FALSE)

  # calculate WAIC and Efficient approximate leave-one-out cross-validation (LOO)
  log_lik = extract_log_lik(fit) 
  WAIC = waic(log_lik)
  LOO = loo(log_lik)
  save("WAIC", "LOO", file = sprintf("%s_waic.RData", outputFile))
  
  # summarise posterior parameters and total log likelihood
  fitSummary <- summary(fit, pars = c(paraNames, "totalLL"), use_cache = F)$summary

  # detect participants with low ESSs and high Rhats 
  ESSCols = which(str_detect(colnames(fitSummary), "Effe")) # columns recording ESSs
  if(any(fitSummary[,ESSCols] < nChain * 100)){
    write(paste(modelName, subName, "Low ESS"), warningFile, append = T, sep = "\n")
  }
  RhatCols = which(str_detect(colnames(fitSummary), "Rhat")) # columns recording ESSs
  if(any(fitSummary[,RhatCols] > 1.01)){
    write(paste(modelName, subName, "High Rhat"), warningFile, append = T, sep = "\n")
  } 
  
  # check divergent transitions
  sampler_params <- get_sampler_params(fit, inc_warmup=FALSE)
  divergent <- do.call(rbind, sampler_params)[,'divergent__']
  nDt = sum(divergent)
  fitSummary = cbind(fitSummary, nDt = rep(nDt, nrow(fitSummary)))

  # write outputs  
  write.table(fitSummary, file = sprintf("%s_summary.txt", outputFile), 
              sep = ",", col.names = F, row.names=FALSE)
}

