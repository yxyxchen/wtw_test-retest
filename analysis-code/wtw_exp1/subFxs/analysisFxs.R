# plots a single subject's trial-by-trial data
trialPlots <- function(thisTrialData) {
  source('./subFxs/plotThemes.R')
  nBlock = length(unique(thisTrialData$blockNum))
  # num of trials in each block
  nTrials = sapply(1:nBlock, function(i) sum(thisTrialData$blockNum == i))
  # num of trials accumulated across blocks
  ac_nTrials = cumsum(nTrials)
  # ensure timeWaited = scheduledWait on rewarded trials
  thisTrialData = within(thisTrialData, {timeWaited[trialEarnings!= 0] = scheduledWait[trialEarnings!= 0]})

  p = thisTrialData %>%
    ggplot(aes(trialNum, timeWaited,color = factor(trialEarnings))) +
    geom_point(size = 4) + geom_line(size = 1.5) + scale_color_manual(values = c("#737373", "#fb6a4a")) +
    geom_point(data = thisTrialData[thisTrialData$trialEarnings == 0, ],
               aes(trialNum, scheduledWait),
               color = 'black', size = 4) +
    xlab("Trial") + ylab("Time (s)") + 
    myTheme 
  print(p)
  return(p)
}


# using kaplan-meier survival analysis to measure:
# average WTW (area under the curve)
# std WTW
kmsc <- function(thisTrialData, tMax, plotKMSC=FALSE, grid) {
  library(survival)
  load("expParas.RData")
  # ensure timeWaited = scheduledWait on rewarded trials
  thisTrialData = within(thisTrialData, {timeWaited[trialEarnings == tokenValue] = scheduledWait[trialEarnings == tokenValue]})
  # fit a kaplan-meier survival curve
  kmfit = survfit(Surv(thisTrialData$timeWaited, (thisTrialData$trialEarnings ==0), type='right') ~ 1, 
            type='kaplan-meier', conf.type='none', start.time=0, se.fit=FALSE)
  # extract elements of the survival curve object 
  kmT = kmfit$time # time value 
  kmF = kmfit$surv # function value
  # add a point at zero, since "kaplan-meier" starts from the first event
  kmT = c(0, kmT) 
  kmF = c(1, kmF)
  # keep only points up through tMax 
  keepIdx = kmT<=tMax
  kmT <- kmT[keepIdx]
  kmF <- kmF[keepIdx]
  # extend the last value to exactly tMax
  # notice that kmT is not evenly spaced
  kmT <- c(kmT, tMax)
  kmF <- c(kmF, tail(kmF,1))
  # calculate auc
  auc <- sum(diff(kmT) * head(kmF,-1))
  # calculate std WTW
  ## (1-kmF) gives the cdf of WTW
  cdf_WTW = 1 - kmF
  ## by adding 1 at the end, we truncate WTW at tMax
  cdf_WTW = c(cdf_WTW, 1)
  # calcualte the pdf of WTW
  pdf_WTW = diff(cdf_WTW)
  # calculate the std of WTW 
  varWTW = sum((kmT - auc) ^2 * pdf_WTW)
  stdWTW = sqrt(varWTW)
  # plot the k-m survival curve
  if (plotKMSC) {
    p = data.frame(kmT = kmT, kmF = kmF) %>%
      ggplot(aes(kmT, kmF)) + geom_line() + xlab('Delay (s)') +
      ylab('Survival rate') + ylim(c(0,1)) + xlim(c(0,tMax)) +
      ggtitle(sprintf('AUC = %1.1f',auc)) + 
      myTheme
    print(p)
  }
  # resample the survival curve for averaging 
  kmOnGrid = resample(kmF, kmT, kmGrid)
  return(list(kmT=kmT, kmF=kmF, auc=auc, kmOnGrid = kmOnGrid, stdWTW = stdWTW))
}

# calculate willingness to wait (WTW) time-series
wtwTS <- function(thisTrialData, tGrid, wtwCeiling, plotWTW = F) {
  # ensure timeWaited = scheduledWait on rewarded trials
  thisTrialData = within(thisTrialData, {timeWaited[trialEarnings!= 0] = scheduledWait[trialEarnings!= 0]})
  
  # initialize the per-trial estimate of WTW
  nTrial = length(thisTrialData$trialEarnings)
  trialWTW = numeric(length = nTrial) 
  
  ## find the longest time waited up through the first quit trial
  ## (or, if there were no quit trials, the longest time waited at all)
  ## that will be the WTW estimate for all trials up to the first quit
  quitIdx = thisTrialData$trialEarnings == 0
  firstQuitTrial = which(quitIdx)[1] 
  if (is.na(firstQuitTrial)) {firstQuitTrial = nTrial} 
  currentTrial = firstQuitTrial
  currentWTW = max(thisTrialData$timeWaited[1 : currentTrial]) 
  trialWTW[1:currentTrial] = currentWTW 
  ## iterate through the remaining trials, updating currentWTW
  ## which is the longest time waited since the recentest quit trial
  if(currentTrial < nTrial){
    for(tIdx in (currentTrial + 1) : nTrial){
      if(quitIdx[tIdx]) {currentWTW = thisTrialData$timeWaited[tIdx]}
      else {currentWTW = max(currentWTW, thisTrialData$timeWaited[tIdx])}
      trialWTW[tIdx] = currentWTW 
    }
  }
  
  # impose a ceiling value, since WTW exceeding some value may be infrequent
  trialWTW = pmin(trialWTW, wtwCeiling)
  
  # convert from per-trial to per-second 
  timeWTW = resample(trialWTW, thisTrialData$sellTime, tGrid)
  
  # plot time WTW
  if(plotWTW){
    p = ggplot(data.frame(tGrid, timeWTW), aes(tGrid, timeWTW)) + geom_line() +
      xlab("Time in block (s)") + ylab("WTW (s)")  + myTheme
    print(p)
  }
  
  # return outputs 
  outputs = list(timeWTW = timeWTW,  trialWTW = trialWTW)
  return(outputs)
}

# resample pair-wise sequences
# inputs:
# ys: y in the original sequence
# xs: x in the original sequence
# Xs: x in the new sequence
# outputs: 
# Ys : y in the new sequence 
resample = function(ys, xs, Xs){
  isBreak = F
  # initialize Ys
  Ys = rep(NA, length = length(Xs))
  for(i in 1 : length(Xs)){
    # for each X in Xs
    X = Xs[i]
    # find the index of cloest x value on the right
    # if closest_right_x_idx exists 
    if(X <= tail(xs,1)) {
      # Y takes the corresonding y value
      closest_right_x_idx = min(which(xs >= X))
      Ys[i] = ys[closest_right_x_idx]
    }else{
      isBreak = T
      lastY = i - 1
      break
    }
  }
  # fill the remaining elements in Ys by the exisiting last element
  if(isBreak){
    Ys[(lastY + 1) : length(Xs)] = Ys[lastY]
  }
  return(Ys)
}

# this function can truncate trials in the simualtion object
# which enables us to zoom in and look and specific trials
truncateTrials = function(data, startTidx, endTidx){
  nVar = length(data)
  varNames = names(data)
  outputs = vector(mode = "list", length = nVar)
  anyMatrix = F
  for(i in 1 : nVar){
    junk = data[[varNames[i]]]
    if(is.matrix(junk)){
      outputs[[i]] = junk[, startTidx:endTidx]
      anyMatrix  = T 
    }else{
      outputs[[i]] = junk[startTidx:endTidx]
    }
  }
  names(outputs) = varNames
  if(!anyMatrix )   outputs = as.data.frame(outputs)
  return(outputs)
}

############# help functions to plot the correlation figure 
my.reg <- function(x, y, digits = 2, prefix = "", use="pairwise.complete.obs", method = "kendall", cex.cor, nCmp = 1, ...) {
  points(x,y, pch=20, col = "grey")
  abline(lm(y~x), col = "black") 
  
  # position to put the correlation coefficient 
  usr <- par("usr")
  rX = usr[1] + 0.4 * (usr[2] - usr[1])
  rY = usr[3] + 0.75 * (usr[4] - usr[3])
  
  corrRes <- cor.test(x, y, method=method) # MG: remove abs here
  txt = substitute(italic(r)~"="~corrEst~","~italic(p)~"="~pvalue,
                    list(corrEst = round(corrRes$estimate, 2),
                         pvalue = round(corrRes$p.value, 2)))
  txt = as.expression(txt)
  if(missing(cex.cor)) cex <- 0.8/strwidth(txt)
  
  test <- cor.test(as.numeric(x),as.numeric(y), method=method)
  # borrowed from printCoefmat
  Signif <- symnum(test$p.value, corr = FALSE, na = FALSE,
                   cutpoints = c(0, 0.001 /nCmp, 0.01 / nCmp, 0.05 / nCmp, 1),
                   symbols = c("***", "**", "*", " "))

  # position to put the sig sign
  if(Signif == "***"){
    sigX = usr[1] + 0.85 * (usr[2] - usr[1])
  }else if(Signif == "**"){
    sigX = usr[1] + 0.80 * (usr[2] - usr[1])
  }else{
    sigX = usr[1] + 0.75 * (usr[2] - usr[1])
  }
  if(corrRes$estimate > 0){
    sigX = sigX - 0.025 * (usr[2] - usr[1])
  }
  sigY = usr[3] + 0.75 * (usr[4] - usr[3])
  # MG: add abs here and also include a 30% buffer for small numbers
  # text(0.5, 0.5, txt, cex = cex * (abs(r) + .3) / 1.3)
  text(rX, rY, txt, cex = 1)
  text(sigX, sigY, Signif, cex = 1.5, col=2)
}
my.panel.cor <- function(x, y, digits=2,  prefix="", use="pairwise.complete.obs", method = "Kendall", cex.cor, ...)
{
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
}

# integrate stacked data from several blocks 
block2session = function(thisTrialData){
  load("expParas.RData")
  nBlock = length(unique(thisTrialData$blockNum))
  nTrials = sapply(1:nBlock, function(i) sum(thisTrialData$blockNum == i))
  # accumulated trials 
  ac_nTrials = c(0, cumsum(head(nTrials, -1)))
  # accumulated task durations
  ac_taskTimes = (1:nBlock - 1) * blockSec
  # accumulated totalEarnings sofar
  ac_totalEarnings_s = c(0, thisTrialData$totalEarnings[cumsum(nTrials)[1:(nBlock - 1)]])
  
  # convert within-block variables to accumualting-across-block variables
  thisTrialData = within(thisTrialData,
                         {trialNum = trialNum + rep(ac_nTrials , time = nTrials);
                         sellTime = sellTime + rep(ac_taskTimes, time = nTrials);
                         totalEarnings = totalEarnings + rep(ac_totalEarnings_s, time = nTrials)});
  return(thisTrialData)
}


