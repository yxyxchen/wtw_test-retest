# unupdated Qwaits were not recorded, so here we manually
# set them identical with the last updated value

actionValueViewer = function(blockData){
  Qwaits = blockData$Qwaits
  Qquits = blockData$Qquits
  nTrial =  length(blockData$trialEarnings)
  nTimeStep = dim(Qwaits)[1]
  for(tIdx in 1: nTrial){
    plotData = data.frame(va =c(Qwaits[,tIdx], rep(Qquits[tIdx], nTimeStep)),
                          time = rep( 1 : nTimeStep, 2),
                          action = rep(c('wait', 'quit'),
                                       each = nTimeStep))
    trialTitle =  sprintf('Trial %d', tIdx)
    if(tIdx == 1){
      preRewardTitle = ""
    }else{
      preRewardTitle = sprintf(', preR = %d, preT = %.2f',
                               blockData$trialEarnings[tIdx-1], blockData$timeWaited[tIdx-1])
    }

    nowRewardTitle = sprintf(', nowR = %d, nowT =%.2f',
                             blockData$trialEarnings[tIdx], blockData$timeWaited[tIdx])  
    label = paste(trialTitle, preRewardTitle, nowRewardTitle, sep = '')
    
    
    # calculate in which step did the trial stops(so waiting values before it won't be updated)
    endStep = round(ifelse(blockData$trialEarnings[tIdx]>0, ceiling(blockData$scheduledWait[tIdx] / stepDuration),
                                                                 floor(blockData$timeWaited[tIdx] / stepDuration) + 1))
    
    p = ggplot(plotData, aes(time, va, color = action)) + geom_line() +
      geom_vline(xintercept = endStep) + xlim(c(-1, nTimeStep)) +
      ggtitle(label) + xlab('time step') + ylab('action value') + displayTheme
    print(p)
    
    input = readline(prompt = paste(tIdx, '(ENTER to continue, Q to quit)'))
    if(input == 'Q'){
      break
    }
  }  
}