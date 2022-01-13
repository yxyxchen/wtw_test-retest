# this scripts contains some supporting functions for the simulations
drawSample = function(cond){
  # generates a sample from the designated delay distribution
  k = pareto[['k']]
  mu = pareto[['mu']]
  sigma = pareto[['sigma']]
  
  if(cond == 'HP'){
    sample = runif(1, min = 0, max = delayMaxs[1])
  }else{
    sample = min(mu + sigma * (runif(1) ^ (-k) - 1) / k, delayMaxs[2])
  }
  return(sample)
}

