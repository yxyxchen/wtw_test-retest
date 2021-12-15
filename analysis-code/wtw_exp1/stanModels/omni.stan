data {
  // experiment parameters
  real stepSec;// duration between two decision points
  real iti;// iti duration
  int  nWaitOrQuit; // number of possible wait-or-quit choices
  real tWaits[nWaitOrQuit]; // time for each decision point 
  
  // empirical data
  int N; // number of trials
  int Rs[N]; // payoff in each trial
  real Ts[N]; // a trial ends at t == T
  int nMadeActions[N];// number of made actions in each trial 
  int cIdxs[N];
  
  // theoretical values
  real HPvalues[nWaitOrQuit];
  real LPvalues[nWaitOrQuit];
}
transformed data {
  // total number of decision points in all trials
  int nTotalAction = sum(nMadeActions);
}
parameters {
  // parameters:
  // tau: action consistency 
  
  // for computational efficiency,we sample raw parameters from unif(-0.5, 0.5)
  // which are later transformed into actual parameters
  real<lower = -0.5, upper = 0.5> raw_tau;

}
transformed parameters{
  // transfer paras
  real tau = (raw_tau + 0.5) * 21.9 + 0.1 ; // tau ~ unif(0.1, 22)
}
model {
  // delcare variables 
  int action; 
  vector[2] actionValues;
  real values[nWaitOrQuit];
  actionValues[2] = 0; // decision threshold 
  
  
  // sample
  raw_tau ~ uniform(-0.5, 0.5);
  
  // loop over trials
  for(tIdx in 1 : N){
    real T = Ts[tIdx]; // this trial ends on t = T
    int R = Rs[tIdx]; // payoff in this trial
    int nMadeAction = nMadeActions[tIdx]; // last decision point in this trial
    
    // determine decision values 
    if(cIdxs[tIdx] == 1){
      values = HPvalues;
    }else{
      values = LPvalues;
    }
    
    // loop over decision points
    for(i in 1 : nMadeAction){
      // the agent wait in every decision point in rewarded trials
      // and wait except for the last decision point in non-rewarded trials
      if(R == 0 && i == nMadeAction){
        action = 2; // quit
      }else{
        action = 1; // wait
      }
      // calculate the likelihood 
      actionValues[1] = values[i] * tau;
      target +=  categorical_logit_lpmf(action | actionValues);
    } 
  }
}
generated quantities {
 // initialize variables
  int action;
  vector[2] actionValues;
  real values[nWaitOrQuit];
  vector[nTotalAction] log_lik = rep_vector(0, nTotalAction); // trial-wise log likelihood 
  real totalLL; // total log likelihood
  int no = 1; // action index
  
  actionValues[2] = 0; // decision threshold 
  // loop over trials
  for(tIdx in 1 : N){
    real T = Ts[tIdx]; // this trial ends on t = T
    int R = Rs[tIdx]; // payoff in this trial
    int nMadeAction = nMadeActions[tIdx]; // last decision point in this trial

    // determine decision values 
    if(cIdxs[tIdx] == 1){
      values = HPvalues;
    }else{
      values = LPvalues;
    }
    
    // loop over decision points
    for(i in 1 : nMadeAction){
      if(R == 0 && i == nMadeAction){
        action = 2; // quit
      }else{
        action = 1; // wait
      }
      // calculate the likelihood 
      actionValues[1] = values[i] * tau;
      log_lik[no] = categorical_logit_lpmf(action | actionValues);
      no = no + 1;
    }
  }
  // calculate total log likelihood
  totalLL =sum(log_lik);
}



