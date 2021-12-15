data {
  // experiment parameters
  real stepSec;// duration of one step
  real iti;// iti duration, unit = second
  int  nWaitOrQuit; // number of possible wait-or-quit choices
  real tWaits[nWaitOrQuit]; // time for each decision point
  
  // experiment data
  int N; // number of trials
  int Rs[N]; // reward in each trial
  real Ts[N]; // terminal state in each trial
  int nMadeActions[N];// number of made actions in each trial 
  
}
transformed data {
  // total number of decision points in all trials
  int nTotalAction = sum(nMadeActions);
}
parameters {
  // parameters:
  // theta: probability of waiting at each stepSep
  real<lower = -0.5, upper = 0.5> raw_theta;
}
transformed parameters{
  // transfer paras
  real theta = (raw_theta + 0.5) ; // theta ~ unif(0, 1)
}
model {
  int action;
  raw_theta ~ uniform(-0.5, 0.5);
  
  // calculate the likelihood 
  for(tIdx in 1 : N){
    int R = Rs[tIdx]; // payoff in this trial
    int nMadeAction = nMadeActions[tIdx]; // total number of actions in a trial
    for(i in 1 : nMadeAction){
      if(R == 0 && i == nMadeAction){
        action = 0; // quit
      }else{
        action = 1; // wait
      }
      target += bernoulli_lpmf(action | theta);// theta defines the prob of 1
    } 
  }
}
generated quantities {
  // initialize variables
  int action;
  vector[nTotalAction] log_lik = rep_vector(0, nTotalAction); // trial-wise log likelihood 
  real totalLL; // total log likelihood
  int no = 1; // action index
  
  // loop over trials
  for(tIdx in 1 : N){
    int R = Rs[tIdx]; // payoff in this trial
    int nMadeAction = nMadeActions[tIdx]; // total number of actions in a trial
    
    // loop over decision points
    for(i in 1 : nMadeAction){
      if(R == 0 && i == nMadeAction){
        action = 0; // quit
      }else{
        action = 1; // wait
      }
      // calculate the likelihood using the soft-max function
      log_lik[no] = bernoulli_lpmf(action | theta);
      no = no + 1;
    }
  }// end of the loop
  
  // calculate total log likelihood
  totalLL =sum(log_lik);
}



