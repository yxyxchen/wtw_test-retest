QL1 = """
data {
  // experiment parameters
  real iti;// iti duration
  int  nt; // number of time steps
  real ts[nt]; // time steps
  
  // initial value for Qquit
  real Qquit_ini; 
  
  // empirical data
  int N; // number of trials
  int Rs[N]; // payoff in each trial
  real Ts[N]; // duration of each trial
  int nMadeActions[N];// number of invoked actions in each trial 
}
transformed data {
  // total number of actions made in all trials
  int nTotalAction = sum(nMadeActions);
}
parameters {
  // parameters:
  // alpha : learning rate
  // tau : action consistency, namely the soft-max temperature parameter
  // gamma: discount factor
  // eta: prior belief parameter
  
  // for computational efficiency,we sample raw parameters from unif(-0.5, 0.5)
  // which are later transformed into actual parameters
  real<lower = -0.5, upper = 0.5> raw_alpha;
  real<lower = -0.5, upper = 0.5> raw_tau;
  real<lower = -0.5, upper = 0.5> raw_gamma;
  real<lower = -0.5, upper = 0.5> raw_eta;
}
transformed parameters {
  // scale raw parameters into real parameters
  real alpha = (raw_alpha + 0.5) * 0.3; // alpha ~ unif(0, 0.3)
  real tau = (raw_tau + 0.5) * 21.9 + 0.1; // tau ~ unif(0.1, 22)
  real gamma = (raw_gamma + 0.5) * 0.3 + 0.7; // gamma ~ unif(0.7, 1)
  real eta = (raw_eta + 0.5) * 6.5; // eta ~ unif(0, 6.5)
  
  // declare variables 
  // // value of quitting
  real Qquit; 
  // // action value of waiting in each decision points 
  vector[nt] Qwaits; 
  // // variables to record action values 
  matrix[nt, N] Qwaits_ = rep_matrix(0, nt, N);
  vector[N] Qquit_ = rep_vector(0, N);
  // // expected return 
  real Gt;
  
  // initialize action values 
  //// the initial value of t = 0 
  Qquit = Qquit_ini; 
  // the initial waiting value delines with elapsed time 
  // and the prior parameter determines at which step it falls below Qquit
  for(i in 1 : nt){
    Qwaits[i] = - ts[i] * 0.1 + eta + Qquit;
  }
  
  // record initial action values
  Qwaits_[,1] = Qwaits;
  Qquit_[1] = Qquit;
 
  //loop over trials
  for(tIdx in 1 : (N - 1)){
    real T = Ts[tIdx]; // this trial ends on t = T
    int R = Rs[tIdx]; // payoff in this trial
    int nMadeAction = nMadeActions[tIdx]; // last decision point in this trial
    
    // update Qwaits towards the discounted returns
    for(i in 1 : nMadeAction){
      real t = ts[i]; // time for this decision points 
      Gt = exp(log(gamma) * (T - t)) * (R + Qquit);
      Qwaits[i] = Qwaits[i] + alpha * (Gt - Qwaits[i]);
    }
    
    // update Qquit towards the discounted returns 
    Gt = exp(log(gamma) * (T - (- iti))) * (R + Qquit);
    Qquit = Qquit + alpha * (Gt - Qquit);
    
    // save action values
    Qwaits_[,tIdx+1] = Qwaits;
    Qquit_[tIdx+1] = Qquit;
  }
}
model {
  // delcare variables 
  int action; 
  vector[2] actionValues; 
  // distributions for raw parameters
  raw_alpha ~ uniform(-0.5, 0.5);
  raw_tau ~ uniform(-0.5, 0.5);
  raw_gamma ~ uniform(-0.5, 0.5);
  raw_eta ~ uniform(-0.5, 0.5);
  
  // loop over trials
  for(tIdx in 1 : N){
    real T = Ts[tIdx]; // this trial ends on t = T
    int R = Rs[tIdx]; // payoff in this trial
    int nMadeAction = nMadeActions[tIdx]; // total number of actions in a trial
    
    // loop over decision points
    for(i in 1 : nMadeAction){
      // the agent wait in every decision point in rewarded trials
      // and wait except for the last decision point in non-rewarded trials
      if(R == 0 && i == nMadeAction){
        action = 2; // quit
      }else{
        action = 1; // wait
      }
      // calculate the likelihood using the soft-max function
      actionValues[1] = Qwaits_[i, tIdx] * tau;
      actionValues[2] = Qquit_[tIdx] * tau;
      target += categorical_logit_lpmf(action | actionValues);
    } 
  }
}
generated quantities {
 // initialize variables
  vector[2] actionValues;
  int action;
  vector[nTotalAction] log_lik = rep_vector(0, nTotalAction); // trial-wise log likelihood 
  real totalLL; // total log likelihood
  int no = 1; // action index
  
  // loop over trials
  for(tIdx in 1 : N){
    real T = Ts[tIdx]; // this trial ends on t = T
    int R = Rs[tIdx]; // payoff in this trial
    int nMadeAction = nMadeActions[tIdx]; // total number of actions in a trial
    
    // loop over decision points
    for(i in 1 : nMadeAction){
      if(R == 0 && i == nMadeAction){
        action = 2; // quit
      }else{
        action = 1; // wait
      }
      // calculate the likelihood using the soft-max function
      actionValues[1] = Qwaits_[i, tIdx] * tau;
      actionValues[2] = Qquit_[tIdx] * tau;
      log_lik[no] =categorical_logit_lpmf(action | actionValues);
      no = no + 1;
    }
  }
  // calculate total log likelihood
  totalLL =sum(log_lik);
}
"""

