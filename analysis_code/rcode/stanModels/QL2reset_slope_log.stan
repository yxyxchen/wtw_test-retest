data {
  // experiment parameters
  real stepSec;// duration between two decision points
  real iti;// iti duration
  int  nWaitOrQuit; // number of possible wait-or-quit choices
  real tWaits[nWaitOrQuit]; // time for each decision point 
  
  // initial value for V0
  real V0_ini; 
  
  // empirical data
  int N; // number of trials
  int N_block1; // number of trials in block1
  int Rs[N]; // payoff in each trial
  real Ts[N]; // a trial ends at t == T
  int nMadeActions[N];// number of made actions in each trial 
}
transformed data {
  // total number of decision points in all trials
  int nTotalAction = sum(nMadeActions);
}
parameters {
  // parameters:
  // alpha : learning rate 
  // nu : valence-dependent bias
  // tau : action consistency, namely the soft-max temperature parameter
  // gamma: discount factor
  // eta: prior belief parameter
  
  // for computational efficiency,we sample raw parameters from normal(0, 1)
  // which are then transformed via the probit function into bounded parameters
  real raw_alpha;
  real raw_nu; // ratio between alphaR and alphaU
  real raw_tau;
  real raw_gamma;
  real raw_eta; 
}
transformed parameters{
  // scale raw parameters into real parameters
  real <lower=0, upper=0.3> alpha = Phi_approx(raw_alpha) * 0.3; 
  real <lower=0, upper=1> alphaU = min([pow(alpha, Phi_approx(raw_nu)) * 10, 1]'); 
  real <lower=0, upper=10> nu = log(alphaU) / log(alpha);
  real <lower=0, upper=42> tau = Phi_approx(raw_tau) * 42; 
  real <lower=0.5, upper=1> gamma = Phi_approx(raw_gamma)* 0.5 + 0.5;  
  real <lower=0, upper=15> eta = Phi_approx(raw_eta) * 15; 

  // declare variables 
  // // state value of t = 0
  real V0; 
  // // action value of waiting in each decision points 
  vector[nWaitOrQuit] Qwaits; 
  // // variables to record action values 
  matrix[nWaitOrQuit, N] Qwaits_ = rep_matrix(0, nWaitOrQuit, N);
  vector[N] V0_ = rep_vector(0, N);
  // // expected return 
  real G0;
  
  // initialize action values 
  //// the initial value of t = 0 
  V0 = V0_ini; 
  // the initial waiting value delines with elapsed time 
  // and the eta parameter determines at which step it falls below V0
  for(i in 1 : nWaitOrQuit){
    Qwaits[i] = - tWaits[i] * 0.1 / eta + 1 + V0;
  }
  
  // record initial action values
  Qwaits_[,1] = Qwaits;
  V0_[1] = V0;
 
  //loop over trials
  if(N_block1 > 0){
    for(tIdx in 1 : (N_block1 - 1)){
      real T = Ts[tIdx]; // this trial ends on t = T
      int R = Rs[tIdx]; // payoff in this trial
      int lastDecPoint = nMadeActions[tIdx]; // last decision point in this trial
      real LR; 
    
      // determine the learning rate 
      if(R > 0){
        LR = alpha;
      }else{
        LR = alphaU;
      }
      // update Qwaits towards the discounted returns
      for(i in 1 : lastDecPoint){
        real t = tWaits[i]; // time for this decision points 
        real Gt = exp(log(gamma) * (T - t)) * (R + V0);
        Qwaits[i] = Qwaits[i] + LR * (Gt - Qwaits[i]);
      }
      
      // update V0 towards the discounted returns 
      G0 = exp(log(gamma) * (T - (-iti))) * (R + V0);
      V0 = V0 + LR * (G0 - V0);
      
      // save action values
      Qwaits_[,tIdx+1] = Qwaits;
      V0_[tIdx+1] = V0;
    }
  }
  
  if(N > N_block1){
    // reset
    V0 = V0_ini; 
    for(i in 1 : nWaitOrQuit){
      Qwaits[i] = - tWaits[i] * 0.1/ eta  + 1 + V0;
    }
    Qwaits_[,N_block1 + 1] = Qwaits;
    V0_[N_block1 + 1] = V0; 
    
    for(tIdx in (1 + N_block1): (N - 1)){
      real T = Ts[tIdx]; // this trial ends on t = T
      int R = Rs[tIdx]; // payoff in this trial
      int lastDecPoint = nMadeActions[tIdx]; // last decision point in this trial
      real LR; 
    
      // determine the learning rate 
      if(R > 0){
        LR = alpha;
      }else{
        LR = alphaU;
      }
      // update Qwaits towards the discounted returns
      for(i in 1 : lastDecPoint){
        real t = tWaits[i]; // time for this decision points 
        real Gt = exp(log(gamma) * (T - t)) * (R + V0);
        Qwaits[i] = Qwaits[i] + LR * (Gt - Qwaits[i]);
      }
      
      // update V0 towards the discounted returns 
      G0 = exp(log(gamma) * (T - (-iti))) * (R + V0);
      V0 = V0 + LR * (G0 - V0);
      
      // save action values
      Qwaits_[,tIdx+1] = Qwaits;
      V0_[tIdx+1] = V0;
    }
  }
}
model {
  // delcare variables 
  int action; 
  vector[2] actionValues; 
  // distributions for raw parameters
  raw_alpha ~ normal(0, 1);
  raw_nu ~ normal(0, 1);
  raw_tau ~ normal(0, 1);
  raw_gamma ~ normal(0, 1);
  raw_eta ~ normal(0, 1);
  
  // loop over trials
  for(tIdx in 1 : N){
    real T = Ts[tIdx]; // this trial ends on t = T
    int R = Rs[tIdx]; // payoff in this trial
    int lastDecPoint = nMadeActions[tIdx]; // total number of actions in a trial
    
    // loop over decision points
    for(i in 1 : lastDecPoint){
      // the agent wait in every decision point in rewarded trials
      // and wait except for the last decision point in non-rewarded trials
      if(R == 0 && i == lastDecPoint){
        action = 2; // quit
      }else{
        action = 1; // wait
      }
      // calculate the likelihood using the soft-max function
      actionValues[1] = Qwaits_[i, tIdx] * tau;
      actionValues[2] = V0_[tIdx] * tau;
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
      actionValues[2] = V0_[tIdx] * tau;
      log_lik[no] =categorical_logit_lpmf(action | actionValues);
      no = no + 1;
    }
  }
  // calculate total log likelihood
  totalLL =sum(log_lik);
}

