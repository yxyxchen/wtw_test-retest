data {
  // experiment parameters
  real stepSec;// duration between two decision points
  real iti;// iti duration
  int  nWaitOrQuit; // number of possible wait-or-quit choices
  real tWaits[nWaitOrQuit]; // time for each decision point 
  
  // initial value for reward rate
  real rewardRate_ini; 
  
  // empirical data
  int N; // number of trials
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
  // nu: valence based bias
  // tau : action consistency, namely the soft-max temperature parameter
  // eta: eta belief parameter
  // beta : learning rate for the task reward rate
  
  // for computational efficiency,we sample raw parameters from unif(-0.5, 0.5)
  // which are later transformed into actual parameters
  real raw_alpha;
  real raw_nu;
  real raw_tau;
  real raw_eta;
  real raw_beta_alpha;
}
transformed parameters{
  // scale raw parameters into real parameters
  real <lower=0, upper=0.3> alpha = Phi_approx(raw_alpha) * 0.3; 
  real <lower=0, upper=1> alphaU = min([alpha * Phi_approx(raw_nu) * 10, 1]'); 
  real <lower=0, upper=10> nu = alphaU / alpha;
  real <lower=0, upper=42> tau = Phi_approx(raw_tau) * 42; 
  real <lower=0.5, upper=1> eta = Phi_approx(raw_eta)* 15; 
  real <lower=0, upper=1> beta_alpha = Phi_approx(raw_beta_alpha); 
  real <lower=0, upper=1> beta = beta_alpha * alpha; 
  
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
  // // prediction error
  real delta0;
  // reward rate 
  real rewardRate;
  
  
  // initialize action values 
  //// the initial value of t = 0 
  V0 = 0; 
  rewardRate = rewardRate_ini;
  // the initial waiting value delines with elapsed time 
  // and the eta parameter determines at which step it falls below V0
  for(i in 1 : nWaitOrQuit){
    Qwaits[i] = - tWaits[i] * 0.1 + eta + V0;
  }
  
  // record initial action values
  Qwaits_[,1] = Qwaits;
  V0_[1] = V0;
 
  //loop over trials
  for(tIdx in 1 : (N - 1)){
    real T = Ts[tIdx]; // this trial ends on t = T
    int R = Rs[tIdx]; // payoff in this trial
    int nMadeAction = nMadeActions[tIdx]; // last decision point in this trial
    
    // determine alpha
    real LR;
    if(R > 0){
      LR = alpha;
    }else{
      LR = alphaU;
    }
    
    
    // update Qwaits towards the discounted returns
    for(i in 1 : nMadeAction){
      real t = tWaits[i]; // time for this decision points 
      real Gt = R - rewardRate * (T - t) + V0;
      Qwaits[i] = Qwaits[i] + LR * (Gt - Qwaits[i]);
    }
    
    // update V0 towards the discounted returns 
    G0 = R - rewardRate * (T - (-iti)) + V0;
    delta0 = G0 - V0;
    V0 = V0 + LR * delta0;
    
    // update reward rate 
    rewardRate = rewardRate + beta * delta0;
    
    // save action values
    Qwaits_[,tIdx+1] = Qwaits;
    V0_[tIdx+1] = V0;
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
  raw_eta ~ normal(0, 1);
  raw_beta_alpha ~ normal(0, 1);

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
