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
  
  // for computational efficiency,we sample raw parameters from unif(-0.5, 0.5)
  // which are later transformed into actual parameters
  real raw_alpha;
  real raw_nu; // ratio between alphaR and alphaU
  real raw_tau;
  real raw_gamma;
  real raw_eta; 
  
}
transformed parameters{
  // scale raw parameters into real parameters
  real alpha = Phi_approx(raw_alpha) * 0.3; // alpha ~ unif(0, 0.3)
  real alphaU = min([alpha * Phi_approx(raw_nu) * 5, 1]'); // 
  real nu = alphaU / alpha;
  real tau = Phi_approx(raw_tau) * 42; // tau ~ unif(0.1, 42)
  real gamma = Phi_approx(raw_gamma)* 0.5 + 0.5; // gamma ~ unif(0.5, 1)
  real eta = Phi_approx(raw_eta) * 15; // eta ~ unif(0, 15)
}
model {
  // declare variables 
  vector[nWaitOrQuit] Qwaits; 
  real V0; 
  vector[2] actionValues;
  int action;
  real LR;
  real G0;
  // distributions for raw parameters
  raw_alpha ~ normal(0, 1);
  raw_nu ~ normal(0, 1);
  raw_tau ~ normal(0, 1);
  raw_gamma ~ normal(0, 1);
  raw_eta ~ normal(0, 1);

 
 // loop over trials 
 for(tIdx in 1 : N){
   // reset if necessary 
   if(tIdx == 1 || tIdx == N_block1){
      // set initial values
      V0 = V0_ini; 
      // the initial waiting value delines with elapsed time 
      // and the eta parameter determines at which step it falls below V0
      for(i in 1 : nWaitOrQuit){
        Qwaits[i] = - tWaits[i] * 0.1 + eta + V0;
      }
   }
   // make choices 
    for(i in 1 : nMadeActions[tIdx]){
      // the agent wait in every decision point in rewarded trials
      // and wait except for the last decision point in non-rewarded trials
      if(Rs[tIdx] == 0 && i == nMadeActions[tIdx]){
        action = 2; // quit
      }else{
        action = 1; // wait
      }
      // calculate the likelihood using the soft-max function
      actionValues[1] = Qwaits[i] * tau;
      actionValues[2] = V0 * tau;
      target += categorical_logit_lpmf(action | actionValues);
    }
    // update value functions
    if(Rs[tIdx] > 0){
      LR = alpha;
    }else{
      LR = alphaU;
    }
    // update Qwaits towards the discounted returns
    for(i in 1 : nMadeActions[tIdx]){
      real t = tWaits[i]; // time for this decision points 
      real Gt = exp(log(gamma) * (Ts[tIdx] - t)) * (Rs[tIdx] + V0);
      Qwaits[i] = Qwaits[i] + LR * (Gt - Qwaits[i]);
    }
    
    // update V0 towards the discounted returns 
    G0 = exp(log(gamma) * (Ts[tIdx] - (-iti))) * (Rs[tIdx] + V0);
    V0 = V0 + LR * (G0 - V0);
 }
}
generated quantities {
  // declare variables 
  vector[nWaitOrQuit] Qwaits; 
  real V0; 
  vector[2] actionValues;
  int action;
  real LR;
  real G0;
  vector[nTotalAction] log_lik = rep_vector(0, nTotalAction); // trial-wise log likelihood 
  real totalLL; // total log likelihood
  int no = 1; // action index
  
 // loop over trials 
 for(tIdx in 1 : N){
   // reset if necessary 
   if(tIdx == 1 || tIdx == N_block1){
      // set initial values
      V0 = V0_ini; 
      // the initial waiting value delines with elapsed time 
      // and the eta parameter determines at which step it falls below V0
      for(i in 1 : nWaitOrQuit){
        Qwaits[i] = - tWaits[i] * 0.1 + eta + V0;
      }
   }
   // make choices 
    for(i in 1 : nMadeActions[tIdx]){
      // the agent wait in every decision point in rewarded trials
      // and wait except for the last decision point in non-rewarded trials
      if(Rs[tIdx] == 0 && i == nMadeActions[tIdx]){
        action = 2; // quit
      }else{
        action = 1; // wait
      }
      // calculate the likelihood using the soft-max function
      actionValues[1] = Qwaits[i] * tau;
      actionValues[2] = V0 * tau;
      log_lik[no] =categorical_logit_lpmf(action | actionValues);
      no = no + 1;
    }
    // update value functions
    if(Rs[tIdx] > 0){
      LR = alpha;
    }else{
      LR = alphaU;
    }
    // update Qwaits towards the discounted returns
    for(i in 1 : nMadeActions[tIdx]){
      real t = tWaits[i]; // time for this decision points 
      real Gt = exp(log(gamma) * (Ts[tIdx] - t)) * (Rs[tIdx] + V0);
      Qwaits[i] = Qwaits[i] + LR * (Gt - Qwaits[i]);
    }
    
    // update V0 towards the discounted returns 
    G0 = exp(log(gamma) * (Ts[tIdx] - (-iti))) * (Rs[tIdx] + V0);
    V0 = V0 + LR * (G0 - V0);
 }
  // calculate total log likelihood
  totalLL =sum(log_lik);
}

