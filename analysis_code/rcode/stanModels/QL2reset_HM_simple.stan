data {
  // experiment parameters
  real stepSec;// duration between two decision points
  real iti;// iti duration
  int  nWaitOrQuit; // number of possible wait-or-quit choices
  real tWaits[nWaitOrQuit]; // time for each decision point 
  
  // initial value for V0
  real V0_ini; 
  
  // empirical data
  int S; 
  int N; // max number of trials
  int N_subj[S]; // number of trials for each participant 
  int block1_N_subj[S]; // number of trials in block1 for each participant
  int R_[S, N]; // payoff in each trial
  real T_[S, N]; // a trial ends at t == T
  int nMadeActions_[S, N];// number of made actions in each trial 
  int nTotalAction;
}
parameters {
  // parameters:
  // alpha : learning rate 
  // nu : valence-dependent bias
  // tau : action consistency, namely the soft-max temperature parameter
  // gamma: discount factor
  // eta: prior belief parameter
  
  // Declare all parameters as vectors for vectorizing
  // Hyper(group)-parameters
  vector[3] mu; # mean of raw parameters
  vector<lower=0>[3] sigma; # std of raw parameters
  
  vector[S] raw_alpha;
  vector[S] raw_nu; // ratio between alphaR and alphaU
  vector[S] raw_tau;
  vector[S] raw_gamma;
  vector[S] raw_eta; 
  
}
transformed parameters{
  // scale raw parameters into real parameters
  vector<lower=0, upper=0.3>[S] alpha;
  vector<lower=0, upper=1>[S] alphaU; // auxiliary variable 
  vector<lower=0, upper=10>[S] nu;
  vector<lower=0, upper=42>[S] tau;
  vector<lower=0.5, upper=1>[S] gamma;
  vector<lower=0, upper=15>[S] eta;
  
  for(sIdx in 1 : S){
    alpha[sIdx] = inv_logit(raw_alpha[sIdx]) * 0.3; // alpha ~ unif(0, 0.3)
    alphaU[sIdx] = min([alpha[sIdx] * inv_logit(raw_nu[sIdx]) * 10, 1]'); // 
    nu[sIdx] = alphaU[sIdx] / alpha[sIdx];
    tau[sIdx] = inv_logit(raw_tau[sIdx]) * 42; // tau ~ unif(0.1, 42)
    gamma[sIdx] = inv_logit(raw_gamma[sIdx])* 0.5 + 0.5; // gamma ~ unif(0.5, 1)
    eta[sIdx] = inv_logit(raw_eta[sIdx]) * 15; // eta ~ unif(0, 15)
  }
}
model {
  // distributions for raw parameters
  mu ~ normal(0, 1);
  sigma ~ normal(0, 1);
  raw_alpha ~ normal(mu[1], sigma[1]);
  raw_nu ~ normal(0, 1.5);
  raw_tau ~ normal(mu[2], sigma[2]);
  raw_gamma ~ normal(0, 1.5);
  raw_eta ~ normal(mu[3], sigma[3]);

 for(sIdx in 1 : S){
    // declare variables 
    vector[nWaitOrQuit] Qwaits; 
    real V0; 
    vector[2] actionValues;
    int action;
    real LR;
    real G0;
   // loop over trials 
   for(tIdx in 1 : N_subj[sIdx]){
     // reset if necessary 
     if(tIdx == 1 || tIdx == block1_N_subj[sIdx]){
        // set initial values
        V0 = V0_ini; 
        // the initial waiting value delines with elapsed time 
        // and the eta parameter determines at which step it falls below V0
        for(i in 1 : nWaitOrQuit){
          Qwaits[i] = - tWaits[i] * 0.1 + eta[sIdx] + V0;
        }
     }
     // make choices 
      for(i in 1 : nMadeActions_[sIdx, tIdx]){
        // the agent wait in every decision point in rewarded trials
        // and wait except for the last decision point in non-rewarded trials
        if(R_[sIdx, tIdx] == 0 && i == nMadeActions_[sIdx, tIdx]){
          action = 2; // quit
        }else{
          action = 1; // wait
        }
        // calculate the likelihood using the soft-max function
        actionValues[1] = Qwaits[i] * tau[sIdx];
        actionValues[2] = V0 * tau[sIdx];
        target += categorical_logit_lpmf(action | actionValues);
      }
      // update value functions
      if(R_[sIdx, tIdx] > 0){
        LR = alpha[sIdx];
      }else{
        LR = alphaU[sIdx];
      }
      // update Qwaits towards the discounted returns
      for(i in 1 : nMadeActions_[sIdx, tIdx]){
        real t = tWaits[i]; // time for this decision points 
        real Gt = exp(log(gamma[sIdx]) * (T_[sIdx, tIdx] - t)) * (R_[sIdx, tIdx] + V0);
        Qwaits[i] = Qwaits[i] + LR * (Gt - Qwaits[i]);
      }
      
      // update V0 towards the discounted returns 
      G0 = exp(log(gamma[sIdx]) * (T_[sIdx, tIdx] - (-iti))) * (R_[sIdx, tIdx] + V0);
      V0 = V0 + LR * (G0 - V0);
   }
 }

}
generated quantities {
  vector[nTotalAction] log_lik = rep_vector(0, nTotalAction); // trial-wise log likelihood 
  real totalLL; // total log likelihood
  int no = 1; // action index
  
  // For group level parameters
  real mu_raw_alpha = mu[1];
  real mu_raw_tau = mu[2];
  real mu_raw_eta = mu[3];
  
  real sigma_raw_alpha = sigma[1];
  real sigma_raw_tau = sigma[2];
  real sigma_raw_eta = sigma[3];
  
 for(sIdx in 1 : S){
    // declare variables 
    vector[nWaitOrQuit] Qwaits; 
    real V0; 
    vector[2] actionValues;
    int action;
    real LR;
    real G0;
   // loop over trials 
   for(tIdx in 1 : N_subj[sIdx]){
     // reset if necessary 
     if(tIdx == 1 || tIdx == block1_N_subj[sIdx]){
        // set initial values
        V0 = V0_ini; 
        // the initial waiting value delines with elapsed time 
        // and the eta parameter determines at which step it falls below V0
        for(i in 1 : nWaitOrQuit){
          Qwaits[i] = - tWaits[i] * 0.1 + eta[sIdx] + V0;
        }
     }
     // make choices 
      for(i in 1 : nMadeActions_[sIdx, tIdx]){
        // the agent wait in every decision point in rewarded trials
        // and wait except for the last decision point in non-rewarded trials
        if(R_[sIdx, tIdx] == 0 && i == nMadeActions_[sIdx, tIdx]){
          action = 2; // quit
        }else{
          action = 1; // wait
        }
        // calculate the likelihood using the soft-max function
        actionValues[1] = Qwaits[i] * tau[sIdx];
        actionValues[2] = V0 * tau[sIdx];
        log_lik[no] =categorical_logit_lpmf(action | actionValues);
        no = no + 1;
      }
      // update value functions
      if(R_[sIdx, tIdx] > 0){
        LR = alpha[sIdx];
      }else{
        LR = alphaU[sIdx];
      }
      // update Qwaits towards the discounted returns
      for(i in 1 : nMadeActions_[sIdx, tIdx]){
        real t = tWaits[i]; // time for this decision points 
        real Gt = exp(log(gamma[sIdx]) * (T_[sIdx, tIdx] - t)) * (R_[sIdx, tIdx] + V0);
        Qwaits[i] = Qwaits[i] + LR * (Gt - Qwaits[i]);
      }
      
      // update V0 towards the discounted returns 
      G0 = exp(log(gamma[sIdx]) * (T_[sIdx, tIdx] - (-iti))) * (R_[sIdx, tIdx] + V0);
      V0 = V0 + LR * (G0 - V0);
   }
 }
  // calculate total log likelihood
  totalLL =sum(log_lik);
}

