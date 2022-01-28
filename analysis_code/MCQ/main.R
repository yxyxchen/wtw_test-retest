# Set contrasts to sum-to-zero codes. 
options(contrasts = c("contr.sum", "contr.poly")) # contr.sum is necessary for type 3 ANOVA, and type 3 ANOVA is necessary for interaction terms

# Load libraries (make sure everything is previously installed, I would restart R a couple of times during the process)
library(brms)
#library(cmdstanr)
library(posterior)
library(bayesplot)
library(lme4)
library(ggplot2)
library(tidyverse)
library(logistf)

# questionaire constants
Vi = c(54, 55, 19, 31, 14, 47, 15, 25, 78, 40, 11, 67, 34, 27, 69, 49, 80, 24 ,33, 28, 34, 25, 41, 54, 54, 22, 20)
Vd = c(55, 75, 25, 85, 25, 50, 35, 60, 80, 55, 30, 75, 35, 50, 85, 60, 85, 35, 80, 30, 50, 30, 75, 60, 80, 25, 55) 
T = c(117, 61, 53, 7, 19, 160, 13, 14, 162, 62, 7, 119, 186, 21, 91, 89, 157, 29, 14, 179, 30, 80, 20, 111, 30, 136, 7)
R = Vi / Vd #reward ratio
TR = 1 - 1/R # transformed reward ratio

# functions
calc_k_simple_GLM = function(filepath){
  df = read.csv(filepath, row.names = 1)
  nsub = dim(df)[1]
  glm_beta_T = rep(NA, length = nsub)
  glm_beta_TR = rep(NA, length = nsub)
  glm_k = rep(NA, length = nsub)
  
  for(i in 1 : nsub){
    # choices = df[i, 4 : 30] 
    choices = df[i, paste0('DD.', seq(1, 27))]
    if(all(choices == 1) || ! any(choices == 1)){
      print(paste(i, " All later or all sooner"))
    }else{
      # regular GLM methods 
      tmp = data.frame(
        pD = as.numeric((choices = df[i, paste0('DD.', seq(1, 27))] - 1)),
        T = T,
        TR = TR
      ) 
      tryCatch(
        expr = {
          fit = glm(pD ~ TR + T - 1, data = tmp, family = "binomial")
          glm_beta_TR[i] = fit$coefficients[1]
          glm_beta_T[i] = fit$coefficients[2]
          glm_k[i] = fit$coefficients[2] / fit$coefficients[1]
        },warning = function(w){
          print("perfect separation or not enough iterations")
          glm_beta_TR[i] = NA
          glm_beta_T[i] = NA
          glm_k[i] = NA}) 
    }
  }
  return(glm_k)
}

calc_k_robust_GLM = function(filepath){
  df = read.csv(filepath, row.names = 1)
  nsub = dim(df)[1]
  rglm_beta_T = rep(NA, length = nsub)
  rglm_beta_TR = rep(NA, length = nsub)
  rglm_k = rep(NA, length = nsub)
  for(i in 1 : nsub){
    # choices = choices = df[i, 4 : 30]  
    choices = df[i, paste0('DD.', seq(1, 27))]
    if(all(choices == 1) || ! any(choices == 1)){
      print(paste(i, " All later or all sooner"))
    }else{
      # regular GLM methods 
      tmp = data.frame(
        pD = as.numeric((df[i, paste0('DD.', seq(1, 27))] - 1)),
        T = T,
        TR = TR
      ) 
      tryCatch(
        expr = {
          fit = logistf(pD ~ TR + T - 1, data = tmp)
          rglm_beta_TR[i] = fit$coefficients[1]
          rglm_beta_T[i] = fit$coefficients[2]
          rglm_k[i] = fit$coefficients[2] / fit$coefficients[1]
        },warning = function(w){
          print("not enough iterations")
          rglm_beta_TR[i] = NA
          rglm_beta_T[i] = NA
          rglm_k[i] = NA}) 
    }
  }
  return(glm_k)
}


calc_k_lookup_table = function(filepath){
  df = read.csv(filepath, row.names = 1)
  source("MCQ/calc_MCQ.R")
  nsub = dim(df)[1]
  resdf = matrix(NA, ncol = 13, nrow = nsub)
  for(i in 1 : nsub){
    choices = as.numeric(df[i, paste0('DD.', seq(1, 27))])
    savedf = data.frame(matrix(c(i, choices), ncol = 28))
    colnames(savedf) = c("SubjID", paste0("MCQ", seq(1, 27, 1)))
    resdf[i,] = as.numeric(calc_MCA(savedf))
  }
  resdf = as.data.frame(resdf)
  colnames(resdf) = c("SubjID", "SmlSeq", "SmlK", "SmlCons", "SmlICR",  "MedSeq", "MedK", "MedCons", "MedICR", "LrgSeq",
                      "LrgK", "LrgCons", "LrgICR")
  resdf$SubjID = rownames(df)
  resdf['GMK'] = (resdf$SmlK*resdf$MedK*resdf$LrgK) ^ (1/3)
  return(resdf)
}


# read table and run
df = read.csv(file.path("data", "active", "selfreport_sess1.csv"), row.names = 1)


####### main script #########
if (!interactive()) {
  expname = "passive"
  sess = 1
  filepath = file.path("data", expname, sprintf("selfreport_sess%d.csv", sess))
  resdf = calc_k_lookup_table(filepath)
  write_csv(resdf, file.path("..",  "analysis_results", expname, 'selfreport',  'MCQ.csv'))
}

################# simple GLM，sum(is.na(glm_k)) = 117 ########################

###############  robust regressions, 41 NA ###########


############## use the original scoring method ###############


##################
# data.frame(
#   rglm_k,
#   resdf['GMK']
# ) %>% ggplot(aes(log(GMK), log(rglm_k))) + geom_point()


# priors = c(prior(cauchy(0, 1), class = "b", coef = TR), prior(cauchy(0, 1), class = "b", coef = T))
# 
# res = brm(pD ~ TR + T - 1, data = tmp,  family = bernoulli(link = 'logit'), warmup = 50, iter = 1000, chains = 2, backend = "rstan", prior = priors)
# 


