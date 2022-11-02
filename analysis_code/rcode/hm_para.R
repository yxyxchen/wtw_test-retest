library('plyr'); 
library('rstan');library("loo");library("coda") 
library(tidyverse)
library(ggplot2)
source("subFxs/helpFxs.R")

expname = "passive"
sess = 1
stepsize = 0.5
chainIdxs = seq(1, 4)
S = 50
modelname = "QL2reset_HM_short"
fitMethod = "whole"
paraNames = getParaNames(modelname)
npara = length(paraNames)

fit_ = list()
for(chainIdx in chainIdxs){
  load(sprintf("../../analysis_results/%s/modelfit_hm/%s/stepsize%.2f/%s/chain%d/sess%d_fit.RData", expname, fitMethod, stepsize,  modelname, chainIdx, sess))
  fit_[[chainIdx]] = fit
}
fit = sflist2stanfit(fit_)

outputFile = sprintf("../../analysis_results/%s/modelfit_hm/%s/stepsize%.2f/%s/combined/sess%d", expname, fitMethod, stepsize,  modelname, sess)
dir.create(sprintf("../../analysis_results/%s/modelfit_hm/%s/stepsize%.2f/%s/combined/", expname, fitMethod, stepsize,  modelname))


samples = read_csv(sprintf("%s_para_sample.txt", outputFile))

samples = fit %>% rstan::extract(permuted = F, pars = c(paste0("mu_raw_", paraNames), paste0("sigma_raw_", paraNames), paraNames, "totalLL")) 
samples = rbind(samples[,1,], samples[,2,], samples[,3,], samples[,4,])
write.table(samples, file = sprintf("%s_para_sample.txt", outputFile), 
            sep = ",", row.names=FALSE)

sampler_params <- get_sampler_params(fit, inc_warmup=FALSE)
divergent <- do.call(rbind, sampler_params)[,'divergent__']
nDt = sum(divergent)
fitSummary <- summary(fit, pars = c(paste0("mu_raw_", paraNames), paste0("sigma_raw_", paraNames), paraNames, "totalLL"), use_cache = F)$summary
fitSummary = cbind(fitSummary, nDt = rep(nDt, nrow(fitSummary)))
write.table(fitSummary, file = sprintf("%s_para_summary.txt", outputFile), 
            sep = ",")

#### raw estimate #####
mu_raw_ = fit %>% rstan::extract(sprintf('mu_raw_%s', paraNames), permuted = F, inc_warmup = FALSE, include = T) %>% adply(2, function(x) x) %>% dplyr::select(-chains) 
sigma_raw_ = fit %>% rstan::extract(sprintf('sigma_raw_%s', paraNames), permuted = F, inc_warmup = FALSE, include = T) %>% adply(2, function(x) x) %>% dplyr::select(-chains) 

mu_raw_means = apply(mu_raw_, FUN= mean, MARGIN = 2)
mu_raw_medians= apply(mu_raw_, FUN= median, MARGIN = 2)
sigma_raw_means = apply(sigma_raw_, FUN= mean, MARGIN = 2)
sigma_raw_medians = apply(sigma_raw_, FUN= median, MARGIN = 2)

indv_paraname_ = vector(mode = "list", length = npara)
paraname_ = vector(mode = "list", length = npara)
for(i in 1 : npara){
  para = paraNames[i]
  indv_paraname_[[i]] = sprintf("raw_%s[%d]", para, 1:S)
  paraname_[[i]] = rep(para, S)
}
indv_paranames = unlist(indv_paraname_)

raw_ = fit %>% rstan::extract(indv_paranames, permuted = F, inc_warmup = FALSE, include = TRUE)  %>% adply(2, function(x) x) %>% dplyr::select(-chains)
raw_pe_ = apply(raw_, FUN = mean, MARGIN = 2)


plotdf = data.frame(
  "val" = raw_pe_,
  "para" = unlist(paraname_),
  "paralabel" = factor(paste0("raw_", unlist(paraname_)), levels = paste0("raw_", paraNames))
)

plotdf %>% ggplot(aes(val)) + geom_histogram() + facet_grid(~paralabel) + theme_bw()
x = seq(-8, 5, by = 0.1)
x_ = vector(mode = "list", length = npara)
y_ = vector(mode = "list", length = npara)
paraname_ = vector(mode = "list", length = npara)
for(i in 1 : npara){
  x_[[i]] = x
  y_[[i]] = dnorm(x, mu_raw_means[i], sigma_raw_means[i]) 
  paraname_[[i]] = rep(paraNames[i], length(x))
}
normdf = data.frame(
  "x" = unlist(x_),
  "y" = unlist(y_),
  "para" = unlist(paraname_),
  "paralabel" = factor(paste0("raw_", unlist(paraname_)),
                       levels = paste0("raw_", paraNames))
)
plotdf %>% ggplot(aes(val)) + geom_histogram() + facet_grid(~paralabel) + theme_bw() 
plotdf %>% ggplot(aes(val)) + geom_density() + facet_grid(~paralabel) + theme_bw() +
  geom_line(data = normdf, aes(x , y), color = "red")



###### not raw estimates 
indv_paraname_ = vector(mode = "list", length = npara)
paraname_ = vector(mode = "list", length = npara)
for(i in 1 : npara){
  para = paraNames[i]
  indv_paraname_[[i]] = sprintf("%s[%d]", para, 1:S)
  paraname_[[i]] = rep(para, S)
}
indv_paranames = unlist(indv_paraname_)

para_ = fit %>% rstan::extract(indv_paranames, permuted = F, inc_warmup = FALSE, include = TRUE)  %>% adply(2, function(x) x) %>% dplyr::select(-chains)
para_pe_ = apply(para_, FUN = mean, MARGIN = 2)


plotdf = data.frame(
  "val" = para_pe_,
  "para" = factor(unlist(paraname_), levels = paraNames)
)

plotdf %>% ggplot(aes(val)) + geom_histogram(bins = 10) + facet_grid(~para, scales = "free_x") + theme_bw()
