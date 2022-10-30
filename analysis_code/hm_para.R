

library('plyr'); 
library('rstan');library("loo");library("coda") 
library(tidyverse)
library(ggplot2)

expname = "passive"
sess = 2
stepsize = 0.5
chainIdxs = seq(1, 4)
S = 50
modelname = "QL2reset_HM_newnew"
fitMethod = "whole"


chainIdx = 2

load(sprintf("../../analysis_results/%s/modelfit_hm/%s/stepsize%.2f/%s/chain%d/sess%d_fit.RData", expname, fitMethod, stepsize,  modelname, chainIdx, sess))

var = 'eta'
#### raw estimate #####
mu_raw_ = fit %>% rstan::extract(sprintf('mu_raw_%s', var), permuted = F, inc_warmup = FALSE, include = T) %>% adply(2, function(x) x) %>% dplyr::select(-chains) %>% as.numeric()
sigma_raw_ = fit %>% rstan::extract(sprintf('sigma_raw_%s', var), permuted = F, inc_warmup = FALSE, include = T) %>% adply(2, function(x) x) %>% dplyr::select(-chains) %>% as.numeric()

mean(mu_raw_)
median(mu_raw_)
mean(sigma_raw_)
median(sigma_raw_)

raw_ = matrix(unlist(fit %>% rstan::extract(sprintf("raw_%s[%d]", var, 1:50), permuted = F, inc_warmup = FALSE, include = TRUE)  %>% adply(2, function(x) x) %>% dplyr::select(-chains)), 1000, 50)
raw_pe_ = apply(raw_, FUN = mean, MARGIN = 2)


plotdf = data.frame(
	"val" = raw_pe_
	)
plotdf %>% ggplot(aes(val)) + geom_histogram()  
x = seq(-20, 2, by = 0.001)
normdf = data.frame(
	"x" = x,
	"y" = dnorm(x, mean(mu_raw_), mean(sigma_raw_)) 
	)
plotdf %>% ggplot(aes(val)) + geom_density() +
geom_line(data = normdf, aes(x , y))


###### not raw estimates 
#### raw estimate #####
var = 'nu'
raw_ = matrix(unlist(fit %>% rstan::extract(sprintf("%s[%d]", var, 1:50), permuted = F, inc_warmup = FALSE, include = TRUE)  %>% adply(2, function(x) x) %>% dplyr::select(-chains)), 1000, 50)
raw_pe_ = apply(raw_, FUN = mean, MARGIN = 2)


plotdf = data.frame(
	"val" = raw_pe_
	)
plotdf %>% ggplot(aes(val)) + geom_histogram()



