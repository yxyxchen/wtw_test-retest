

library("tidyverse")
library(latex2exp)
library(ggplot2)
library(igraph)
library("ggpubr")
source("plotThemes.R")

################ MSD for selfreport measures and task measures ###########
data = read.csv("all_measures.csv")
self_vars =  c("discount_logk","NU", "PU", "PM", "PS", "SS", "attention", "cogstable", "motor", "perseverance", "selfcontrol", "cogcomplex")
task_vars = c('auc', "std_wtw", "auc_delta")
paranames = c("alpha", "alphaU", "eta", "tau")
n_selfv = length(self_vars)
n_taskv = length(task_vars)
n_para = length(paranames)


vars = c(task_vars, self_vars)
X = data[vars]
X = scale(X)
types = c(rep("Task", n_taskv), rep("Self-report", n_selfv))
colors = c(rep("#dd1c77", n_taskv), rep("#addd8e", n_selfv))
labels = c("AUC", TeX("$\\sigma_{wtw}$"), TeX("$\\Delta$AUC"), TeX("$log(k)$"), "NU", "PU", "PM", "PS", "SS", "At", "Ci", "Mt", "Pe", "Sc", "Cc")
labels = c("AUC", "sigma", "delta", "k", "NU", "PU", "PM", "PS", "SS", "At", "Ci", "Mt", "Pe", "Sc", "Cc")

rho_df = read.csv("all_reliability.csv")
rhos = rep(0, length(vars))
for(i in 1 : length(vars)){
  var = vars[i]
  rhos[i] = rho_df[rho_df$"var" == var, "rho"]
  
}

mds <- t(X) %>%
  dist() %>% cmdscale() %>%
  as_tibble()
colnames(mds) <- c("Dim.1", "Dim.2")
mds$type = types
# Plot MDS

p = ggscatter(mds, x = "Dim.1", y = "Dim.2", 
          label = labels,
          parse = TRUE,
          size = rhos * 5,
          color = "type",
          palette = c("#2ca25f", "#dd1c77"),
          shape = 16,
          repel = TRUE) + myTheme 

ggsave("../../figures/msd_task.eps", width = 4, height = 4) 


############# MDS for selfreport measures and model parameters ##################
self_vars =  c("discount_logk","NU", "PU", "PM", "PS", "SS", "attention", "cogstable", "motor", "perseverance", "selfcontrol", "cogcomplex")
task_vars = c('auc', "std_wtw", "auc_delta")
paranames = c("alpha", "alphaU", "eta", "tau")
n_selfv = length(self_vars)
n_taskv = length(task_vars)
n_para = length(paranames)
data = read.csv("all_measures.csv")
vars = c(self_vars, paranames)
X = data[vars]
X = scale(X)
# X = X[is.nan(X$"discount_logk")]
types = c(rep("Self-report", n_selfv), rep("RL parameter", n_para))
colors = c(rep("#addd8e", n_selfv), rep("#ffeda0", n_para))
labels = c("k", "NU", "PU", "PM", "PS", "SS", "At", "Ci", "Mt", "Pe", "Sc", "Cc", "alpha", "alphaU", "eta", "tau")

rho_df = read.csv("all_reliability.csv")
rhos = rep(0, length(vars))
for(i in 1 : length(vars)){
  var = vars[i]
  rhos[i] = rho_df[rho_df$"var" == var, "rho"]
  
}

mds <- t(X) %>%
  dist() %>% cmdscale() %>%
  as_tibble()
colnames(mds) <- c("Dim.1", "Dim.2")
mds$type = types
mds$Dim.1 = -mds$Dim.1
mds$Dim.2 = -mds$Dim.2
# Plot MDS
ggscatter(mds, x = "Dim.1", y = "Dim.2", 
          label = labels,
          size = rhos * 5,
          color = "type",
          palette = c("#feb24c", "#2ca25f"),
          shape = 16,
          repel = TRUE) + myTheme
ggsave("../../figures/msd_para.eps", width = 4, height = 4) 


