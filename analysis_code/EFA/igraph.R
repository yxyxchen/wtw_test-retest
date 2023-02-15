# let me formalize these analyses

library("tidyverse")
library(latex2exp)
library(ggplot2)
library(igraph)
library("ggpubr")
source("plotThemes.R")
data = read.csv("measures.csv")


self_vars =  c("discount_logk","NU", "PU", "PM", "PS", "Attentional", "Nonplanning", "Motor")
task_vars = c('auc', "std_wtw", "auc_delta")
paranames = c("alpha", "alphaU", "eta", "tau")
n_selfv = length(self_vars)
n_taskv = length(task_vars)
n_para = length(paranames)


vars = c(task_vars, self_vars)
X = data[vars]
X = scale(X)
types = c(rep("task", n_taskv), rep("selfreport", n_selfv))
colors = c(rep("#dd1c77", n_taskv), rep("#addd8e", n_selfv))
labels = c("AUC", TeX("$\\sigma_{wtw}$"), TeX("$\\Delta$AUC"), TeX("$log(k)$"), "NU", "PU", "PM", "PS", "SS", "Attentional", "Nonplanning", "Motor")
rho_df = read.csv("all_reliability.csv")
rhos = rep(0, length(vars))
for(i in 1 : length(vars)){
  var = vars[i]
  rhos[i] = rho_df[rho_df$"var" == var, "rho"]
  
}
  
cor_mat<-abs(cor(X, X, method = 'spearman'))
diag(cor_mat)<-0
graph<-graph.adjacency(cor_mat,weighted=TRUE,mode="lower")
plot(graph, vertex.color = colors, 
     vertex.size = rhos * 10,
     vertex.label = labels,
     edge.width = E(graph)$weight) # how to change the base width unite?



# this is not a right way to visualize it. 
# I mean it changes everytime? 

mds <- t(X) %>%
  dist() %>% cmdscale() %>%
  as_tibble()
colnames(mds) <- c("Dim.1", "Dim.2")
mds$type = types
# Plot MDS

p = ggscatter(mds, x = "Dim.1", y = "Dim.2", 
          label = row.names(t(X)),
          size = rhos * 5,
          color = "type",
          palette = c("#2ca25f", "#dd1c77"),
          shape = 16,
          repel = TRUE) + myTheme 

ggsave("../../figures/msd_task_1.eps", width = 4, height = 4) 


############ the modeling version ###############












###########################
############# selfreport and with discounting
self_vars =  c("discount_logk","NU", "PU", "PM", "PS", "Attentional", "Nonplanning", "Motor")
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
types = c(rep("selfreport", n_selfv), rep("parameter", n_para))
colors = c(rep("#addd8e", n_selfv), rep("#ffeda0", n_para))
labels = c(TeX("$log(k)$"), "NU", "PU", "PM", "PS", "SS", "Attentional", "Nonplanning", "Motor",
           TeX("$log(\\alpha)$"),  TeX("$log(\\phi)$"),  TeX("$log(\\tau)$"),  TeX("$log(\\eta)$"))

cor_mat<-abs(cor(X, X, method = 'pearson'))
diag(cor_mat)<-0
graph<-graph.adjacency(cor_mat,weighted=TRUE,mode="lower")

plot(graph, vertex.color = colors, 
     vertex.label = labels,
     edge.width = E(graph)$weight) # how to change the base width unite?


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
ggscatter(mds, x = "Dim.1", y = "Dim.2", 
          label = row.names(t(X)),
          size = rhos * 5,
          color = "type",
          palette = c("#feb24c", "#2ca25f"),
          shape = 16,
          repel = TRUE) + myTheme
ggsave("../../figures/msd_para.eps", width = 4, height = 4) 


