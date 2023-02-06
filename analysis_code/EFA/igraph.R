library("tidyverse")
library(latex2exp)
library(ggplot2)
library(igraph)
source("plotThemes.R")
data = read.csv("measures.csv")
vars = c('auc', "std_wtw", "auc_delta", "discount_logk","NU", "PU", "PM", "PS", "SS", "Attentional", "Nonplanning", "Motor")
X = data[vars]
types = c(rep("task", 3), rep("selfreport", 3+5+1))
colors = c(rep("#dd1c77",3), rep("#addd8e", 3+5+1))
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

ggsave("../../figures/msd_task.eps", width = 4, height = 4) 


############ the modeling version ###############
data = read.csv("all_measures.csv")
vars = c('auc', "std_wtw", "auc_delta", "discount_logk","NU", "PU", "PM", "PS", "SS", "Attentional", "Nonplanning", "Motor", "alpha", "alphaU", "tau", "eta")
X = data[vars]
# X = X[is.nan(X$"discount_logk")]
types = c(rep("task", 3), rep("selfreport", 3+5+1), rep("parameter", 4))
colors = c(rep("#dd1c77",3), rep("#addd8e", 3+5+1), rep("#ffeda0", 4))
labels = c("AUC", TeX("$\\sigma_{wtw}$"), TeX("$\\Delta$AUC"), TeX("$log(k)$"), "NU", "PU", "PM", "PS", "SS", "Attentional", "Nonplanning", "Motor",
           TeX("$log(\\alpha)$"),  TeX("$log(\\phi)$"),  TeX("$log(\\tau)$"),  TeX("$log(\\eta)$"))

cor_mat<-abs(cor(X, X, method = 'pearson'))
diag(cor_mat)<-0
graph<-graph.adjacency(cor_mat,weighted=TRUE,mode="lower")

plot(graph, vertex.color = colors, 
     vertex.label = labels,
     edge.width = E(graph)$weight) # how to change the base width unite?

length(V(graph))









###########################
############# selfreport and with discounting
data = read.csv("all_measures.csv")
vars = c("discount_logk", "NU", "PU", "PM", "PS", "SS", "Attentional", "Nonplanning", "Motor", "alpha", "alphaU", "tau", "eta")
X = data[vars]
# X = X[is.nan(X$"discount_logk")]
types = c(rep("selfreport", 3+5+1), rep("parameter", 4))
colors = c(rep("#addd8e", 3+5+1), rep("#ffeda0", 4))
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


