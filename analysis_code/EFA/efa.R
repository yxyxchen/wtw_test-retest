library("psych")
library("tidyverse")

data = read.csv("measures.csv")
# "Attentional", "Nonplanning", "Motor"
X = data[c('auc', "std_wtw", "auc_delta", "discount_logk","NU", "PU", "PM", "PS", "SS", "attention", "cogstable", "motor", "perseverance", "selfcontrol", "cogcomplex")]

cor_mat<-cor(X, X, method = 'spearman')
diag(cor_mat)<-0
cor_mat = abs(cor_mat)
graph<-graph.adjacency(cor_mat,weighted=TRUE,mode="lower")
plot(graph)

cortest.bartlett(X)
KMO(r=cor(X))

library(ggplot2)
fafitfree <- fa(X,nfactors = ncol(X), rotate = "none")
n_factors <- length(fafitfree$e.values)
scree     <- data.frame(
  Factor_n =  as.factor(1:n_factors), 
  Eigenvalue = fafitfree$e.values)
ggplot(scree, aes(x = Factor_n, y = Eigenvalue, group = 1)) + 
  geom_point() + geom_line() +
  xlab("Number of factors") +
  ylab("Initial eigenvalue") +
  labs( title = "Scree Plot", 
        subtitle = "(Based on the unreduced correlation matrix)")

parallel <- fa.parallel(X)
fa.none <- fa(r=X, 
              nfactors = 4, 
              fm = "pa", # type of factor analysis we want to use (“pa” is principal axis factoring)
              max.iter=1000, # (50 is the default, but we have changed it to 100,
              rotate="promax") # none rotation
print(fa.none)


