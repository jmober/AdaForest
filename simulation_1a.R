### Electronic Supplement to 'AdaForest: Adaptive random forests for sparse
### high-dimensional biomedical data'

## The following contains the R code to reproduce the simulation study (scenario 1a)
## presented in Section 3 of the paper.

## The function first applies adaforest() and then fits 
# i) a simple tree
# ii) the classical random forest 
# iii) the AdaForest with all available variables 
# iv) the AdaForest including the variables with the highest final weights, only. 
#     The optimal number is determined by the MSPE on the validation sample. 

## The simulations were run on a linux system: 
## Hardware: Dell Inc. Precision 5820 Tower; Processor: Intel Xeon(R)W-2155 CPU@3.30GHz x 20

## The resuls obtained by parallel computing using the R package "batchtools" 
## is stored in 'simulation_1a.rda'. 

# library("batchtools")
# r <- makeRegistry(file.dir="./simulation_1a")
# r <- loadRegistry(file.dir="./simulation_1a", writeable=T)

## A single simulation run can be performed by calling oneSim(). 

## Input: 
# seed          random seed
# n             sample size 
# nvar          number of variables 
# ninf          number of informative variables
# sigma2        error variance
# B             number of trees per iteration
# L             learning rate
# eps           stopping tolerance

## Ouput: 
## A list containing 
# pred          vector of mean squared prediction errors from test set
# r_adaptD_lv   vector of sampling weights obtained from learning and validation set
# r_adaptD_l    vector of sampling weights obtained from learning set

oneSim <- function(seed, n, nvar, ninf, sigma2, B, L, eps){
  
  ms   <- 0.05*n

  simDat <- function(n, nvar, ninf, sigma2){
    
    X <- matrix(rnorm(n*nvar, 0, 1), nrow = n, ncol = nvar)
    sigg <- toeplitz(0.4^(0:(nvar-1)))
    L    <- chol(sigg)
    X    <- X %*% L
    colnames(X) <- paste0("x",1:nvar)
    betas1 <- runif(ninf,-0.5,0.5)
    eta <- 0 
    for(i in 1:ninf){
      eta <- eta+betas1[i]*X[,i]
    }
    
    eta <- c(scale(eta))
    sigma <- sqrt(sigma2)
    y <- eta + rnorm(n, 0, sigma)
    
    return(as.data.frame(cbind(X,y)))
  }
  
  set.seed(seed+022025)
  data    <- simDat(n*2, nvar, ninf, sigma2)
  data_lv <- data[1:n,]           # n
  data_t  <- data[(n+1):(n*2),]   # n 
  data_l  <- data_lv[1:(0.8*n),]  # 0.8*n
  data_v  <- data_lv[(0.8*n+1):n,]# 0.2*n
  
  form  <- paste(names(data_l)[1:nvar], collapse="+")
  form  <- formula(paste("y~",form))
  
  source("adaforest.R")
  require("ranger")
  
  mtry0       <- sqrt(nvar)
  r_adaptD_lv <- adaforest(form, data_lv, mtry0, ms, B, L, eps)
  r_adaptD_l  <- adaforest(form, data_l, mtry0, ms, B, L, eps)
  
  pred <- numeric(4)
  
  # simple tree
  mod1    <- ranger(form, data=data_lv, mtry=nvar, replace=FALSE, sample.fraction=1, min.node.size=ms, num.trees=1)
  pred1   <- predict(mod1, data=data_t)$prediction
  pred[1] <- mean((data_t$y-pred1)^2)
  
  # classical random forest 
  mod2    <- ranger(form, data=data_lv, mtry=mtry0, min.node.size=ms, num.trees=500, importance="permutation")
  pred2   <- predict(mod2, data=data_t)$prediction
  pred[2] <- mean((data_t$y-pred2)^2)
  
  # the AdaForest with all available variables
  mod3    <- ranger(form, data=data_lv, mtry=mtry0, min.node.size=ms, num.trees=500, split.select.weights=r_adaptD_lv)
  pred3   <- predict(mod3, data=data_t)$prediction
  pred[3] <- mean((data_t$y-pred3)^2)
  
  # the AdaForest including the variables with the highest final weights
  seq_max <- floor(log(nvar)/log(2))
  thres   <- 2^(seq_max:1)
  pred_ll <- numeric(length(thres))
  for(th in 1:length(thres)){
    vars_freqD  <- order(r_adaptD_l, decreasing=T)[1:thres[th]]
    form4  <- paste(names(data_l)[vars_freqD], collapse="+")
    form4  <- formula(paste("y~",form4))
    mod4   <- ranger(form4, data=data_l, mtry=sqrt(length(vars_freqD)), min.node.size=ms, num.trees=500, split.select.weights=r_adaptD_l[vars_freqD])
    pred4  <- predict(mod4, data=data_v)$prediction
    pred_ll[th] <- mean((data_v$y-pred4)^2)
  }
  min_thres <- which.min(pred_ll)
  vars_freqD  <- order(r_adaptD_l, decreasing=T)[1:thres[min_thres]]
  form4  <- paste(names(data_l)[vars_freqD], collapse="+")
  form4  <- formula(paste("y~",form4))
  mod4    <- ranger(form4, data=data_lv, mtry=sqrt(length(vars_freqD)), min.node.size=ms, num.trees=500, split.select.weights=r_adaptD_l[vars_freqD])
  pred4   <- predict(mod4, data=data_t)$prediction
  pred[4] <- mean((data_t$y-pred4)^2)
  
  return(list(pred, r_adaptD_lv, r_adaptD_l))
}
  
## The simulation was run with the following arguments.
args <- expand.grid(seed=c(1:100), 
                    n=c(100,500,1000),
                    nvar=c(25,100,500),
                    ninf=c(3,8,25), 
                    sigma2=c(0.5,1,2)) 

# JobIDs <- batchMap(oneSim, args=args, more.args=list(B=1, L=1, eps=1e-5), reg=r)
# JobIDs$chunk <- chunk(JobIDs$job.id, n.chunks=10)
# submitJobs(ids=JobIDs, reg=r)

## One exemplary call yields: 
source("adaforest.R")
oneSim(seed=1, n=100, nvar=25, ninf=3, sigma2=1, B=1, L=1, eps=1e-5)

## The full list of results is loaded by 
load("simulation_1a.rda")

library(ggplot2)
library(dplyr)

## prepare results for plots 
preds <- c(sapply(1:8100, function(j) res[[j]][[1]][-1]))
methods <- c("RandomForest","AdaForest","AdaForest (s)")
for_plot <- data.frame(y=preds, method=factor(rep(methods, 8100)), n=rep(args$n, each=3), nvar=rep(args$nvar, each=3),
                       ninf=rep(args$ninf, each=3), sigma=rep(args$sigma, each=3))
for_plot$method <- factor(for_plot$method, levels=c("RandomForest","AdaForest","AdaForest (s)"))

## Figure 2 
for_plot_s <- for_plot[for_plot$sigma==1 & for_plot$n==500 & for_plot$ninf %in% c(3,8,25) & for_plot$nvar %in% c(25,100,500),]

nvarl <- c("25" = "low-dimensional", "100" = "moderate-dimensional", "500"= "high-dimensional")
ninfl <- c("3" = "sparse", "8" = "medium-sparse", "25" = "non-sparse")

for_plot_s %>% mutate(method=factor(method, levels=c("RandomForest","AdaForest","AdaForest (s)"))) %>%
  ggplot(aes(x=method,y=y, fill=method))+
  geom_boxplot()+
  facet_grid(nvar~ninf, labeller=labeller(nvar=nvarl, ninf=ninfl))+
  ylab("MSPE")+
  xlab("")+
  scale_fill_manual(values=c(grey(1),grey(0.9),grey(0.7),grey(0.7),grey(0.7),grey(0.7),grey(0.7),grey(0.7)))+
  scale_x_discrete(labels = c("RandomForest","AdaForest",expression(AdaForest~(p[s]))))+
  theme_bw()+
  theme(axis.text.x = element_text(size=14, angle=45, vjust=1, hjust=1),
        axis.text.y = element_text(size=12),
        axis.title.x = element_text(size=18),
        axis.title.y = element_text(size=16),
        strip.text.x = element_text(size = 14),
        strip.text.y = element_text(size = 14),
        plot.title = element_text(size=20, vjust=2, hjust=0.5),
        legend.position="none")



## Figure 1 
weights <- t(sapply(2801:2900, function(j) res[[j]][[2]]))
for_plot <- data.frame("w"=c(weights),"v"=factor(rep(paste0("x",1:25), each=100), levels = paste0("x",1:25)),"inf"=factor(rep(c(0,0,0,rep(1,22)), each=100)))

ggplot(for_plot, aes(y=w, x=v))+
  geom_boxplot(aes(fill=inf), show.legend=F)+
  scale_fill_manual(values=c("darkgrey","lightgrey"))+
  theme_bw()+
  geom_hline(yintercept=1/25, lty="dashed")+
  xlab("Covariate")+
  ylab(bquote(hat(r)))+
  theme(strip.text.y = element_text(size=18),
        strip.text.x = element_text(size= 18),
        axis.title.y = element_text(angle = 0, vjust=0.5, size=18),
        axis.title.x = element_text(size= 18),
        axis.text = element_text(size=15))

weights <- t(sapply(3701:3800, function(j) res[[j]][[2]]))
for_plot <- data.frame("w"=c(weights),"v"=factor(rep(paste0("x",1:25), each=100), levels = paste0("x",1:25)),"inf"=factor(rep(c(rep(0,8),rep(1,17)), each=100)))

ggplot(for_plot, aes(y=w, x=v))+
  geom_boxplot(aes(fill=inf), show.legend=F)+
  scale_fill_manual(values=c("darkgrey","lightgrey"))+
  theme_bw()+
  geom_hline(yintercept=1/25, lty="dashed")+
  xlab("Covariate")+
  ylab(bquote(hat(r)))+
  theme(strip.text.y = element_text(size=18),
        strip.text.x = element_text(size= 18),
        axis.title.y = element_text(angle = 0, vjust=0.5, size=18),
        axis.title.x = element_text(size= 18),
        axis.text = element_text(size=15))








