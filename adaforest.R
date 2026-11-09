### Electronic Supplement to 'AdaForest: Adaptive random forests for sparse
### high-dimensional biomedical data'

## The following contains the main R function 'adaforest()' to fit the AdaForest.
## The function internally calls 'ranger()' from R add-on package "ranger", 
## which needs to be installed. See also its documentation for further details.

## Input: 
# formula                       description of the model to fit
# data                          training data
# mtry0                         number of candidate variables    
# ms                            minimal node size
# B                             number of trees per iteration
# L                             learning rate
# eps                           stopping tolerance
# alpha_start                   (optional) initial unnormalized starting weights
# classification                set to TRUE to grow a classification forest
# probability                   set to TRUE to grow a probability forest
# class.weights                 (optional) class weights for the outcome classes
# dependent.variable.name       (optional) name of the dependent variable
# ...                           further arguments passed to ranger()

## Output: 
# r                             vector of sampling weights


adaforest  <- function(formula,
                       data,
                       mtry0,
                       ms,
                       B=1,
                       L=1,
                       eps=10e-5,
                       alpha_start=NULL,
                       classification=FALSE,
                       probability=FALSE,
                       class.weights=NULL,
                       dependent.variable.name=NULL,
                       ...){

  require("ranger")

  update_jk <- function(splitted, j){
    sumjk   <- sum(sapply(1:B, function(b) sum(splitted[[b]] %in% j)))
    return(sumjk)
  }
  if(is.null(formula)){
    if(is.null(dependent.variable.name)){
      stop("Either formula or dependent.variable.name must be specified.")
    }
    x <- setdiff(colnames(data), dependent.variable.name)
  } else {
    x <- all.vars(formula[[3]])
  }
  p <- length(x)

  if(is.null(alpha_start)){
    alpha <- rep(1, p)
  } else{
    alpha <- alpha_start
  }
  r     <- alpha / sum(alpha)
  d     <- 1

  while(d > eps){
    mod <- ranger(
      formula=formula,
      dependent.variable.name=dependent.variable.name,
      data=data, 
      mtry=mtry0, 
      min.node.size=ms, 
      num.trees=B, 
      split.select.weights=r, 
      write.forest=TRUE, 
      classification=classification,
      probability=probability,
      class.weights=class.weights,
      ...)
    
    # list with trees; each entry is list  with (left or right) child node ID; "0" means no child --> terminal node
    terminals <- lapply(1:B, function(b) which(mod$forest$child.nodeIDs[[b]][[1]] == 0)) 
    
    # list with trees; each entry is list of variable index used for split (+1 since Cpp is zero-indexed)
    splitted  <- lapply(1:B, function(b) mod$forest$split.varIDs[[b]][-terminals[[b]]] + 1)
    
    # vector with values how often each variable has been used for splitting
    all_j   <- sapply(1:p, function(j) update_jk(splitted, j))
    
    # update alpha & r; check for convergence
    alpha   <- alpha + L * all_j
    rOld    <- r
    r       <- alpha/sum(alpha)
    d       <- max(abs(r - rOld))
  }
  return(r)
}

