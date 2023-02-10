likelihood_deletion_ranking<-function(dataset,outcome_name){
  predictor_name<-colnames(dataset)[which(colnames(dataset)!=outcome_name)]
  variable_rank<-rep(0,length(predictor_name))
  variable_left<-predictor_name
  for (i in 1:(length(predictor_name)-1)) {
    loglik_tmp <- rep(0,(length(predictor_name)-i+1))
    for (j in 1:(length(predictor_name)-i+1)) {
      if (i == 1){
        predictor<-predictor_name[-j]
        dataset_tmp<-cbind(dataset[,predictor],label = dataset[,outcome_name])
        model_tmp <- glm(label ~ ., data = dataset_tmp, family = binomial)
        loglik_tmp[j] <- logLik(model_tmp)}
      else {
        predictor<-variable_left[-j]
        # predictor<-c(variable_rank[1:i-1],variable_left[-j])
        dataset_tmp<-data.frame(cbind(dataset[,predictor],label = dataset[,outcome_name]))
        model_tmp <- glm(label ~ ., data = dataset_tmp, family = binomial)
        loglik_tmp[j] <- logLik(model_tmp)
      }
    }
    variable_rank[i]<-variable_left[which.min(loglik_tmp)]
    variable_left<-variable_left[which(variable_left!=variable_rank[i])]
  }
  variable_rank_1<-c(length(predictor_name):1)
  names(variable_rank_1)<-variable_rank
  return(variable_rank_1)
}

rank_likelihood_deletion<-likelihood_deletion_ranking(dataset =new_data_train,outcome_name<-"label")
