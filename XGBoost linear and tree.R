library(xgboost)
XGBoost_tree_ranking<-function(dataset,outcome_name){
  predictor_name<-colnames(dataset)[which(colnames(dataset)!=outcome_name)]

  xgb_data<-as.matrix(dataset[,predictor_name])
  bst_gbtree <- xgboost(data = xgb_data, label = dataset[,outcome_name], max_depth = 10,
                        eta = 0.3, nthread = 1, nrounds = 500, objective = "binary:logistic")
  bst_gbtree_importance<-xgb.importance(model = bst_gbtree)
  rank_gbtree<-bst_gbtree_importance$Gain
  names(rank_gbtree)<-bst_gbtree_importance$Feature
  return(rank_gbtree)
}

XGBoost_linear_ranking<-function(dataset,outcome_name){
  predictor_name<-colnames(dataset)[which(colnames(dataset)!=outcome_name)]
  xgb_data<-as.matrix(dataset[,predictor_name])
  bst_gblinear <- xgboost(data = xgb_data, label = dataset[,outcome_name], max_depth = 10, booster = "gblinear",
                          eta = 0.3, nthread = 1, nrounds = 500, objective = "binary:logistic")
  bst_gblinear_importance<-xgb.importance(model = bst_gblinear)
  rank_gblinear<-bst_gblinear_importance$Weight
  names(rank_gblinear)<-bst_gblinear_importance$Feature
  return(rank_gblinear)
}

rank_xgb_tree<-XGBoost_tree_ranking(dataset=new_data_train,outcome_name="label")
rank_xgb_linear<-XGBoost_linear_ranking(dataset=new_data_train,outcome_name="label")
