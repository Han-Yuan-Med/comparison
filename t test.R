# t test ranking
t_test_ranking<-function(dataset,outcome_name){
  predictor_name<-colnames(dataset)[which(colnames(dataset)!=outcome_name)]
  posi_id<-dataset[which(dataset[,outcome_name]==1),predictor_name]
  nega_id<-dataset[which(dataset[,outcome_name]==0),predictor_name]
  t_test_result<-rep(0,length(predictor_name))
  for (i in 1:length(predictor_name)) {
    t_test_result[i]<-t.test(posi_id[,i],nega_id[,i])$p.value
  }
  rank_t_test<-cbind("test value" = as.numeric(t_test_result),"variable names" = predictor_name)
  rank_t_test<-rank_t_test[order(as.numeric(rank_t_test[,"test value"], decreasing = FALSE)),]
  # Change ranking format to rank_rf
  rank_t_test_1<-as.numeric(rank_t_test[,"test value"])
  names(rank_t_test_1)<-rank_t_test[,"variable names"]
  return(rank_t_test_1)
}


rank_t_test<-t_test_ranking(dataset=new_data_train,outcome_name="label")
