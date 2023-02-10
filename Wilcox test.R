# Wilcoxon's rank sum test
wilcox_ranking<-function(dataset,outcome_name){
  predictor_name<-colnames(dataset)[which(colnames(dataset)!=outcome_name)]
  posi_id<-dataset[which(dataset[,outcome_name]==1),predictor_name]
  nega_id<-dataset[which(dataset[,outcome_name]==0),predictor_name]
  wilcox_result<-rep(0,length(predictor_name))
  for (i in 1:length(predictor_name)) {
    wilcox_result[i]<-wilcox.test(posi_id[,i],nega_id[,i])$p.value
  }
  rank_wilcox<-cbind("test value" = as.numeric(wilcox_result),"variable names" = predictor_name)
  rank_wilcox<-rank_wilcox[order(as.numeric(rank_wilcox[,"test value"], decreasing = FALSE)),]
  # Change ranking format to rank_rf
  rank_wilcox_1<-as.numeric(rank_wilcox[,"test value"])
  names(rank_wilcox_1)<-rank_wilcox[,"variable names"]
  return(rank_wilcox_1)
}


rank_wilcox<-wilcox_ranking(dataset=new_data_train,outcome_name="label")
