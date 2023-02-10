new_data_train<-read.csv("data_train_4w.csv")
new_data_validation<-read.csv("data_validation_4w.csv")
new_data_test<-read.csv("data_test_4w.csv")

# separate training data into positive samples and negative samples for statistical ranking methods
posi_id<-new_data_train[which(new_data_train$label==1),]
nega_id<-new_data_train[which(new_data_train$label==0),]

# random forest variable ranking (baseline)
set.seed(1234)
rank_rf<-AutoScore_rank(new_data_train)

# AutoScore other modules
AutoScore_parsimony_plot(new_data_train,new_data_validation,rank_rf)
auc_parsimony<-AutoScore_parsimony(new_data_train,new_data_validation,rank_rf)

# set 95% of max AUC as variable numbers
num_rf<-length(rank_rf[1:which(auc_parsimony>=max(auc_parsimony)*0.95)[1]])

# finally selected variables by random forest: var_rf
var_rf<-names(rank_rf)[1:num_rf]
auto_weight<-AutoScore_weighting(new_data_train,new_data_train,var_rf)
myvec<-AutoScore_fine_tuning(new_data_train,new_data_train,var_rf,auto_weight)
test_result_rf<-AutoScore_testing(new_data_test,var_rf,auto_weight,myvec)