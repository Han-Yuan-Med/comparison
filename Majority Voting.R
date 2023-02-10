# majority voting is based on 95% cutoff on the parsimony plot
# for each variable ranking, 95% cutoff might return different numbers of variables
 
# variable selection by random forest
auc_parsimony_rf<-AutoScore_parsimony(new_data_train,new_data_validation,rank_rf)
# set 95% of max AUC
num_rf<-length(rank_rf[1:which(auc_parsimony_rf>=max(auc_parsimony_rf)*0.95)[1]])
var_rf<-names(rank_rf)[1:num_rf]

# variable selection by t test
auc_parsimony_t_test<-AutoScore_parsimony(new_data_train,new_data_validation,rank_t_test)
# set 95% of max AUC
num_t_test<-length(rank_t_test[1:which(auc_parsimony_t_test>=max(auc_parsimony_t_test)*0.95)[1]])
var_t_test<-names(rank_t_test)[1:num_t_test]

# variable selection by Wilcoxon's rank sum test
auc_parsimony_wilcox<-AutoScore_parsimony(new_data_train,new_data_validation,rank_wilcox)
# set 95% of max AUC
num_wilcox<-length(rank_wilcox[1:which(auc_parsimony_wilcox>=max(auc_parsimony_wilcox)*0.95)[1]])
var_wilcox<-names(rank_wilcox)[1:num_wilcox]

# variable selection by likelihood increment
auc_parsimony_likelihood_increment<-AutoScore_parsimony(new_data_train,new_data_validation,rank_likelihood_increment)
# set 95% of max AUC
num_likelihood_increment<-length(rank_likelihood_increment[1:which(auc_parsimony_likelihood_increment>=max(auc_parsimony_likelihood_increment)*0.95)[1]])
var_likelihood_increment<-names(rank_likelihood_increment)[1:num_likelihood_increment]

# variable selection by likelihood deletion
auc_parsimony_likelihood_deletion<-AutoScore_parsimony(new_data_train,new_data_validation,rank_likelihood_deletion)
# set 95% of max AUC
num_likelihood_deletion<-length(rank_likelihood_deletion[1:which(auc_parsimony_likelihood_deletion>=max(auc_parsimony_likelihood_deletion)*0.95)[1]])
var_likelihood_deletion<-names(rank_likelihood_deletion)[1:num_likelihood_deletion]

# variable selection by XGBoost linear
auc_parsimony_xgb_linear<-AutoScore_parsimony(new_data_train,new_data_validation,rank_xgb_linear)
# set 95% of max AUC
num_xgb_linear<-length(rank_xgb_linear[1:which(auc_parsimony_xgb_linear>=max(auc_parsimony_xgb_linear)*0.95)[1]])
var_xgb_linear<-names(rank_xgb_linear)[1:num_xgb_linear]

# variable selection by XGBoost tree
auc_parsimony_xgb_tree<-AutoScore_parsimony(new_data_train,new_data_validation,rank_xgb_tree)
# set 95% of max AUC
num_xgb_tree<-length(rank_xgb_tree[1:which(auc_parsimony_xgb_tree>=max(auc_parsimony_xgb_tree)*0.95)[1]])
var_xgb_tree<-names(rank_xgb_tree)[1:num_xgb_tree]

# count each variable's time by ranking methods above
var_total<-c(var_rf,var_t_test,var_wilcox,var_likelihood_increment,var_likelihood_deletion,var_xgb_linear,var_xgb_tree)

var_unique<-unique(var_total)
var_unique_count<-rep(0,length(var_unique))
for (i in 1:length(var_unique)) {
  var_unique_count[i]<-length(which(var_total==var_unique[i]))
}

names(var_unique_count)<-var_unique
var_unique_count_1<-sort(var_unique_count, decreasing = T)

# we have 7 ranking measures here. We include variables selected by 50% of all measures, and that is 4(round(7/2))
rank_majority_voting<-var_unique_count_1[1:(which(var_unique_count_1<4)[1]-1)]

# run AutoScore
auto_weight<-AutoScore_weighting(new_data_train,new_data_train,names(rank_majority_voting))
myvec<-AutoScore_fine_tuning(new_data_train,new_data_train,names(rank_majority_voting),auto_weight)
test_result_rf<-AutoScore_testing(new_data_test,names(rank_majority_voting),auto_weight,myvec)
