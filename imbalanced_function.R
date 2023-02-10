#Load necessary packages
library(randomForest)
library(ggplot2)
library(pROC)
library(PRROC)
library(reticulate)
library(tsne)
library(rpart)
library(DMwR)
library(mltools)
library(knitr)
#When meeting "Error in plot.new() : figure margins too large", users can try clear all plots

# Data visulization function --------------------------------------------------------
#Preliminary visualization through tSNE
imbalanced_tsne<-function(dataset, ratio = 5, random_seed = 1234){
  set.seed(random_seed)
  data_positive<-dataset[which(dataset$label==1), ]
  data_negative<-dataset[which(dataset$label==0), ]
  data_negative_tsne<-data_negative[sample(1:nrow(data_negative), size = ratio*nrow(data_positive)), ]
  data_tsne<-rbind(data_positive,data_negative_tsne)
  character_label<-c(1:nrow(data_tsne))
  character_label[which(data_tsne$label==1)]<-"1"
  character_label[which(data_tsne$label==0)]<-"0"
  colors = rainbow(length(unique(character_label)))
  names(colors) = unique(character_label)
  result_tsne<-tsne(data_tsne[,-which(names(data_negative)=="label")])
  # visualize results 
  plot(result_tsne,col=colors[character_label],pch=16,
       xlab = "tSNE_1",ylab = "tSNE_2",main = "tSNE plot of dataset")
  # add lines
  abline(h=0,v=0,lty=2,col="gray")
  # add explanations
  legend("topright",title = "Labels",inset = 0.01,
         legend = unique(character_label),pch=16,
         col = unique(colors[character_label]))
}

# Data generation function --------------------------------------------------------
#Data generation by SMOTE
smote_generation<-function(dataset, ratio, random_seed = 1234){
  set.seed(random_seed)
  dataset$label<-as.factor(dataset$label)#important due to smote requirement
  ratio_smote<-c(0,0)
  result<-c(0,0)
  a<-length(which(dataset$label==1))
  b<-length(which(dataset$label==0))
  c<-(b/(1-ratio)-b)%/%a
  int<-(c-1)*100
  residual<-round(((b/(1-ratio)-b)/a-c)*100)
  result[1]<-int
  result[2]<-residual
  ratio_smote<-result
  if (ratio_smote[1]<=0 & a*ratio_smote[2]/100<2){return(dataset)}
  if (ratio_smote[1]<=0 & a*ratio_smote[2]/100>=2){
    set.seed(random_seed)
    data_generation<-SMOTE(label~ ., dataset, perc.over = ratio_smote[2], perc.under = 100)
    data_all<-rbind(data_generation[which(data_generation$label==1),], dataset[which(dataset$label==0),])
    data_all<-data_all[complete.cases(data_all),]
    return(data_all)
  }
  if (ratio_smote[1]>0 & a*ratio_smote[2]/100<2){
    set.seed(random_seed)
    data_generation<-SMOTE(label~ ., dataset, perc.over = ratio_smote[1], perc.under = 100)
    data_all<-rbind(data_generation[which(data_generation$label==1),], dataset[which(dataset$label==0),])
    data_all<-data_all[complete.cases(data_all),]
    return(data_all)
  }
  d1<-SMOTE(label~., dataset, perc.over = ratio_smote[1], perc.under = 100)
  d2<-SMOTE(label~., dataset, perc.over = ratio_smote[2], perc.under = 100)
  d1<-d1[complete.cases(d1),]
  d2<-d2[complete.cases(d2),]
  {if (nrow(d1)==0) {return(d2)}}
  {if (nrow(d2)==0) {return(d1)}}
  data_generation<-rbind(d1,d2)
  data_all<-rbind(data_generation[which(data_generation$label==1),], dataset[which(dataset$label==0),])
  return(data_all)
}
  


#Data generation by upsampling
upsampling_generation<-function(dataset, ratio, random_seed = 1234){
  set.seed(random_seed)
  positive_data<-dataset[which(dataset$label==1),]
  negative_data<-dataset[which(dataset$label==0),]
  a<-length(which(dataset$label==1))
  b<-length(which(dataset$label==0))
  c<-round(b/(1-ratio)-b)
  d<-c%%a
  e<-c%/%a
  data_all<-rbind(positive_data[as.vector(sample(c(1:nrow(positive_data)),size = d,replace = F)),],negative_data)
  for (i in 1:e) {
    data_all<-rbind(data_all,positive_data)
  }
  return(data_all)
}

#Data generation by downsampling
downsampling_generation<-function(dataset, ratio, random_seed = 1234){
  set.seed(random_seed)
  positive_data<-dataset[which(dataset$label==1),]
  negative_data<-dataset[which(dataset$label==0),]
  a<-length(which(dataset$label==1))
  c<-round(a/ratio-a)
  data_all<-rbind(positive_data,negative_data[as.vector(sample(c(1:nrow(negative_data)),size = c,replace = F)),])
  return(data_all)
}

up_and_down_generation<-function(dataset_out, ratio_out, random_seed_out = 1234){
  set.seed(random_seed_out)
  raw_ratio<-length(which(dataset_out$label==1))/nrow(dataset_out)
  dataset_up<-upsampling_generation(dataset = dataset_out,ratio = (ratio_out+raw_ratio)/2, random_seed = random_seed_out)
  dataset_down<-downsampling_generation(dataset = dataset_up,ratio = ratio_out, random_seed = random_seed_out)
  return(dataset_down)
}

smote_and_down_generation<-function(dataset_out, ratio_out, random_seed_out = 1234){
  set.seed(random_seed_out)
  raw_ratio<-length(which(dataset_out$label==1))/nrow(dataset_out)
  dataset_smote<-smote_generation(dataset = dataset_out,ratio = ((ratio_out+raw_ratio)/2), random_seed = random_seed_out)
  if (length(which(dataset_smote$label==1))/nrow(dataset_smote)>=ratio_out) 
    {return(dataset_smote)}
  else {
  dataset_down<-downsampling_generation(dataset = dataset_smote,ratio = ratio_out, random_seed = random_seed_out)
  return(dataset_down)}
}

gan_and_down_generation<-function(dataset_out, ratio_out, random_seed_out = 1234,ctgan_positive_results){
  set.seed(random_seed_out)
  raw_ratio<-length(which(dataset_out$label==1))/nrow(dataset_out)
  dataset_gan<-ctgan_generation(dataset = dataset_out,ctgan_positive_result = ctgan_positive_results,ratio = ((ratio_out+raw_ratio)/2), random_seed = random_seed_out)
  if (length(which(dataset_gan$label==1))/nrow(dataset_gan)>=ratio_out) 
  {return(dataset_gan)}
  else {
    dataset_down<-downsampling_generation(dataset = dataset_gan,ratio = ratio_out, random_seed = random_seed_out)
    return(dataset_down)}
}

ctgan_preparation<-function(python_route){
  #set python environment
  use_condaenv(python_route)
  if (py_module_available("pandas")==T) {
    print("Python environment is available")
    print("Module pandas is available")} 
  else {print("error: Module pandas is not available")}
  if (py_module_available("ctgan")==T) {print("MOdule ctgan is available")} else {print("error: MOdule ctgan is not available")}
}

ctgan_positive_generation<-function(dataset, epoch, sample_num){
  ctgan<-import("ctgan")
  CTGANSynthesizer<-ctgan$CTGANSynthesizer
  discrete_column = "label"
  gan_fun = CTGANSynthesizer(epochs=epoch)
  gan_fun$fit(data_train, discrete_column)
  # Synthetic copy
  generated_data = gan_fun$sample(sample_num)
  print("Mutiple warning could be generated by CTGAN in this step")
  positive_generated_sample<-generated_data[which(generated_data$label==1),]
  return(positive_generated_sample)
}

ctgan_generation<-function(dataset, ctgan_positive_result, ratio, random_seed = 1234){
  set.seed(random_seed)
  dataset_negative<-dataset[which(data_train$label==0),]
  a<-length(which(dataset$label==1))
  b<-length(which(dataset$label==0))
  c<-round(b/(1-ratio)-b-a)
  data_positive<-ctgan_positive_result[as.vector(sample(c(1:nrow(ctgan_positive_result)),
                                                        size = c,replace = T)),]
  data_all<-rbind(data_positive, dataset)
  return(data_all)
}

get_ratio<-function(data_train,steps = 1,serial_num){
  a<-length(which(data_train$label==1))
  b<-nrow(data_train)
  generated_numbers<-((50-ceiling(a/b*100))/steps+1)
  generated_dataset<-list()
  generated_dataset[[1]]<-data_train
  generated_ratio<-seq(from = ceiling((a/b)*100), to = 50, by = steps)/100
  return(generated_ratio[serial_num-1])
}

# Auto calculator function --------------------------------------------------------
#auto calculator for different imbalanced adjustment measures
auto_calculator<-function(data_train, data_validation,
                          method, var_num, 
                          random_seed_out = 1234, steps = 1, ctgan_positive_results = 1){
  set.seed(random_seed_out)
  a<-length(which(data_train$label==1))
  b<-nrow(data_train)
  generated_numbers<-((50-ceiling(a/b*100))/steps+1)
  generated_dataset<-list()
  generated_dataset[[1]]<-data_train
  generated_ratio<-seq(from = ceiling((a/b)*100), to = 50, by = steps)/100
  all_ratio<-c(round((a/b),4), generated_ratio)
  #AUPRC_values<-c(1:(length(generated_ratio)+1))
  precision_values<-c(1:(length(generated_ratio)+1))
  AUROC_values<-c(1:(length(generated_ratio)+1))
  specificity_values<-c(1:(length(generated_ratio)+1))
  sensitivity_values<-c(1:(length(generated_ratio)+1))
  accuracy_values<-c(1:(length(generated_ratio)+1))
  npv_values<-c(1:(length(generated_ratio)+1))
  F1_values<-c(1:(length(generated_ratio)+1))
  Fmeasure_values<-c(1:(length(generated_ratio)+1))
  if (method == "SMOTE"){
    for (i in 1:length(generated_ratio)) {
      generated_dataset[[i+1]]<-smote_generation(data_train,generated_ratio[i],random_seed = random_seed_out)
      rr<-length(which(generated_dataset[[i+1]]$label==1))/nrow(generated_dataset[[i+1]])
      print(paste("real generated raio is",rr))
      print(paste("SMOTE dataset", i, "has been generated"))
    }}
  if (method == "upsampling"){
    for (i in 1:length(generated_ratio)) {
      generated_dataset[[i+1]]<-upsampling_generation(data_train,generated_ratio[i],random_seed = random_seed_out)
      rr<-length(which(generated_dataset[[i+1]]$label==1))/nrow(generated_dataset[[i+1]])
      print(paste("real generated raio is",rr))
      print(paste("Upsampling dataset", i, "has been generated"))
    }}
  if (method == "downsampling"){
    for (i in 1:length(generated_ratio)) {
      generated_dataset[[i+1]]<-downsampling_generation(data_train,generated_ratio[i],random_seed = random_seed_out)
      rr<-length(which(generated_dataset[[i+1]]$label==1))/nrow(generated_dataset[[i+1]])
      print(paste("real generated raio is",rr))
      print(paste("Downsampling dataset", i, "has been generated"))
    }}
  if (method == "up_and_down_sampling"){
    for (i in 1:length(generated_ratio)) {
      generated_dataset[[i+1]]<-up_and_down_generation(data_train,generated_ratio[i],random_seed_out)
      rr<-length(which(generated_dataset[[i+1]]$label==1))/nrow(generated_dataset[[i+1]])
      print(paste("real generated raio is",rr))
      print(paste("up_and_down_sampling dataset", i, "has been generated"))
    }}
  if (method == "smote_and_down_sampling"){
    for (i in 1:length(generated_ratio)) {
      generated_dataset[[i+1]]<-smote_and_down_generation(data_train,generated_ratio[i],random_seed_out)
      rr<-length(which(generated_dataset[[i+1]]$label==1))/nrow(generated_dataset[[i+1]])
      print(paste("real generated raio is",rr))
      print(paste("smote_and_down_sampling dataset", i, "has been generated"))
    }}
  if (method == "ctgan"){
    for (i in 1:length(generated_ratio)) {
      generated_dataset[[i+1]]<-ctgan_generation(data_train,ctgan_positive_results,generated_ratio[i],random_seed = random_seed_out)
      rr<-length(which(generated_dataset[[i+1]]$label==1))/nrow(generated_dataset[[i+1]])
      print(paste("real generated raio is",rr))
      print(paste("ctgan dataset", i, "has been generated"))
    }}
  if (method == "gan_and_downsampling"){
    for (i in 1:length(generated_ratio)) {
      generated_dataset[[i+1]]<-gan_and_down_generation(dataset_out = data_train, ratio_out = generated_ratio[i], 
                              random_seed_out,ctgan_positive_results)
      rr<-length(which(generated_dataset[[i+1]]$label==1))/nrow(generated_dataset[[i+1]])
      print(paste("real generated raio is",rr))
      print(paste("gan_and_downsampling dataset", i, "has been generated"))
    }}
  for (j in 1:(length(generated_ratio)+1)) {
    rr<-length(which(generated_dataset[[j]]$label==1))/nrow(generated_dataset[[j]])
    print(paste("current raio is",rr))
    Ranking<-AutoScore_rank(generated_dataset[[j]])
    #AUC<-AutoScore_parsimony(generated_dataset[[j]],data_validation,Ranking)
    FinalVariable<-names(Ranking)[1:var_num]
    CutVec<-AutoScore_weighting(generated_dataset[[j]],data_validation,FinalVariable)
    ScoringTable<-AutoScore_fine_tuning(generated_dataset[[j]],data_validation,FinalVariable,CutVec)
    test_result<-AutoScore_testing(data_validation,FinalVariable,CutVec,ScoringTable)
    #AUPRC_values[j]<-test_result[[7]]$auc.integral
    AUROC_values[j]<-test_result[[1]]
    specificity_values[j]<-test_result[[2]]
    sensitivity_values[j]<-test_result[[3]]
    accuracy_values[j]<-test_result[[4]]
    npv_values[j]<-test_result[[5]]
    precision_values[j]<-test_result[[6]]
    F1_values[j]<-(sensitivity_values[j]*precision_values[j])*2/(sensitivity_values[j]+precision_values[j])
    Fmeasure_values[j]<-(specificity_values[j]*sensitivity_values[j])*2/(specificity_values[j]+sensitivity_values[j])
    print(paste("Dataset", j, "Autoscore has been calculated"))
  }
  optimal_data<-generated_dataset[[which(AUROC_values==max(AUROC_values[-1]))]]
  return_values<-list(precision = precision_values,
                      AUROC = AUROC_values, specificity = specificity_values,
                      sensitivity = sensitivity_values, accuracy = accuracy_values,
                      npv = npv_values, F1 = F1_values, Fmeasure = Fmeasure_values,
                      optimal_dataset = optimal_data)
  #print(paste("AUPRC reached max values in minority samples ratio of",
              #all_ratio[which(AUPRC_values==max(AUPRC_values))]))
  print(paste("Precision reached max values in minority samples ratio of",
              all_ratio[which(precision_values==max(precision_values))]))
  print(paste("F1 score reached max values in minority samples ratio of",
              all_ratio[which(F1_values==max(F1_values))]))
  print(paste("F measure reached max values in minority samples ratio of",
              all_ratio[which(Fmeasure_values==max(Fmeasure_values))]))
  #plotauprc<-data.frame(x = all_ratio, y = AUPRC_values)
  #plotprecision<-data.frame(x = all_ratio, y = precision_values)
  #plotf1<-data.frame(x = all_ratio, y = F1_values)
  plotfm<-data.frame(x = all_ratio, y = Fmeasure_values)
  #print(ggplot(plotauprc, aes(x,y))+geom_line(color="blue",size = 1)+
  #geom_point(size=2)+
  #geom_point(color = "orange")+
  #xlab("Minority samples ratio")+ylab("AUPRC value")+
  #labs(title = "AUPRC under different data generation"))
  #print(ggplot(plotprecision, aes(x,y))+geom_line(color="blue",size = 1)+
    #geom_point(size=2)+
    #geom_point(color = "orange")+
    #xlab("Minority samples ratio")+ylab("Precision value")+
    #labs(title = "Precision under different data generation"))
  #print(ggplot(plotf1, aes(x,y))+geom_line(color="blue",size = 1)+
  #geom_point(size=2)+
  #geom_point(color = "orange")+
  #xlab("Minority samples ratio")+ylab("Precision value")+
  #labs(title = "F1 score under different data generation"))
  print(ggplot(plotfm, aes(x,y))+geom_line(color="blue",size = 1)+
          geom_point(size=2)+
          geom_point(color = "orange")+
          xlab("Minority samples ratio")+ylab("Precision value")+
          labs(title = "F measure under different data generation"))
  return(return_values)
}

auto_calculator_plot<-function(data_train, result, evaluation, steps = 1){
  a<-length(which(data_train$label==1))
  b<-nrow(data_train)
  generated_ratio<-seq(from = ceiling(a/b*100), to = 50, by = steps)/100
  all_ratio<-c(round((a/b),4), generated_ratio)
  if (evaluation == "AUPRC"){
  data_plot<-data.frame(x = all_ratio, y = result$AUPRC)
  print(ggplot(data_plot, aes(x,y))+geom_line(color="blue",size = 1)+
    geom_point(size=2)+
    geom_point(color = "orange")+
    xlab("Minority samples ratio")+ylab("AUPRC value")+
    labs(title = "AUPRC under different data generation"))
  a<-all_ratio[which(result$AUPRC==max(result$AUPRC))]
  print(paste("AUPRC reached max value",max(result$AUPRC),"in minority samples ratio of",a))
  return(a)
  }
  if (evaluation == "Precision"){
    data_plot<-data.frame(x = all_ratio, y = result$precision)
    print(ggplot(data_plot, aes(x,y))+geom_line(color="blue",size = 1)+
      geom_point(size=2)+
      geom_point(color = "orange")+
      xlab("Minority samples ratio")+ylab("Precision value")+
      labs(title = "Precision under different data generation"))
    a<-all_ratio[which(result$precision==max(result$precision))]
    print(paste("Precision reached max value",max(result$precision)," in minority samples ratio of",
                a))
    return(a)
  }
  if (evaluation == "AUROC"){
    data_plot<-data.frame(x = all_ratio, y = result$AUROC)
    print(ggplot(data_plot, aes(x,y))+geom_line(color="blue",size = 1)+
      geom_point(size=2)+
      geom_point(color = "orange")+
      xlab("Minority samples ratio")+ylab("AUROC value")+
      labs(title = "AUROC under different data generation"))
    a<-all_ratio[which(result$AUROC==max(result$AUROC))]
    print(paste("AUROC reached max value",max(result$AUROC)," in minority samples ratio of",
                a))
    return(a)
  }
  if (evaluation == "Specificity"){
    data_plot<-data.frame(x = all_ratio, y = result$specificity)
    print(ggplot(data_plot, aes(x,y))+geom_line(color="blue",size = 1)+
      geom_point(size=2)+
      geom_point(color = "orange")+
      xlab("Minority samples ratio")+ylab("Specificity value")+
      labs(title = "Specificity under different data generation"))
    a<-all_ratio[which(result$specificity==max(result$specificity))]
    print(paste("Specificity reached max value",max(result$specificity)," in minority samples ratio of",
                a))
    return(a)
  }
  if (evaluation == "Sensitivity"){
    data_plot<-data.frame(x = all_ratio, y = result$sensitivity)
    print(ggplot(data_plot, aes(x,y))+geom_line(color="blue",size = 1)+
      geom_point(size=2)+
      geom_point(color = "orange")+
      xlab("Minority samples ratio")+ylab("Sensitivity value")+
      labs(title = "Sensitivity under different data generation"))
    a<-all_ratio[which(result$sensitivity==max(result$sensitivity))]
    print(paste("Sensitivity reached max value",max(result$sensitivity)," in minority samples ratio of",
                a))
    return(a)
  }
  if (evaluation == "Accuracy"){
    data_plot<-data.frame(x = all_ratio, y = result$accuracy)
    print(ggplot(data_plot, aes(x,y))+geom_line(color="blue",size = 1)+
      geom_point(size=2)+
      geom_point(color = "orange")+
      xlab("Minority samples ratio")+ylab("Accuracy value")+
      labs(title = "Accuracy under different data generation"))
    a<-all_ratio[which(result$accuracy==max(result$accuracy))]
    print(paste("Accuracy reached max value",max(result$accuracy)," in minority samples ratio of",
                a))
    return(a)
  }
  if (evaluation == "NPV"){
    data_plot<-data.frame(x = all_ratio, y = result$npv)
    print(ggplot(data_plot, aes(x,y))+geom_line(color="blue",size = 1)+
      geom_point(size=2)+
      geom_point(color = "orange")+
      xlab("Minority samples ratio")+ylab("NPV value")+
      labs(title = "NPV under different data generation"))
    a<-all_ratio[which(result$npv==max(result$npv))]
    print(paste("NPV reached max value",max(result$npv)," in minority samples ratio of",
                a))
    return(a)
  }
}

auto_calculator_robust<-function(data_train_out, data_validation_out,
                                 data_test_out, method_out, var_num_out, 
                                 steps_out = 1, ctgan_positive_results_out = 1, num_out = 10){
  a_out<-length(which(data_train_out$label==1))
  b_out<-nrow(data_train_out)
  generated_ratio_out<-seq(from = ceiling(a_out/b_out*100), to = 50, by = steps_out)/100
  all_ratio_out<-c(round((a_out/b_out),4), generated_ratio_out)
  AUPRC_values_out<-matrix(nrow = num_out, ncol = (length(generated_ratio_out)+1))
  precision_values_out<-matrix(nrow = num_out, ncol = (length(generated_ratio_out)+1))
  AUROC_values_out<-matrix(nrow = num_out, ncol = (length(generated_ratio_out)+1))
  specificity_values_out<-matrix(nrow = num_out, ncol = (length(generated_ratio_out)+1))
  sensitivity_values_out<-matrix(nrow = num_out, ncol = (length(generated_ratio_out)+1))
  accuracy_values_out<-matrix(nrow = num_out, ncol = (length(generated_ratio_out)+1))
  npv_values_out<-matrix(nrow = num_out, ncol = (length(generated_ratio_out)+1))
  
  for (i in 1:num_out) {
    seed <- i
    result_in<-auto_calculator(data_train_out, data_validation_out, data_test_out, method_out, 
                               var_num_out, random_seed_out = seed, steps_out, ctgan_positive_results = ctgan_positive_results_out)
    AUPRC_values_out[i,]<-result_in$AUPRC
    precision_values_out[i,]<-result_in$precision
    AUROC_values_out[i,]<-result_in$AUROC
    specificity_values_out[i,]<-result_in$specificity
    sensitivity_values_out[i,]<-result_in$sensitivity
    accuracy_values_out[i,]<-result_in$accuracy
    npv_values_out[i,]<-result_in$npv
    print(paste("Robust scenario", i, "Autoscore has been calculated"))
  }

  AUPRC_values_final<-colMeans(AUPRC_values_out)
  precision_values_final<-colMeans(precision_values_out)
  AUROC_values_final<-colMeans(AUROC_values_out)
  specificity_values_final<-colMeans(specificity_values_out)
  sensitivity_values_final<-colMeans(sensitivity_values_out)
  accuracy_values_final<-colMeans(accuracy_values_out)
  npv_values_final<-colMeans(npv_values_out)
  
  library("matrixStats")
  
  AUPRC_values_var<-colVars(AUPRC_values_out)
  precision_values_var<-colVars(precision_values_out)
  AUROC_values_var<-colVars(AUROC_values_out)
  specificity_values_var<-colVars(specificity_values_out)
  sensitivity_values_var<-colVars(sensitivity_values_out)
  accuracy_values_var<-colVars(accuracy_values_out)
  npv_values_var<-colVars(npv_values_out)
  
  return_values<-list(AUPRC = AUPRC_values_final, precision = precision_values_final,
                      AUROC = AUROC_values_final, specificity = specificity_values_final,
                      sensitivity = sensitivity_values_final, accuracy = accuracy_values_final,
                      npv = npv_values_final,
                      AUPRC_var = AUPRC_values_var, precision_var = precision_values_var,
                      AUROC_var = AUROC_values_var, specificity_var = specificity_values_var,
                      sensitivity_var = sensitivity_values_var, accuracy_var = accuracy_values_var,
                      npv_var = npv_values_var,
                      AUPRC_all = AUPRC_values_out, precision_all = precision_values_out,
                      AUROC_all = AUROC_values_out, specificity_all = specificity_values_out,
                      sensitivity_all = sensitivity_values_out, accuracy_all = accuracy_values_out,
                      npv_all = npv_values_out
                      )
  return(return_values)
}

#custom evaluations
custom_evaluation<-function(data_train,result,steps = 1,quant,base1,base2,
                            base3 = 1,base4 = 1,base5 = 1,base6 = 1,base7 = 1,
                            weight1,weight2,weight3 = 1,weight4 = 1,weight5 = 1,weight6 = 1,weight7 = 1){
  if (quant == 2){
    base1_value<-result[[which(names(result)==base1)]]
    base2_value<-result[[which(names(result)==base2)]]
    new_evaluation_value<-(weight1*base1_value+weight2*base2_value)/(weight1+weight2)
    a<-length(which(data_train$label==1))
    b<-nrow(data_train)
    generated_ratio<-seq(from = ceiling(a/b*100), to = 50, by = steps)/100
    all_ratio<-c(round((a/b),4), generated_ratio)
    data_plot<-data.frame(ratio = all_ratio, value = new_evaluation_value)
    print(ggplot(data_plot, aes(ratio,value))+geom_line(color="blue",size = 1)+
            geom_point(size=2)+
            geom_point(color = "orange")+
            xlab("Minority samples ratio")+ylab("AUPRC value")+
            labs(title = "Custom evaluation including two base evaluations under different data generation"))
    a<-all_ratio[which(new_evaluation_value==max(new_evaluation_value))]
    print(paste("Custom evaluation reached max value",max(new_evaluation_value),"in minority samples ratio of",a))
    return(data_plot)
  }
  if (quant == 3){
    base1_value<-result[[which(names(result)==base1)]]
    base2_value<-result[[which(names(result)==base2)]]
    base3_value<-result[[which(names(result)==base3)]]
    new_evaluation_value<-(weight1*base1_value+weight2*base2_value+weight3*base3_value)/(weight1+weight2+weight3)
    a<-length(which(data_train$label==1))
    b<-nrow(data_train)
    generated_ratio<-seq(from = ceiling(a/b*100), to = 50, by = steps)/100
    all_ratio<-c(round((a/b),4), generated_ratio)
    data_plot<-data.frame(ratio = all_ratio, value = new_evaluation_value)
    print(ggplot(data_plot, aes(ratio,value))+geom_line(color="blue",size = 1)+
            geom_point(size=2)+
            geom_point(color = "orange")+
            xlab("Minority samples ratio")+ylab("AUPRC value")+
            labs(title = "Custom evaluation including three base evaluations under different data generation"))
    a<-all_ratio[which(new_evaluation_value==max(new_evaluation_value))]
    print(paste("Custom evaluation reached max value",max(new_evaluation_value),"in minority samples ratio of",a))
    return(data_plot)
  }
  if (quant == 4){
    base1_value<-result[[which(names(result)==base1)]]
    base2_value<-result[[which(names(result)==base2)]]
    base3_value<-result[[which(names(result)==base3)]]
    base4_value<-result[[which(names(result)==base4)]]
    new_evaluation_value<-(weight1*base1_value+weight2*base2_value+weight3*base3_value+
                             weight4*base4_value)/(weight1+weight2+weight3+weight4)
    a<-length(which(data_train$label==1))
    b<-nrow(data_train)
    generated_ratio<-seq(from = ceiling(a/b*100), to = 50, by = steps)/100
    all_ratio<-c(round((a/b),4), generated_ratio)
    data_plot<-data.frame(ratio = all_ratio, value = new_evaluation_value)
    print(ggplot(data_plot, aes(ratio,value))+geom_line(color="blue",size = 1)+
            geom_point(size=2)+
            geom_point(color = "orange")+
            xlab("Minority samples ratio")+ylab("AUPRC value")+
            labs(title = "Custom evaluation including four base evaluations under different data generation"))
    a<-all_ratio[which(new_evaluation_value==max(new_evaluation_value))]
    print(paste("Custom evaluation reached max value",max(new_evaluation_value),"in minority samples ratio of",a))
    return(data_plot)
  }
  if (quant == 5){
    base1_value<-result[[which(names(result)==base1)]]
    base2_value<-result[[which(names(result)==base2)]]
    base3_value<-result[[which(names(result)==base3)]]
    base4_value<-result[[which(names(result)==base4)]]
    base5_value<-result[[which(names(result)==base5)]]
    new_evaluation_value<-(weight1*base1_value+weight2*base2_value+weight3*base3_value+
                             weight4*base4_value+weight5*base5_value)/(weight1+weight2+weight3+weight4+weight5)
    a<-length(which(data_train$label==1))
    b<-nrow(data_train)
    generated_ratio<-seq(from = ceiling(a/b*100), to = 50, by = steps)/100
    all_ratio<-c(round((a/b),4), generated_ratio)
    data_plot<-data.frame(ratio = all_ratio, value = new_evaluation_value)
    print(ggplot(data_plot, aes(ratio,value))+geom_line(color="blue",size = 1)+
            geom_point(size=2)+
            geom_point(color = "orange")+
            xlab("Minority samples ratio")+ylab("AUPRC value")+
            labs(title = "Custom evaluation including five base evaluations under different data generation"))
    a<-all_ratio[which(new_evaluation_value==max(new_evaluation_value))]
    print(paste("Custom evaluation reached max value",max(new_evaluation_value),"in minority samples ratio of",a))
    return(data_plot)
  }
  if (quant == 6){
    base1_value<-result[[which(names(result)==base1)]]
    base2_value<-result[[which(names(result)==base2)]]
    base3_value<-result[[which(names(result)==base3)]]
    base4_value<-result[[which(names(result)==base4)]]
    base5_value<-result[[which(names(result)==base5)]]
    base6_value<-result[[which(names(result)==base6)]]
    new_evaluation_value<-(weight1*base1_value+weight2*base2_value+weight3*base3_value+
                             weight4*base4_value+weight5*base5_value+weight6*base6_value)/(weight1+weight2+weight3+weight4+weight5+weight6)
    a<-length(which(data_train$label==1))
    b<-nrow(data_train)
    generated_ratio<-seq(from = ceiling(a/b*100), to = 50, by = steps)/100
    all_ratio<-c(round((a/b),4), generated_ratio)
    data_plot<-data.frame(ratio = all_ratio, value = new_evaluation_value)
    print(ggplot(data_plot, aes(ratio,value))+geom_line(color="blue",size = 1)+
            geom_point(size=2)+
            geom_point(color = "orange")+
            xlab("Minority samples ratio")+ylab("AUPRC value")+
            labs(title = "Custom evaluation including six base evaluations under different data generation"))
    a<-all_ratio[which(new_evaluation_value==max(new_evaluation_value))]
    print(paste("Custom evaluation reached max value",max(new_evaluation_value),"in minority samples ratio of",a))
    return(data_plot)
  }
  if (quant == 7){
    base1_value<-result[[which(names(result)==base1)]]
    base2_value<-result[[which(names(result)==base2)]]
    base3_value<-result[[which(names(result)==base3)]]
    base4_value<-result[[which(names(result)==base4)]]
    base5_value<-result[[which(names(result)==base5)]]
    base6_value<-result[[which(names(result)==base6)]]
    base7_value<-result[[which(names(result)==base7)]]
    new_evaluation_value<-(weight1*base1_value+weight2*base2_value+weight3*base3_value+
                             weight4*base4_value+weight5*base5_value+weight6*base6_value+weight7*base7_value)/(weight1+weight2+weight3+weight4+weight5+weight6+weight7)
    a<-length(which(data_train$label==1))
    b<-nrow(data_train)
    generated_ratio<-seq(from = ceiling(a/b*100), to = 50, by = steps)/100
    all_ratio<-c(round((a/b),4), generated_ratio)
    data_plot<-data.frame(ratio = all_ratio, value = new_evaluation_value)
    print(ggplot(data_plot, aes(ratio,value))+geom_line(color="blue",size = 1)+
            geom_point(size=2)+
            geom_point(color = "orange")+
            xlab("Minority samples ratio")+ylab("AUPRC value")+
            labs(title = "Custom evaluation including seven base evaluations under different data generation"))
    a<-all_ratio[which(new_evaluation_value==max(new_evaluation_value))]
    print(paste("Custom evaluation reached max value",max(new_evaluation_value),"in minority samples ratio of",a))
    return(data_plot)
  }
}

adjust_weight<-function(data_train,data_validation,predictor,predictor_num,steps = 1,random_seed_out = 1234){
  set.seed(random_seed_out)
  weight_max<-length(which(data_train$label==0))/length(which(data_train$label==1))
  weight_list<-c(seq(from = 1, to = ceiling(weight_max), by = steps))
  weight_list<-round(weight_list,0)
  sample_weight<-c(rep(1,nrow(data_train)))
  auroc_list<-c(rep(1,length(weight_list)))
  specificity_list<-c(rep(1,length(weight_list)))
  sensitivity_list<-c(rep(1,length(weight_list)))
  for (i in 1:length(weight_list)) {
    weight_tmp<-weight_list[i]
    sample_weight_tmp<-sample_weight
    sample_weight_tmp[which(data_train$label==1)]<-weight_tmp
    #auc_parsimony<-AutoScore_parsimony_weight(data_train,data_validation,predictor,weight = sample_weight_tmp)
    final_variable<-names(predictor)[1:predictor_num]
    auto_weight<-AutoScore_weighting_weight(data_train,data_validation,final_variable,weight = sample_weight_tmp)
    myvec<-AutoScore_fine_tuning_weight(data_train,data_validation,final_variable,auto_weight,weight = sample_weight_tmp)
    test_result<-AutoScore_testing(data_validation,final_variable,auto_weight,myvec)
    auroc_list[i]<-test_result[[1]]
    specificity_list[i]<-test_result[[2]]
    sensitivity_list[i]<-test_result[[3]]
    print(paste("Adjusted weight", i,"in",length(weight_list), "has been calculated"))
  }
  weight_final_tmp<-c()
  for (i in 1:length(weight_list)) {
    #if (specificity_list[i]>=specificity_list[1] & sensitivity_list[i]>=sensitivity_list[1]){weight_final_tmp<-c(weight_final_tmp,i)}
    if (auroc_list[i]>=auroc_list[1]){weight_final_tmp<-c(weight_final_tmp,i)}
    }
  weight_final<-weight_list[which(auroc_list==max(auroc_list[weight_final_tmp]))]
  sample_weight_tmp<-sample_weight
  sample_weight_tmp[which(data_train$label==1)]<-weight_final
  #auc_parsimony<-AutoScore_parsimony_weight(data_train,data_validation,predictor,weight = sample_weight_tmp)
  final_variable<-names(predictor)[1:predictor_num]
  auto_weight<-AutoScore_weighting_weight(data_train,data_validation,final_variable,weight = sample_weight_tmp)
  myvec<-AutoScore_fine_tuning_weight(data_train,data_validation,final_variable,auto_weight,weight = sample_weight_tmp)
  results<-list(final_variable,auto_weight,myvec,sample_weight_tmp)
  return(results)
}

adjust_weight_harmonic<-function(data_train,data_validation,predictor,predictor_num,steps = 1){
  weight_max<-length(which(data_train$label==0))/length(which(data_train$label==1))
  weight_list<-c(seq(from = 1, to = ceiling(weight_max), by = steps))
  weight_list<-round(weight_list,0)
  sample_weight<-c(rep(1,nrow(data_train)))
  auroc_list<-c(rep(1,length(weight_list)))
  specificity_list<-c(rep(1,length(weight_list)))
  sensitivity_list<-c(rep(1,length(weight_list)))
  for (i in 1:length(weight_list)) {
    weight_tmp<-weight_list[i]
    sample_weight_tmp<-sample_weight
    sample_weight_tmp[which(data_train$label==1)]<-weight_tmp
    auc_parsimony<-AutoScore_parsimony_weight(data_train,data_validation,predictor,weight = sample_weight_tmp)
    final_variable<-names(predictor)[1:predictor_num]
    auto_weight<-AutoScore_weighting_weight(data_train,data_validation,final_variable,weight = sample_weight_tmp)
    myvec<-AutoScore_fine_tuning_weight(data_train,data_validation,final_variable,auto_weight,weight = sample_weight_tmp)
    test_result<-AutoScore_testing(data_validation,final_variable,auto_weight,myvec)
    auroc_list[i]<-test_result[[1]]
    specificity_list[i]<-test_result[[2]]
    sensitivity_list[i]<-test_result[[3]]
    print(paste("Adjusted weight", i,"in",length(weight_list), "has been calculated"))
  }
  weight_final_tmp<-c()
  harmonic_list<-(2*specificity_list*sensitivity_list/(specificity_list+sensitivity_list))
  for (i in 1:length(weight_list)) {
    if (harmonic_list[i]>=harmonic_list[1]){weight_final_tmp<-c(weight_final_tmp,i)}
  }
  weight_final<-weight_list[which(harmonic_list==max(harmonic_list[weight_final_tmp]))]
  sample_weight_tmp<-sample_weight
  sample_weight_tmp[which(data_train$label==1)]<-weight_final
  auc_parsimony<-AutoScore_parsimony_weight(data_train,data_validation,predictor,weight = sample_weight_tmp)
  final_variable<-names(predictor)[1:predictor_num]
  auto_weight<-AutoScore_weighting_weight(data_train,data_validation,final_variable,weight = sample_weight_tmp)
  myvec<-AutoScore_fine_tuning_weight(data_train,data_validation,final_variable,auto_weight,weight = sample_weight_tmp)
  results<-list(final_variable,auto_weight,myvec)
  return(results)
}

AutoScore_parsimony_imbalance <- function(train_set, validation_set, rank, max_score = 100, n = 5, fold = 10,
                                split = "quantile", quantiles = c(0, 0.05, 0.2, 0.8, 0.95, 1), do_trace = FALSE) {
    AUC <- c()
    
    # Go through AUtoScore Module 2/3/4 in the loop
    i<-n
    cat(paste("Select the number of Variables",i,":  "))
      
    variable_list<-names(rank)[1:i]
    train_set_1 <- train_set[, c(variable_list, "label")]
    validation_set_1 <- validation_set[, c(variable_list, "label")]
    model_roc<-compute_auc_val(train_set_1, validation_set_1,variable_list, split, quantiles, max_score)
    print(auc(model_roc))
    AUC <- c(AUC, auc(model_roc))
    return(AUC)
  }

AutoScore_weighting_weight <- function(train_set, validation_set, final_variables, weight, max_score = 100,split = "quantile", quantiles = c(0, 0.05, 0.2, 0.8, 0.95, 1)) {
  # prepare train_set and VadalitionSet
  cat("****Included Variables: \n")
  print(data.frame(variable_name = final_variables))
  train_set_1 <- train_set[, c(final_variables, "label")]
  validation_set_1 <- validation_set[, c(final_variables, "label")]
  
  # AutoScore Module 2 : cut numeric and transfer categories and generate "cut_vec"
  df_transformed <- transform_df(train_set_1, validation_set_1, split = split, quantiles = quantiles, Print_categories = TRUE)
  train_set_2 <- df_transformed[[1]]
  validation_set_2 <- df_transformed[[2]]
  cut_vec_tmp <- df_transformed[[3]]
  cut_vec <- cut_vec_tmp
  for (i in 1:length(cut_vec)) cut_vec[[i]] <- cut_vec[[i]][2:(length(cut_vec[[i]]) - 1)]
  
  # AutoScore Module 3 : Score weighting
  score_table<-compute_score_table_weight(train_set_2,validation_set_2,max_score,final_variables,weight)
  # Revised scoring table representation if possible @YuanHan
  cat("****The initial generated Scores are shown below \n")
  #print(as.data.frame(score_table))
  print_score_table(scoring_table = score_table, final_variable = final_variables)
  
  # Using "auto_test" to generate score based on new dataset and Scoring table "score_table"
  validation_set_3 <- auto_test(validation_set_2, score_table)
  validation_set_3$total_score <- rowSums(subset(validation_set_3, select = -label))
  y_validation <- validation_set_3$label
  
  # Intermediate evaluation based on Validation Set
  plot_roc_curve(validation_set_3$total_score, as.numeric(y_validation) - 1)
  cat("***Performance using AutoScore (based on validation set):\n")
  model_roc <- roc(y_validation, validation_set_3$total_score, quiet = T)
  print_roc_performance(model_roc)
  cat("***The cut-offs generated by the AutoScore are saved in cut_vec. You can decide whether to revise or fine-tune them \n")
  #print(cut_vec)
  return(cut_vec)
}

AutoScore_fine_tuning_weight <- function(train_set, validation_set, final_variables, weight, cut_vec, max_score = 100) {
  # Prepare train_set and VadalitionSet
  train_set_1 <- train_set[, c(final_variables, "label")]
  validation_set_1 <- validation_set[, c(final_variables, "label")]
  
  # AutoScore Module 2 : cut numeric and transfer categories (based on fix "cut_vec" vector)
  train_set_2 <- transform_df_fixed(train_set_1, cut_vec = cut_vec)
  validation_set_2 <- transform_df_fixed(validation_set_1, cut_vec = cut_vec)
  
  # AutoScore Module 3 : Score weighting
  score_table<-compute_score_table_weight(train_set_2,validation_set_2,max_score,final_variables,weight)
  # Revised scoring table representation if possible @YuanHan
  cat("***The generated Scores are shown below \n")
  #print(as.data.frame(score_table))
  #print_score_table(scoring_table = score_table, final_variable = final_variables)
  
  # Using "auto_test" to generate score based on new dataset and Scoring table "score_table"
  validation_set_3 <- auto_test(validation_set_2, score_table)
  validation_set_3$total_score <- rowSums(subset(validation_set_3, select = -label))
  y_validation <- validation_set_3$label
  
  # Intermediate evaluation based on Validation Set after fine-tuning
  plot_roc_curve(validation_set_3$total_score, as.numeric(y_validation) - 1)
  cat("***Performance using AutoScore (based on Validation Set (After fine-tuning)):\n")
  model_roc <- roc(y_validation, validation_set_3$total_score, quiet = T)
  print_roc_performance(model_roc)
  return(score_table)
}

compute_score_table_weight<-function(train_set_2,validation_set_2,max_score,variable_list,weight){
  
  #AutoScore Module 3 : Score weighting
  # First-step logistic regression
  model <- glm(label ~ ., family = binomial(link = "logit"), data = train_set_2,weights = weight)
  y_validation <- validation_set_2$label
  coef_vec <- coef(model)
  if (length(which(is.na(coef_vec)))>0) {warning(" WARNING: GLM output contains NULL, Replace NULL with 1")
    coef_vec[which(is.na(coef_vec))]<-1}
  train_set_2 <- change_reference(train_set_2, coef_vec)
  
  # Second-step logistic regression
  model <- glm(label ~ ., family = binomial(link = "logit"), data = train_set_2,weights = weight)
  coef_vec <- coef(model)
  if (length(which(is.na(coef_vec)))>0) {warning(" WARNING: GLM output contains NULL, Replace NULL with 1")
    coef_vec[which(is.na(coef_vec))]<-1}
  
  # rounding for final scoring table "score_table"
  coef_vec_tmp <- round(coef_vec/min(coef_vec[-1]))
  score_table <- add_baseline(train_set_2, coef_vec_tmp)
  
  # normalization according to "max_score" and regenerate score_table
  total_max <- max_score
  total <- 0
  for (i in 1:length(variable_list)) total <- total + max(score_table[grepl(variable_list[i], names(score_table))])
  score_table <- round(score_table/(total/total_max))
  return(score_table)}

AutoScore_testing_imbalance_evaluation <- function(test_set, final_variables, cut_vec, scoring_table,evaluation,evaluation_value) {
  # prepare testset: categorization and "auto_test"
  test_set_1 <- test_set[, c(final_variables, "label")]
  test_set_2 <- transform_df_fixed(test_set_1, cut_vec = cut_vec)
  test_set_3 <- auto_test(test_set_2, scoring_table)
  test_set_3$total_score <- rowSums(subset(test_set_3, select = -label))
  test_set_3$total_score[which(is.na(test_set_3$total_score))]<-0
  y_test <- test_set_3$label
  
  # Final evaluation based on testing set
  plot_roc_curve(test_set_3$total_score, as.numeric(y_test) - 1)
  cat("***Performance using AutoScore (based on unseen test Set):\n")
  model_roc <- roc(y_test, test_set_3$total_score, quiet = T)
  print_roc_performance(model_roc)
  {if (evaluation == "sensitivity")
  {threshold<-coords(model_roc, evaluation_value, input = "sens",transpose = TRUE)
  values2<-ci.coords(model_roc,x=threshold[1],input = "threshold",ret = c("sensitivity","specificity"))
  return_values<-list(threshold[1],auc(model_roc),paste("(",round(ci.auc(model_roc)[1],3),"-",round(ci.auc(model_roc)[3],3),")",sep = ""),
                      threshold[2],paste("(",round(values2$sensitivity[,1],3),"-",round(values2$sensitivity[,3],3),")",sep = ""),
                      threshold[3],paste("(",round(values2$specificity[,1],3),"-",round(values2$specificity[,3],3),")",sep = ""))
  }
  else
  {threshold<-coords(model_roc, "best", ret = "threshold", transpose = TRUE)
  #Modelprc <- pr.curve(test_set_3$total_score[which(y_test == 1)],test_set_3$total_score[which(y_test == 0)],curve = TRUE)
  values<-coords(model_roc, "best", ret = c("specificity", "sensitivity"), transpose = TRUE)
  values2<-ci.coords(model_roc,x=threshold,input = "threshold",ret = c("sensitivity","specificity"))
  return_values<-list(threshold,auc(model_roc),paste("(",round(ci.auc(model_roc)[1],3),"-",round(ci.auc(model_roc)[3],3),")",sep = ""),
                      values["sensitivity"],paste("(",round(values2$sensitivity[,1],3),"-",round(values2$sensitivity[,3],3),")",sep = ""),
                      values["specificity"],paste("(",round(values2$specificity[,1],3),"-",round(values2$specificity[,3],3),")",sep = ""))
  }}
  return(return_values)
}

AutoScore_testing_imbalance <- function(test_set, final_variables, cut_vec, scoring_table) {
  # prepare testset: categorization and "auto_test"
  test_set_1 <- test_set[, c(final_variables, "label")]
  test_set_2 <- transform_df_fixed(test_set_1, cut_vec = cut_vec)
  test_set_3 <- auto_test(test_set_2, scoring_table)
  test_set_3$total_score <- rowSums(subset(test_set_3, select = -label))
  test_set_3$total_score[which(is.na(test_set_3$total_score))]<-0
  y_test <- test_set_3$label
  
  # Final evaluation based on testing set
  plot_roc_curve(test_set_3$total_score, as.numeric(y_test) - 1)
  cat("***Performance using AutoScore (based on unseen test Set):\n")
  model_roc <- roc(y_test, test_set_3$total_score, quiet = T)
  print_roc_performance(model_roc)

  threshold<-coords(model_roc, "best", ret = "threshold", transpose = TRUE)
    #Modelprc <- pr.curve(test_set_3$total_score[which(y_test == 1)],test_set_3$total_score[which(y_test == 0)],curve = TRUE)
  label_pred<-c(rep(0,length(test_set_3$total_score)))
  label_pred[which(test_set_3$total_score>=threshold)]<-1
  mcc_values<-mcc(preds = label_pred, actuals = y_test)  
  values<-coords(model_roc, "best", ret = c("specificity", "sensitivity","accuracy","npv","ppv"), transpose = TRUE)
    values2<-ci.coords(model_roc,x=threshold,input = "threshold", ret = c("specificity", "sensitivity","accuracy","npv","ppv"))
    return_values<-list(threshold,round(auc(model_roc),3),paste("(",round(ci.auc(model_roc)[1],3),"-",round(ci.auc(model_roc)[3],3),")",sep = ""),
                        round(values["sensitivity"],3),paste("(",round(values2$sensitivity[,1],3),"-",round(values2$sensitivity[,3],3),")",sep = ""),
                        round(values["specificity"],3),paste("(",round(values2$specificity[,1],3),"-",round(values2$specificity[,3],3),")",sep = ""),
                        round(values["accuracy"],3),paste("(",round(values2$accuracy[,1],3),"-",round(values2$accuracy[,3],3),")",sep = ""),
                        round(values["npv"],3),paste("(",round(values2$npv[,1],3),"-",round(values2$npv[,3],3),")",sep = ""),
                        round(values["ppv"],3),paste("(",round(values2$ppv[,1],3),"-",round(values2$ppv[,3],3),")",sep = ""),round(mcc_values,3))
  return(return_values)
}

AutoScore_parsimony_plot <- function(train_set, validation_set, rank, max_score = 100, n_min = 1, n_max = 20, 
                                split = "quantile", quantiles = c(0, 0.05, 0.2, 0.8, 0.95, 1), do_trace = FALSE) {
  # if Cross validation is FALSE

    AUC <- c()
    
    # Go through AUtoScore Module 2/3/4 in the loop
    for (i in n_min:n_max) {
      cat(paste("Select the number of Variables",i,":  "))
      
      variable_list<-names(rank)[1:i]
      train_set_1 <- train_set[, c(variable_list, "label")]
      validation_set_1 <- validation_set[, c(variable_list, "label")]
      model_roc<-compute_auc_val(train_set_1, validation_set_1,variable_list, split, quantiles, max_score)
      print(auc(model_roc))
      AUC <- c(AUC, auc(model_roc))
    }
    par(mar=c(5,5,2,2))
    names(AUC) <- n_min:n_max
    #cat("list of AUC values are shown below")
    #print(data.frame(AUC))
    plot(AUC, xlab = "Number of Variables", ylab = "Area Under the Curve", cex.axis=1.5, cex.lab=2, col = "blue",
         lwd = 2, type = "o")
    
    return(AUC)

}
