## "Interpretable Machine Learning-Based Risk Scoring with Individual and Ensemble Model Selection for Clinical Decision Making"
### Overview
Clinical scores are highly interpretable and widely used in clinical risk stratification.
AutoScore was previously developed as a clinical score generator, integrating the interpretability of clinical scores and the discriminability of machine learning (ML).
Although a basic framework has been established, AutoScore still leaves room for enhancement: variable ranking via the random forest and manual model selection.
In this respository, we improved them with additional variable ranking methods.
### R script description
#### Two-sample Student’s t test and Wilcoxon's rank sum test
Statistical tests such as the two-sample Student’s t test [1] and Wilcoxon's rank sum test [2] are classic methods for 
continuous variable ranking and the univariable p-values derived from them are often employed to rank variables in medicine [3, 4]. 
Since all variables in our dataset are continuous, t test and Wilcoxon's rank sum test work well. However, t test cannot be applied to 
datasets that contain both continuous and categorical variables. Users should therefore avoid using this ranking method on mixed-type 
datasets. Variables with small p-values suggest significant differences between outcomes and are therefore assumed to have top rankings 
in the variable ranking module and great discriminatory power for the scoring model.
#### Likelihood increment and deletion
We also explore two stepwise regression-based ranking: increment and deletion algorithms based on log-likelihood. 
The likelihood deletion algorithm begins by including all variables into a saturated model and then eliminates the most significant 
variables in a stepwise manner. In the i^th step, the eliminated variable with the greatest likelihood of decreasing will be determined 
as the i^th important variable. Similarly, likelihood increment algorithm constructs a model with a single variable first, then adds 
a variable at a time. The variable that results in the greatest likelihood increases in the i^th step will be considered as the i^th 
important variable. 
#### XGBoost
XGBoost is a scalable boosting system that provides linear and tree-based modeling. In linear booster models, 
the variable importance is determined by the absolute magnitude of linear coefficients [5]. In tree booster models, 
the variable importance is calculated through the percentage of gain, which measures the fractional contribution of each 
variable to the model, which optimizes the total gain of the variable’s splits [5]. We included both types for variable ranking in AutoScore.
#### Majority Voting
While individual ranking methods in conjunction with “human observation” approach provide flexibility in model 
selection, ensemble learning may provide an “automated” solution that could not only remove bias introduced by individual methods, 
but also improve the robustness of the model [6]. As part of our study, we propose using majority voting, an ensemble learning 
strategy, to aggregate the outputs of individual ranking methods for model selection.

Figure 1 illustrates how a decision ensemble is constructed based on nine individual variable ranking methods. 
Using the "thresholding" strategy described in the previous section, we can identify the most significant variables for each 
ranking method. Several methods may select similar sets of variables (such as Methods 2 and 8) while others could produce 
quite different results (such as Methods 6 and 7). For each variable, we count its occurrences among the nine methods. The 
frequency of occurrences indicates the importance of the variable. According to a majority voting strategy, variables that are 
selected by at least five out of nine ranking methods will be kept. In the example shown in Figure 3, variables #1-4 are selected 
to construct a prediction model.

![image](https://github.com/Han-Yuan-Med/comparison/blob/main/ensemble.jpg)
Figure 1: A hypothetical example to illustrate the mechanism of majority voting-based ensemble learning for model selection.

#### References
[1] Student: The Probable Error of a Mean. Biometrika 1908, 6(1):1-25.  
[2] Wilcoxon F: Probability Tables for Individual Comparisons by Ranking Methods. Biometrics 1947, 3(3):119-122.  
[3] Padma Shri TK, Sriraam N: Comparison of t-test ranking with PCA and SEPCOR feature selection for wake and stage 1 sleep pattern 
recognition in multichannel electroencephalograms. Biomedical Signal Processing and Control 2017, 31:499-512.  
[4] Saha S, Seal DB, Ghosh A, Dey KN: A novel gene ranking method using Wilcoxon rank sum test and genetic algorithm. 
International Journal of Bioinformatics Research and Applications 2016, 12(3):263-279.  
[5] Chen T, Guestrin C: XGBoost: A Scalable Tree Boosting System. In: Proceedings of the ACM SIGKDD International Conference 
on Knowledge Discovery and Data Mining 2016: 785–794.  
[6] Bhowan U, Johnston M, Zhang M, Yao X: Evolving Diverse Ensembles Using Genetic Programming for Classification With Unbalanced Data. 
IEEE Transactions on Evolutionary Computation 2013, 17(3):368-386.
