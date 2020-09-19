# To import libraries
import numpy as np # for computation
import pandas as pd # for data manipulation
import matplotlib.pyplot as plt # for visualization

dataset = pd.read_csv('Texture-Features.csv')

# Preliminary analysis of dataset
# A. To know if there is missing data
dataset.isnull().sum().sort_values(ascending=False)

# To check column names and total records
dataset_count = dataset.count()

# C. To view the info about the dataset 
print(dataset.info())

# D. To view statistical salary of the dataset
dataset_statistics = dataset.iloc[:,2:10]
statistics = dataset_statistics.describe() #one-way of checking for outliers

# To set independent variables from feature selection

#features_uni = ['Variance_H4','Entropy','Energy','Info_meas_corr2_H13']
features_rec = ['Variance_H4','Entropy','Homogeneity','Info_meas_corr2_H13']
#features_fi = ['Biomass Perimeter','Biomass Area','Biomass Major Axis Length','No of Leaves (Segmentation)']

#features = ['Contrast', 'Correlation', 'Energy', 'Homogeneity', 'Entropy', 'Variance_H4', 'Info_meas_corr1_H12', 'Info_meas_corr2_H13']

# To create the matrix of independent variable,
X = dataset[features_rec].values

# To create the matrix of independent variable, x with feature selection
# X = dataset[features_fi].values

# To create the matrix of dependent variable, y
Y = dataset.iloc[:,0].values

# To encode the categorical data (Country) in the dependent variable, Y
from sklearn.preprocessing import LabelEncoder
label_encoder_Y = LabelEncoder()
Y = label_encoder_Y.fit_transform(Y)

# To split the whole dataset into training dataset and testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0) #train_size=0.8, you can either still put this or not since test_size is already defined. By default, remaining is for training

# To perform feature scaling 
# A. For standardization feature scaling
from sklearn.preprocessing import StandardScaler # for not normally distributed samples
standard_scaler = StandardScaler ()
X_train_standard = X_train.copy()
X_test_standard = X_test.copy()
X_train_standard = standard_scaler.fit_transform(X_train_standard) # X_train_standard[:,3:5] -> for specifying features to be scaled (age and salary)
X_test_standard = standard_scaler.fit_transform(X_test_standard)  # X_test_standard[:,3:5] -> for specifying features to be scaled (age and salary)

# To view the scatterplot of our dataset
import seaborn as sns
sns.pairplot(dataset)

# To determine the Pearson's Coefficient of Correlation for the whole dataset
# Hindi dapat talaga ito i-perform sa classification dahil discrete ang output. Hindi scatter. Para lang kay regression
dataset_correlation = dataset.corr()
##plt.figure(figsize=(3,3))
sns.heatmap(dataset_correlation, annot=True, linewidths=3)
############################################## DECISION TREE CLASSIFIER #######################################################


# To fit the training dataset into a Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(X_train_standard,Y_train)

# To predict the output of the testing dataset
Y_predict_dtc_def = dtc.predict_proba(X_test_standard) #predict proba is in terms of probability kaya continuous ang output. Tanggalin dapat para categorical
Y_predict_dtc_def = dtc.predict(X_test_standard)

# To show the confusion matrix.
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_predict_dtc_def)

import seaborn as sns
sns.heatmap(confusion_matrix, annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')

TP = confusion_matrix[1,1]
TN = confusion_matrix[0,0]
FP = confusion_matrix[0,1]
FN = confusion_matrix[1,0]

# To apply K-fold cross-validation for the logistic regression model
from sklearn.model_selection import KFold
k_fold = KFold(n_splits=10, shuffle=True, random_state=0) # shuffle = True para sa small dataset

from sklearn.model_selection import cross_val_score

# For the accuracy as scoring for for cross-validation 
accuracies = cross_val_score(estimator=dtc, X=X_train_standard, y=Y_train, cv=k_fold, scoring='accuracy')

accuracies_average = accuracies.mean()
accuracies_variance = accuracies.std()

print('Accuracy of K-FOLDS:')
print (accuracies)
print(' ')
print('Average Accuracy of K-FOLDS:')
print(accuracies_average)
print(' ')
print('Accuracy Variance of K-FOLDS:')
print(accuracies_variance)
print(' ')

# For the F1 as scoring for for cross-validation 
F1 = cross_val_score(estimator=dtc, X=X_train_standard, y=Y_train, cv=k_fold, scoring='f1_weighted')

F1_average = F1.mean()
F1_variance = F1.std()

print('F1 of K-FOLDS:')
print (F1)
print(' ')
print('Average F1 of K-FOLDS:')
print(F1_average)
print(' ')
print('F1 Variance of K-FOLDS:')
print(F1_variance)
print(' ')

# To visualize DTC, 

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(dtc, 
                     out_file=dot_data,  
                     filled=True, rounded=True,
                     special_characters=True,feature_names = features_rec,class_names=['2','1', '0'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Lettuce__RFE_DTC_Def.png')
Image(graph.create_png())

# To evaluate the performance of the Decision Tree Classifier using holdout
# A. For the Classification Accuracy
from sklearn.metrics import accuracy_score
classification_accuracy = accuracy_score(Y_test, Y_predict_dtc_def)
print('Classification Accuracy: %.4f'
      % classification_accuracy)
print(' ')

# B. For the Classification Error
from sklearn.metrics import accuracy_score
classification_error = 1-classification_accuracy
print('Classification Error: %.4f'
      % classification_error)
print(' ')

# C. For the Sensitivity or Recall Score / True Positive Rate (Kung posotive ang hinahanap, gano kadalas ang positive) Dapat same ang performance ng predicting ng + and - para walang bias sa isa
# True Positive Rate:  Actual Value +, How often Correct
from sklearn.metrics import recall_score
sensitivity = recall_score(Y_test, Y_predict_dtc_def, pos_label = 'positive', average = 'weighted')
print('Sensitivity or Recall Score: %.4f'
      % sensitivity)
print(' ')

# D. For the Specficity (kung ang actual value is negative, gaano kadalas negative. Counterpart ni sensitivity)
# True Negative Rate: Actual Value -, How often Correct
specificity = TN/(TN+FP)
print('Specificity: %.4f'
      % specificity)
print(' ') 

# yung result bias kay negative compared to positive

# E. For the FP rate .
# False Positive Rate: Actual Value -, How often Inorrect
false_positve_rate = 1-specificity
print('False Positive Rate: %.4f'
      % false_positve_rate)
print(' ')  

# F. For the precision.
# False Negative Rate: Predicted Value +, How often the prediction is Correct
from sklearn.metrics import precision_score
precision = precision_score(Y_test, Y_predict_dtc_def, pos_label = 'positive', average = 'weighted')
print('Precision: %.4f'
      % precision)
print(' ')  

#Wag ma amaze sa accuracy 

# G. For the F1 score. Relating precision and sensitivity
# False Negative Rate: Predicted Value and Actual Value+, How often the prediction is Correct relation
from sklearn.metrics import f1_score
f1_score = f1_score(Y_test, Y_predict_dtc_def, pos_label = 'positive', average = 'weighted')
print('F1 Score: %.4f'
      % f1_score)
print(' ')  

# H. For classification report
from sklearn.metrics import classification_report
classification_report = classification_report(Y_test, Y_predict_dtc_def)


######################################## DECISION TREE CLASSIFIER OPTIMIZATION #############################################################

# To Import the kFold Class
from sklearn.model_selection import StratifiedKFold
k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0) 

# To Import the GridSearch Class
from sklearn.model_selection import GridSearchCV

Criterion = ['gini','entropy']
Splitter = ['best','random']
Max_Depth = list(range(1,10))
# To Set Parameters to be Optimized Under the Decision Tree Classifier Model
parameters = dict(criterion=Criterion, splitter=Splitter, max_depth=Max_Depth)   
                                                                                                                                                 
grid_search = GridSearchCV(estimator = dtc,
              param_grid = parameters,
              scoring = 'accuracy',
              cv = k_fold,
              n_jobs = -1)
grid_search = grid_search.fit(X,Y)
print(grid_search)

# To View the Results of the GridSearch
pd.DataFrame(grid_search.cv_results_)[['mean_test_score','std_test_score','params']]

# To Identify the Best Accuracy and Best Parameters
best_accuracy =grid_search.best_score_
best_parameters = grid_search.best_params_

print("Best Accuracy Score:")
print(best_accuracy)
print('')

print("Best Parameters Score:")
print(best_parameters)
print('')

#dtc_opt = DecisionTreeClassifier(max_depth=5, criterion='entropy', splitter = 'best') #all
dtc_opt = DecisionTreeClassifier(max_depth=5, criterion='gini', splitter = 'best') #-> UNI and RFE

dtc_opt.fit(X_train_standard,Y_train)

# To Predict the Output of the Whole Dataset
Y_predict_dtc_opt = dtc_opt.predict(X_test_standard)

# To Show the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_predict_dtc_opt)

import seaborn as sns
sns.heatmap(confusion_matrix, annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')

TP = confusion_matrix[1,1]
TN = confusion_matrix[0,0]
FP = confusion_matrix[0,1]
FN = confusion_matrix[1,0]

# To apply K-fold cross-validation for the logistic regression model
from sklearn.model_selection import KFold
k_fold = KFold(n_splits=10, shuffle=True, random_state=0) # shuffle = True para sa small dataset

from sklearn.model_selection import cross_val_score

# For the accuracy as scoring for for cross-validation 
accuracies = cross_val_score(estimator=dtc_opt, X=X_train_standard, y=Y_train, cv=k_fold, scoring='accuracy')

accuracies_average = accuracies.mean()
accuracies_variance = accuracies.std()

print('Accuracy of K-FOLDS:')
print (accuracies)
print(' ')
print('Average Accuracy of K-FOLDS:')
print(accuracies_average)
print(' ')
print('Accuracy Variance of K-FOLDS:')
print(accuracies_variance)
print(' ')

# For the F1 as scoring for for cross-validation 
F1 = cross_val_score(estimator=dtc_opt, X=X_train_standard, y=Y_train, cv=k_fold, scoring='f1_weighted')

F1_average = F1.mean()
F1_variance = F1.std()

print('F1 of K-FOLDS:')
print (F1)
print(' ')
print('Average F1 of K-FOLDS:')
print(F1_average)
print(' ')
print('F1 Variance of K-FOLDS:')
print(F1_variance)
print(' ')

# To visualize DTC, 

# To visualize DTC, 

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(dtc_opt, 
                     out_file=dot_data,  
                     filled=True, rounded=True,
                     special_characters=True,feature_names = features_rec,class_names=['2','1', '0'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Lettuce__RFE_DTC_Opt.png')
Image(graph.create_png())

# To evaluate the performance of the Decision Tree Classifier using holdout
# A. For the Classification Accuracy
from sklearn.metrics import accuracy_score
classification_accuracy = accuracy_score(Y_test, Y_predict_dtc_opt)
print('Classification Accuracy: %.4f'
      % classification_accuracy)
print(' ')

# B. For the Classification Error
from sklearn.metrics import accuracy_score
classification_error = 1-classification_accuracy
print('Classification Error: %.4f'
      % classification_error)
print(' ')

# C. For the Sensitivity or Recall Score / True Positive Rate (Kung posotive ang hinahanap, gano kadalas ang positive) Dapat same ang performance ng predicting ng + and - para walang bias sa isa
# True Positive Rate:  Actual Value +, How often Correct
from sklearn.metrics import recall_score
sensitivity = recall_score(Y_test, Y_predict_dtc_opt, pos_label = 'positive', average = 'weighted')
print('Sensitivity or Recall Score: %.4f'
      % sensitivity)
print(' ')

# D. For the Specficity (kung ang actual value is negative, gaano kadalas negative. Counterpart ni sensitivity)
# True Negative Rate: Actual Value -, How often Correct
specificity = TN/(TN+FP)
print('Specificity: %.4f'
      % specificity)
print(' ') 

# yung result bias kay negative compared to positive

# E. For the FP rate .
# False Positive Rate: Actual Value -, How often Inorrect
false_positve_rate = 1-specificity
print('False Positive Rate: %.4f'
      % false_positve_rate)
print(' ')  

# F. For the precision.
# False Negative Rate: Predicted Value +, How often the prediction is Correct
from sklearn.metrics import precision_score
precision = precision_score(Y_test, Y_predict_dtc_opt, pos_label = 'positive', average = 'weighted')
print('Precision: %.4f'
      % precision)
print(' ')  

#Wag ma amaze sa accuracy 

# G. For the F1 score. Relating precision and sensitivity
# False Negative Rate: Predicted Value and Actual Value+, How often the prediction is Correct relation
from sklearn.metrics import f1_score
f1_score = f1_score(Y_test, Y_predict_dtc_opt, pos_label = 'positive', average = 'weighted')
print('F1 Score: %.4f'
      % f1_score)
print(' ')  

# H. For classification report
from sklearn.metrics import classification_report
classification_report = classification_report(Y_test, Y_predict_dtc_opt)


############################################## NAIVE BAYES #######################################################

# To fit the training dataset into a Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train_standard,Y_train)

# To predict the output of the testing dataset
Y_predict_nb_def = nb.predict_proba(X_test_standard) #predict proba is in terms of probability kaya continuous ang output. Tanggalin dapat para categorical
Y_predict_nb_def = nb.predict(X_test_standard)

# To show the confusion matrix.
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_predict_nb_def)

import seaborn as sns
sns.heatmap(confusion_matrix, annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')

TP = confusion_matrix[1,1]
TN = confusion_matrix[0,0]
FP = confusion_matrix[0,1]
FN = confusion_matrix[1,0]

# To apply K-fold cross-validation for the logistic regression model
from sklearn.model_selection import KFold
k_fold = KFold(n_splits=10, shuffle=True, random_state=0) # shuffle = True para sa small dataset

from sklearn.model_selection import cross_val_score

# For the accuracy as scoring for for cross-validation 
accuracies = cross_val_score(estimator=nb, X=X_train_standard, y=Y_train, cv=k_fold, scoring='accuracy')

accuracies_average = accuracies.mean()
accuracies_variance = accuracies.std()

print('Accuracy of K-FOLDS:')
print (accuracies)
print(' ')
print('Average Accuracy of K-FOLDS:')
print(accuracies_average)
print(' ')
print('Accuracy Variance of K-FOLDS:')
print(accuracies_variance)
print(' ')

# For the F1 as scoring for for cross-validation 
F1 = cross_val_score(estimator=nb, X=X_train_standard, y=Y_train, cv=k_fold, scoring='f1_weighted')

F1_average = F1.mean()
F1_variance = F1.std()

print('F1 of K-FOLDS:')
print (F1)
print(' ')
print('Average F1 of K-FOLDS:')
print(F1_average)
print(' ')
print('F1 Variance of K-FOLDS:')
print(F1_variance)
print(' ')

# To visualize DTC, 

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(nb, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = features,class_names=['2','1', '0'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Lettuce__ALL_NB_Def.png')
Image(graph.create_png())

# To evaluate the performance of the Decision Tree Classifier using holdout
# A. For the Classification Accuracy
from sklearn.metrics import accuracy_score
classification_accuracy = accuracy_score(Y_test, Y_predict_nb_def)
print('Classification Accuracy: %.4f'
      % classification_accuracy)
print(' ')

# B. For the Classification Error
from sklearn.metrics import accuracy_score
classification_error = 1-classification_accuracy
print('Classification Error: %.4f'
      % classification_error)
print(' ')

# C. For the Sensitivity or Recall Score / True Positive Rate (Kung posotive ang hinahanap, gano kadalas ang positive) Dapat same ang performance ng predicting ng + and - para walang bias sa isa
# True Positive Rate:  Actual Value +, How often Correct
from sklearn.metrics import recall_score
sensitivity = recall_score(Y_test, Y_predict_nb_def, pos_label = 'positive', average = 'weighted')
print('Sensitivity or Recall Score: %.4f'
      % sensitivity)
print(' ')

# D. For the Specficity (kung ang actual value is negative, gaano kadalas negative. Counterpart ni sensitivity)
# True Negative Rate: Actual Value -, How often Correct
specificity = TN/(TN+FP)
print('Specificity: %.4f'
      % specificity)
print(' ') 

# yung result bias kay negative compared to positive

# E. For the FP rate .
# False Positive Rate: Actual Value -, How often Inorrect
false_positve_rate = 1-specificity
print('False Positive Rate: %.4f'
      % false_positve_rate)
print(' ')  

# F. For the precision.
# False Negative Rate: Predicted Value +, How often the prediction is Correct
from sklearn.metrics import precision_score
precision = precision_score(Y_test, Y_predict_nb_def, pos_label = 'positive', average = 'weighted')
print('Precision: %.4f'
      % precision)
print(' ')  

#Wag ma amaze sa accuracy 

# G. For the F1 score. Relating precision and sensitivity
# False Negative Rate: Predicted Value and Actual Value+, How often the prediction is Correct relation
from sklearn.metrics import f1_score
f1_score = f1_score(Y_test, Y_predict_nb_def, pos_label = 'positive', average = 'weighted')
print('F1 Score: %.4f'
      % f1_score)
print(' ')  

# H. For classification report
from sklearn.metrics import classification_report
classification_report = classification_report(Y_test, Y_predict_nb_def)

######################################## NAIVE BAYES OPTIMIZATION #############################################################

# To Import the kFold Class
from sklearn.model_selection import StratifiedKFold
k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0) 

# To Import the GridSearch Class
from sklearn.model_selection import GridSearchCV

Priors = ['array-like','shape']
# To Set Parameters to be Optimized Under the Decision Tree Classifier Model
parameters = dict(priors=Priors)   
                                                                                                                                                 
grid_search = GridSearchCV(estimator = nb,
              param_grid = parameters,
              scoring = 'accuracy',
              cv = k_fold,
              n_jobs = -1)
grid_search = grid_search.fit(X,Y)
print(grid_search)

# To View the Results of the GridSearch
pd.DataFrame(grid_search.cv_results_)[['mean_test_score','std_test_score','params']]

# To Identify the Best Accuracy and Best Parameters
best_accuracy =grid_search.best_score_
best_parameters = grid_search.best_params_

print("Best Accuracy Score:")
print(best_accuracy)
print('')

print("Best Parameters Score:")
print(best_parameters)
print('')

nb_opt = DecisionTreeClassifier(max_depth=5, criterion='entropy', splitter = 'best')

nb_opt.fit(X_train_standard,Y_train)

# To Predict the Output of the Whole Dataset
Y_predict_nb_opt = dtc_opt.predict(X_test_standard)

# To Show the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_predict_nb)

import seaborn as sns
sns.heatmap(confusion_matrix, annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')

TP = confusion_matrix[1,1]
TN = confusion_matrix[0,0]
FP = confusion_matrix[0,1]
FN = confusion_matrix[1,0]

# To apply K-fold cross-validation for the logistic regression model
from sklearn.model_selection import KFold
k_fold = KFold(n_splits=10, shuffle=True, random_state=0) # shuffle = True para sa small dataset

from sklearn.model_selection import cross_val_score

# For the accuracy as scoring for for cross-validation 
accuracies = cross_val_score(estimator=nb_opt, X=X_train_standard, y=Y_train, cv=k_fold, scoring='accuracy')

accuracies_average = accuracies.mean()
accuracies_variance = accuracies.std()

print('Accuracy of K-FOLDS:')
print (accuracies)
print(' ')
print('Average Accuracy of K-FOLDS:')
print(accuracies_average)
print(' ')
print('Accuracy Variance of K-FOLDS:')
print(accuracies_variance)
print(' ')

# For the F1 as scoring for for cross-validation 
F1 = cross_val_score(estimator=nb_opt, X=X_train_standard, y=Y_train, cv=k_fold, scoring='f1_weighted')

F1_average = F1.mean()
F1_variance = F1.std()

print('F1 of K-FOLDS:')
print (F1)
print(' ')
print('Average F1 of K-FOLDS:')
print(F1_average)
print(' ')
print('F1 Variance of K-FOLDS:')
print(F1_variance)
print(' ')

# To visualize DTC, 

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(nb_opt, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = features,class_names=['2','1', '0'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Lettuce_ALL_DTC_Opt.png')
Image(graph.create_png())

# To evaluate the performance of the Decision Tree Classifier using holdout
# A. For the Classification Accuracy
from sklearn.metrics import accuracy_score
classification_accuracy = accuracy_score(Y_test, Y_predict_nb_opt)
print('Classification Accuracy: %.4f'
      % classification_accuracy)
print(' ')

# B. For the Classification Error
from sklearn.metrics import accuracy_score
classification_error = 1-classification_accuracy
print('Classification Error: %.4f'
      % classification_error)
print(' ')

# C. For the Sensitivity or Recall Score / True Positive Rate (Kung posotive ang hinahanap, gano kadalas ang positive) Dapat same ang performance ng predicting ng + and - para walang bias sa isa
# True Positive Rate:  Actual Value +, How often Correct
from sklearn.metrics import recall_score
sensitivity = recall_score(Y_test, Y_predict_nb_opt, pos_label = 'positive', average = 'weighted')
print('Sensitivity or Recall Score: %.4f'
      % sensitivity)
print(' ')

# D. For the Specficity (kung ang actual value is negative, gaano kadalas negative. Counterpart ni sensitivity)
# True Negative Rate: Actual Value -, How often Correct
specificity = TN/(TN+FP)
print('Specificity: %.4f'
      % specificity)
print(' ') 

# yung result bias kay negative compared to positive

# E. For the FP rate .
# False Positive Rate: Actual Value -, How often Inorrect
false_positve_rate = 1-specificity
print('False Positive Rate: %.4f'
      % false_positve_rate)
print(' ')  

# F. For the precision.
# False Negative Rate: Predicted Value +, How often the prediction is Correct
from sklearn.metrics import precision_score
precision = precision_score(Y_test, Y_predict_nb_opt, pos_label = 'positive', average = 'weighted')
print('Precision: %.4f'
      % precision)
print(' ')  

#Wag ma amaze sa accuracy 

# G. For the F1 score. Relating precision and sensitivity
# False Negative Rate: Predicted Value and Actual Value+, How often the prediction is Correct relation
from sklearn.metrics import f1_score
f1_score = f1_score(Y_test, Y_predict_dtc_opt, pos_label = 'positive', average = 'weighted')
print('F1 Score: %.4f'
      % f1_score)
print(' ')  

# H. For classification report
from sklearn.metrics import classification_report
classification_report = classification_report(Y_test, Y_predict_nb_opt)


############################################## STOCHASTIC GRADIENT DESCENT #######################################################


# To fit the training dataset into a Decision Tree Classifier
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(random_state=0)
sgd.fit(X_train_standard,Y_train)

# To predict the output of the testing dataset
#Y_predict_sgd_def = sgd.predict_proba(X_test_standard) #predict proba is in terms of probability kaya continuous ang output. Tanggalin dapat para categorical
Y_predict_sgd_def = sgd.predict(X_test_standard)

# To show the confusion matrix.
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_predict_sgd_def)

import seaborn as sns
sns.heatmap(confusion_matrix, annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')

TP = confusion_matrix[1,1]
TN = confusion_matrix[0,0]
FP = confusion_matrix[0,1]
FN = confusion_matrix[1,0]

# To apply K-fold cross-validation for the logistic regression model
from sklearn.model_selection import KFold
k_fold = KFold(n_splits=10, shuffle=True, random_state=0) # shuffle = True para sa small dataset

from sklearn.model_selection import cross_val_score

# For the accuracy as scoring for for cross-validation 
accuracies = cross_val_score(estimator=sgd, X=X_train_standard, y=Y_train, cv=k_fold, scoring='accuracy')

accuracies_average = accuracies.mean()
accuracies_variance = accuracies.std()

print('Accuracy of K-FOLDS:')
print (accuracies)
print(' ')
print('Average Accuracy of K-FOLDS:')
print(accuracies_average)
print(' ')
print('Accuracy Variance of K-FOLDS:')
print(accuracies_variance)
print(' ')

# For the F1 as scoring for for cross-validation 
F1 = cross_val_score(estimator=sgd, X=X_train_standard, y=Y_train, cv=k_fold, scoring='f1_weighted')

F1_average = F1.mean()
F1_variance = F1.std()

print('F1 of K-FOLDS:')
print (F1)
print(' ')
print('Average F1 of K-FOLDS:')
print(F1_average)
print(' ')
print('F1 Variance of K-FOLDS:')
print(F1_variance)
print(' ')

# To visualize DTC, 

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(dtc, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = features,class_names=['2','1', '0'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Lettuce__ALL_DTC_Def.png')
Image(graph.create_png())

# To evaluate the performance of the Decision Tree Classifier using holdout
# A. For the Classification Accuracy
from sklearn.metrics import accuracy_score
classification_accuracy = accuracy_score(Y_test, Y_predict_sgd_def)
print('Classification Accuracy: %.4f'
      % classification_accuracy)
print(' ')

# B. For the Classification Error
from sklearn.metrics import accuracy_score
classification_error = 1-classification_accuracy
print('Classification Error: %.4f'
      % classification_error)
print(' ')

# C. For the Sensitivity or Recall Score / True Positive Rate (Kung posotive ang hinahanap, gano kadalas ang positive) Dapat same ang performance ng predicting ng + and - para walang bias sa isa
# True Positive Rate:  Actual Value +, How often Correct
from sklearn.metrics import recall_score
sensitivity = recall_score(Y_test, Y_predict_sgd_def, pos_label = 'positive', average = 'weighted')
print('Sensitivity or Recall Score: %.4f'
      % sensitivity)
print(' ')

# D. For the Specficity (kung ang actual value is negative, gaano kadalas negative. Counterpart ni sensitivity)
# True Negative Rate: Actual Value -, How often Correct
specificity = TN/(TN+FP)
print('Specificity: %.4f'
      % specificity)
print(' ') 

# yung result bias kay negative compared to positive

# E. For the FP rate .
# False Positive Rate: Actual Value -, How often Inorrect
false_positve_rate = 1-specificity
print('False Positive Rate: %.4f'
      % false_positve_rate)
print(' ')  

# F. For the precision.
# False Negative Rate: Predicted Value +, How often the prediction is Correct
from sklearn.metrics import precision_score
precision = precision_score(Y_test, Y_predict_sgd_def, pos_label = 'positive', average = 'weighted')
print('Precision: %.4f'
      % precision)
print(' ')  

#Wag ma amaze sa accuracy 

# G. For the F1 score. Relating precision and sensitivity
# False Negative Rate: Predicted Value and Actual Value+, How often the prediction is Correct relation
from sklearn.metrics import f1_score
f1_score = f1_score(Y_test, Y_predict_sgd_def, pos_label = 'positive', average = 'weighted')
print('F1 Score: %.4f'
      % f1_score)
print(' ')  

# H. For classification report
from sklearn.metrics import classification_report
classification_report = classification_report(Y_test, Y_predict_sgd_def)


######################################## STOCHASTIC GRADIENT DESCENT OPTIMIZATION #############################################################

# To Import the kFold Class
from sklearn.model_selection import StratifiedKFold
k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0) 

# To Import the GridSearch Class
from sklearn.model_selection import GridSearchCV

Alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10]
Loss = ['hinge','log','modified_huber', 'squared_hinge', 'perceptron']
Penalty = ['l2','l1','elasticnet']
L1_Ratio = [0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1]
# To Set Parameters to be Optimized Under the Decision Tree Classifier Model
parameters = dict(alpha=Alpha, loss=Loss, penalty=Penalty, l1_ratio=L1_Ratio)   
                                                                                                                                                 
grid_search = GridSearchCV(estimator = sgd,
              param_grid = parameters,
              scoring = 'accuracy',
              cv = k_fold,
              n_jobs = -1)
grid_search = grid_search.fit(X,Y)
print(grid_search)

# To View the Results of the GridSearch
pd.DataFrame(grid_search.cv_results_)[['mean_test_score','std_test_score','params']]

# To Identify the Best Accuracy and Best Parameters
best_accuracy =grid_search.best_score_
best_parameters = grid_search.best_params_

print("Best Accuracy Score:")
print(best_accuracy)
print('')

print("Best Parameters Score:")
print(best_parameters)
print('')

#sgd_opt = SGDClassifier(alpha=0.0001, loss='hinge', penalty='l2', l1_ratio=0) #all
sgd_opt = SGDClassifier(alpha=0.0001, loss='log', penalty='elasticnet', l1_ratio=0.3) #uni and rfe

sgd_opt.fit(X_train_standard,Y_train)

# To Predict the Output of the Whole Dataset
Y_predict_sgd_opt = sgd_opt.predict(X_test_standard)

# To Show the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_predict_sgd_opt)

import seaborn as sns
sns.heatmap(confusion_matrix, annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')

TP = confusion_matrix[1,1]
TN = confusion_matrix[0,0]
FP = confusion_matrix[0,1]
FN = confusion_matrix[1,0]

# To apply K-fold cross-validation for the logistic regression model
from sklearn.model_selection import KFold
k_fold = KFold(n_splits=10, shuffle=True, random_state=0) # shuffle = True para sa small dataset

from sklearn.model_selection import cross_val_score

# For the accuracy as scoring for for cross-validation 
accuracies = cross_val_score(estimator=sgd_opt, X=X_train_standard, y=Y_train, cv=k_fold, scoring='accuracy')

accuracies_average = accuracies.mean()
accuracies_variance = accuracies.std()

print('Accuracy of K-FOLDS:')
print (accuracies)
print(' ')
print('Average Accuracy of K-FOLDS:')
print(accuracies_average)
print(' ')
print('Accuracy Variance of K-FOLDS:')
print(accuracies_variance)
print(' ')

# For the F1 as scoring for for cross-validation 
F1 = cross_val_score(estimator=sgd_opt, X=X_train_standard, y=Y_train, cv=k_fold, scoring='f1_weighted')

F1_average = F1.mean()
F1_variance = F1.std()

print('F1 of K-FOLDS:')
print (F1)
print(' ')
print('Average F1 of K-FOLDS:')
print(F1_average)
print(' ')
print('F1 Variance of K-FOLDS:')
print(F1_variance)
print(' ')

# To visualize DTC, 

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(dtc_opt, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = features,class_names=['2','1', '0'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Lettuce_ALL_DTC_Opt.png')
Image(graph.create_png())

# To evaluate the performance of the Decision Tree Classifier using holdout
# A. For the Classification Accuracy
from sklearn.metrics import accuracy_score
classification_accuracy = accuracy_score(Y_test, Y_predict_sgd_opt)
print('Classification Accuracy: %.4f'
      % classification_accuracy)
print(' ')

# B. For the Classification Error
from sklearn.metrics import accuracy_score
classification_error = 1-classification_accuracy
print('Classification Error: %.4f'
      % classification_error)
print(' ')

# C. For the Sensitivity or Recall Score / True Positive Rate (Kung posotive ang hinahanap, gano kadalas ang positive) Dapat same ang performance ng predicting ng + and - para walang bias sa isa
# True Positive Rate:  Actual Value +, How often Correct
from sklearn.metrics import recall_score
sensitivity = recall_score(Y_test, Y_predict_sgd_opt, pos_label = 'positive', average = 'weighted')
print('Sensitivity or Recall Score: %.4f'
      % sensitivity)
print(' ')

# D. For the Specficity (kung ang actual value is negative, gaano kadalas negative. Counterpart ni sensitivity)
# True Negative Rate: Actual Value -, How often Correct
specificity = TN/(TN+FP)
print('Specificity: %.4f'
      % specificity)
print(' ') 

# yung result bias kay negative compared to positive

# E. For the FP rate .
# False Positive Rate: Actual Value -, How often Inorrect
false_positve_rate = 1-specificity
print('False Positive Rate: %.4f'
      % false_positve_rate)
print(' ')  

# F. For the precision.
# False Negative Rate: Predicted Value +, How often the prediction is Correct
from sklearn.metrics import precision_score
precision = precision_score(Y_test, Y_predict_sgd_opt, pos_label = 'positive', average = 'weighted')
print('Precision: %.4f'
      % precision)
print(' ')  

#Wag ma amaze sa accuracy 

# G. For the F1 score. Relating precision and sensitivity
# False Negative Rate: Predicted Value and Actual Value+, How often the prediction is Correct relation
from sklearn.metrics import f1_score
f1_score = f1_score(Y_test, Y_predict_sgd_opt, pos_label = 'positive', average = 'weighted')
print('F1 Score: %.4f'
      % f1_score)
print(' ')  

# H. For classification report
from sklearn.metrics import classification_report
classification_report = classification_report(Y_test, Y_predict_sgd_opt)


########################################## LINEAR DISCRIMINANT ANALYSIS ###########################################################


# To fit the training dataset into a Decision Tree Classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_standard,Y_train)

# To predict the output of the testing dataset
Y_predict_lda_def = lda.predict_proba(X_test_standard) #predict proba is in terms of probability kaya continuous ang output. Tanggalin dapat para categorical
Y_predict_lda_def = lda.predict(X_test_standard)

# To show the confusion matrix.
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_predict_lda_def)

import seaborn as sns
sns.heatmap(confusion_matrix, annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')

TP = confusion_matrix[1,1]
TN = confusion_matrix[0,0]
FP = confusion_matrix[0,1]
FN = confusion_matrix[1,0]

# To apply K-fold cross-validation for the logistic regression model
from sklearn.model_selection import KFold
k_fold = KFold(n_splits=10, shuffle=True, random_state=0) # shuffle = True para sa small dataset

from sklearn.model_selection import cross_val_score

# For the accuracy as scoring for for cross-validation 
accuracies = cross_val_score(estimator=lda, X=X_train_standard, y=Y_train, cv=k_fold, scoring='accuracy')

accuracies_average = accuracies.mean()
accuracies_variance = accuracies.std()

print('Accuracy of K-FOLDS:')
print (accuracies)
print(' ')
print('Average Accuracy of K-FOLDS:')
print(accuracies_average)
print(' ')
print('Accuracy Variance of K-FOLDS:')
print(accuracies_variance)
print(' ')

# For the F1 as scoring for for cross-validation 
F1 = cross_val_score(estimator=lda, X=X_train_standard, y=Y_train, cv=k_fold, scoring='f1_weighted')

F1_average = F1.mean()
F1_variance = F1.std()

print('F1 of K-FOLDS:')
print (F1)
print(' ')
print('Average F1 of K-FOLDS:')
print(F1_average)
print(' ')
print('F1 Variance of K-FOLDS:')
print(F1_variance)
print(' ')

# To visualize DTC, 

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(dtc, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = features_rfe,class_names=['2','1', '0'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Lettuce__ALL_DTC_Def.png')
Image(graph.create_png())

# To evaluate the performance of the Decision Tree Classifier using holdout
# A. For the Classification Accuracy
from sklearn.metrics import accuracy_score
classification_accuracy = accuracy_score(Y_test, Y_predict_lda_def)
print('Classification Accuracy: %.4f'
      % classification_accuracy)
print(' ')

# B. For the Classification Error
from sklearn.metrics import accuracy_score
classification_error = 1-classification_accuracy
print('Classification Error: %.4f'
      % classification_error)
print(' ')

# C. For the Sensitivity or Recall Score / True Positive Rate (Kung posotive ang hinahanap, gano kadalas ang positive) Dapat same ang performance ng predicting ng + and - para walang bias sa isa
# True Positive Rate:  Actual Value +, How often Correct
from sklearn.metrics import recall_score
sensitivity = recall_score(Y_test, Y_predict_lda_def, pos_label = 'positive', average = 'weighted')
print('Sensitivity or Recall Score: %.4f'
      % sensitivity)
print(' ')

# D. For the Specficity (kung ang actual value is negative, gaano kadalas negative. Counterpart ni sensitivity)
# True Negative Rate: Actual Value -, How often Correct
specificity = TN/(TN+FP)
print('Specificity: %.4f'
      % specificity)
print(' ') 

# yung result bias kay negative compared to positive

# E. For the FP rate .
# False Positive Rate: Actual Value -, How often Inorrect
false_positve_rate = 1-specificity
print('False Positive Rate: %.4f'
      % false_positve_rate)
print(' ')  

# F. For the precision.
# False Negative Rate: Predicted Value +, How often the prediction is Correct
from sklearn.metrics import precision_score
precision = precision_score(Y_test, Y_predict_lda_def, pos_label = 'positive', average = 'weighted')
print('Precision: %.4f'
      % precision)
print(' ')  

#Wag ma amaze sa accuracy 

# G. For the F1 score. Relating precision and sensitivity
# False Negative Rate: Predicted Value and Actual Value+, How often the prediction is Correct relation
from sklearn.metrics import f1_score
f1_score = f1_score(Y_test, Y_predict_lda_def, pos_label = 'positive', average = 'weighted')
print('F1 Score: %.4f'
      % f1_score)
print(' ')  

# H. For classification report
from sklearn.metrics import classification_report
classification_report = classification_report(Y_test, Y_predict_lda_def)


######################################## LINEAR DISCRIMINANT ANALYSIS OPTIMIZATION #############################################################

# To Import the kFold Class
from sklearn.model_selection import StratifiedKFold
k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0) 

# To Import the GridSearch Class
from sklearn.model_selection import GridSearchCV

#N_Components = ['none','auto','float']
#Priors = ['hinge','log','modified_huber', 'squared_hinge', 'perceptron']
Shrinkage = ['none','auto',0.0001, 0.001, 0.01,]
Solver = ['svd','lsqr','eigen']
Tol = [0.0001, 0.001, 0.01, 0.1, 1, 10]
# To Set Parameters to be Optimized Under the Decision Tree Classifier Model
parameters = dict(solver=Solver, tol=Tol)   
                                                                                                                                                 
grid_search = GridSearchCV(estimator = lda,
              param_grid = parameters,
              scoring = 'accuracy',
              cv = k_fold,
              n_jobs = -1)
grid_search = grid_search.fit(X,Y)
print(grid_search)

# To View the Results of the GridSearch
pd.DataFrame(grid_search.cv_results_)[['mean_test_score','std_test_score','params']]

# To Identify the Best Accuracy and Best Parameters
best_accuracy =grid_search.best_score_
best_parameters = grid_search.best_params_

print("Best Accuracy Score:")
print(best_accuracy)
print('')

print("Best Parameters Score:")
print(best_parameters)
print('')

sgd_opt = SGDClassifier(alpha=0.0001, loss='hinge', penalty='l2', l1_ratio=0)

sgd_opt.fit(X_train_standard,Y_train)

# To Predict the Output of the Whole Dataset
Y_predict_sgd_opt = sgd_opt.predict(X_test_standard)

# To Show the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_predict_sgd_opt)

import seaborn as sns
sns.heatmap(confusion_matrix, annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')

TP = confusion_matrix[1,1]
TN = confusion_matrix[0,0]
FP = confusion_matrix[0,1]
FN = confusion_matrix[1,0]

# To apply K-fold cross-validation for the logistic regression model
from sklearn.model_selection import KFold
k_fold = KFold(n_splits=10, shuffle=True, random_state=0) # shuffle = True para sa small dataset

from sklearn.model_selection import cross_val_score

# For the accuracy as scoring for for cross-validation 
accuracies = cross_val_score(estimator=sgd_opt, X=X_train_standard, y=Y_train, cv=k_fold, scoring='accuracy')

accuracies_average = accuracies.mean()
accuracies_variance = accuracies.std()

print('Accuracy of K-FOLDS:')
print (accuracies)
print(' ')
print('Average Accuracy of K-FOLDS:')
print(accuracies_average)
print(' ')
print('Accuracy Variance of K-FOLDS:')
print(accuracies_variance)
print(' ')

# For the F1 as scoring for for cross-validation 
F1 = cross_val_score(estimator=sgd_opt, X=X_train_standard, y=Y_train, cv=k_fold, scoring='f1_weighted')

F1_average = F1.mean()
F1_variance = F1.std()

print('F1 of K-FOLDS:')
print (F1)
print(' ')
print('Average F1 of K-FOLDS:')
print(F1_average)
print(' ')
print('F1 Variance of K-FOLDS:')
print(F1_variance)
print(' ')

# To visualize DTC, 

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(dtc_opt, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = features,class_names=['2','1', '0'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Lettuce_ALL_DTC_Opt.png')
Image(graph.create_png())

# To evaluate the performance of the Decision Tree Classifier using holdout
# A. For the Classification Accuracy
from sklearn.metrics import accuracy_score
classification_accuracy = accuracy_score(Y_test, Y_predict_sgd_opt)
print('Classification Accuracy: %.4f'
      % classification_accuracy)
print(' ')

# B. For the Classification Error
from sklearn.metrics import accuracy_score
classification_error = 1-classification_accuracy
print('Classification Error: %.4f'
      % classification_error)
print(' ')

# C. For the Sensitivity or Recall Score / True Positive Rate (Kung posotive ang hinahanap, gano kadalas ang positive) Dapat same ang performance ng predicting ng + and - para walang bias sa isa
# True Positive Rate:  Actual Value +, How often Correct
from sklearn.metrics import recall_score
sensitivity = recall_score(Y_test, Y_predict_sgd_opt, pos_label = 'positive', average = 'weighted')
print('Sensitivity or Recall Score: %.4f'
      % sensitivity)
print(' ')

# D. For the Specficity (kung ang actual value is negative, gaano kadalas negative. Counterpart ni sensitivity)
# True Negative Rate: Actual Value -, How often Correct
specificity = TN/(TN+FP)
print('Specificity: %.4f'
      % specificity)
print(' ') 

# yung result bias kay negative compared to positive

# E. For the FP rate .
# False Positive Rate: Actual Value -, How often Inorrect
false_positve_rate = 1-specificity
print('False Positive Rate: %.4f'
      % false_positve_rate)
print(' ')  

# F. For the precision.
# False Negative Rate: Predicted Value +, How often the prediction is Correct
from sklearn.metrics import precision_score
precision = precision_score(Y_test, Y_predict_sgd_opt, pos_label = 'positive', average = 'weighted')
print('Precision: %.4f'
      % precision)
print(' ')  

# G. For the F1 score. Relating precision and sensitivity
# False Negative Rate: Predicted Value and Actual Value+, How often the prediction is Correct relation
from sklearn.metrics import f1_score
f1_score = f1_score(Y_test, Y_predict_sgd_opt, pos_label = 'positive', average = 'weighted')
print('F1 Score: %.4f'
      % f1_score)
print(' ')  

# H. For classification report
from sklearn.metrics import classification_report
classification_report = classification_report(Y_test, Y_predict_sgd_opt)


############################################## GRADIENT BOOSTING CLASSIFIER #######################################################


# To fit the training dataset into a Decision Tree Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(random_state=0)
gbc.fit(X_train_standard,Y_train)

# To predict the output of the testing dataset
Y_predict_gbc_def = gbc.predict_proba(X_test_standard) #predict proba is in terms of probability kaya continuous ang output. Tanggalin dapat para categorical
Y_predict_gbc_def = gbc.predict(X_test_standard)

# To show the confusion matrix.
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_predict_gbc_def)

import seaborn as sns
sns.heatmap(confusion_matrix, annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')

TP = confusion_matrix[1,1]
TN = confusion_matrix[0,0]
FP = confusion_matrix[0,1]
FN = confusion_matrix[1,0]

# To apply K-fold cross-validation for the logistic regression model
from sklearn.model_selection import KFold
k_fold = KFold(n_splits=10, shuffle=True, random_state=0) # shuffle = True para sa small dataset

from sklearn.model_selection import cross_val_score

# For the accuracy as scoring for for cross-validation 
accuracies = cross_val_score(estimator=gbc, X=X_train_standard, y=Y_train, cv=k_fold, scoring='accuracy')

accuracies_average = accuracies.mean()
accuracies_variance = accuracies.std()

print('Accuracy of K-FOLDS:')
print (accuracies)
print(' ')
print('Average Accuracy of K-FOLDS:')
print(accuracies_average)
print(' ')
print('Accuracy Variance of K-FOLDS:')
print(accuracies_variance)
print(' ')

# For the F1 as scoring for for cross-validation 
F1 = cross_val_score(estimator=gbc, X=X_train_standard, y=Y_train, cv=k_fold, scoring='f1_weighted')

F1_average = F1.mean()
F1_variance = F1.std()

print('F1 of K-FOLDS:')
print (F1)
print(' ')
print('Average F1 of K-FOLDS:')
print(F1_average)
print(' ')
print('F1 Variance of K-FOLDS:')
print(F1_variance)
print(' ')

# To visualize DTC, 

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(dtc, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = features,class_names=['2','1', '0'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Lettuce__ALL_DTC_Def.png')
Image(graph.create_png())

# To evaluate the performance of the Decision Tree Classifier using holdout
# A. For the Classification Accuracy
from sklearn.metrics import accuracy_score
classification_accuracy = accuracy_score(Y_test, Y_predict_gbc_def)
print('Classification Accuracy: %.4f'
      % classification_accuracy)
print(' ')

# B. For the Classification Error
from sklearn.metrics import accuracy_score
classification_error = 1-classification_accuracy
print('Classification Error: %.4f'
      % classification_error)
print(' ')

# C. For the Sensitivity or Recall Score / True Positive Rate (Kung posotive ang hinahanap, gano kadalas ang positive) Dapat same ang performance ng predicting ng + and - para walang bias sa isa
# True Positive Rate:  Actual Value +, How often Correct
from sklearn.metrics import recall_score
sensitivity = recall_score(Y_test, Y_predict_gbc_def, pos_label = 'positive', average = 'weighted')
print('Sensitivity or Recall Score: %.4f'
      % sensitivity)
print(' ')

# D. For the Specficity (kung ang actual value is negative, gaano kadalas negative. Counterpart ni sensitivity)
# True Negative Rate: Actual Value -, How often Correct
specificity = TN/(TN+FP)
print('Specificity: %.4f'
      % specificity)
print(' ') 

# yung result bias kay negative compared to positive

# E. For the FP rate .
# False Positive Rate: Actual Value -, How often Inorrect
false_positve_rate = 1-specificity
print('False Positive Rate: %.4f'
      % false_positve_rate)
print(' ')  

# F. For the precision.
# False Negative Rate: Predicted Value +, How often the prediction is Correct
from sklearn.metrics import precision_score
precision = precision_score(Y_test, Y_predict_gbc_def, pos_label = 'positive', average = 'weighted')
print('Precision: %.4f'
      % precision)
print(' ')  

#Wag ma amaze sa accuracy 

# G. For the F1 score. Relating precision and sensitivity
# False Negative Rate: Predicted Value and Actual Value+, How often the prediction is Correct relation
from sklearn.metrics import f1_score
f1_score = f1_score(Y_test, Y_predict_gbc_def, pos_label = 'positive', average = 'weighted')
print('F1 Score: %.4f'
      % f1_score)
print(' ')  

# H. For classification report
from sklearn.metrics import classification_report
classification_report = classification_report(Y_test, Y_predict_gbc_def)


######################################## GRADIENT BOOSTING CLASSIFIER OPTIMIZATION #############################################################

# To Import the kFold Class
from sklearn.model_selection import StratifiedKFold
k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0) 

# To Import the GridSearch Class
from sklearn.model_selection import GridSearchCV

Loss = ['deviance','exponential']
Learning_rate = [0.0001, 0.001, 0.01, 0.1, 1, 10]
N_estimators = [80, 90, 100, 110, 120]
Criterion = ['friedman_mse','friedman_mae']
# To Set Parameters to be Optimized Under the Decision Tree Classifier Model
parameters = dict(criterion=Criterion, n_estimators=N_estimators, learning_rate=Learning_rate, loss=Loss)   
                                                                                                                                                 
grid_search = GridSearchCV(estimator = gbc,
              param_grid = parameters,
              scoring = 'accuracy',
              cv = k_fold,
              n_jobs = -1)
grid_search = grid_search.fit(X,Y)
print(grid_search)

# To View the Results of the GridSearch
pd.DataFrame(grid_search.cv_results_)[['mean_test_score','std_test_score','params']]

# To Identify the Best Accuracy and Best Parameters
best_accuracy =grid_search.best_score_
best_parameters = grid_search.best_params_

print("Best Accuracy Score:")
print(best_accuracy)
print('')

print("Best Parameters Score:")
print(best_parameters)
print('')

#dtc_opt = DecisionTreeClassifier(max_depth=5, criterion='entropy', splitter = 'best') #all
dtc_opt = DecisionTreeClassifier(max_depth=5, criterion='gini', splitter = 'best') #-> UNI

dtc_opt.fit(X_train_standard,Y_train)

# To Predict the Output of the Whole Dataset
Y_predict_dtc_opt = dtc_opt.predict(X_test_standard)

# To Show the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_predict_dtc_opt)

import seaborn as sns
sns.heatmap(confusion_matrix, annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')

TP = confusion_matrix[1,1]
TN = confusion_matrix[0,0]
FP = confusion_matrix[0,1]
FN = confusion_matrix[1,0]

# To apply K-fold cross-validation for the logistic regression model
from sklearn.model_selection import KFold
k_fold = KFold(n_splits=10, shuffle=True, random_state=0) # shuffle = True para sa small dataset

from sklearn.model_selection import cross_val_score

# For the accuracy as scoring for for cross-validation 
accuracies = cross_val_score(estimator=dtc_opt, X=X_train_standard, y=Y_train, cv=k_fold, scoring='accuracy')

accuracies_average = accuracies.mean()
accuracies_variance = accuracies.std()

print('Accuracy of K-FOLDS:')
print (accuracies)
print(' ')
print('Average Accuracy of K-FOLDS:')
print(accuracies_average)
print(' ')
print('Accuracy Variance of K-FOLDS:')
print(accuracies_variance)
print(' ')

# For the F1 as scoring for for cross-validation 
F1 = cross_val_score(estimator=dtc_opt, X=X_train_standard, y=Y_train, cv=k_fold, scoring='f1_weighted')

F1_average = F1.mean()
F1_variance = F1.std()

print('F1 of K-FOLDS:')
print (F1)
print(' ')
print('Average F1 of K-FOLDS:')
print(F1_average)
print(' ')
print('F1 Variance of K-FOLDS:')
print(F1_variance)
print(' ')

# To visualize DTC, 

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(dtc_opt, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = features,class_names=['2','1', '0'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Lettuce_ALL_DTC_Opt.png')
Image(graph.create_png())

# To evaluate the performance of the Decision Tree Classifier using holdout
# A. For the Classification Accuracy
from sklearn.metrics import accuracy_score
classification_accuracy = accuracy_score(Y_test, Y_predict_dtc_opt)
print('Classification Accuracy: %.4f'
      % classification_accuracy)
print(' ')

# B. For the Classification Error
from sklearn.metrics import accuracy_score
classification_error = 1-classification_accuracy
print('Classification Error: %.4f'
      % classification_error)
print(' ')

# C. For the Sensitivity or Recall Score / True Positive Rate (Kung posotive ang hinahanap, gano kadalas ang positive) Dapat same ang performance ng predicting ng + and - para walang bias sa isa
# True Positive Rate:  Actual Value +, How often Correct
from sklearn.metrics import recall_score
sensitivity = recall_score(Y_test, Y_predict_dtc_opt, pos_label = 'positive', average = 'weighted')
print('Sensitivity or Recall Score: %.4f'
      % sensitivity)
print(' ')

# D. For the Specficity (kung ang actual value is negative, gaano kadalas negative. Counterpart ni sensitivity)
# True Negative Rate: Actual Value -, How often Correct
specificity = TN/(TN+FP)
print('Specificity: %.4f'
      % specificity)
print(' ') 

# yung result bias kay negative compared to positive

# E. For the FP rate .
# False Positive Rate: Actual Value -, How often Inorrect
false_positve_rate = 1-specificity
print('False Positive Rate: %.4f'
      % false_positve_rate)
print(' ')  

# F. For the precision.
# False Negative Rate: Predicted Value +, How often the prediction is Correct
from sklearn.metrics import precision_score
precision = precision_score(Y_test, Y_predict_dtc_opt, pos_label = 'positive', average = 'weighted')
print('Precision: %.4f'
      % precision)
print(' ')  

#Wag ma amaze sa accuracy 

# G. For the F1 score. Relating precision and sensitivity
# False Negative Rate: Predicted Value and Actual Value+, How often the prediction is Correct relation
from sklearn.metrics import f1_score
f1_score = f1_score(Y_test, Y_predict_dtc_opt, pos_label = 'positive', average = 'weighted')
print('F1 Score: %.4f'
      % f1_score)
print(' ')  

# H. For classification report
from sklearn.metrics import classification_report
classification_report = classification_report(Y_test, Y_predict_dtc_opt)