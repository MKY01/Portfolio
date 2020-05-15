#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Copyright (c) 2019 Man kit Yip. All rights reserved. License under: Apache License 2.0

======================================================Documentation for the implementation of Machine Learning algorithm======================================================

Supervised learning - Classification problem - multi-class

Dataset source: https://archive.ics.uci.edu/ml/datasets/ecoli

Modules required: numpy, pandas, matplotlib, scikitlearn, mglearn, graphviz

For documentation on sklearn: https://scikit-learn.org/stable/supervised_learning.html

Steps:

(Optional) Pre-processing which may or may not be required depending on the dataset: 4 techniques for pre-processing of raw data
(A) Binarization
(B) Standardisation
(C) Scaling
(D) Normalisation


Feature engineering & feature selection, 2 approaches:
(1) Use the full set of features.
(2) Use only some of the more 'important' features selected by either (I) Univeriate statistics or (II) Recursive Feature Elimination.


ML algorithms to try out:
(I) Logistic regression: +works well for large dataset +faster train and predict -other algorithm for lower dimensional spaces
(II) KNN: +easy to understand and implement
(III) Decision Trees: +easy to understand, little pre-processing -often overfit and poor generalization
(IV) Ensemble method - (Bagging/ parallel) Random Forest: +perform well withour parameter optimisation, better than a single DT. -poor on high dimensional sparse data, takes longer time
(V) Ensemble method - (Boosting/sequential) Gradient Boosting: +reduce bias & variance -can overfit


(Optional) Uncertainty Estimation - for prediction with probabilistic approach:
(A)The decision function - https://scikit-learn.org/stable/modules/calibration.html
(B)Predicting Probabilities - https://scikit-learn.org/stable/modules/calibration.html


2 different modes for model evaluation: https://scikit-learn.org/0.15/modules/model_evaluation.html
(I) Estimator score method - this is classifier dependentusing the evaluation appropriate criterion - see https://scikit-learn.org/stable/modules/classes.html
(II) Scoring parameters using cross-validation - see https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation

==============================================================================================================================================================================
'''


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_selection import RFE, SelectPercentile
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import mglearn
import graphviz


#set seed for reproducibility
seed = 1
np.random.seed(seed)

#load dataset & encode the labels
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data", delim_whitespace = True)

df.columns = ['seq_name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'loc_site'] #these are the column labels. see https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.names

df = df.drop('seq_name', axis = 1) #drop the descriptive name from each row

le = preprocessing.LabelEncoder() #create a category encoder object and fit it to the target variable's column
le.fit(df['loc_site'])
le.transform(df['loc_site']) #apply the fitted encoder
print(list(le.classes_)) #view the labels
df.loc_site.replace(('cp', 'im', 'pp', 'imU', 'om', 'omL', 'imL', 'imS'), (1, 2, 3, 4, 5, 6, 7, 8), inplace = True) #the 'location sites'/classes are now represented by numbers




##pre-processing: (A)Binarisation, (B)Standardisation, (C)Scaling, (D) Normalisation
#(A)Binarisation
#bindata = preprocessing.Binarizer(threshold=0.5).transform(data)
#print('Binarised data:\n\n', bindata)


#(B)Standardisation
print('Mean (before)= ', (df[0:7]).mean(axis=0))
print('Standard Deviation (before)= ',(df[0:7]).std(axis=0)) #STD and mean not centered around 1

scaled_data = preprocessing.scale(df[0:7])
print('Mean (after)= ', scaled_data.mean(axis=0))
print('Standard Deviation (after)= ', scaled_data.std(axis=0)) #STD mean now centred around 1


#(C)feature scaling - StandardScaler or MinMaxScaler
minmax_scaler = preprocessing.MinMaxScaler(copy=True, feature_range=(0,1)) #transform them by scaling each feature between 0 and 1
data_minmax = minmax_scaler.fit_transform(df[['mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2']]) #only scale the predictor variables
data_minmax2 = (df[['loc_site']]) #do not scale the y target variables which are categories/classes

print('MinMaxScaler applied on the data: \n', data_minmax)


#(D) Normalisation - L1 - least absolute deviations ignore outliers, L2 least squres include outliers
data_l1 = preprocessing.normalize((df[0:7]), norm='l1') #I will choose L1 if the outliers are unimportant

data_l2 = preprocessing.normalize((df[0:7]), norm='l2') #I will choose L2 if the outliers are important

print('L1-normalised data:\n', data_l1)
print('L2-normalised data:\n', data_l2)


#These are the pre-processed dataset in numpy array format, which maybe optionally tested for certain machine learning algorithms.
#scaled_data
#data_minmax, data_minmax2
#data_l1
#data_l2


#basic statistics & description of dataset
print(df.describe()) #display basic statistics about the dataset
print(df.head()) #display only the first few rows of the dataset to check it is in the format
print(df.shape) #display the shape of the dataset (rows,columns)
print(df.dtypes) #datatype of the columns




##instantiate the predictor variables & the target variable (with known classes encoded in numbers)
dataset = df.values #returns a numpy array format of the dataset
x = dataset[:, 0:7] #each column from 1 - 7 contains the predictor variables (x)
y = dataset[:, 7] #the last column 8 contains the target variable which is the "class" of the localisation site (y)


#(Optional) feature engineering: automatic feature selection to reduce dimensionality
#(I)Univeriate statistics method by SelectPercentile
select = SelectPercentile(percentile=50) #(B) this automatically selected half of the features: lip, alm1, alm2
select.fit(x,y)
x_selected_bypercent = select.transform(x)

x_chosen1 = select.get_support() #this shows the 3 features selected from the 7 possible
print(x_chosen1)
print(x_selected_bypercent) #the 3 automatically selected features to be used for the train data


#(II)Recursive Feature Elimination - to keep the most important features, important for some Machine learning algorithms.
print(df.corr(method = 'pearson')) #other options: 'kendall', 'spearman'. highest correlation in descending order: mcg, gvh, alm1, lip, aac, chg, alm2.
estimator = SVR(kernel="linear") #Choose the model it is appropriate for: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’.
select = RFE(estimator, n_features_to_select=3, step=1) #feature ranking with RFE to get only 3 features
x_selected_bymodel = select.fit(x, y)
x_selected_bymodel = select.transform(x)

x_chosen2 = select.get_support() #this shows the 3 features selected from the 7 possible
print(x_chosen2)
print(x_selected_bymodel) #the 3 manually selected features to be used for the train data


#splitting the dataset into the training set & test set https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state = 1) #I have decided to uses all 7 selected features as the dataset is small
#x_train_, x_test, y_train, y_test = train_test_split(x_selected_bypercent, y, train_size = 0.8, test_size = 0.2, random_state = 1) #this only uses the 3 features selected by percentile as above
#x_train_, x_test, y_train, y_test = train_test_split(x_selected_bymodel, y, train_size = 0.8, test_size = 0.2, random_state = 1) #this uses 3 different features selected by RFE as above




##picking a machine learning algorithm

#(1a)fitting the Logistic Regression classifier to the training set https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
log_reg01 = LogisticRegression(C=0.1, solver='liblinear',multi_class='auto', max_iter=1000) #Use regularisation to prevent overfitting, L2 considersd all features, low C increases the level of regularisation.
log_reg01.fit(x_train, y_train)

log_reg1 = LogisticRegression(C=1, solver='liblinear',multi_class='auto', max_iter=1000) #Use regularisation to prevent overfitting, C=1 is the default.
log_reg1.fit(x_train, y_train)

log_reg10 = LogisticRegression(C=10, solver='liblinear',multi_class='auto', max_iter=1000) #Use regularisation to prevent overfitting, L1 only assumes a few features are important, higher C emphasise correct classifcation of each data point
log_reg10.fit(x_train, y_train)

print('Accuracy of the LogisticRegression train subset using estimator score: {:.3f}'.format(log_reg10.score(x_train, y_train))) #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
print('Accuracy of the LogisticRegression test subset using estimator score: {:.3f}'.format(log_reg10.score(x_test, y_test))) #log_reg10 was chosen

cv_score = cross_val_score(log_reg10, x_train, y_train, cv=2) #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
print('Accuracy of the LogisticRegression train subset using cv score: {:.3f}'.format(cv_score.mean()))

cv_score = cross_val_score(log_reg10, x_test, y_test, cv=2) #log_reg10 was chosen
print('Accuracy of the LogisticRegression test subset using cv score: {:.3f}'.format(cv_score.mean()))

#(1b)logistic regression visualisation, evaluation & optimisation - adjusting the C parameter
#for scoring metrics see https://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html
plt.plot(log_reg01.coef_.T[3], 'x', label ='C=0.1')
plt.plot(log_reg1.coef_.T[3], 'x', label ='C=1')
plt.plot(log_reg10.coef_.T[3], 'x', label ='C=10') #high C allows better interpretations
plt.xticks(range(df.shape[1]), df.columns, rotation = 90)
plt.hlines(0, 0, df.shape[1])
plt.ylim(-15, 15)
plt.xlabel('Coefficient Index')
plt.ylabel('Coefficient Magnitude')
plt.legend()
plt.show()




#(2a)fitting the K Nearest Neighbour classifier to the training set https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', n_jobs=None)
knn.fit(x_train, y_train)

print('Accuracy of the KNN n-7 train subset using estimator score: {:.3f}'.format(knn.score(x_train, y_train))) # optimal n = 7 for both training and test set
print('Accuracy of the KNN n-7 test subset using estimator score: {:.3f}'.format(knn.score(x_test, y_test)))

cv_score = cross_val_score(knn, x_train, y_train, cv=2) #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
print('Accuracy of the KNN n-7 train subset using cv score: {:.3f}'.format(cv_score.mean()))

cv_score = cross_val_score(knn, x_test, y_test, cv=2) #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
print('Accuracy of the KNN n-7 test subset using cv score: {:.3f}'.format(cv_score.mean()))


#(2b)KNN visualisation, evaluation & optimisation - how the number of neighbours affects it's accuracy
#for scoring metrics see https://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html
training_accuracy = []
test_accuracy = []

neighbors_settings = range(1,11)

for n_neighbors in neighbors_settings:
	clf = KNeighborsClassifier(n_neighbors = n_neighbors)
	clf.fit(x_train, y_train)
	training_accuracy.append(clf.score(x_train, y_train))
	test_accuracy.append(clf.score(x_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label = 'Accuracy of the training set')
plt.plot(neighbors_settings, test_accuracy, label = 'Accuracy of the test set')
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbours')
plt.legend()
plt.show()




#(3a)fitting the Decision Tree
tree = DecisionTreeClassifier(random_state=1)
tree.fit(x_train, y_train)

print('Accuracy of the DecisionTree train subset using estimator score: {:.3f}'.format(tree.score(x_train, y_train))) # a score of 1 or 100% means overfitting
print('Accuracy of the DecisionTree test subset using estimator score: {:.3f}'.format(tree.score(x_test, y_test)))

cv_score = cross_val_score(tree, x_train, y_train, cv=2) #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
print('Accuracy of the DecisionTree train subset using cv score: {:.3f}'.format(cv_score.mean()))

cv_score = cross_val_score(tree, x_test, y_test, cv=2) #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
print('Accuracy of the DecisionTree test subset using cv score: {:.3f}'.format(cv_score.mean()))


#(3b)Decision Tree visualisation, evaluation & optimisation - apply restrictions such as setting max depth e.g. pre- post- pruning.
#for scoring metrics see https://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html
tree_pruned = DecisionTreeClassifier(max_depth=7, random_state=1) #max_leaf_nodes. see https://scikit-learn.org/stable/modules/tree.html#classification
tree_pruned.fit(x_train, y_train)

print('Accuracy of the PrunedTree train subset using estimator score: {:.3f}'.format(tree_pruned.score(x_train, y_train))) # a score of 1 or 100% means overfitting
print('Accuracy of the PrunedTree test subset using estimator score: {:.3f}'.format(tree_pruned.score(x_test, y_test)))

cv_score = cross_val_score(tree_pruned, x_train, y_train, cv=2) #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
print('Accuracy of the PrunedTree train subset using cv score: {:.3f}'.format(cv_score.mean()))

cv_score = cross_val_score(tree_pruned, x_test, y_test, cv=2) #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
print('Accuracy of the PrunedTree test subset using cv score: {:.3f}'.format(cv_score.mean()))

export_graphviz(tree_pruned, out_file='ecolitree.dot', class_names=['cp', 'im', 'pp', 'imU', 'om', 'omL', 'imL', 'imS'], feature_names=df.columns[0:7], impurity=False, filled=True)


#(3c)To convert the dot file to png, on linux type in bash: $ dot -Tpng ecolitree.dot -o ecolitree.png
#see https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
n_features = 7 #7 predictor features (x)
plt.barh(range(n_features), tree_pruned.feature_importances_)
plt.yticks(np.arange(n_features), df.columns[0:7]) #feature names
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show() #importances in descending order: alm1, gvh, mcg
print('Feature importances: {}'.format(tree_pruned.feature_importances_))#the depth of the tree is 7. Feature importances




#(4a) Ensemble methods - fitting the random forests
forest = RandomForestClassifier(n_estimators=100, random_state=1) # n_jobs , max_depth, max_features=sqrt(n_features),
forest.fit(x_train, y_train)

print('Accuracy of the RandomForest train subset using estimator score: {:.3f}'.format(forest.score(x_train, y_train)))
print('Accuracy of the RandomForest test subset using estimator score: {:.3f}'.format(forest.score(x_test, y_test)))

cv_score = cross_val_score(forest, x_train, y_train, cv=2) #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
print('Accuracy of the RandomForest train subset using cv score: {:.3f}'.format(cv_score.mean()))

cv_score = cross_val_score(forest, x_test, y_test, cv=2) #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
print('Accuracy of the RandomForest test subset using cv score: {:.3f}'.format(cv_score.mean()))


#(4b)Ensemble methods - random forests visualisation, evaluation & optimisation. https://scikit-learn.org/stable/modules/ensemble.html#ensemble
#for scoring metrics see https://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html
n_features = 7 #7 predictor features (x)
plt.barh(range(n_features), forest.feature_importances_)
plt.yticks(np.arange(n_features), df.columns[0:7]) #feature names
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show() #importances in descending order: alm1, mcg, alm2
print('Feature importances: {}'.format(forest.feature_importances_))#the depth of the tree is 7. Feature importances




#(5) Ensemble methods - gradient boosting classifier. #https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=1) #learning rate should not be too high
gbrt.fit(x_train, y_train)

print('Accuracy of the GradientBoostingClassifier train subset using estimator score: {:.3f}'.format(gbrt.score(x_train, y_train)))
print('Accuracy of the GradientBoostingClassifier test subset using estimator score: {:.3f}'.format(gbrt.score(x_test, y_test)))

cv_score = cross_val_score(gbrt, x_train, y_train, cv=2) #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
print('Accuracy of the GradientBoostingClassifier train subset using cv score: {:.3f}'.format(cv_score.mean()))

cv_score = cross_val_score(gbrt, x_test, y_test, cv=2) #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
print('Accuracy of the GradientBoostingClassifier test subset using cv score: {:.3f}'.format(cv_score.mean()))

#print('The decision function for the samples in the 8-class ecoli dataset:\n\n{}'.format(gbrt.decision_function(x_test[:10])))
print('Predicted probabilities for the samples in the 8-class ecoli dataset:\n\n{}'.format(gbrt.predict_proba(x_test[:10]))) #multi-class classification prediction




##(1) Logistic Regression appears to be the best model overall for this dataset, so it will be explored further with their parameters further tuned.
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
#https://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html


#logistic Regression using pre-processed data scaled between 0-1 instead
x = data_minmax
y = data_minmax2

#tuning the parameters to get higher training and testing accuracy on both scores
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.85, test_size = 0.15, random_state = 1,shuffle=True) #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

log_reg15 = LogisticRegression(penalty='l2', C=15, solver='lbfgs', multi_class='multinomial', max_iter=1000) #https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
log_reg15.fit(x_train, y_train.values.ravel()) #flatten the returned numpy array

print('Accuracy of the LogisticRegression train subset using estimator score: {:.3f}'.format(log_reg15.score(x_train, y_train)))
print('Accuracy of the LogisticRegression test subset using estimator score: {:.3f}'.format(log_reg15.score(x_test, y_test)))

cv_score = cross_val_score(log_reg15, x_train, y_train, cv=2)
print('Accuracy of the LogisticRegression train subset using cv score: {:.3f}'.format(cv_score.mean()))

cv_score = cross_val_score(log_reg15, x_test, y_test, cv=2)
print('Accuracy of the LogisticRegression test subset using cv score: {:.3f}'.format(cv_score.mean()))


#Before tuning:
#train subset using Estimator score = 0.877
#test subset using Estimator score = 0.940
#train subset using cv score = 0.843
#test subset using cv score = 0.879

#After tuning:
#train subset using Estimator score = 0.884
#test subset using Estimator score = 0.922
#train subset using cv score = 0.863
#test subset using cv score = 0.921
