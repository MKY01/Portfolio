#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
================================================================================
The Breast Cancer dataset
================================================================================
This data sets consist of 2 types of breast cancer (Malignant and Benign) stored
in a 569x32 csv file.

The rows being individual breast tissue sample and the columns being the Mean,
Standard Error, and “Worse” of the 10 real-valued features:
radius, texture, perimeter, area, smoothness,
compactness,
concavity, concave points, symmetry and fractal dimensions.

The plot below uses the 3 features out of 30 possible features:

concavity_worst
perimeter_worst
area_worst

for further info on this dataset see:
https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research
'''

#Code source: Man Kit Yip
#Documentation: Man kit Yip
#License: Apache 2.0


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score


#load the dataset from the dataset downloaded from the kaggle webpage ('https://www.kaggle.com/uciml/breast-cancer-wisconsin-data/version/2')
df = pd.read_csv('/Users/mankityip/Documents/BBK/Data-Science_Techniques_and_Applications/coursework2/kaggle_breast-cancer_data.csv')

print(df.head()) #check the first several rows of of the dataset


#standardise the data, so that the mean = 0 and variance =1
all_features = df.iloc[:,2:32]    #this is all of the 30 features excluding the ID and known diagnosis
selected_features = df.loc[:, ['perimeter_worst','area_worst','concavity_worst']]   #these are the 3 features I have selected to use

y = df.loc[:, ['diagnosis']].values     #the y axis will be the target feature 'diagnosis'

x2 = selected_features.iloc[:,:].values     #the x axis will be the 3 aforementioned predictor features 
x2 = StandardScaler().fit_transform(x2) #normalize the datasets
transformed_df = pd.DataFrame(data = x2, columns = ['perimeter_worst', 'area_worst','concavity_worst'])

x1 = all_features.iloc[:,:].values #this x axis contain all of the 30 features which may be called upon later on if need be
x1 = StandardScaler().fit_transform(x1) #normalize the datasets
transformed_df2 = pd.DataFrame(data = x1)

#print(transformed_df)


#PCA projection to 2 dimensions (reduced from 3 predictor features)
pca = PCA(n_components = 2)
principal_components = pca.fit_transform(x2)
principal_df = pd.DataFrame(data = principal_components, columns = ['principal_component_1', 'principal_component_2'])

#print(principal_df)
#print(y)


#concatenate the Dataframe along axis 1, which gives the final DF
final_df = pd.concat([principal_df, df[['diagnosis']]], axis = 1)
print(final_df)


#Visualize 2D projection
fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_title('2 Component PCA', fontsize = 20)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)

diagnosis = ['M','B'] #these represents the known classes labels instead of just PC1 and PC2
diagnosis2 = ['B','M']

colors = ['g','r',] 
for diagnosis, color in zip(diagnosis, colors):
    indices_to_keep = final_df['diagnosis'] == diagnosis
    ax.scatter(final_df.loc[indices_to_keep, 'principal_component_1'], final_df.loc[indices_to_keep, 'principal_component_2'], c = color, s = 10)

plt.legend(diagnosis)
plt.legend(diagnosis2) #this will ensure it correctly show the legend for both markers

plt.show()


#explained variance 
print(pca.explained_variance_ratio_)


#PCA projections to 3 Dimensions (using all of the 3 selected features)
pca2 = PCA(n_components=3)
principal_components2 = pca2.fit_transform(x2)
principal_df2 = pd.DataFrame(data = principal_components2, columns = ['principal_component_1', 'principal_component_2', 'principal_component_3'])
final_df2 = pd.concat([principal_df2, df[['diagnosis']]], axis = 1)
print(final_df2)


#Visualize 3D projection
fig2 = plt.figure(figsize = (8,8))
ax = fig2.gca(projection='3d')
ax.set_title('First three PCA directions', fontsize = 20)
ax.set_xlabel('1st Eigenvector', fontsize = 15)
ax.set_ylabel('2nd Eigenvector', fontsize = 15)
ax.set_zlabel('3rd Eigenvector', fontsize = 15)

diagnosis = ['M','B'] #these represents the known classes labels instead of just PC1 and PC2
diagnosis2 = ['B','M']

colors = ['g', 'r']
                  
pc1 = (final_df2.loc[:,'principal_component_1'])
pc2 = (final_df2.loc[:,'principal_component_2'])
pc3 = (final_df2.loc[:,'principal_component_3'])

for diagnosis,color in zip(diagnosis,colors):
    indices_to_keep = final_df2['diagnosis'] == diagnosis
    ax.scatter(final_df2.loc[indices_to_keep, 'principal_component_1'],
               final_df2.loc[indices_to_keep, 'principal_component_2'],
               final_df2.loc[indices_to_keep, 'principal_component_3'],
               alpha = 0.8, edgecolor='none',c = color, s = 10)
 
plt.legend(diagnosis)
plt.legend(diagnosis2) #this will ensure it correctly show the legend for both markers
plt.show()

# rotate the axes and update
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)


#explained variance 
print(pca2.explained_variance_ratio_)


#An alternate approach would be to use PCA to select the 3 Dimensions out of the possible 30.
#Another approach would be to use PCA to select as many Dimensions as possible which explains 95% of the variance.
#However both of these are outside the scope of the requirement of the coursework.



