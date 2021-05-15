# -*- coding: utf-8 -*-
"""
Created on Sun May  9 22:30:17 2021

@author: Shubham
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

forestfire = pd.read_csv("D:\\Data Science study\\Assignment of Data Science\\Sent\\15 Neural Network\\forestfires.csv")
print(forestfire)

forestfire.head
forestfire.info()
forestfire.columns
forestfire.select_dtypes(object)   # Getting only the object type of column from the dataframe.
# checking if there are any null values in the dataframe
forestfire.isnull().sum()    
# We can say that there are no null values in the dataset

import seaborn as sb

sb.boxplot(forestfire.FFMC)
sb.boxplot(forestfire.DMC)
sb.boxplot(forestfire.DC)
sb.boxplot(forestfire.ISI)
sb.boxplot(forestfire.temp)
sb.boxplot(forestfire.RH)
sb.boxplot(forestfire.wind)

# Let's create a function for detecting the outliers present in the above graphs.

outliers=[]
def detect_outlier(data_1):         # Creating a function for finding the outliers
    outliers.clear()
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

#Let's see the outliers now.

detect_outlier(forestfire.FFMC)
len(outliers)    # There are 7 outliers in this column

detect_outlier(forestfire.DMC)
len(outliers)    # There are no outliers in this column

detect_outlier(forestfire.DC)
len(outliers)    # There are no outliers in this column

detect_outlier(forestfire.ISI)
len(outliers)    # There are 2 outliers in this column

detect_outlier(forestfire.temp)
len(outliers)    # There are no outliers in this column

detect_outlier(forestfire.RH)
len(outliers)    # There are 5 outliers in this column

detect_outlier(forestfire.wind)
len(outliers)    # There are 4 outliers in this column

# As per my study Neural Network gets affected by the outliers if we use the Relu function for the model
# But this condition stands true only for the less number of the layers and hidden layers 
# If we increase the number of layers in the model the impact of the outliers becomes less significant
# So we will try the model on this data as it is first to see how it works.

# first let's drop the unnecessary columns from the data frame.

data = forestfire.iloc[:,2:]
print(data)

# Now let's assign the values to the bicategorical output column for the model building

data['size_category'].describe()
data['size_category'].unique()

#Let's assign 0 to the 'small' and 1 to the 'large'

data.loc[data.size_category=='small','size_category'] = 0
data.loc[data.size_category=='large','size_category'] = 1

# Let's create X and Y for the model

x = data.iloc[:,:28]
y = data.iloc[:,28]

plt.hist(y)

data.size_category.value_counts()

# Let's split the data into train and test data

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2)

# We will standardize the values in the dataframe for the model
# We will be using the StandardScaler function for that.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(x_train) # giving data to the function to make it familiar with the data

# Now let's transform the data using the function scaler

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Let's import the MLP Classifier for model building
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(50,50),activation = 'tanh')

# let's fit the data in the above model

mlp.fit(x_train,y_train)

train_predict = mlp.predict(x_train)
test_predict = mlp.predict(x_test)

# Training accuracy
train_accu = np.mean(y_train==train_predict)
train_accu

# Now we will find out the accuracy of our model 

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test,test_predict))

pd.crosstab(y_test,test_predict)

# Accuracy

accuracy = np.mean(y_test==test_predict)
accuracy  # 98.07%

# This accuracy keeps changing according to the number of hidden layer and activation function.

# Above is the best result I have got so far using the 'tanh' activation function and (50,50) number of the hidden layers.

