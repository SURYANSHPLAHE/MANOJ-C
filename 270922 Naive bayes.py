# Naive bayers

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import *

#Logistic regression
# Loading dataset

data=pd.read_csv("D:\\Apurva Studies\\3rd SEM\\Lectures\\ML using python and R\\loan prediction.CSV")

# Pre processing 
data.isnull().sum()

#fill in with most frequent categgory (mode)
data['Gender'].fillna(data['Gender'].mode()[0],inplace=True)
data['Married'].fillna(data['Married'].mode()[0],inplace=True)
data['Dependents'].fillna(data['Dependents'].mode()[0],inplace=True)
data['Self_Employed'].fillna(data['Self_Employed'].mode()[0],inplace=True)
data['Credit_History'].fillna(data['Credit_History'].mode()[0],inplace=True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0],inplace=True)
data['LoanAmount'].fillna(data['LoanAmount'].median(),inplace=True)

#Dummy variables= text to numeric, variable is not made by original data. label encoder can be used, comp both label encoders and this step to get accu
data1=data.iloc[:,1:-1]
data1=pd.get_dummies(data1,columns=['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area'],drop_first=True)
X=data1.values
y=data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=.25, random_state=10)

from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
clf.fit(X_train, y_train)

y_pred=clf.predict(X_test)

print('Training Accuracy: {:.3f}'.format(clf.score(X_train, y_train)))
print('Testing Accuracy: {:.3f}'.format(clf.score(X_train, y_train)))

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred,labels=('Y','N'))
print(cm)

# Cross Validation
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(clf, X,y,cv=10)
print('{:.3f}'.format(accuracies.mean()))

# This is a probability based algorithm 






