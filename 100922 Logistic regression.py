# Logistic regression
# sigmoid
# 0 and 1 are the probability values

#y=b0+b1X normal regression

#log(p/1-p)=y p is probabiliy of being approved (since we have to put it in sigmoid form we are taking the log)
#its log of odds
#exponential is just opposite of log

# exxp of above= p/1-p=e^y
#p=1/(1+e^-y)

#P=(E^-y)*(1-p)
#= e^y-e^y*p
#p(1+e^y)=e^y
#p=e^y/(1+e^y)
#divide both by e^y
#p=1/(1+e^-y)
#p=1/(1+(1/e^y))


import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import *

#Logistic regression
# Loading dataset

data=pd.read_csv("E:\ML DATASETS\loan prediction.CSV")

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

#splitting data=train test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=.25, random_state=10)


#model traininig
from sklearn.linear_model import LogisticRegression 
log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)
y_pred=log_reg.predict(X_test)

# you can get probability of other models as well instead of just logistic
y_prob=log_reg.predict_proba(X_test)


print('Training data accurcy :{:.3f}'.format(log_reg.score(X_train,y_train)))

print('Test data accuracy on test data: {:.3f}'.format(log_reg.score(X_test,y_test)))

#confusing matrix
from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test, y_pred)
print(cm1) # here python by defaults takes n as 1 and y as o so that we get mirrored kinda metrix
cm2=confusion_matrix(y_test, y_pred, labels=['Y','N'])
print(cm2)# we did this to get proper confusion metrics
acc=(112+13)/(112+13+1+28)
#testing acc and this acc is same so success
# CM helps you to identify the accuracy 
# other terminologies like recall, precisions 



