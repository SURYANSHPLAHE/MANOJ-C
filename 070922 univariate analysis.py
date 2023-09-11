#HERE WHAT WE ACTUALLY DID
# pehle pure data ko split kiya, 25% data testing ko bheja aur 75% training ke liye
#model abhi 75% data se sikhega ke sales itna raha toh op itna rahega (kind of ML)
# Baki 25% sales woh predict karega matlab usko pata nahi rhega usko ke sales kitna hora h 
# fir accuracy dekhenge training ki aur testing ki 
#training ki yaha 62% ayi h pr we need above 75% 
# we are checking if change advertising cost of TV has any relation to sales


# data exploration
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

data=pd.read_csv("D:\\Apurva Studies\\3rd SEM\Lectures\\ML using python and R\\Advertising.csv")

data.isnull().sum()

#simple linear regression
#input variable
X=data.iloc[:,1:2].values

#output variable
y=data.iloc[:,4].values

#split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=.25, random_state=1)
# why random state=1 ? bcoz it will do not give the random state and next time when you will pick up data you will get different data
# why 1, you will get 1 set data everytime. Usually number doesn't have any significance
# every number has different set od data 

#model building & prediction
from sklearn.linear_model import LinearRegression
reg= LinearRegression()
reg.fit(X_train, y_train)

# x test mai only input variable and we are predicting it

y_pred=reg.predict(X_test)

#mode evaluation
print('Training data Accuracy: {:.3f}'.format(reg.score(X_train, y_train)))
print('Test data Accuracy : {:.3f}'.format(reg.score(X_test, y_test)))

# coefficient and intercept values
print(reg.coef_)  # lowest accuracy we can get from the data
print(reg.intercept_) # 

#calculating y for new value of x
#y= b0 +b1x
x=100
y1=reg.intercept_+reg.coef_ *X
y1


# Cross Validation (K fold cross validation)
from sklearn.model_selection import cross_val_score
scores=cross_val_score(reg, X_train, y_train, scoring='r2',cv=10)# these are accuracies
mean_r2=np.mean(scores)
round(mean_r2,2)

# Scatter plot
plt.scatter(X_train,y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='blue')
plt.title('TV Adverstising')
plt.xlabel('TV')
plt.ylabel('Sales')

plt.scatter(X_test,y_test,color='green')
plt.plot(X_test,reg.predict(X_test),color='blue')
plt.title('TV Adverstising')
plt.xlabel('TV')
plt.ylabel('Sales')

###############################################################################


