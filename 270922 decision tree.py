# Decision tree
import pandas as pd
from sklearn import tree
import pydotplus as pdtp
from io  import StringIO
import os
os.chdir("D:\\Apurva Studies\\3rd SEM\\Lectures\\ML using python and R")
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

data1=data.iloc[:,1:-1]
data1=pd.get_dummies(data1,columns=['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area'],drop_first=True)
X=data1.values
y=data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=.25, random_state=10)

from sklearn.tree import DecisionTreeClassifier
clf= DecisionTreeClassifier(criterion='gini',max_depth=3,min_samples_leaf=5,random_state=10)
#gini
clf= DecisionTreeClassifier( max_depth=3, min_samples_leaf=5,random_state=10)
# max depth= 

# Entropy
clf=DecisionTreeClassifier(criterion='entropy',max_depth=3,min_samples_leaf=5)

#common to run once you run gini or entropy
clf.fit(X_train,y_train)
print('Training Accuracy: {:.3f}'.format(clf.score(X_train, y_train)))
print('Testing Accuracy: {:.3f}'.format(clf.score(X_test, y_test)))


#################################################3
from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(criterion='gini',n_estimators=100 ,
                           max_depth=3,min_samples_leaf=5,random_state=10)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)
print('Training Accuracy: {:.3f}'.format(classifier.score(X_train, y_train)))
print('Testing Accuracy: {:.3f}'.format(classifier.score(X_test, y_test)))
################################################


#chart
len2= len(data1.columns)
features=list(data1.columns[0:len2])

dot_data=StringIO();
tree.export_graphviz(clf,out_file=dot_data,feature_names=features,impurity=False)
graph=pdtp.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('tdemo.pdf')


df1=pd.DataFrame(X_train,columns=data1.columns)
df1['Loan_Status']=y_train
df1.to_csv('TreeTrain.csv')








