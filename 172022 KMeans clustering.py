# K Means clustering 

import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir("E:\\Apurva Studies\\3rd SEM\\Revision lectures\\Revision Python ")
data= pd.read_csv('E:\\Apurva Studies\\3rd SEM\\Revision lectures\\Revision Python\\loan prediction.csv')

data['Total Income']=data['ApplicantIncome']+data['CoapplicantIncome']
data1=data[['Total Income','LoanAmount']]
data1.isnull().sum()
med1=int(data['LoanAmount'].median())
data1['LoanAmount'].fillna(med1,inplace=True)
X=data1.values



plt.scatter(data1['Total Income'],data1['LoanAmount'],s=100)
plt.title('Clusters of customers')
plt.xlabel('Annual Income')
plt.ylabel('Loan Income')



from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)


#Using the elbow method to find optimal number of cluster

from sklearn.cluster import KMeans
list=[]
for i in range(1, 11):
    kmeans=KMeans(n_clusters=i,random_state=42)
    kmeans.fit(X)
    list.append(kmeans.inertia_)

plt.plot(range(1,11),list,marker='o')
plt.title('Elbow method')
plt.xlabel("Number of clusters")
plt.ylabel('Within cluster distance')

kmeans1=KMeans(n_clusters=5,random_state=10,n_init=42)
y_kmeans=kmeans1.fit_predict(X)


data['kmeans']=y_kmeans
data.replace({'kmeans':{0: 'Red',1:'Blue',2:'Green',3:'Orange',4:'purple'}},inplace=True)
data['kmeans'].value_counts()

# Visualising the clusters
plt.scatter(X[y_kmeans==0,0], X[y_kmeans==0,1],s=100, c='red',label='Cluster 1')
plt.scatter(X[y_kmeans==1,0], X[y_kmeans==1,1],s=100, c='blue',label='Cluster 2')
plt.scatter(X[y_kmeans==2,0], X[y_kmeans==2,1],s=100, c='green',label='Cluster 3')
plt.scatter(X[y_kmeans==3,0], X[y_kmeans==3,1],s=100, c='orange',label='Cluster 4')
plt.scatter(X[y_kmeans==4,0], X[y_kmeans==4,1],s=100, c='purple',label='Cluster 5')
plt.scatter(kmeans1.cluster_centers_[:,0], kmeans1.cluster_centers_[:,1],s=300,c='yellow', label='Centroids')
plt.title("Clusters of customers")

# uSING seaborn
import seaborn as sns
sns.scatterplot('Total Income','LoanAmount',data=data, hue= 'kmeans1')


