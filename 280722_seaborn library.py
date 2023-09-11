import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.chdir("E:\Apurva Studies\3rd SEM\Revision lectures\Revision Python")

data1=pd.read_csv("E:\\Apurva Studies\\3rd SEM\\Revision lectures\\Revision Python\\loan prediction.csv")

sns.barplot(x="Dependents",y="LoanAmount",data=data1)
sns.barplot(x="LoanAmount",y="Dependents",data=data1)

sns.barplot(x="LoanAmount",y="Dependents",hue="Education",  data=data1, ci=None)

sns.barplot(x="LoanAmount",y="Dependents",hue="Education", data=data1)

sns.barplot(x="LoanAmount",y="Dependents", data=data1, estimator=(sum))


#Factor plot
sns.factorplot(x="Dependents",y="LoanAmount",data=data1, col="Education",kind='bar',ci=None)

sns.boxplot(x="Married", y="LoanAmount",data=data1)
sns.boxplot(x="Dependents", y="LoanAmount",data=data1)
sns.boxenplot(x="Married", y="LoanAmount",data=data1,hue=("Education"))

sns.distplot(data1['LoanAmount'])
sns.distplot(data1['LoanAmount'], bins=20,kde=False)#Kernel density estimate
sns.distplot(data1['LoanAmount'], bins=20,hist=False)
# distplot() represents the univariate distribution of data, data distribution of a variable against the density distribution
sns.distplot(data1["LoanAmount"])
plt.savefig("snsChart1.png")

dFPP=pd.read_csv("E:\\Apurva Studies\\3rd SEM\\Revision lectures\\Revision Python\\280722_auto_mpg_seaborn.csv")
sns.regplot(dFPP["weight"],dFPP["mpg"],fit_reg=False)
# regplot() is used to plot data and a linear regression model fit.

#Multi varient analysis
sns.pairplot(dFPP[["mpg","weight","horsepower"]])

#sworn plot
sns.swarmplot(x="Married",y="LoanAmount",data=data1)
sns.factorplot(x="Dependents",y="LoanAmount",data=data1, col="Education", kind="swarm") #2 plots in 1 page

corrmat=dFPP[["mpg","weight","horsepower","displacement"]].corr()
plt.figure(figsize=(12,6))
sns.heatmap(corrmat,annot=True)
