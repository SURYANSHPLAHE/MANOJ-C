import nltk
nltk.download() 


# VADER sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyser= SentimentIntensityAnalyzer()

str2= 'This is excellent car'
str2= 'This is excellent car with great mileage'
str2= 'This is good car with great mileage'
str2= 'This is not good car with average mileage'
score= analyser.polarity_scores(str2)
score

#Twitter airlines
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyser=SentimentIntensityAnalyzer()
data=pd.read_csv("E:\\Apurva Studies\\3rd SEM\\Lectures\\ML using python and R\\twitterAirlines.csv")
data=data[['airline','text']]
data=data[data['airline']=='United'] #Filtering data set for united

#computed VADER score
data['score']=[analyser.polarity_scores(x)['compound'] for x in data ['text']]
data['Predicated']=''
data.loc[data['score']>=0.1, 'Predicated']= 'positive'
data.loc[(data['score']>=0) & (data['score']<0.1),'Predicated']='neutral'
data.loc[data['score']<0,'Predicated']='negative'









