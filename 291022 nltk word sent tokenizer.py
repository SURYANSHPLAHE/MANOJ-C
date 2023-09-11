import nltk
#nltk.download() 

from nltk.tokenize import word_tokenize

str1="India is exporting $10 million software services to USA"

str1=str1.lower()
str1

#word Tokenization

wt=word_tokenize(str1)
print(wt)
# Sentence tokenization
from nltk.tokenize import sent_tokenize
str2='India is exporting $10 million software to USA. Software is one of the most growing sector.'
ws= sent_tokenize(str2)
print(ws)
ws[0]


# Frequency distribution
from nltk.probability import FreqDist
wt1= word_tokenize(str2)
fdist= FreqDist(wt1)
fdist
fdist.most_common(2)
fdist.plot(10)

# part of speech (pos)
pos= nltk.pos_tag(wt)
pos
# nltk.download('tagsets')
#nltk.help.upenn_tagset() # list of all tags 

# stop words
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
print(stop_words)

filtered1=[]
for w in wt:
    if w not in stop_words:
        filtered1.append(w)
print('Tokensized:',wt)
print('Filtered:',filtered1)      

# Lemmatization ( it needs context)
str3= "I am a runner running in the race as i love to run since I ran past years"
wt1= word_tokenize(str3)
from nltk.stem.wordnet import WordNetLemmatizer
lem= WordNetLemmatizer()
lem_words=[]
for w in wt1: 
    lem_words.append(lem.lemmatize(w,'v')) # lemmatizing the verb only, v is verb, w is word in wt1
lem_words

# Steaming
str4= 'Connection connectivity connnected connecting'
str4= 'studying studies studied'
str4= 'likes liked likely'
str4= 'I am a runner running in the race as I love to run since I ran past years'

from nltk.stem import PorterStemmer
wt1= word_tokenize(str4)
ps= PorterStemmer()
stemmed_words=[]
for w in wt:
    stemmed_words.append(ps.stem(w))
stemmed_words

#CountVectorizer
# Each word acts as a column header and number of occurance of that word is mentioned in it sentence wise.
# counting the frequency of that word
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
cv1= CountVectorizer()
lstVect= ['Hi How are you How are you doing',
          'I am doing very very good',
          'Wow thats awesome really awesome']

x_traincv=cv1.fit_transform(lstVect)
x_traincv_df= pd.DataFrame(x_traincv.toarray(), columns=list(cv1.get_feature_names()))
x_traincv_df

# TF - IDF Vectorizer= Term Frequency, Inverse Document Frequency
tf1= TfidfVectorizer()
x_traintv= tf1.fit_transform(lstVect)
x_traintv_df= pd.DataFrame(x_traincv.toarray(), columns=list(tf1.get_feature_names()))
x_traintv_df








        
