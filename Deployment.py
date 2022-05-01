import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import numpy as np
import string # special operations on strings
import spacy # language models
import pandas as pd
import streamlit as st
from matplotlib.pyplot import imread
from matplotlib import pyplot as plt



st.title('Sentiment Analysis of')
st.sidebar.header('User Input Parameters')

url = st.sidebar.text_input("Insert the URL")



#code


r=requests.get(url)
soup=BeautifulSoup(r.text,'html.parser')
reviews=soup.find_all('div',{'data-hook':'review'})
pr={'product':soup.title.text.replace('Amazon.in:Customer reviews:','').strip()}
Product_Name=pr.get('product')
print("Product_Name:",Product_Name)
st.subheader(Product_Name)

pg=re.findall(r"all_reviews&pageNumber=1",url)
if len(pg)==0:
    url=url+'&pageNumber=1'

review_list=[]

#To create soup for url
def get_soup(url):
    r=requests.get(url)
    soup=BeautifulSoup(r.text,'html.parser')
    return soup

#To extract title, body and ratings of review for given URL
def get_reviews(soup):
    reviews=soup.find_all('div',{'data-hook':'review'})
    for item in reviews:
        review={
        #'product':soup.title.text.replace('Amazon.in:Customer reviews:','').strip(),
        'title':item.find('a',{'data-hook':'review-title'}).text.strip(),
        'rating':float(item.find('i',{'data-hook':'review-star-rating'}).text.replace('out of 5 stars','').strip()),
        'body':item.find('span',{'data-hook':'review-body'}).text.strip()
        }
        review_list.append(review)

#To repeat above function for all pages of review for particular product.
for x in range(0,51):
    y='pageNumber={}'.format(x+1)
    url=url.replace(f'pageNumber={x}',y)
    #print(url)
    soup=get_soup(url)
    #print('getting_page={}'.format(x+1))
    get_reviews(soup)
    #print(len(review_list))
    #To end the loop on last page
    end=soup.find('li',{'class':'a-disabled a-last'})
    if end==None:
        pass
    else:
        break

df=pd.DataFrame(review_list)



df['review']=df['title']+" "+df['body']    
df=df[['rating','review']]
df=df.dropna()


import numpy as np # linear algebra
import string # special operations on strings
import spacy # language models
import pandas as pd



#To remove special character,punctuation emoji etc
import re
clean=[]
for i in df['review']:
    cl=re.sub(r'[^\w\s]','',i)
    if cl !='':
        clean.append(cl)
book2 = [x.strip() for x in clean] # remove both the leading and the trailing characters
#book3 = [x for x in book2 if x]
#To convert into lower case
lower=[]
for i in book2:
    lower.append(str.lower(i))

#Tokenization
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
nltk.download('punkt')
#word tokenize
c2=[word_tokenize(i) for i in lower]


#Dowmload stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words=stopwords.words('english')

#remove no, nor, not from stop words
new_stop_words=[]
for i in stop_words:
    if i not in ['no','nor','not']:
        new_stop_words.append(i)

#Remove stop in text
c3=[]
for words in c2:
    w=[]
    for j in words:
        if not j in new_stop_words:
            w.append(j)
    c3.append(w)


#stemming using lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
lem=[]
for i in c3:
    w=[]
    for j in i:
        x=lemmatizer.lemmatize(j)
        w.append(x)
    lem.append(w)



y=[]
for i in lem:
    x=','.join(i)
    y.append(x)
no_comma=[]
for i in y:
    c=i.replace(','," ")
    no_comma.append(c) 


df['NLP_Review']=pd.DataFrame(no_comma)


#Import affin Data
afinn = pd.read_csv('Afinn.csv', sep=',', encoding='latin-1')
affinity_scores = afinn.set_index('word')['value'].to_dict()


#!python -m spacy download en_core_web_sm
spacy.cli.download("en_core_web_sm")
nlp = spacy.load('en_core_web_sm')
sentiment_lexicon = affinity_scores

def calculate_sentiment(text: str = None):
    sent_score = 0
    if text:
        sentences = nlp(text)
        for word in sentences:
            sent_score += sentiment_lexicon.get(word.lemma_, 0)
    return sent_score

print("yes")
df['NLP_Review']=df['NLP_Review'].astype(str)
print(type(df['NLP_Review']))
df['Sentiment_Value']=df['NLP_Review'].apply(calculate_sentiment) #calculate sentiment value
df['word_count'] = df['NLP_Review'].str.split().apply(len) #word count of each review after nlp

df['rating']=df['rating'].astype(int)
df['NLP_Review']=df['NLP_Review'].astype(str)
df['word_count']=df['word_count'].astype(int)

dg=df.groupby('rating').count()
dz=dg.reset_index()
dz=dz.iloc[:,[0,1]]


#plot histogram
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle







dff=df.sort_values(by='Sentiment_Value')
dff=dff.reset_index()
df=dff.iloc[:,1:]


#To find out the index for which sentiment is zero
ix=0
for i in df.iloc[:,3]:
    if i<1:
        ix=ix+1

#Bar and sentiment graph
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(20,8))


patches = ax2.bar(data=dz,x='rating',height='review')
#patches=dz.plot.bar(ax=ax1,x='rating',y='review')

plt.xlabel('Ratings')
plt.ylabel('Count_Ratings')
#plt.title('abc')

low = 'r'
medium ='b'
high = 'g'

for i in range(0,2):
    patches[i].set_facecolor(low)
    
for i in range(2,3):    
    patches[i].set_facecolor(medium)
    

for i in range(3, len(patches)):
    patches[i].set_facecolor(high)

#For legends
handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [low,medium, high]]
labels= ["Bad","Medium", "Good"]
ax2.legend(handles, labels)

import seaborn as sns
#fig2, ax2 = plt.subplots(1,1)

sns.lineplot(ax=ax1,y='Sentiment_Value',x=df.index,data=df)
x=df.index
y=df.iloc[0,3]
ax1.fill_between(x,y, 0,
                 facecolor="orange", # The fill color
                 color='blue',
                 where= (x < ix),
                 alpha=0.2)          # Transparency of the fill
ax1.annotate("Transition_Pt",(ix,0))
st.pyplot(fig)




import matplotlib.pyplot as plt
#%matplotlib inline
 
import sys








#Experiment


dff= pd.DataFrame(df['NLP_Review'][df['Sentiment_Value'] > 0])
dff.rename(columns = {'NLP_Review':'Positive_Review'}, inplace = True)



book2=[x.strip() for x in dff.Positive_Review]
book3 = [i for i in book2 if i]
textP = ' '.join(book3)


from sklearn.feature_extraction.text import CountVectorizer
vectorizer_ngram_range = CountVectorizer(analyzer='word',ngram_range=(1,3),max_features = 100)
vectorizer_ngram_range.fit_transform(book3)
textPBI=vectorizer_ngram_range.get_feature_names()
textPBI=' '.join(textPBI) 

#using bigram
#from PIL import Image
from wordcloud import WordCloud, STOPWORDS
fig1,ax = plt.subplots(1,1,figsize=(20,8))
# Generate wordcloud
#fig3,ax3= plt.subplots(1,2)
#stopwords = 

wordcloud =WordCloud(width = 3000, height = 3000, background_color='black', max_words=100,colormap='Set2',stopwords=None).generate(textPBI)
# Plot
#plot_cloud(wordcloud)
plt.title("Positive_Sentiment")
plt.imshow(wordcloud)
st.pyplot(fig1)


#Negative wordcloud
dfN= pd.DataFrame(df['NLP_Review'][df['Sentiment_Value'] < 0])
dfN.rename(columns = {'NLP_Review':'Negative_Review'}, inplace = True)
#dfN

bookN2=[x.strip() for x in dfN.Negative_Review]
bookN3 = [i for i in bookN2 if i]
textN = ' '.join(bookN3)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer_ngram_range = CountVectorizer(analyzer='word',ngram_range=(1,3),max_features = 100)
vectorizer_ngram_range.fit_transform(bookN3)
textNBI=vectorizer_ngram_range.get_feature_names()
textNBI=' '.join(textNBI)

# Generate wordcloud using Bigram
fig2,ax4 = plt.subplots(1,1,figsize=(20,8))
stopwords = STOPWORDS
#stopwords.add('will')
wordcloud_N = WordCloud(width = 3000, height = 3000, background_color='black', max_words=100,colormap='Set2',stopwords=stopwords).generate(textNBI)
plt.title("Negative_Sentiment")
# Plot
plt.imshow(wordcloud_N)
st.pyplot(fig2)


c=df['rating'].count()
percentage=100-(ix*100/c)
p=round(percentage,2)
print("Positive_Sentiment_Percentage:",p)
st.subheader('Positive_Sentiment_Percentage')
st.write(p)