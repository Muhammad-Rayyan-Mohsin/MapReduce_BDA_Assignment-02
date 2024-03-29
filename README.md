# MapReduce

# Data preprocessing code

import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Read the dataset
df = pd.read_csv("Dataset.csv")

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

df['SECTION_TEXT'] = df['SECTION_TEXT'].apply(preprocess_text)

df.to_csv("Cleaned_Dataset.csv", index=False)


#Code for generating  tf/idf values for each word in a document or sentence

sentence1="I wonder how many miles"
sentence2="I’ve fallen by this time?"
vocabulary=sentence1+" "+sentence2
uniquewords=[]
for x in vocabulary.split():
    if x not in uniquewords:
        uniquewords.append(x)
wordcounts=[]
u=0
wordcounts = {}
for word in uniquewords:
        wordcounts[u] = word
        u=u+1
idfrequency={}
for x in range(len(wordcounts)):
  idfrequency[x]=sentence1.count(wordcounts[x])
idfrequency2={}
for x in range(len(wordcounts)):
  idfrequency2[x]=sentence2.count(wordcounts[x])
for x in range(len(wordcounts)):
  if idfrequency[x]==0:
    del idfrequency[x]
for x in range(len(wordcounts)):
  if idfrequency2[x]==0:
    del idfrequency2[x]
weights1={}
for x in range(len(wordcounts)):
 tf=sentence1.count(wordcounts[x])
 idf=vocabulary.count(wordcounts[x])
 weights1[x]=tf/idf
print(weights1)
weights2={}
for x in range(len(wordcounts)):
 tf=sentence2.count(wordcounts[x])
 idf=vocabulary.count(wordcounts[x])
 weights2[x]=tf/idf
print(weights2)
