import streamlit as st
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import spacy
import pickle
import pandas as pd

import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


from nltk.stem import WordNetLemmatizer

import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# import en_core_web_sm
# nlp = en_core_web_sm.load()
df = pd.read_csv('episodes-sample.csv')

def text_transform(text):
    text = str(text)
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    
    lemmatizer = WordNetLemmatizer()
    
    # removing special chars
    for i in text:
        if i.isalnum() and (i not in stopwords.words('english') and i not in string.punctuation):
            i = lemmatizer.lemmatize(i)
            y.append(i)

    return " ".join(y) # return the list in the form of string    

def search_query(query , tfidf_matrix , vectorizer):
    preprocessed_query = text_transform(query)
    query_vector = vectorizer.transform([preprocessed_query])
    
    similarity_scores = cosine_similarity(query_vector , tfidf_matrix)
    top_indices = similarity_scores.argsort()[0][::-1]
    
    return df.iloc[top_indices[:5]]
    
vectorizer = pickle.load(open('vectorizer.pkl' , 'rb'))
tfidf_matrix = pickle.load(open('tfidf-matrix.pkl' , 'rb'))

# Streamlit interface
st.title("Search Engine")
query = st.text_input("Enter your search query:")


if query:
    search_results = search_query(query, tfidf_matrix, vectorizer)
    for index, row in search_results.iterrows():
        st.write(f"**Title**: {row['title']}")
        st.write(f"**Description**: {row['description']}")
        st.write(f"[Link]({row['link']})")
        st.write("---")
