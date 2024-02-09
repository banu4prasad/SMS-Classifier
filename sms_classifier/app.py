import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
#lets load 
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model =  pickle.load(open('model.pkl','rb'))

#transform text
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower() #converting to lower case
    text = nltk.word_tokenize(text) #tokenization

    #removing spcl characters/punctuations/stopwords
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    #applying stemming
    text = [ps.stem(word) for word in text]
    return " ".join(text)

#streamlit app
st.title("SMS Spam Classifier")
input_sms = st. text_area("Enter the message")

if st.button('Predict'):
    #preprocess the input message
    transformed_sms = transform_text(input_sms)

    #vectorize the preporcessed message
    vector_input = tfidf.transform([transformed_sms])

    #predict
    result = model.predict(vector_input)[0]

    #display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not spam")