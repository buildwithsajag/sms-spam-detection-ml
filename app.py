import nltk
nltk.download('punkt_tab')

# ... the rest of your imports (streamlit, pickle, etc.) continue below ...
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the message")

if st.button("Predict"):

    # 1. preprocess (Indented by 4 spaces)
    transformed_sms = transform_text(input_sms)

    # 2. vectorize (Indented by 4 spaces)
    vector_input = tfidf.transform([transformed_sms])

    # 3. predict (Indented by 4 spaces)
    result = model.predict(vector_input)[0]

    # 4. Display (Indented by 4 spaces)
    if result == 1:
        st.header("Spam")  # Indented further because it lives inside the 'if result == 1' block
    else:
        st.header("Not Spam")
