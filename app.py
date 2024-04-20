import streamlit as st
from nltk.corpus import stopwords
from nltk import PorterStemmer 
import pickle
import string


model = pickle.load(open('model.pkl' , 'rb'))
n_gram = pickle.load(open('Vectorizer.pkl' , 'rb')  )


def preprocessing(text):
    text = text.lower()
    ps = PorterStemmer()
    
    y = []

    for word in text.split():
        if word.isalnum():
            y.append(word)
    text = " ".join(y[:])
    y.clear()

    for word in text.split():
        if word not in stopwords.words('english') and word not in string.punctuation:
            y.append(ps.stem(word))

    text = " ".join(y[:])
    y.clear()

    return text

st.title("Sms Spam Classifier" )

st.sidebar.title("About")
st.sidebar.info(
    "This is a simple SMS spam classifier app. Enter an SMS message in the text area and click 'Predict' to classify whether it's spam or not."
)
st.sidebar.markdown(
    "<a href='https://github.com/Shantnu-singh/textGuard' target='_blank'><i class='fab fa-github'></i> GitHub Repo</a>", 
    unsafe_allow_html=True
)

input = st.text_area("Enter Spam messages here...")


if st.button('Predict', key="predict_button", help="Click to classify the message"):
    if input:
        # preprocess the input
        preprocessed_input = preprocessing(input)

        # Vectorlization
        vectorize_input = n_gram.transform([preprocessed_input])

        # Predic
        ans = model.predict(vectorize_input)[0]

        # Finding probability 
        prediction_proba = model.predict_proba(vectorize_input)[0]


        if prediction_proba[1] > .5 :
            st.error(f"Spam Message (Probability: {prediction_proba[1]:.2f})")

        else :
            st.success(f"Not a Spam Message (Probability: {prediction_proba[0]:.2f})")
    else:
        st.warning("Please Enter a text for Prediction")

