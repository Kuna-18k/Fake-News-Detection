import streamlit as st
import joblib

vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("model.jb")

st.title("Fake News Detector")
st.write("Enter a News Article below to check whether it is Fake or Real. ")

inputn = st.text_area("News Article:","")

if st.button("Check News"):
    if inputn.strip():
        transform_input = vectorizer.transform([inputn])
        prediction = model.predict(transform_input)

        if prediction[0] == 1:
            st.success("The News is Real! ")
        else:
            st.error("The News is Fake! ")
    else:
        st.warning("Please enter some text to Analyze. ") 