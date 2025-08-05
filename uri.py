import streamlit as st
import joblib


m1 = joblib.load("sentiment_model.pkl")
vec = joblib.load("tfidf_vectorizer.pkl")

Sentiment = {
    1: "Positive Feedback 😊",
    0: "Neutral Feedback 😉",
    -1: "Negative Feedback 😞"
}


st.title("📃 Comments Analyzer")
st.slider("Rate us", 1, 10, 7)

user_input = st.text_area("Enter your comments")

if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
      
        v = vec.transform([user_input])
        
        
        prd = m1.predict(v)[0]  
        fb = Sentiment.get(prd, "Unknown Sentiment 🤔")
        
      
        st.subheader("Prediction:")
        st.write(fb)
    else:
        st.warning("Please enter a comment")

    

