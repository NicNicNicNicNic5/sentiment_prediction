import streamlit as st
import joblib
import re
import pandas as pd
import plotly.express as px
model = joblib.load('model.pkl')
st.set_page_config(layout='wide')
st.title('Sentiment Prediction')
col1, col2 = st.columns(2)

with col1:
    text = st.text_area("Text to analyze: ")

prediction = model.predict([text])[0]
proba = model.predict_proba([text])[0]

output = re.sub(r"[^\w\s]", "", str(prediction))
print(type(output))
if st.button("Predict"):
    st.markdown("Prediction here: **{}**".format(output))