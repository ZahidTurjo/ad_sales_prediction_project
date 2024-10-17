import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import pickle

model=pickle.load(open('ad_sales_prediction.pkl','rb'))
st.title("Ad Sales Prediction..")
tv=st.text_input('Enter Tv Sales:')
radio=st.text_input('Enter Radio Sales:')
newspaper=st.text_input('Enter Newspaper Sales:')
if st.button('predict'):
    features=np.array([[tv,radio,newspaper]],dtype=np.float64)
    result=model.predict(features).reshape(1,-1)
    st.write("Predited sale:",result[0])
    

