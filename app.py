import streamlit as st
import numpy as np
from pickle import load

scaler = load(open('models/standard_scaler.pkl', 'rb'))
lr_model = load(open('models/nb_model.pkl', 'rb'))

st.title(':red[Penguin] Species Predictor')

cl = st.text_input("Culmen Length", placeholder="Enter value in mm")
cd = st.text_input("Culmen Depth", placeholder="Enter value in mm")
fl = st.text_input("Flipper Length", placeholder="Enter value in mm")
bm = st.text_input("Body Mass (g)", placeholder="Enter value in grams")

btn_click = st.button("Predict")

if btn_click == True:
    if cl and cd and fl and bm:
        query_point = np.array([float(cl), float(cd), float(fl), float(bm)]).reshape(1, -1)
        query_point_transformed = scaler.transform(query_point)
        pred = lr_model.predict(query_point_transformed)
        st.success(pred)
    else:
        st.error("Enter the values properly.")