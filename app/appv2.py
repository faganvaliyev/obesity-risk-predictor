import streamlit as st
import joblib
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np

model = joblib.load('app/model.joblib')
scaler = joblib.load('app/scaler.joblib')

file_path = 'app/clean_df.csv'
df = pd.read_csv(file_path)

st.set_page_config(page_title="Obesity Risk Predictor", layout="wide", page_icon="🧑‍⚕️")

st.title("🧑‍⚕️ Obesity Risk Predictor")
st.markdown("Use this tool to assess your potential obesity risk based on your health and lifestyle inputs.")

with st.sidebar:
    st.header("📝 Input Information")
    age = st.slider("Age", int(df['Age'].min()), 65, int(df['Age'].mean()))
    gender = st.selectbox("Gender", ["Male", "Female"])
    height = st.slider("Height (meters)", float(df['Height'].min()), float(df['Height'].max()), float(df['Height'].mean()))
    weight = st.slider("Weight (kg)", float(df['Weight'].min()), float(df['Weight'].max()), float(df['Weight'].mean()))
    family_history_with_overweight = st.radio("Family member has overweight?", ["yes", "no"])
    favc = st.radio("High-caloric food consumption?", ["yes", "no"])
    fcvc = st.slider("Vegetable consumption frequency (0–3)", int(df['FCVC'].min()), int(df["FCVC"].max()), 1)
    ncp = st.slider("Main meals/day", int(df['NCP'].min()), int(df['NCP'].max()), 3)
    caec = st.selectbox("Snacking frequency", ['Sometimes', 'Frequently', 'Always', 'no'])
    smoke = st.radio("Do you smoke?", ['yes', 'no'])
    ch20 = st.slider("Daily water intake (cups)", int(df['CH2O'].min()), int(df['CH2O'].max()), 2)
    scc = st.radio("Do you monitor calories?", ['yes', 'no'])
    faf = st.slider("Physical activity (hours/week)", int(df['FAF'].min()), int(df["FAF"].max()), 1)
    tue = st.slider("Tech usage (hours/day)", int(df['TUE'].min()), int(df['TUE'].max()), 3)
    calc = st.selectbox("Alcohol consumption", ['Sometimes', 'Frequently', 'Always', 'no'])
    mtrans = st.selectbox("Transportation", ('Public_Transportation', 'Walking', 'Motorbike', 'Automobile', 'Bike'))

# Organize input
input_data = {
    "Gender": [gender],
    "Age": [age],
    "Height": [height],
    "Weight": [weight],
    "family_history_with_overweight": [family_history_with_overweight],
    "FAVC": [favc],
    "FCVC": [fcvc],
    "NCP": [ncp],
    "CAEC": [caec],
    "SMOKE": [smoke],
    "CH2O": [ch20],
    "SCC": [scc],
    "FAF": [faf],
    "TUE": [tue],
    "CALC": [calc],
    "MTRANS": [mtrans]
}

input_df = pd.DataFrame(input_data)

with st.expander("🔍 Show Raw Input Data"):
    st.dataframe(input_df)

binary_map = {'Male': 1, 'Female': 0, 'yes': 1, 'no': 0}
ordinal_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
binary_cols = ['family_history_with_overweight', 'Gender', 'FAVC', 'SCC', 'SMOKE']
ordinal_cols = ['CALC', 'CAEC']
categories = ['Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking']

for col in binary_cols:
    input_df[col] = input_df[col].map(binary_map)

for col in ordinal_cols:
    input_df[col] = input_df[col].map(ordinal_map)

input_df['MTRANS'] = input_df['MTRANS'].astype(CategoricalDtype(categories=categories))
input_df = pd.get_dummies(input_df, columns=['MTRANS'], prefix='MTRANS')

scaled_input = scaler.transform(input_df)

with st.expander("⚙️ Preprocessed Input for Model"):
    st.dataframe(pd.DataFrame(scaled_input, columns=input_df.columns))

prediction = model.predict(scaled_input)
prediction_proba = model.predict_proba(scaled_input)

classes = [
    "Insufficient_Weight", "Normal_Weight", "Obesity_Type_I", 
    "Obesity_Type_II", "Obesity_Type_III", 
    "Overweight_Level_I", "Overweight_Level_II"
]

df_proba = pd.DataFrame(prediction_proba, columns=classes).T.reset_index()
df_proba.columns = ['Category', 'Probability']

st.markdown("### 🧠 Prediction Result")
st.success(f"💡 Based on your inputs, you are classified as: **{classes[prediction[0]]}**")

st.markdown("### 📊 Prediction Probability")
st.bar_chart(df_proba.set_index('Category'))

