import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE
import catboost as cb

# ---------------- LOAD DATA ----------------
dataset = pd.read_csv("Dataset/healthcare-dataset-stroke-data.csv")
dataset.fillna(0, inplace=True)

# Encoding
enc1, enc2, enc3, enc4, enc5 = LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder()

dataset['gender'] = enc1.fit_transform(dataset['gender'].astype(str))
dataset['ever_married'] = enc2.fit_transform(dataset['ever_married'].astype(str))
dataset['work_type'] = enc3.fit_transform(dataset['work_type'].astype(str))
dataset['Residence_type'] = enc4.fit_transform(dataset['Residence_type'].astype(str))
dataset['smoking_status'] = enc5.fit_transform(dataset['smoking_status'].astype(str))

Y = dataset['stroke']
dataset.drop(['id','stroke'], axis=1, inplace=True)

# Pipeline
scaler = MinMaxScaler()
X = scaler.fit_transform(dataset.values)
X, Y = SMOTE().fit_resample(X, Y)

selector = SelectKBest(chi2, k=9)
X = selector.fit_transform(X, Y)

model = cb.CatBoostClassifier(iterations=200, verbose=0)
model.fit(X, Y)

explainer = shap.TreeExplainer(model)

# ---------------- UI ----------------
st.title("🧠 Stroke Prediction System")

st.sidebar.header("Enter Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 1, 100, 30)
hypertension = st.sidebar.selectbox("Hypertension", [0,1])
heart_disease = st.sidebar.selectbox("Heart Disease", [0,1])
married = st.sidebar.selectbox("Married", ["Yes","No"])
work = st.sidebar.selectbox("Work Type", ["Private","Self-employed","Govt_job"])
res = st.sidebar.selectbox("Residence", ["Urban","Rural"])
glucose = st.sidebar.number_input("Glucose Level", 50.0, 300.0)
bmi = st.sidebar.number_input("BMI", 10.0, 50.0)
smoking = st.sidebar.selectbox("Smoking", ["never smoked","smokes","formerly smoked"])

if st.button("Predict"):

    df = pd.DataFrame([[
        gender, age, hypertension, heart_disease,
        married, work, res, glucose, bmi, smoking
    ]], columns=[
        'gender','age','hypertension','heart_disease',
        'ever_married','work_type','Residence_type',
        'avg_glucose_level','bmi','smoking_status'
    ])

    # Encode
    df['gender']=enc1.transform(df['gender'])
    df['ever_married']=enc2.transform(df['ever_married'])
    df['work_type']=enc3.transform(df['work_type'])
    df['Residence_type']=enc4.transform(df['Residence_type'])
    df['smoking_status']=enc5.transform(df['smoking_status'])

    # Pipeline
    df = scaler.transform(df)
    df = selector.transform(df)

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    st.subheader("Result")
    st.write("Prediction:", "Stroke" if pred==1 else "Normal")
    st.write("Probability:", round(prob*100,2), "%")

    # SHAP
    shap_values = explainer.shap_values(df)

    st.subheader("Explanation")
    fig = plt.figure()
    shap.summary_plot(shap_values, df, show=False)
    st.pyplot(fig)
