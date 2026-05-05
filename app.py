import streamlit as st
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Stroke Prediction", layout="wide")

st.title("🧠 Stroke Prediction System")

# ---------------- LOAD DATA ----------------
dataset = pd.read_csv("Dataset/healthcare-dataset-stroke-data.csv")
dataset.fillna(0, inplace=True)

# ---------------- ENCODING ----------------
encoders = {}
for col in ['gender','ever_married','work_type','Residence_type','smoking_status']:
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col].astype(str))
    encoders[col] = le

# ---------------- SPLIT ----------------
Y = dataset['stroke']
X = dataset.drop(['id','stroke'], axis=1)

# ---------------- SCALE ----------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- FEATURE SELECT ----------------
selector = SelectKBest(score_func=chi2, k=9)
X_selected = selector.fit_transform(X_scaled, Y)

# ---------------- MODEL ----------------
X_train, X_test, y_train, y_test = train_test_split(X_selected, Y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

st.success(f"Model Accuracy: {round(model.score(X_test, y_test)*100,2)}%")

# ---------------- UI ----------------
st.subheader("Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 1, 100, 25)
    hypertension = st.selectbox("Hypertension", [0,1])
    heart = st.selectbox("Heart Disease", [0,1])
    married = st.selectbox("Ever Married", ["Yes","No"])

with col2:
    work = st.selectbox("Work Type", ["Private","Self-employed","Govt_job"])
    residence = st.selectbox("Residence", ["Urban","Rural"])
    glucose = st.slider("Glucose Level", 50, 300, 100)
    bmi = st.slider("BMI", 10.0, 50.0, 25.0)
    smoke = st.selectbox("Smoking", ["never smoked","smokes","formerly smoked"])

# ---------------- PREDICT ----------------
if st.button("🔍 Predict"):

    try:
        # -------- ENCODE --------
        gender = encoders['gender'].transform([gender])[0]
        married = encoders['ever_married'].transform([married])[0]
        work = encoders['work_type'].transform([work])[0]
        residence = encoders['Residence_type'].transform([residence])[0]
        smoke = encoders['smoking_status'].transform([smoke])[0]

        # -------- INPUT --------
        input_data = np.array([[gender, age, hypertension, heart,
                                married, work, residence,
                                glucose, bmi, smoke]])

        # -------- SCALE --------
        input_scaled = scaler.transform(input_data)
        input_selected = selector.transform(input_scaled)

        # -------- PREDICT --------
        prob = model.predict_proba(input_selected)[0][1]

        # -------- RISK --------
        if prob < 0.3:
            risk = "Low"
        elif prob < 0.7:
            risk = "Medium"
        else:
            risk = "High"

        result = "Stroke" if prob > 0.5 else "Normal"

        # -------- SAFETY RULE --------
        if glucose < 140 and bmi < 30 and hypertension == 0:
            result = "Normal"
            risk = "Low"
            prob = min(prob, 0.3)

        # ---------------- OUTPUT UI ----------------
        st.success("Prediction Complete")

        col1, col2, col3 = st.columns(3)

        col1.metric("Prediction", result)
        col2.metric("Probability", f"{round(prob*100,2)}%")
        col3.metric("Risk Level", risk)

        st.progress(int(prob * 100))

        st.bar_chart(pd.DataFrame({
            "Values": [prob, 1-prob]
        }, index=["Stroke Risk", "Normal"]))

    except Exception as e:
        st.error(str(e))
