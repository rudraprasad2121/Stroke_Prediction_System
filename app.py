import streamlit as st
import numpy as np
import pandas as pd
import time

st.set_page_config(page_title="Stroke Prediction", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
.big-title {font-size:32px; font-weight:bold; color:#4F46E5;}
.card {
    padding:20px;
    border-radius:10px;
    background:#f8fafc;
    box-shadow:0px 2px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown('<p class="big-title">🧠 Stroke Prediction System</p>', unsafe_allow_html=True)

# ---------- SIDEBAR ----------
menu = st.sidebar.radio("Navigation", ["Prediction", "About"])

# ---------- PREDICTION PAGE ----------
if menu == "Prediction":

    st.subheader("Enter Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 1, 100, 25)
        hypertension = st.selectbox("Hypertension", [0,1])
        heart = st.selectbox("Heart Disease", [0,1])
        married = st.selectbox("Ever Married", ["Yes","No"])

    with col2:
        work = st.selectbox("Work Type", ["Private","Govt","Self-employed"])
        residence = st.selectbox("Residence", ["Urban","Rural"])
        glucose = st.slider("Glucose Level", 50, 300, 100)
        bmi = st.slider("BMI", 10.0, 50.0, 25.0)
        smoke = st.selectbox("Smoking", ["never","smokes","formerly smoked"])

    if st.button("🔍 Predict"):

        # Fake prediction (replace with your model)
        prob = np.random.rand()

        risk = "Low" if prob < 0.3 else "Medium" if prob < 0.7 else "High"

        # ---------- LOADING ----------
        with st.spinner("Analyzing data..."):
            time.sleep(1.5)

        st.success("Prediction Completed")

        # ---------- METRICS ----------
        col1, col2, col3 = st.columns(3)

        col1.metric("Prediction", "Stroke" if prob>0.5 else "Normal")
        col2.metric("Probability", f"{round(prob*100,2)}%")
        col3.metric("Risk Level", risk)

        # ---------- PROGRESS BAR ----------
        st.subheader("Risk Level Indicator")
        st.progress(int(prob * 100))

        # ---------- CHART ----------
        st.subheader("Result Visualization")

        chart_data = pd.DataFrame({
            "Category": ["Stroke Risk", "Normal"],
            "Value": [prob, 1-prob]
        })

        st.bar_chart(chart_data.set_index("Category"))

# ---------- ABOUT PAGE ----------
elif menu == "About":
    st.subheader("About Project")
    st.write("""
    This system predicts stroke risk using Machine Learning.
    It includes Explainable AI (SHAP) and interactive dashboards.
    """)
