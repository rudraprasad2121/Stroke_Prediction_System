import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, render_template, request, redirect, session
import pymysql

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import chi2, SelectKBest

import catboost as cb
import shap
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'welcome'

# ---------------- DATABASE ----------------
def getConnection():
    return pymysql.connect(host='localhost', user='root', password='', database='stroke_db')


# ---------------- LOAD + TRAIN MODEL ----------------
dataset = pd.read_csv("Dataset/healthcare-dataset-stroke-data.csv")
dataset.fillna(0, inplace=True)

labels = ['Normal', 'Stroke']

# Label Encoding
enc1, enc2, enc3, enc4, enc5 = LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder()

dataset['gender'] = enc1.fit_transform(dataset['gender'].astype(str))
dataset['ever_married'] = enc2.fit_transform(dataset['ever_married'].astype(str))
dataset['work_type'] = enc3.fit_transform(dataset['work_type'].astype(str))
dataset['Residence_type'] = enc4.fit_transform(dataset['Residence_type'].astype(str))
dataset['smoking_status'] = enc5.fit_transform(dataset['smoking_status'].astype(str))

Y = dataset['stroke']
dataset.drop(['id','stroke'], axis=1, inplace=True)

# -------- CORRECT PIPELINE --------
scaler = MinMaxScaler()
X = scaler.fit_transform(dataset.values)

X, Y = SMOTE().fit_resample(X, Y)

selector = SelectKBest(chi2, k=9)
X = selector.fit_transform(X, Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Better model (handling imbalance)
model = cb.CatBoostClassifier(
    iterations=300,
    learning_rate=0.1,
    scale_pos_weight=5,
    verbose=0
)

model.fit(X_train, y_train)

# SHAP Explainer (create once)
explainer = shap.TreeExplainer(model)

# Ensure folder exists
os.makedirs("static/images", exist_ok=True)


# ---------------- ROUTES ----------------

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        con = getConnection()
        cur = con.cursor()
        cur.execute("INSERT INTO users(name,email,password) VALUES(%s,%s,%s)",
                    (request.form['name'], request.form['email'], request.form['password']))
        con.commit()
        con.close()
        return redirect('/login')

    return render_template('register.html')


@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        con = getConnection()
        cur = con.cursor()
        cur.execute("SELECT * FROM users WHERE email=%s AND password=%s",
                    (request.form['email'], request.form['password']))
        user = cur.fetchone()
        con.close()

        if user:
            session['user'] = request.form['email']
            return redirect('/dashboard')
        else:
            return render_template('login.html', msg="Invalid login")

    return render_template('login.html')


@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/login')

    con = getConnection()
    cur = con.cursor()

    # Total predictions
    cur.execute("SELECT COUNT(*) FROM predictions WHERE user_email=%s", (session['user'],))
    total = cur.fetchone()[0]

    # Stroke count
    cur.execute("SELECT COUNT(*) FROM predictions WHERE result='Stroke' AND user_email=%s", (session['user'],))
    stroke = cur.fetchone()[0]

    # Normal count
    cur.execute("SELECT COUNT(*) FROM predictions WHERE result='Normal' AND user_email=%s", (session['user'],))
    normal = cur.fetchone()[0]

    con.close()

    return render_template(
        'dashboard.html',
        total=total,
        stroke=stroke,
        normal=normal
    )

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')


# ---------------- PREDICTION ----------------

@app.route('/predict', methods=['GET','POST'])
def predict():
    if 'user' not in session:
        return redirect('/login')

    if request.method == 'POST':

        # Input dataframe
        df = pd.DataFrame([[
            request.form['gender'],
            float(request.form['age']),
            int(request.form['hypertension']),
            int(request.form['heart_disease']),
            request.form['ever_married'],
            request.form['work_type'],
            request.form['Residence_type'],
            float(request.form['avg_glucose_level']),
            float(request.form['bmi']),
            request.form['smoking_status']
        ]], columns=[
            'gender','age','hypertension','heart_disease',
            'ever_married','work_type','Residence_type',
            'avg_glucose_level','bmi','smoking_status'
        ])

        # Encode (same as training)
        df['gender'] = enc1.transform(df['gender'])
        df['ever_married'] = enc2.transform(df['ever_married'])
        df['work_type'] = enc3.transform(df['work_type'])
        df['Residence_type'] = enc4.transform(df['Residence_type'])
        df['smoking_status'] = enc5.transform(df['smoking_status'])

        # Apply SAME scaler + selector
        df_scaled = scaler.transform(df)
        df_selected = selector.transform(df_scaled)

        # Prediction
        pred = model.predict(df_selected)[0]
        prob = model.predict_proba(df_selected)[0][1]

        risk = "Low" if prob < 0.3 else "Medium" if prob < 0.7 else "High"

        # -------- SHAP (single prediction waterfall) --------
        shap_values = explainer.shap_values(df_selected)

        plt.figure()
        shap.summary_plot(shap_values, df_selected, show=False)

        img_path = "static/images/shap.png"
        plt.savefig(img_path, bbox_inches='tight')
        plt.close()

        # Save to DB
        con = getConnection()
        cur = con.cursor()
        cur.execute(
            "INSERT INTO predictions(user_email,result,probability,risk) VALUES(%s,%s,%s,%s)",
            (session['user'], labels[pred], float(prob), risk)
        )
        con.commit()
        con.close()

        return render_template(
            'result.html',
            result=labels[pred],
            prob=round(prob*100, 2),
            risk=risk
        )

    return render_template('predict.html')


# ---------------- HISTORY ----------------

@app.route('/history')
def history():
    if 'user' not in session:
        return redirect('/login')

    con = getConnection()
    cur = con.cursor()

    cur.execute("SELECT result,probability,risk,created_at FROM predictions WHERE user_email=%s",
                (session['user'],))
    data = cur.fetchall()

    cur.execute("SELECT COUNT(*) FROM predictions WHERE risk='High' AND user_email=%s",(session['user'],))
    high = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM predictions WHERE risk='Medium' AND user_email=%s",(session['user'],))
    medium = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM predictions WHERE risk='Low' AND user_email=%s",(session['user'],))
    low = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM predictions WHERE result='Stroke' AND user_email=%s",(session['user'],))
    stroke = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM predictions WHERE result='Normal' AND user_email=%s",(session['user'],))
    normal = cur.fetchone()[0]

    con.close()

    return render_template(
        'history.html',
        data=data,
        high=high,
        medium=medium,
        low=low,
        stroke=stroke,
        normal=normal
    )


# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(debug=True)