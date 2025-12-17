import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("student_stress_data.csv")

data['stress_level'] = data['stress_level'].map({
    'Low': 0,
    'Medium': 1,
    'High': 2
})

X = data[['study_hours', 'sleep_hours', 'attendance',
          'exam_pressure', 'mental_health']]
y = data['stress_level']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

st.title("🎓 Student Stress Level Prediction")

study_hours = st.slider("Study Hours", 0, 10, 4)
sleep_hours = st.slider("Sleep Hours", 0, 10, 6)
attendance = st.slider("Attendance (%)", 0, 100, 70)
exam_pressure = st.slider("Exam Pressure (1-10)", 1, 10, 7)
mental_health = st.slider("Mental Health Score (1-10)", 1, 10, 4)

if st.button("Predict Stress Level"):
    result = model.predict([[study_hours, sleep_hours,
                              attendance, exam_pressure,
                              mental_health]])

    if result[0] == 0:
        st.success("Stress Level: Low")
    elif result[0] == 1:
        st.warning("Stress Level: Medium")
    else:
        st.error("Stress Level: High")
