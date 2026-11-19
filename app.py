
import streamlit as st
import pandas as pd
import joblib

# Load trained files
model = joblib.load("employee_attrition_model.pkl")
columns = joblib.load("model_columns.pkl")
defaults = joblib.load("model_defaults.pkl")

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 Employee Attrition Prediction")
st.write("AI Powered HR Analytics Dashboard")

st.divider()

st.subheader("Enter Employee Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Employee Age", 18, 60, 30)
    income = st.number_input(
        "Monthly Income",
        min_value=1000,
        max_value=20000,
        value=5000
    )
    years = st.slider("Years at Company", 0, 40, 5)
    overtime = st.selectbox("OverTime", ["No", "Yes"])

with col2:
    department = st.selectbox(
        "Department",
        ["Sales", "Research & Development", "Human Resources"]
    )
    job_role = st.selectbox(
        "Job Role",
        [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Research Director",
            "Human Resources"
        ]
    )
    marital = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )
    distance = st.slider("Distance From Home", 1, 29, 5)

if st.button("Predict Employee Attrition", type="primary"):

    input_data = pd.DataFrame([defaults], columns=columns)

    # Update numerical values
    numeric_values = {
        "Age": age,
        "MonthlyIncome": income,
        "YearsAtCompany": years,
        "DistanceFromHome": distance
    }

    for feature, value in numeric_values.items():
        if feature in input_data.columns:
            input_data.loc[0, feature] = value

    # Update categorical dummy columns
    categories = {
        "OverTime": overtime,
        "Department": department,
        "JobRole": job_role,
        "MaritalStatus": marital
    }

    for feature, value in categories.items():
        prefix = feature + "_"
        matching = [
            c for c in columns if c.startswith(prefix)
        ]

        for c in matching:
            input_data.loc[0, c] = 0

        dummy = prefix + value

        if dummy in input_data.columns:
            input_data.loc[0, dummy] = 1

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    st.metric(
        "Estimated Attrition Risk Score",
        f"{probability * 100:.2f}%"
    )

    if prediction == 1:
        st.error("⚠️ Model predicts: Employee may leave the company.")
    else:
        st.success("✅ Model predicts: Employee may stay in the company.")

    st.caption(
        "This is a machine learning estimate, not a certainty."
    )
