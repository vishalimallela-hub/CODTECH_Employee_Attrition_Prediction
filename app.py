
import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
analytics_df = pd.read_csv(DATA_PATH)
analytics_df = pd.read_csv(DATA_PATH)

model = joblib.load(BASE_DIR / "attrition_model.joblib")

with open(BASE_DIR / "model_results.json", "r") as f:
    results = json.load(f)

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("Employee Attrition Prediction")
st.write(
    "Explore how employee characteristics relate to a machine-learning "
    "model's prediction of attrition. This is a demonstration, not an "
    "employment decision tool."
)

st.subheader("Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("Test Accuracy", f"{results['accuracy']:.1%}")
col2.metric("ROC-AUC", f"{results['roc_auc']:.3f}")
col3.metric("Test Records", results["testing_records"])

st.divider()
st.subheader("Enter Employee Details")

with st.form("employee_form"):
    st.write("Enter employee details")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 60, 30)
        business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        daily_rate = st.number_input("Daily Rate", 100, 1500, 800)
        department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
        distance = st.slider("Distance From Home", 1, 29, 5)
        education = st.selectbox("Education", [1, 2, 3, 4, 5])
        education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"])
        environment_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
        gender = st.selectbox("Gender", ["Male", "Female"])
        hourly_rate = st.number_input("Hourly Rate", 30, 100, 65)

    with col2:
        job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4])
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
        job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000, step=500)
        monthly_rate = st.number_input("Monthly Rate", 2000, 27000, 14000, step=500)
        num_companies = st.number_input("Number of Previous Companies", 0, 10, 2)
        overtime = st.selectbox("Overtime", ["No", "Yes"])
        salary_hike = st.slider("Percent Salary Hike", 11, 25, 15)

    with col3:
        performance_rating = st.selectbox("Performance Rating", [1, 2, 3, 4])
        relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4])
        stock_option = st.selectbox("Stock Option Level", [0, 1, 2, 3])
        total_working_years = st.slider("Total Working Years", 0, 40, 8)
        training_times = st.slider("Training Times Last Year", 0, 6, 3)
        work_life_balance = st.selectbox("Work-Life Balance", [1, 2, 3, 4])
        years_at_company = st.slider("Years at Company", 0, 40, 5)
        years_in_role = st.slider("Years in Current Role", 0, 18, 3)
        years_since_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
        years_with_manager = st.slider("Years With Current Manager", 0, 17, 3)

    submitted = st.form_submit_button("Predict Attrition")

    st.divider()
st.header("What-if Simulator")
st.write("Compare how changing overtime affects a hypothetical employee.")

sim_overtime = st.selectbox(
    "Overtime for simulation",
    ["Yes", "No"],
    key="sim_overtime"
)

if st.button("Compare Overtime Scenarios"):
    scenario_no = st.session_state.get("employee_sample")

    if scenario_no is None:
        st.info("Submit an employee prediction first, then compare overtime scenarios.")
    else:
        scenario_yes = scenario_no.copy()
        scenario_no = scenario_no.copy()

        scenario_yes["OverTime"] = "Yes"
        scenario_no["OverTime"] = "No"

        prob_yes = model.predict_proba(scenario_yes)[0][1] * 100
        prob_no = model.predict_proba(scenario_no)[0][1] * 100

        st.write(f"Predicted attrition probability with overtime: **{prob_yes:.1f}%**")
        st.write(f"Predicted attrition probability without overtime: **{prob_no:.1f}%**")
        st.metric(
            "Difference (Yes − No)",
            f"{prob_yes - prob_no:+.1f} percentage points"
        )
        
if submitted:
    sample = pd.DataFrame([{
        "Age": age,
        "BusinessTravel": business_travel,
        "DailyRate": daily_rate,
        "Department": department,
        "DistanceFromHome": distance,
        "Education": education,
        "EducationField": education_field,
        "EnvironmentSatisfaction": environment_satisfaction,
        "Gender": gender,
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobRole": job_role,
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies,
        "OverTime": overtime,
        "PercentSalaryHike": salary_hike,
        "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StockOptionLevel": stock_option,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_role,
        "YearsSinceLastPromotion": years_since_promotion,
        "YearsWithCurrManager": years_with_manager
    }])

    expected_columns = model.named_steps["preprocessor"].feature_names_in_
    sample = sample.reindex(columns=expected_columns)
    st.session_state["employee_sample"] = sample.copy()

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][1]
    
    probabilities = model.predict_proba(sample)[0]
    class_labels = model.classes_

    probability_data = pd.DataFrame({
        "Outcome": [
            "No Attrition",
            "Attrition"
        ],
        "Probability (%)": [
            probabilities[list(class_labels).index(0)] * 100,
            probabilities[list(class_labels).index(1)] * 100
        ]
    })

    st.subheader("Predicted Probability Breakdown")
    st.bar_chart(
        probability_data.set_index("Outcome"),
        y="Probability (%)",
        y_label="Probability (%)"
    )


    st.subheader("Prediction Result")
    st.metric("Estimated Attrition Probability", f"{probability:.1%}")

    if prediction == 1:
        st.warning("The model predicts the attrition class for these inputs.")
    else:
        st.success("The model predicts the no-attrition class for these inputs.")

    st.caption(
        "This is an uncertain statistical prediction, not a reliable "
        "judgment about an individual employee."
    )
st.divider()
st.subheader("Project Visualizations")

for filename, caption in [
    ("attrition_distribution.png", "Attrition Distribution"),
    ("confusion_matrix.png", "Model Confusion Matrix")
]:
    image_path = BASE_DIR / "visualizations" / filename
    if image_path.exists():
        st.image(str(image_path), caption=caption, use_container_width=True)

st.divider()
st.header("Interactive Employee Analytics")


st.subheader("Explore Attrition by Department")

selected_department = st.selectbox(
    "Choose a department",
    ["All Departments"] + sorted(analytics_df["Department"].unique().tolist())
)

if selected_department == "All Departments":
    filtered_df = analytics_df
else:
    filtered_df = analytics_df[
        analytics_df["Department"] == selected_department
    ]

department_attrition = (
    filtered_df.groupby("Department")["Attrition"]
    .value_counts()
    .unstack(fill_value=0)
)

st.bar_chart(department_attrition)


department_attrition = (
    analytics_df.groupby("Department")["Attrition"]
    .value_counts()
    .unstack(fill_value=0)
)

st.bar_chart(department_attrition)

st.subheader("Attrition by Overtime")

overtime_attrition = (
    analytics_df.groupby("OverTime")["Attrition"]
    .value_counts(normalize=True)
    .unstack(fill_value=0) * 100
)

st.bar_chart(overtime_attrition)

st.subheader("Explore Attrition by Job Role")

selected_role = st.selectbox(
    "Choose a job role",
    ["All Job Roles"] + sorted(analytics_df["JobRole"].unique().tolist()),
    key="job_role_filter"
)

if selected_role == "All Job Roles":
    role_df = analytics_df
else:
    role_df = analytics_df[analytics_df["JobRole"] == selected_role]

role_attrition = (
    role_df.groupby("JobRole")["Attrition"]
    .value_counts()
    .unstack(fill_value=0)
)

st.bar_chart(role_attrition)
