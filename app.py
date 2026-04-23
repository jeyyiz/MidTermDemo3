import streamlit as st
import joblib
import pandas as pd

clf_model = joblib.load('artifacts/classifier_pipeline.pkl')
reg_model = joblib.load('artifacts/regressor_pipeline.pkl')

st.set_page_config(page_title="Student Placement Prediction", layout="wide")

def main():
    st.title("Student Placement Prediction")
    st.markdown("Predict whether a student will be **placed** and estimate their **expected salary**.")

    with st.sidebar:
        st.header("About")
        st.info(
            "This app predicts student placement status and expected salary "
            "based on academic performance and profile information."
        )
        st.markdown("**Models Used**")
        st.markdown("- Classification: Logistic Regression")
        st.markdown("- Regression: Gradient Boosting")

    with st.form("prediction_form"):
        st.subheader("Academic Profile")
        col1, col2, col3 = st.columns(3)

        with col1:
            cgpa = st.number_input("CGPA", 0.0, 10.0, value=7.5, step=0.01)
            tenth_percentage = st.number_input("10th Percentage", 0.0, 100.0, value=75.0, step=0.1)
            twelfth_percentage = st.number_input("12th Percentage", 0.0, 100.0, value=75.0, step=0.1)
            backlogs = st.number_input("Backlogs", 0, 10, value=0)
            attendance_percentage = st.number_input("Attendance (%)", 0.0, 100.0, value=80.0, step=0.1)

        with col2:
            study_hours_per_day = st.number_input("Study Hours/Day", 0.0, 12.0, value=4.0, step=0.1)
            projects_completed = st.number_input("Projects Completed", 0, 20, value=4)
            internships_completed = st.number_input("Internships Completed", 0, 10, value=1)
            hackathons_participated = st.number_input("Hackathons Participated", 0, 20, value=2)
            certifications_count = st.number_input("Certifications", 0, 20, value=2)

        with col3:
            coding_skill_rating = st.slider("Coding Skill (1-5)", 1, 5, value=3)
            communication_skill_rating = st.slider("Communication Skill (1-5)", 1, 5, value=3)
            aptitude_skill_rating = st.slider("Aptitude Skill (1-5)", 1, 5, value=3)
            sleep_hours = st.number_input("Sleep Hours/Day", 0.0, 12.0, value=7.0, step=0.1)
            stress_level = st.slider("Stress Level (1-10)", 1, 10, value=5)

        st.subheader("Personal Profile")
        col4, col5 = st.columns(2)

        with col4:
            gender = st.selectbox("Gender", ["Male", "Female"])
            branch = st.selectbox("Branch", ["CSE", "ECE", "IT", "ME", "CE"])
            part_time_job = st.selectbox("Part-Time Job", ["No", "Yes"])

        with col5:
            internet_access = st.selectbox("Internet Access", ["Yes", "No"])
            family_income_level = st.selectbox("Family Income Level", ["Low", "Medium", "High"])
            extracurricular_involvement = st.selectbox("Extracurricular Involvement", ["Low", "Medium", "High"])
            city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])

        submitted = st.form_submit_button("Predict")

    if submitted:
        features = {
            'gender': gender,
            'branch': branch,
            'cgpa': float(cgpa),
            'tenth_percentage': float(tenth_percentage),
            'twelfth_percentage': float(twelfth_percentage),
            'backlogs': int(backlogs),
            'study_hours_per_day': float(study_hours_per_day),
            'attendance_percentage': float(attendance_percentage),
            'projects_completed': int(projects_completed),
            'internships_completed': int(internships_completed),
            'coding_skill_rating': int(coding_skill_rating),
            'communication_skill_rating': int(communication_skill_rating),
            'aptitude_skill_rating': int(aptitude_skill_rating),
            'hackathons_participated': int(hackathons_participated),
            'certifications_count': int(certifications_count),
            'sleep_hours': float(sleep_hours),
            'stress_level': int(stress_level),
            'part_time_job': part_time_job,
            'family_income_level': family_income_level,
            'city_tier': city_tier,
            'internet_access': internet_access,
            'extracurricular_involvement': extracurricular_involvement,
        }

        input_df = pd.DataFrame([features])

        placement_pred = clf_model.predict(input_df)[0]
        placement_proba = clf_model.predict_proba(input_df)[0][1]
        salary_pred = reg_model.predict(input_df)[0]

        st.markdown("---")
        st.subheader("Prediction Results")

        col6, col7 = st.columns(2)

        with col6:
            if placement_pred == 1:
                st.success("✅ Placement Status: **Placed**")
            else:
                st.error("❌ Placement Status: **Not Placed**")
            st.metric("Placement Probability", f"{placement_proba * 100:.1f}%")

        with col7:
            if placement_pred == 1:
                st.metric("Estimated Salary", f"₹ {salary_pred:.2f} LPA")
            else:
                st.warning("Salary prediction is only shown for placed students.")

        st.markdown("---")
        st.subheader("Input Summary")
        st.dataframe(input_df, use_container_width=True)


if __name__ == "__main__":
    main()