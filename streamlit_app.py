import os

import requests
import streamlit as st


@st.cache_data(ttl=5, show_spinner=False)
def get_api_status(health_url: str) -> bool:
    try:
        health_response = requests.get(health_url, timeout=1)
        return health_response.status_code == 200
    except requests.RequestException:
        return False


def run_app():
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    predict_url = f"{api_base_url}/predict"
    health_url = f"{api_base_url}/health"

    st.set_page_config(
        page_title="Customer Churn Prediction",
        page_icon="chart_with_upwards_trend",
        layout="wide",
    )

    st.title("Customer Churn Prediction")
    st.markdown(
        """
This application predicts whether a customer will churn based on profile and usage data.
Enter the customer details below and click **Predict** to get churn probability.
"""
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Personal Information")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])

        st.subheader("Service Information")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        if phone_service == "No":
            multiple_lines = st.selectbox("Multiple Lines", ["No phone service"])
        else:
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])

    with col2:
        st.subheader("Internet Service")
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        if internet_service == "No":
            internet_options = ["No internet service"]
        else:
            internet_options = ["Yes", "No"]

        online_security = st.selectbox("Online Security", internet_options)
        online_backup = st.selectbox("Online Backup", internet_options)
        device_protection = st.selectbox("Device Protection", internet_options)
        tech_support = st.selectbox("Tech Support", internet_options)
        streaming_tv = st.selectbox("Streaming TV", internet_options)
        streaming_movies = st.selectbox("Streaming Movies", internet_options)

    st.subheader("Billing Information")
    col3, col4 = st.columns(2)

    with col3:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )

    with col4:
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0, step=0.01)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0, step=0.01)

    if st.button("Predict Churn", type="primary", use_container_width=True):
        customer_data = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        try:
            response = requests.post(predict_url, json=customer_data, timeout=10)

            if response.status_code == 200:
                result = response.json()
                st.success("Prediction completed.")

                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric(
                        label="Churn Prediction",
                        value=result["prediction_text"],
                        delta="High Risk" if result["churn_prediction"] else "Low Risk",
                    )

                with res_col2:
                    st.metric(
                        label="Churn Probability",
                        value=f"{result['churn_probability']:.1%}",
                        delta=f"{result['confidence']:.1%} confidence",
                    )

                st.progress(result["churn_probability"])
                if result["churn_prediction"]:
                    st.warning("This customer is likely to churn. Consider retention strategies.")
                else:
                    st.success("This customer is likely to stay.")
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException:
            st.error(f"Connection Error: Could not connect to API at {predict_url}")
            st.info("To start the API server, run: `uvicorn app:app --reload`")

    st.markdown("---")
    st.markdown(
        """
**About this model:**
- Uses Logistic Regression trained on customer data
- Accuracy: ~79% on test data
- Features: 19 raw input fields transformed via preprocessing pipeline
"""
    )

    with st.sidebar:
        st.header("How to Use")
        st.markdown(
            """
1. Fill in all customer details
2. Click **Predict Churn**
3. Review prediction and probability
4. Use insights for retention strategy
"""
        )

        st.header("API Status")
        if get_api_status(health_url):
            st.success("API server running")
        else:
            st.warning("API server not running")

        st.header("Model Info")
        st.info("Logistic Regression\nPipeline preprocessing enabled")


if __name__ == "__main__":
    run_app()
