import requests
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)

# Title and description
st.title("Customer Churn Prediction")
st.markdown(
    """
This application predicts whether a customer will churn based on profile and usage data.
Enter the customer details below and click **Predict** to get churn probability.
"""
)

API_URL = "http://localhost:8000/predict"

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
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

with col2:
    st.subheader("Internet Service")
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

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
        response = requests.post(API_URL, json=customer_data, timeout=10)

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
        st.error(f"Connection Error: Could not connect to API at {API_URL}")
        st.info("To start the API server, run: `uvicorn app:app --reload`")

st.markdown("---")
st.markdown(
    """
**About this model:**
- Uses Logistic Regression trained on customer data
- Accuracy: ~79% on test data
- Features: 26 input variables including demographics, services, and billing info
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
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        if health_response.status_code == 200:
            st.success("API server running")
        else:
            st.error("API server error")
    except requests.exceptions.RequestException:
        st.warning("API server not running")

    st.header("Model Info")
    st.info("Logistic Regression\nAccuracy: 79.12%\nFeatures: 26")
