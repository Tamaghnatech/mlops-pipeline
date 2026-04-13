import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #262b3e);
        border: 1px solid #2d3250;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .risk-high {
        background: linear-gradient(135deg, #3d1515, #5c1f1f);
        border: 1px solid #ff4444;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .risk-medium {
        background: linear-gradient(135deg, #3d2e10, #5c4515);
        border: 1px solid #ffaa00;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .risk-low {
        background: linear-gradient(135deg, #0d3320, #1a5c35);
        border: 1px solid #00cc66;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(135deg, #6c63ff, #a855f7);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 30px;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #a855f7, #6c63ff);
        transform: translateY(-1px);
    }
    div[data-testid="metric-container"] {
        background: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 10px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "https://churn-prediction-api-ykim.onrender.com/predict"

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔮 Churn Intelligence")
    st.markdown("---")
    page = st.radio("Navigate", ["Single Prediction", "Batch Analysis"])
    st.markdown("---")
    st.markdown("### API Status")
    try:
        r = requests.get("https://churn-prediction-api-ykim.onrender.com/health", timeout=10)
        if r.status_code == 200:
            st.success("API Online")
        else:
            st.error("API Error")
    except:
        st.error("API Offline")
    st.markdown("---")
    st.markdown("**Model:** Random Forest")
    st.markdown("**Dataset:** Telco Churn")
    st.markdown("**Features:** 19")

# ── Single Prediction Page ───────────────────────────────────
if page == "Single Prediction":
    st.markdown("# Customer Churn Predictor")
    st.markdown("Fill in customer details to get an instant churn risk score.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Account Info")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    with col2:
        st.markdown("#### Services")
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    with col3:
        st.markdown("#### Charges")
        monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
        total = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)
        st.markdown("&nbsp;", unsafe_allow_html=True)
        predict_btn = st.button("Analyze Churn Risk")

    if predict_btn:
        payload = {
            "gender": gender, "SeniorCitizen": senior,
            "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone,
            "MultipleLines": lines, "InternetService": internet,
            "OnlineSecurity": security, "OnlineBackup": backup,
            "DeviceProtection": protection, "TechSupport": support,
            "StreamingTV": tv, "StreamingMovies": movies,
            "Contract": contract, "PaperlessBilling": billing,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total
        }
        try:
            response = requests.post(API_URL, json=payload)
            result = response.json()
            prob = result["churn_probability"]
            risk = result["risk_level"]
            will_churn = result["will_churn"]

            st.markdown("---")
            st.markdown("## Prediction Result")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Churn Probability", f"{prob:.1%}")
            c2.metric("Risk Level", risk)
            c3.metric("Verdict", "Will Churn" if will_churn else "Will Stay")
            c4.metric("Tenure", f"{tenure} months")

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Churn Risk Score", "font": {"color": "white"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "white"},
                    "bar": {"color": "#ff4444" if prob > 0.7 else "#ffaa00" if prob > 0.4 else "#00cc66"},
                    "steps": [
                        {"range": [0, 40], "color": "#1a3a2a"},
                        {"range": [40, 70], "color": "#3a2a10"},
                        {"range": [70, 100], "color": "#3a1010"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 4}, "value": prob * 100}
                },
                number={"suffix": "%", "font": {"color": "white"}}
            ))
            fig.update_layout(
                paper_bgcolor="#0e1117",
                font={"color": "white"},
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

            if risk == "High":
                st.error("🚨 High churn risk! Recommend: Offer contract upgrade discount immediately.")
            elif risk == "Medium":
                st.warning("⚠️ Medium churn risk. Recommend: Proactive outreach within 7 days.")
            else:
                st.success("✅ Low churn risk. Customer is healthy — no action needed.")

        except Exception as e:
            st.error(f"API error: {e}")

# ── Batch Analysis Page ──────────────────────────────────────
else:
    st.markdown("# Batch Churn Analysis")
    st.markdown("Upload your customer CSV to score all customers at once.")
    st.markdown("---")

    uploaded = st.file_uploader("Drop your customer CSV here", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df):,} customers")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Customers", len(df))
        c2.metric("Features", len(df.columns))
        c3.metric("Missing Values", df.isnull().sum().sum())

        with st.expander("Preview Data"):
            st.dataframe(df.head(10))

        if st.button("Run Batch Predictions"):
            results = []
            progress = st.progress(0)
            status = st.empty()

            for i, row in df.iterrows():
                try:
                    payload = row.drop(["customerID", "Churn"], errors="ignore").to_dict()
                    payload["SeniorCitizen"] = int(payload.get("SeniorCitizen", 0))
                    payload["tenure"] = int(payload.get("tenure", 0))
                    payload["MonthlyCharges"] = float(payload.get("MonthlyCharges", 0))
                    tc = str(payload.get("TotalCharges", "0")).strip()
                    payload["TotalCharges"] = float(tc) if tc else 0.0
                    r = requests.post(API_URL, json=payload)
                    res = r.json()
                    results.append({
                        "customerID": row.get("customerID", i),
                        "churn_probability": res.get("churn_probability", 0),
                        "risk_level": res.get("risk_level", "Unknown"),
                        "will_churn": res.get("will_churn", False)
                    })
                except:
                    results.append({"customerID": row.get("customerID", i),
                                     "churn_probability": 0,
                                     "risk_level": "Error",
                                     "will_churn": False})
                progress.progress((i + 1) / len(df))
                status.text(f"Scoring customer {i+1} of {len(df)}...")

            status.empty()
            result_df = pd.DataFrame(results)
            merged = df.merge(result_df, on="customerID", how="left") if "customerID" in df.columns else result_df

            st.markdown("---")
            st.markdown("## Results Summary")

            high = len(result_df[result_df.risk_level == "High"])
            med  = len(result_df[result_df.risk_level == "Medium"])
            low  = len(result_df[result_df.risk_level == "Low"])

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Scored", len(result_df))
            c2.metric("High Risk", high, delta=f"{high/len(result_df):.0%}")
            c3.metric("Medium Risk", med)
            c4.metric("Low Risk", low)

            col1, col2 = st.columns(2)

            with col1:
                fig = px.pie(
                    result_df, names="risk_level",
                    title="Risk Distribution",
                    color="risk_level",
                    color_discrete_map={"High": "#ff4444", "Medium": "#ffaa00", "Low": "#00cc66"}
                )
                fig.update_layout(paper_bgcolor="#0e1117", font_color="white")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig2 = px.histogram(
                    result_df, x="churn_probability",
                    nbins=20, title="Churn Probability Distribution",
                    color_discrete_sequence=["#6c63ff"]
                )
                fig2.update_layout(paper_bgcolor="#0e1117", font_color="white")
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("### Top 20 High-Risk Customers")
            st.dataframe(
                result_df.sort_values("churn_probability", ascending=False).head(20),
                use_container_width=True
            )

            csv = result_df.to_csv(index=False)
            st.download_button(
                "Download Full Results CSV",
                csv, "churn_predictions.csv", "text/csv"
            )