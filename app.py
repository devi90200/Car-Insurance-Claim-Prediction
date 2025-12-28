import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Insurance Risk Assessment Engine",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================
def load_model():
    return joblib.load(
        "notebook/model/final_insurance_claim_model.pkl",
        mmap_mode="r"
    )

model = load_model()

# =====================================================
# UI STYLES (UNCHANGED)
# =====================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #fff1f2, #fdf4ff);
    font-family: 'Segoe UI', sans-serif;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fde2e4, #e0e7ff);
}
.hero {
    background: linear-gradient(135deg, #fde68a, #fbcfe8);
    padding: 35px;
    border-radius: 24px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}
.kpi {
    background: linear-gradient(135deg, #ecfeff, #f0fdf4);
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0 6px 16px rgba(0,0,0,0.06);
}
.badge-low {background:#dcfce7; padding:8px 16px; border-radius:18px;}
.badge-mid {background:#fef9c3; padding:8px 16px; border-radius:18px;}
.badge-high {background:#fee2e2; padding:8px 16px; border-radius:18px;}
.badge-vhigh {background:#fecaca; padding:8px 16px; border-radius:18px;}
.marquee {
    background: #fdf2f8;
    padding: 14px;
    border-radius: 16px;
    overflow: hidden;
    white-space: nowrap;
}
.marquee span {
    display: inline-block;
    animation: scrollText 18s linear infinite;
}
@keyframes scrollText {
    from { transform: translateX(100%); }
    to { transform: translateX(-100%); }
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# TOP IMAGE
# =====================================================
top_l, top_m, top_r = st.columns([1, 2, 1])
with top_m:
    st.image("assests/card.jpg", use_container_width=True)

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div class="hero">
<h1>🚗 Insurance Risk Assessment Engine</h1>
<p>AI-powered underwriting intelligence for smarter insurance decisions</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR INPUTS
# =====================================================
st.sidebar.image("assests/carj.gif", use_container_width=True)
st.sidebar.header("📝 Policy Inputs")

policy_tenure = st.sidebar.number_input("Policy Tenure (Years)", 0.0, 30.0, 1.0)
age_of_car = st.sidebar.number_input("Vehicle Age (Years)", 0, 20, 3)
age_of_policyholder = st.sidebar.number_input("Policyholder Age", 18, 90, 35)

area_cluster = st.sidebar.selectbox("Area Cluster", ["A", "B", "C", "D"])
segment = st.sidebar.selectbox("Customer Segment", ["A", "B", "C"])
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])

population_density = st.sidebar.number_input("Population Density", 500, 50000, 5000)
make = st.sidebar.selectbox("Vehicle Make", [1, 2, 3])
ncap_rating = st.sidebar.selectbox("NCAP Rating", [0, 1, 2, 3, 4, 5])

predict_btn = st.sidebar.button("🔍 Assess Risk")

# =====================================================
# PREDICTION
# =====================================================
if predict_btn:

    input_data = {
        "policy_tenure": policy_tenure,
        "age_of_car": age_of_car,
        "age_of_policyholder": age_of_policyholder,
        "area_cluster": area_cluster,
        "segment": segment,
        "fuel_type": fuel_type,
        "population_density": population_density,
        "make": make,
        "ncap_rating": ncap_rating
    }

    df = pd.DataFrame([input_data])

    preprocessor = model.named_steps["prep"]
    expected_cols = list(preprocessor.feature_names_in_)

    categorical_cols = []
    for name, transformer, cols in preprocessor.transformers_:
        if name.lower().startswith("cat"):
            categorical_cols = cols

    for col in expected_cols:
        if col not in df.columns:
            df[col] = "No" if col in categorical_cols else 0

    for col in categorical_cols:
        df[col] = df[col].astype(str)

    for col in expected_cols:
        if col not in categorical_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df = df[expected_cols]

    risk_prob = model.predict_proba(df)[0][1] * 100
    predicted_claim = int(model.predict(df)[0])

    # =====================================================
    # RISK LOGIC
    # =====================================================
    if risk_prob < 5:
        level, action, badge = "Low Risk", "Standard premium", "badge-low"
    elif risk_prob < 10:
        level, action, badge = "Moderate Risk", "Monitor customer", "badge-mid"
    elif risk_prob < 20:
        level, action, badge = "High Risk", "Higher premium", "badge-high"
    else:
        level, action, badge = "Very High Risk", "Manual underwriting review", "badge-vhigh"

    # =====================================================
    # DASHBOARD GRID
    # =====================================================
    g_col, kpi1_col, kpi2_col, kpi3_col = st.columns([2, 1, 1, 1])

    # ----- GAUGE -----
    with g_col:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_prob,
            title={"text": "Risk Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#fca5a5"},
                "steps": [
                    {"range": [0, 5], "color": "#dcfce7"},
                    {"range": [5, 10], "color": "#fef9c3"},
                    {"range": [10, 20], "color": "#fee2e2"},
                    {"range": [20, 100], "color": "#fecaca"}
                ],
            }
        ))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # ----- KPI 1 -----
    with kpi1_col:
        st.markdown(f"""
        <div class="kpi">
            <h3>Risk Category</h3>
            <div class="{badge}"><strong>{level}</strong></div>
        </div>
        """, unsafe_allow_html=True)

    # ----- KPI 2 -----
    with kpi2_col:
        st.markdown(f"""
        <div class="kpi">
            <h3>Business Action</h3>
            <p><strong>{action}</strong></p>
        </div>
        """, unsafe_allow_html=True)

    # ----- KPI 3 (NEW) -----
    with kpi3_col:

        if predicted_claim == 1:
            claim_text = "Claim Expected"
            badge_class = "badge-high"
        else:
            claim_text = "No Claim Expected"
            badge_class = "badge-low"

        st.markdown(f"""
        <div class="kpi">
            <h3>Predicted Output</h3>
            <div class="{badge_class}">
                <strong>is_claim = {predicted_claim}</strong>
            </div>
            <p style="margin-top:10px;">{claim_text}</p>
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# LIVE CUSTOMER INSIGHTS
# =====================================================
st.markdown("### 🧠 Live Customer Insights")
st.markdown(f"""
<div class="marquee">
<span>
Customer from Area Cluster {area_cluster} with Segment {segment},
living in population density {population_density},
driving a {fuel_type} vehicle (NCAP {ncap_rating}),
shows risk behavior impacting underwriting decisions.
</span>
</div>
""", unsafe_allow_html=True)

# =====================================================
# THANK YOU NOTE
# =====================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<center><b>🙏 Thank you for visiting — Please review again</b></center>",
    unsafe_allow_html=True
)

