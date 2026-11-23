import streamlit as st
import pandas as pd
import joblib
import shap
import random


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="LoanWise AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD MODEL + DATA
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("loanwise_model.pkl")


@st.cache_data
def load_dataset():
    return pd.read_excel(
        "Bank_Personal_Loan_Modelling.xlsx",
        sheet_name="Data"
    )


model = load_model()
dataset = load_dataset()

# Immersive 3D experience layer (existing ML pipeline remains unchanged)
from loanwise_3d import render_3d_hero
render_3d_hero(len(dataset))


# =========================================================
# SETTINGS
# =========================================================

# Original dataset:
# Income   = USD thousands/year
# CCAvg    = USD thousands/month
# Mortgage = USD thousands

USD_TO_INR = 94.60

FEATURES = [
    "Age",
    "Experience",
    "Income",
    "Family",
    "CCAvg",
    "Education",
    "Mortgage",
    "Securities Account",
    "CD Account",
    "Online",
    "CreditCard"
]


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def format_inr(value):
    return f"₹{value:,.0f}"


def set_profile(row):
    """
    Fill the Streamlit input fields using a dataset row.
    Values remain in the model-compatible scale internally.
    Currency values are converted to INR for the UI.
    """

    st.session_state["age"] = int(row["Age"])
    st.session_state["experience"] = int(row["Experience"])

    st.session_state["income_inr"] = int(
        row["Income"] * USD_TO_INR * 1000
    )

    st.session_state["family"] = int(row["Family"])

    st.session_state["ccavg_inr"] = int(
        row["CCAvg"] * USD_TO_INR * 1000
    )

    st.session_state["education"] = int(row["Education"])

    st.session_state["mortgage_inr"] = int(
        row["Mortgage"] * USD_TO_INR * 1000
    )

    st.session_state["securities"] = int(
        row["Securities Account"]
    )

    st.session_state["cd_account"] = int(
        row["CD Account"]
    )

    st.session_state["online"] = int(
        row["Online"]
    )

    st.session_state["credit_card"] = int(
        row["CreditCard"]
    )


def find_demo_profile(low, high, attempts=5000):
    """
    Find a real feature combination from the dataset whose
    model probability falls inside the requested range.
    """

    demo_data = dataset.copy()

    demo_data = demo_data.drop(
        columns=["ID", "ZIP Code", "Personal Loan"],
        errors="ignore"
    )

    demo_data = demo_data[FEATURES]

    # Check existing dataset rows first
    probabilities = model.predict_proba(
        demo_data
    )[:, 1] * 100

    matches = demo_data[
        (probabilities >= low) &
        (probabilities < high)
    ].copy()

    if not matches.empty:
        selected_index = random.choice(
            matches.index.tolist()
        )

        return matches.loc[selected_index]

    # Fallback: random synthetic combinations
    # based on dataset ranges
    numeric_ranges = {}

    for feature in [
        "Age",
        "Experience",
        "Income",
        "Family",
        "CCAvg",
        "Education",
        "Mortgage",
        "Securities Account",
        "CD Account",
        "Online",
        "CreditCard"
    ]:
        numeric_ranges[feature] = (
            demo_data[feature].min(),
            demo_data[feature].max()
        )

    for _ in range(attempts):

        candidate = {
            "Age": random.randint(
                int(numeric_ranges["Age"][0]),
                int(numeric_ranges["Age"][1])
            ),

            "Experience": random.randint(
                int(numeric_ranges["Experience"][0]),
                int(numeric_ranges["Experience"][1])
            ),

            "Income": random.uniform(
                numeric_ranges["Income"][0],
                numeric_ranges["Income"][1]
            ),

            "Family": random.randint(
                int(numeric_ranges["Family"][0]),
                int(numeric_ranges["Family"][1])
            ),

            "CCAvg": random.uniform(
                numeric_ranges["CCAvg"][0],
                numeric_ranges["CCAvg"][1]
            ),

            "Education": random.randint(
                int(numeric_ranges["Education"][0]),
                int(numeric_ranges["Education"][1])
            ),

            "Mortgage": random.uniform(
                numeric_ranges["Mortgage"][0],
                numeric_ranges["Mortgage"][1]
            ),

            "Securities Account": random.randint(0, 1),

            "CD Account": random.randint(0, 1),

            "Online": random.randint(0, 1),

            "CreditCard": random.randint(0, 1)
        }

        candidate_df = pd.DataFrame(
            [candidate]
        )[FEATURES]

        probability = (
            model.predict_proba(candidate_df)[0][1]
            * 100
        )

        if low <= probability < high:
            return candidate_df.iloc[0]

    return None


# =========================================================
# INITIAL SESSION STATE
# =========================================================

defaults = {
    "age": 35,
    "experience": 10,
    "income_inr": 7500000,
    "family": 4,
    "ccavg_inr": 20000,
    "education": 1,
    "mortgage_inr": 0,
    "securities": 0,
    "cd_account": 0,
    "online": 0,
    "credit_card": 0,
    "prediction_done": False
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# CUSTOM CSS
# =========================================================

st.html("""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

* {
    font-family: 'Inter', sans-serif !important;
}

html,
body {
    background: #06111f !important;
}

.stApp {
    background:
        radial-gradient(
            circle at 8% 5%,
            rgba(37,99,235,0.17),
            transparent 28%
        ),
        radial-gradient(
            circle at 92% 8%,
            rgba(14,165,233,0.12),
            transparent 28%
        ),
        #06111f !important;

    color: #ffffff !important;
}

header,
#MainMenu,
footer {
    visibility: hidden;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}


/* =====================================================
   SIDEBAR
   ===================================================== */

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #071322,
            #081a2d
        ) !important;

    border-right:
        1px solid rgba(148,163,184,0.12);
}

[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}


/* =====================================================
   HERO
   ===================================================== */

.hero {
    background:
        linear-gradient(
            135deg,
            #10223d,
            #123d62
        );

    border:
        1px solid rgba(96,165,250,0.25);

    border-radius: 26px;

    padding: 44px;

    margin-bottom: 26px;

    box-shadow:
        0 25px 70px rgba(0,0,0,0.28);
}

.hero-badge {
    display: inline-block;

    padding: 8px 14px;

    border-radius: 999px;

    background:
        rgba(96,165,250,0.13);

    border:
        1px solid rgba(147,197,253,0.30);

    color: #bfdbfe !important;

    font-size: 11px;

    font-weight: 800;

    letter-spacing: 1px;
}

.hero-title {
    color: #ffffff !important;

    font-size: 46px;

    font-weight: 800;

    line-height: 1.1;

    margin-top: 18px;
}

.hero-title span {
    color: #60a5fa !important;
}

.hero-description {
    color: #dbeafe !important;

    font-size: 15px;

    line-height: 1.8;

    max-width: 850px;

    margin-top: 14px;
}


/* =====================================================
   SECTION
   ===================================================== */

.section-title {
    color: #ffffff !important;

    font-size: 24px;

    font-weight: 800;

    margin-top: 30px;

    margin-bottom: 6px;
}

.section-subtitle {
    color: #cbd5e1 !important;

    font-size: 13px;

    margin-bottom: 20px;
}


/* =====================================================
   CARDS
   ===================================================== */

.metric-card {
    background:
        linear-gradient(
            145deg,
            #10213a,
            #122d49
        );

    border:
        1px solid rgba(96,165,250,0.16);

    border-radius: 18px;

    padding: 20px;

    min-height: 116px;

    box-shadow:
        0 10px 32px rgba(0,0,0,0.17);
}

.metric-label {
    color: #cbd5e1 !important;

    font-size: 11px;

    font-weight: 800;

    letter-spacing: 0.8px;

    text-transform: uppercase;
}

.metric-value {
    color: #ffffff !important;

    font-size: 27px;

    font-weight: 800;

    margin-top: 7px;

    word-break: break-word;
}

.metric-note {
    color: #94a3b8 !important;

    font-size: 11px;

    margin-top: 6px;
}


/* =====================================================
   DEMO CARDS
   ===================================================== */

.demo-card {
    background:
        linear-gradient(
            145deg,
            #0e1d32,
            #102944
        );

    border:
        1px solid rgba(96,165,250,0.15);

    border-radius: 16px;

    padding: 16px;

    margin-bottom: 10px;
}

.demo-title {
    color: #ffffff !important;

    font-size: 14px;

    font-weight: 800;
}

.demo-text {
    color: #94a3b8 !important;

    font-size: 11px;

    margin-top: 4px;

    line-height: 1.5;
}


/* =====================================================
   INPUTS
   ===================================================== */

label {
    color: #f8fafc !important;

    font-size: 13px !important;

    font-weight: 700 !important;
}

div[data-baseweb="input"] {
    background: #f8fafc !important;

    border-radius: 10px !important;
}

div[data-baseweb="input"] input {
    color: #111827 !important;

    background: #f8fafc !important;
}

div[data-baseweb="select"] {
    background: #f8fafc !important;

    border-radius: 10px !important;
}

div[data-baseweb="select"] * {
    color: #111827 !important;
}


/* =====================================================
   BUTTON
   ===================================================== */

.stButton > button {
    width: 100%;

    height: 56px;

    border-radius: 14px;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #0ea5e9
        ) !important;

    color: #ffffff !important;

    font-size: 14px;

    font-weight: 800;

    border:
        1px solid rgba(147,197,253,0.28);

    box-shadow:
        0 10px 30px rgba(37,99,235,0.25);
}

.stButton > button:hover {
    color: #ffffff !important;

    transform: translateY(-1px);
}


/* =====================================================
   RESULT
   ===================================================== */

.result-card {
    background:
        linear-gradient(
            145deg,
            #10233d,
            #123b5a
        );

    border:
        1px solid rgba(96,165,250,0.22);

    border-radius: 22px;

    padding: 30px;

    min-height: 225px;

    box-shadow:
        0 15px 45px rgba(0,0,0,0.20);
}

.result-label {
    color: #cbd5e1 !important;

    font-size: 11px;

    font-weight: 800;

    letter-spacing: 0.9px;

    text-transform: uppercase;
}

.result-number {
    color: #ffffff !important;

    font-size: 48px;

    font-weight: 800;

    margin-top: 8px;
}

.result-status {
    display: inline-block;

    padding: 9px 15px;

    margin-top: 13px;

    border-radius: 999px;

    font-size: 12px;

    font-weight: 800;
}

.high {
    color: #bbf7d0 !important;

    background: rgba(34,197,94,0.14);

    border:
        1px solid rgba(34,197,94,0.35);
}

.medium {
    color: #fef3c7 !important;

    background: rgba(251,191,36,0.14);

    border:
        1px solid rgba(251,191,36,0.35);
}

.low {
    color: #fecaca !important;

    background: rgba(239,68,68,0.14);

    border:
        1px solid rgba(239,68,68,0.35);
}


/* =====================================================
   SNAPSHOT
   ===================================================== */

.snapshot-box {
    background:
        rgba(8,22,39,0.86);

    border:
        1px solid rgba(148,163,184,0.12);

    border-radius: 22px;

    padding: 24px;

    min-height: 225px;
}

.snapshot-title {
    color: #cbd5e1 !important;

    font-size: 11px;

    font-weight: 800;

    letter-spacing: 0.9px;

    text-transform: uppercase;

    margin-bottom: 10px;
}

.snapshot-row {
    display: flex;

    justify-content: space-between;

    align-items: center;

    padding: 9px 0;

    border-bottom:
        1px solid rgba(148,163,184,0.08);
}

.snapshot-row:last-child {
    border-bottom: none;
}

.snapshot-name {
    color: #94a3b8 !important;

    font-size: 12px;
}

.snapshot-value {
    color: #ffffff !important;

    font-size: 13px;

    font-weight: 800;

    text-align: right;
}


/* =====================================================
   INSIGHT
   ===================================================== */

.insight {
    background:
        rgba(18,34,56,0.92);

    border-left:
        3px solid #60a5fa;

    border-radius: 10px;

    padding: 16px 18px;

    margin-top: 16px;

    color: #e2e8f0 !important;

    font-size: 13px;

    line-height: 1.7;
}


/* =====================================================
   RECOMMENDATION
   ===================================================== */

.recommendation {
    background:
        linear-gradient(
            135deg,
            rgba(13,52,67,0.92),
            rgba(16,42,67,0.96)
        );

    border:
        1px solid rgba(45,212,191,0.18);

    border-radius: 18px;

    padding: 20px;

    margin-top: 18px;
}

.recommendation-title {
    color: #99f6e4 !important;

    font-size: 11px;

    font-weight: 800;

    letter-spacing: 0.8px;

    text-transform: uppercase;
}

.recommendation-text {
    color: #e2e8f0 !important;

    font-size: 14px;

    line-height: 1.7;

    margin-top: 8px;
}


/* =====================================================
   EXPLAINABLE AI
   ===================================================== */

.explain-card {
    background:
        linear-gradient(
            145deg,
            #0e1e34,
            #102944
        );

    border:
        1px solid rgba(96,165,250,0.18);

    border-radius: 18px;

    padding: 22px;

    margin-top: 18px;
}

.explain-title {
    color: #ffffff !important;

    font-size: 15px;

    font-weight: 800;
}

.explain-text {
    color: #94a3b8 !important;

    font-size: 12px;

    margin-top: 5px;
}


/* =====================================================
   FACTORS
   ===================================================== */

.factor {
    margin-top: 15px;
}

.factor-header {
    display: flex;

    justify-content: space-between;

    margin-bottom: 6px;
}

.factor-name {
    color: #e2e8f0 !important;

    font-size: 12px;

    font-weight: 700;
}

.factor-direction {
    color: #94a3b8 !important;

    font-size: 10px;
}

.factor-bg {
    width: 100%;

    height: 8px;

    background: #1e293b;

    border-radius: 999px;

    overflow: hidden;
}

.factor-fill {
    height: 100%;

    border-radius: 999px;

    background:
        linear-gradient(
            90deg,
            #2563eb,
            #38bdf8
        );
}


/* =====================================================
   FOOTER
   ===================================================== */

.footer {
    text-align: center;

    color: #94a3b8 !important;

    font-size: 11px;

    line-height: 1.7;

    margin-top: 45px;

    padding-top: 22px;

    border-top:
        1px solid rgba(255,255,255,0.08);
}

</style>
""")


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.html("""
    <div style="
        font-size:28px;
        font-weight:800;
        color:#ffffff;
    ">
        💳 LoanWise <span style="color:#60a5fa;">AI</span>
    </div>
    """)

    st.html("""
    <div style="
        color:#94a3b8;
        font-size:12px;
        margin-top:5px;
        margin-bottom:28px;
    ">
        Personal Loan Acceptance Intelligence
    </div>
    """)

    st.markdown("### 🤖 Model Information")

    st.html("""
    <div class="sidebar-card">

        <b style="color:#ffffff;">Algorithm</b>
        <br>
        Gradient Boosting Classifier

        <br><br>

        <b style="color:#ffffff;">Validation Accuracy</b>
        <br>
        <span style="
            color:#86efac;
            font-size:17px;
            font-weight:800;
        ">
            98.78%
        </span>

        <br><br>

        <b style="color:#ffffff;">Precision</b>
        <br>
        96.49%

        <br><br>

        <b style="color:#ffffff;">Recall</b>
        <br>
        90.62%

        <br><br>

        <b style="color:#ffffff;">ROC-AUC</b>
        <br>
        0.9982

        <br><br>

        <b style="color:#ffffff;">Training Records</b>
        <br>
        5,000 customers

    </div>
    """)

    st.markdown("### 💱 Currency")

    st.html("""
    <div class="sidebar-card">

        Display:
        <b style="color:#ffffff;">
            Indian Rupee (₹)
        </b>

        <br>

        Model data:
        <b style="color:#ffffff;">
            USD thousands
        </b>

        <br>

        Reference:
        <b style="color:#ffffff;">
            ₹94.60 / $1
        </b>

    </div>
    """)


# =========================================================
# HERO
# =========================================================

st.html("""
<div class="hero">

    <div class="hero-badge">
        AI-POWERED FINANCIAL INTELLIGENCE
    </div>

    <div class="hero-title">
        Smarter loan insights with
        <span>LoanWise AI</span>
    </div>

    <div class="hero-description">
        Analyze customer financial and banking profiles using
        machine learning and estimate the likelihood of accepting
        a personal loan offer.
    </div>

</div>
""")


# =========================================================
# PERFORMANCE CARDS
# =========================================================

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.html("""
    <div class="metric-card">
        <div class="metric-label">
            Model Accuracy
        </div>
        <div class="metric-value">
            98.78%
        </div>
        <div class="metric-note">
            5-fold validated
        </div>
    </div>
    """)

with m2:
    st.html("""
    <div class="metric-card">
        <div class="metric-label">
            ROC-AUC
        </div>
        <div class="metric-value">
            0.9982
        </div>
        <div class="metric-note">
            Excellent discrimination
        </div>
    </div>
    """)

with m3:
    st.html("""
    <div class="metric-card">
        <div class="metric-label">
            Training Records
        </div>
        <div class="metric-value">
            5,000
        </div>
        <div class="metric-note">
            Customer profiles
        </div>
    </div>
    """)

with m4:
    st.html("""
    <div class="metric-card">
        <div class="metric-label">
            Input Features
        </div>
        <div class="metric-value">
            11
        </div>
        <div class="metric-note">
            Financial + banking
        </div>
    </div>
    """)


# =========================================================
# DEMO PROFILES
# =========================================================

st.html("""
<div class="section-title">
    Quick Demo Profiles
</div>

<div class="section-subtitle">
    These buttons find profiles using your actual trained model.
    No guessing required.
</div>
""")


d1, d2, d3 = st.columns(3)


with d1:

    st.html("""
    <div class="demo-card">
        <div class="demo-title">🔴 Low Likelihood</div>
        <div class="demo-text">
            Automatically find a customer below 50%.
        </div>
    </div>
    """)

    if st.button(
        "Load Low Profile",
        key="low_demo"
    ):

        row = find_demo_profile(0, 50)

        if row is not None:

            set_profile(row)

            st.session_state["prediction_done"] = False

            st.rerun()

        else:

            st.error(
                "Could not find a low-likelihood profile."
            )


with d2:

    st.html("""
    <div class="demo-card">
        <div class="demo-title">🟡 Moderate Likelihood</div>
        <div class="demo-text">
            Automatically find a customer between 50% and 80%.
        </div>
    </div>
    """)

    if st.button(
        "Load Moderate Profile",
        key="moderate_demo"
    ):

        row = find_demo_profile(50, 80)

        if row is not None:

            set_profile(row)

            st.session_state["prediction_done"] = False

            st.rerun()

        else:

            st.error(
                "Could not find a moderate profile."
            )


with d3:

    st.html("""
    <div class="demo-card">
        <div class="demo-title">🟢 High Likelihood</div>
        <div class="demo-text">
            Automatically find a customer at 80% or above.
        </div>
    </div>
    """)

    if st.button(
        "Load High Profile",
        key="high_demo"
    ):

        row = find_demo_profile(80, 100.0001)

        if row is not None:

            set_profile(row)

            st.session_state["prediction_done"] = False

            st.rerun()

        else:

            st.error(
                "Could not find a high-likelihood profile."
            )


# =========================================================
# CUSTOMER PROFILE
# =========================================================

st.html("""
<div class="section-title">
    Customer Profile
</div>

<div class="section-subtitle">
    Enter the customer's financial and banking information.
</div>
""")


left, middle, right = st.columns(3)


# =========================================================
# LEFT COLUMN
# =========================================================

with left:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        step=1,
        key="age"
    )

    experience = st.number_input(
        "Professional Experience (Years)",
        min_value=0,
        max_value=60,
        step=1,
        key="experience"
    )

    income_inr = st.number_input(
        "Annual Income (₹)",
        min_value=100000,
        max_value=50000000,
        step=50000,
        key="income_inr"
    )

    family = st.number_input(
        "Family Size",
        min_value=1,
        max_value=10,
        step=1,
        key="family"
    )


# =========================================================
# MIDDLE COLUMN
# =========================================================

with middle:

    ccavg_inr = st.number_input(
        "Credit Card Spending (₹/month)",
        min_value=0,
        max_value=1000000,
        step=1000,
        key="ccavg_inr"
    )

    education = st.selectbox(
        "Education Level",
        [1, 2, 3],
        format_func=lambda x: {
            1: "Undergraduate",
            2: "Graduate",
            3: "Advanced / Professional"
        }[x],
        key="education"
    )

    mortgage_inr = st.number_input(
        "Mortgage (₹)",
        min_value=0,
        max_value=70000000,
        step=50000,
        key="mortgage_inr"
    )


# =========================================================
# RIGHT COLUMN
# =========================================================

with right:

    securities = st.selectbox(
        "Securities Account",
        [0, 1],
        format_func=lambda x:
            "Yes" if x else "No",
        key="securities"
    )

    cd_account = st.selectbox(
        "CD Account",
        [0, 1],
        format_func=lambda x:
            "Yes" if x else "No",
        key="cd_account"
    )

    online = st.selectbox(
        "Online Banking",
        [0, 1],
        format_func=lambda x:
            "Yes" if x else "No",
        key="online"
    )

    credit_card = st.selectbox(
        "Credit Card",
        [0, 1],
        format_func=lambda x:
            "Yes" if x else "No",
        key="credit_card"
    )


# =========================================================
# ANALYZE BUTTON
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

if st.button(
    "🔮  ANALYZE CUSTOMER",
    use_container_width=True,
    key="predict_button"
):

    st.session_state["prediction_done"] = True


# =========================================================
# PREDICTION
# =========================================================

if st.session_state["prediction_done"]:

    # -----------------------------------------------------
    # Convert INR to original dataset units
    # -----------------------------------------------------

    income_model = (
        income_inr /
        USD_TO_INR /
        1000
    )

    ccavg_model = (
        ccavg_inr /
        USD_TO_INR /
        1000
    )

    mortgage_model = (
        mortgage_inr /
        USD_TO_INR /
        1000
    )


    # -----------------------------------------------------
    # Create model input
    # -----------------------------------------------------

    input_data = pd.DataFrame([{

        "Age": age,
        "Experience": experience,
        "Income": income_model,
        "Family": family,
        "CCAvg": ccavg_model,
        "Education": education,
        "Mortgage": mortgage_model,
        "Securities Account": securities,
        "CD Account": cd_account,
        "Online": online,
        "CreditCard": credit_card

    }])


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    prediction = int(
        model.predict(input_data)[0]
    )

    probability = float(
        model.predict_proba(
            input_data
        )[0][1] * 100
    )


    # =====================================================
    # CLASSIFICATION
    # =====================================================

    if probability >= 80:

        status_text = "HIGH LIKELIHOOD"

        status_class = "high"

        message = (
            "This customer shows a high likelihood "
            "of accepting the personal loan offer."
        )

        recommendation = (
            "Consider this customer as a high-priority "
            "candidate for campaign engagement."
        )

    elif probability >= 50:

        status_text = "MODERATE LIKELIHOOD"

        status_class = "medium"

        message = (
            "This customer shows a moderate likelihood "
            "of accepting the personal loan offer."
        )

        recommendation = (
            "Consider additional customer engagement "
            "before prioritizing the offer."
        )

    else:

        status_text = "LOW LIKELIHOOD"

        status_class = "low"

        message = (
            "This customer shows a low likelihood "
            "of accepting the personal loan offer."
        )

        recommendation = (
            "Consider lower-priority campaign targeting "
            "for this customer profile."
        )


    # =====================================================
    # AI DECISION
    # =====================================================

    st.html("""
    <div class="section-title">
        AI Decision Analysis
    </div>
    """)


    result_left, result_right = st.columns(
        [1.25, 1]
    )


    # =====================================================
    # RESULT CARD
    # =====================================================

    with result_left:

        st.html(
            f"""
            <div class="result-card">

                <div class="result-label">
                    Personal Loan Acceptance Probability
                </div>

                <div class="result-number">
                    {probability:.2f}%
                </div>

                <div class="result-status {status_class}">
                    {status_text}
                </div>

                <div style="
                    color:#dbeafe;
                    font-size:13px;
                    line-height:1.7;
                    margin-top:18px;
                ">
                    {message}
                </div>

            </div>
            """
        )


    # =====================================================
    # CUSTOMER SNAPSHOT
    # =====================================================

    education_name = {
        1: "Undergraduate",
        2: "Graduate",
        3: "Advanced / Professional"
    }[education]


    with result_right:

        st.html(
            f"""
            <div class="snapshot-box">

                <div class="snapshot-title">
                    Customer Snapshot
                </div>

                <div class="snapshot-row">
                    <div class="snapshot-name">
                        Annual Income
                    </div>

                    <div class="snapshot-value">
                        {format_inr(income_inr)}
                    </div>
                </div>

                <div class="snapshot-row">
                    <div class="snapshot-name">
                        Experience
                    </div>

                    <div class="snapshot-value">
                        {experience} years
                    </div>
                </div>

                <div class="snapshot-row">
                    <div class="snapshot-name">
                        Family Size
                    </div>

                    <div class="snapshot-value">
                        {family}
                    </div>
                </div>

                <div class="snapshot-row">
                    <div class="snapshot-name">
                        Credit Spending
                    </div>

                    <div class="snapshot-value">
                        {format_inr(ccavg_inr)}
                    </div>
                </div>

                <div class="snapshot-row">
                    <div class="snapshot-name">
                        Mortgage
                    </div>

                    <div class="snapshot-value">
                        {format_inr(mortgage_inr)}
                    </div>
                </div>

                <div class="snapshot-row">
                    <div class="snapshot-name">
                        Education
                    </div>

                    <div class="snapshot-value">
                        {education_name}
                    </div>
                </div>

            </div>
            """
        )


    # =====================================================
    # PROBABILITY BAR
    # =====================================================

    st.html("""
    <div class="section-title">
        Acceptance Probability
    </div>
    """)

    st.progress(
        min(
            max(
                int(probability),
                0
            ),
            100
        )
    )


    # =====================================================
    # AI INSIGHT
    # =====================================================

    st.html(
        f"""
        <div class="insight">

            💡 <b>AI Insight:</b>
            {message}

        </div>
        """
    )


    # =====================================================
    # RECOMMENDATION
    # =====================================================

    st.html(
        f"""
        <div class="recommendation">

            <div class="recommendation-title">
                Recommended Action
            </div>

            <div class="recommendation-text">
                {recommendation}
            </div>

        </div>
        """
    )


    # =====================================================
    # EXPLAINABLE AI
    # =====================================================

    st.html("""
    <div class="section-title">
        Why This Prediction?
    </div>

    <div class="section-subtitle">
        These factors show which customer attributes
        influenced this individual prediction most.
    </div>
    """)


    try:

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(
            input_data
        )

        # Handle different SHAP output formats
        if isinstance(shap_values, list):

            values = shap_values[-1][0]

        else:

            values = shap_values[0]

            # Some SHAP versions can produce
            # an extra output dimension.
            if hasattr(values, "ndim") and values.ndim > 1:
                values = values[:, -1]

        values = values.reshape(-1)


        explanation = pd.DataFrame({

            "Feature": input_data.columns,

            "Influence": values

        })


        explanation["Absolute"] = (
            explanation["Influence"].abs()
        )


        explanation = explanation.sort_values(
            "Absolute",
            ascending=False
        ).head(6)


        max_value = explanation["Absolute"].max()


        st.html("""
        <div class="explain-card">

            <div class="explain-title">
                Top Influencing Factors
            </div>

            <div class="explain-text">
                Stronger bars represent stronger influence
                on this prediction.
            </div>

        </div>
        """)


        for _, row in explanation.iterrows():

            feature = row["Feature"]

            influence = float(
                row["Influence"]
            )

            absolute_value = float(
                row["Absolute"]
            )


            if max_value > 0:

                width = (
                    absolute_value /
                    max_value *
                    100
                )

            else:

                width = 0


            if influence >= 0:

                direction = "Supports acceptance"

            else:

                direction = "Reduces acceptance likelihood"


            st.html(
                f"""
                <div class="factor">

                    <div class="factor-header">

                        <div class="factor-name">
                            {feature}
                        </div>

                        <div class="factor-direction">
                            {direction}
                        </div>

                    </div>

                    <div class="factor-bg">

                        <div
                            class="factor-fill"
                            style="width:{width:.1f}%;">
                        </div>

                    </div>

                </div>
                """
            )


    except Exception:

        st.html("""
        <div class="insight">

            Explainable AI could not be generated
            for this prediction.

        </div>
        """)


    # =====================================================
    # KEY INDICATORS
    # =====================================================

    st.html("""
    <div class="section-title">
        Key Customer Indicators
    </div>
    """)


    k1, k2, k3, k4 = st.columns(4)


    with k1:

        st.html(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    Annual Income
                </div>

                <div class="metric-value">
                    {format_inr(income_inr)}
                </div>

                <div class="metric-note">
                    Reported annual income
                </div>

            </div>
            """
        )


    with k2:

        st.html(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    Credit Spending
                </div>

                <div class="metric-value">
                    {format_inr(ccavg_inr)}
                </div>

                <div class="metric-note">
                    Average monthly spending
                </div>

            </div>
            """
        )


    with k3:

        st.html(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    Mortgage
                </div>

                <div class="metric-value">
                    {format_inr(mortgage_inr)}
                </div>

                <div class="metric-note">
                    Outstanding mortgage
                </div>

            </div>
            """
        )


    with k4:

        banking_products = (
            securities
            + cd_account
            + online
            + credit_card
        )

        st.html(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    Banking Products
                </div>

                <div class="metric-value">
                    {banking_products}/4
                </div>

                <div class="metric-note">
                    Active banking products
                </div>

            </div>
            """
        )


# =========================================================
# FOOTER
# =========================================================

st.html("""
<div class="footer">

    LoanWise AI • Machine Learning Decision Support System

    <br>

    Predictions are model-based estimates and are not actual
    lending decisions.

</div>
""")