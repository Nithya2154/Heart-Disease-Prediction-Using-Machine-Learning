import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from scipy.stats import ttest_ind, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS — dark clinical aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Main background */
.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] * {
    color: #c9d1d9 !important;
}

/* Header */
.hero {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 50%, #1a0a0a 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(220,50,50,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #f0f6fc;
    margin: 0 0 0.4rem 0;
    line-height: 1.1;
}
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #8b949e;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.heart-pulse {
    font-size: 3rem;
    display: inline-block;
    animation: pulse 1.5s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.15); }
}

/* Metric cards */
.metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #dc3232; }
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #f0f6fc;
}
.metric-accent { color: #dc3232; }

/* Result boxes */
.result-positive {
    background: linear-gradient(135deg, #2d1515, #1a0a0a);
    border: 1px solid #dc3232;
    border-radius: 12px;
    padding: 1.8rem 2rem;
    text-align: center;
    margin: 1rem 0;
}
.result-negative {
    background: linear-gradient(135deg, #0f2318, #0d1117);
    border: 1px solid #2ea043;
    border-radius: 12px;
    padding: 1.8rem 2rem;
    text-align: center;
    margin: 1rem 0;
}
.result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    margin: 0.5rem 0;
}
.result-prob {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: #8b949e;
}

/* Section headers */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #dc3232;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #21262d;
}

/* Streamlit overrides */
.stSlider > div > div > div { background: #dc3232 !important; }
.stSelectbox > div > div { background: #161b22; border-color: #21262d; }
.stButton > button {
    background: #dc3232;
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    padding: 0.6rem 2rem;
    font-size: 1rem;
    width: 100%;
    transition: background 0.2s, transform 0.1s;
}
.stButton > button:hover {
    background: #b82828;
    transform: translateY(-1px);
}
div[data-testid="stMetricValue"] {
    font-family: 'DM Serif Display', serif;
    color: #f0f6fc;
}

/* Warning / info */
.stAlert { background: #161b22; border-color: #21262d; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 8px; }
.stTabs [data-baseweb="tab"] { color: #8b949e; }
.stTabs [aria-selected="true"] { color: #f0f6fc !important; background: #21262d; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Data Loading & Model Training
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model on your dataset…")
def load_and_train(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.drop_duplicates(inplace=True)

    target = 'heart_disease'
    results = []

    # T-Tests for numerical
    numerical_cols = df.select_dtypes(
        include=['int64', 'float64']).columns.tolist()
    if target in numerical_cols:
        numerical_cols.remove(target)

    for col in numerical_cols:
        group0 = df[df[target] == 0][col]
        group1 = df[df[target] == 1][col]
        stat, p_value = ttest_ind(group0, group1)
        results.append({
            'Feature': col, 'Feature_Type': 'Numerical',
            'Test': 'T-Test', 'Statistic': round(stat, 4),
            'P_Value': round(p_value, 6),
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })

    # Chi-Square for categorical
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target in categorical_cols:
        categorical_cols.remove(target)

    for col in categorical_cols:
        contingency_table = pd.crosstab(df[col], df[target])
        stat, p_value, dof, expected = chi2_contingency(contingency_table)
        results.append({
            'Feature': col, 'Feature_Type': 'Categorical',
            'Test': 'Chi-Square', 'Statistic': round(stat, 4),
            'P_Value': round(p_value, 6),
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })

    # Build pipeline
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), ['age', 'cholesterol', 'blood_pressure']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['smoking_status'])
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])

    X = df[['age', 'cholesterol', 'blood_pressure', 'smoking_status']]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
        'precision': round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        'recall': round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        'cm': confusion_matrix(y_test, y_pred).tolist(),
        'report': classification_report(y_test, y_pred, output_dict=True)
    }

    smoking_options = sorted(df['smoking_status'].dropna().unique().tolist())

    return pipeline, metrics, pd.DataFrame(results), df, smoking_options


# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="heart-pulse">🫀</span>
    <div class="hero-title">Heart Disease Predictor</div>
    <div class="hero-sub">Logistic Regression · Clinical Risk Assessment · ML Pipeline</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR — File Upload + Patient Inputs
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Dataset")
    uploaded_file = st.file_uploader(
        "Upload heart_disease CSV", type=["csv"],
        help="Needs columns: age, cholesterol, blood_pressure, smoking_status, heart_disease"
    )

    st.markdown("---")
    st.markdown("### 👤 Patient Details")
    st.caption("Enter values to get a prediction")

    patient_name = st.text_input(
        "Patient Name", placeholder="e.g. John Doe", max_chars=60)

    age = st.slider("Age", min_value=18, max_value=100, value=45, step=1)
    cholesterol = st.slider("Cholesterol (mg/dL)",
                            min_value=100, max_value=400, value=200, step=1)
    blood_pressure = st.slider(
        "Blood Pressure (mmHg)", min_value=60, max_value=200, value=120, step=1)

    # Smoking status — populated after model loads
    smoking_placeholder = st.empty()

    st.markdown("---")
    predict_btn = st.button("🔍 Run Prediction", use_container_width=True)


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
if uploaded_file is None:
    st.info(
        "⬆️ Upload your **heart_disease_50000.csv** file in the sidebar to get started.")
    st.markdown("""
    <div style='background:#161b22;border:1px solid #21262d;border-radius:10px;padding:1.5rem 2rem;margin-top:1rem;'>
    <div class='section-label'>Required CSV Columns</div>
    <p style='font-family:DM Mono,monospace;font-size:0.85rem;color:#8b949e;'>
    age &nbsp;·&nbsp; cholesterol &nbsp;·&nbsp; blood_pressure &nbsp;·&nbsp; smoking_status &nbsp;·&nbsp; heart_disease
    </p>
    </div>
    """, unsafe_allow_html=True)

else:
    pipeline, metrics, results_df, df, smoking_options = load_and_train(
        uploaded_file)

    # Populate smoking dropdown in sidebar
    with smoking_placeholder:
        smoking_status = st.selectbox(
            "Smoking Status", options=smoking_options)

    tab1, tab2, tab3 = st.tabs(
        ["🔮 Prediction", "📊 Model Performance", "🔬 Feature Analysis"])

    # ── TAB 1: PREDICTION ──────────────────────────
    with tab1:
        if predict_btn:
            # Validate name
            display_name = patient_name.strip() if patient_name.strip() else "Unknown Patient"

            input_df = pd.DataFrame([{
                'age': age,
                'cholesterol': cholesterol,
                'blood_pressure': blood_pressure,
                'smoking_status': smoking_status
            }])

            prediction = pipeline.predict(input_df)[0]
            proba = pipeline.predict_proba(input_df)[0]
            risk_pct = round(proba[1] * 100, 1)

            if prediction == 1:
                st.markdown(f"""
                <div class="result-positive">
                    <div style='font-size:2.5rem;'>⚠️</div>
                    <div style='font-family:DM Mono,monospace;font-size:0.75rem;color:#fc8181;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.3rem;'>Patient Report</div>
                    <div class="result-title" style='color:#fc8181;'>{display_name}</div>
                    <div style='font-family:DM Serif Display,serif;font-size:1.2rem;color:#f0f6fc;margin:0.5rem 0;'>Heart Disease Detected</div>
                    <div class="result-prob">Estimated Risk Probability: <strong>{risk_pct}%</strong></div>
                </div>
                """, unsafe_allow_html=True)
                st.warning(
                    f"**{display_name}** is at **high risk**. Please consult a cardiologist immediately.")
            else:
                st.markdown(f"""
                <div class="result-negative">
                    <div style='font-size:2.5rem;'>✅</div>
                    <div style='font-family:DM Mono,monospace;font-size:0.75rem;color:#68d391;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.3rem;'>Patient Report</div>
                    <div class="result-title" style='color:#68d391;'>{display_name}</div>
                    <div style='font-family:DM Serif Display,serif;font-size:1.2rem;color:#f0f6fc;margin:0.5rem 0;'>No Heart Disease Detected</div>
                    <div class="result-prob">Estimated Risk Probability: <strong>{risk_pct}%</strong></div>
                </div>
                """, unsafe_allow_html=True)
                st.success(
                    f"**{display_name}** results look healthy. Continue regular check-ups.")

            # Patient summary
            st.markdown(
                "<div class='section-label' style='margin-top:1.5rem;'>Patient Summary</div>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-label'>Age</div><div class='metric-value'>{age}</div></div>", unsafe_allow_html=True)
            with c2:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-label'>Cholesterol</div><div class='metric-value'>{cholesterol}</div></div>", unsafe_allow_html=True)
            with c3:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-label'>Blood Pressure</div><div class='metric-value'>{blood_pressure}</div></div>", unsafe_allow_html=True)
            with c4:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-label'>Smoking</div><div class='metric-value' style='font-size:1.1rem;padding-top:0.4rem;'>{smoking_status}</div></div>", unsafe_allow_html=True)

            # Probability bar
            st.markdown(
                "<div class='section-label' style='margin-top:1.5rem;'>Risk Score</div>", unsafe_allow_html=True)
            st.progress(int(risk_pct), text=f"Risk: {risk_pct}%")

        else:
            st.markdown("""
            <div style='text-align:center;padding:4rem 2rem;color:#8b949e;'>
                <div style='font-size:3rem;margin-bottom:1rem;'>🩺</div>
                <div style='font-family:DM Serif Display,serif;font-size:1.4rem;color:#c9d1d9;'>Ready for Assessment</div>
                <div style='font-size:0.9rem;margin-top:0.5rem;'>Fill in patient details in the sidebar and click <strong>Run Prediction</strong></div>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 2: MODEL PERFORMANCE ───────────────────
    with tab2:
        st.markdown("<div class='section-label'>Model Metrics</div>",
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Accuracy", f"{metrics['accuracy']}%")
        with c2:
            st.metric("Precision", f"{metrics['precision']}%")
        with c3:
            st.metric("Recall", f"{metrics['recall']}%")

        st.markdown("---")
        st.markdown(
            "<div class='section-label'>Confusion Matrix</div>", unsafe_allow_html=True)
        cm = metrics['cm']
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: No Disease", "Actual: Disease"],
            columns=["Predicted: No Disease", "Predicted: Disease"]
        )
        st.dataframe(cm_df, use_container_width=True)

        st.markdown(
            "<div class='section-label' style='margin-top:1.5rem;'>Classification Report</div>", unsafe_allow_html=True)
        report = metrics['report']
        report_rows = []
        for key in ['0', '1', 'macro avg', 'weighted avg']:
            if key in report:
                row = report[key]
                report_rows.append({
                    'Class': 'No Disease' if key == '0' else ('Disease' if key == '1' else key.title()),
                    'Precision': f"{round(row['precision']*100, 1)}%",
                    'Recall': f"{round(row['recall']*100, 1)}%",
                    'F1-Score': f"{round(row['f1-score']*100, 1)}%",
                    'Support': int(row['support'])
                })
        st.dataframe(pd.DataFrame(report_rows),
                     use_container_width=True, hide_index=True)

        st.markdown(
            "<div class='section-label' style='margin-top:1.5rem;'>Dataset Overview</div>", unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("Total Records", f"{len(df):,}")
        with d2:
            st.metric("Disease Cases", f"{df['heart_disease'].sum():,}")
        with d3:
            st.metric("Healthy Cases", f"{(df['heart_disease'] == 0).sum():,}")

    # ── TAB 3: FEATURE ANALYSIS ────────────────────
    with tab3:
        st.markdown(
            "<div class='section-label'>Statistical Significance Tests</div>", unsafe_allow_html=True)
        st.caption(
            "T-Test for numerical features · Chi-Square for categorical features")

        sig_col = results_df['Significant'].map({'Yes': '✅ Yes', 'No': '❌ No'})
        display_df = results_df.copy()
        display_df['Significant'] = sig_col
        display_df['Statistic'] = display_df['Statistic'].round(4)
        display_df['P_Value'] = display_df['P_Value'].apply(
            lambda x: f"{x:.6f}")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        sig_features = results_df[results_df['Significant'].str.lower(
        ) == 'yes']['Feature'].tolist()
        st.markdown(f"""
        <div style='background:#161b22;border:1px solid #21262d;border-radius:8px;padding:1rem 1.5rem;margin-top:1rem;'>
        <div class='section-label'>Significant Features (p &lt; 0.05)</div>
        <p style='font-family:DM Mono,monospace;font-size:0.9rem;color:#2ea043;margin:0;'>
        {' &nbsp;·&nbsp; '.join(sig_features) if sig_features else 'None found'}
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            "<div class='section-label' style='margin-top:1.5rem;'>Feature Distributions</div>", unsafe_allow_html=True)
        selected_feat = st.selectbox("Select a numerical feature to explore", options=[
                                     'age', 'cholesterol', 'blood_pressure'])

        col_a, col_b = st.columns(2)
        with col_a:
            no_disease = df[df['heart_disease'] == 0][selected_feat]
            st.markdown(f"**No Disease** — {selected_feat}")
            st.write(no_disease.describe().round(2))
        with col_b:
            disease = df[df['heart_disease'] == 1][selected_feat]
            st.markdown(f"**Disease** — {selected_feat}")
            st.write(disease.describe().round(2))

        st.bar_chart(
            pd.DataFrame({
                'No Disease': no_disease.value_counts().sort_index(),
                'Disease': disease.value_counts().sort_index()
            }).fillna(0)
        )

# Footer
st.markdown("""
<div style='text-align:center;color:#30363d;font-family:DM Mono,monospace;font-size:0.7rem;
margin-top:3rem;padding-top:1rem;border-top:1px solid #21262d;'>
⚠️ FOR EDUCATIONAL PURPOSES ONLY · NOT A MEDICAL DIAGNOSIS TOOL
</div>
""", unsafe_allow_html=True)
