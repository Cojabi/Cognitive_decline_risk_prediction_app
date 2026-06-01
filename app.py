import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from model_backend import XGB_Model_Backend
from plotting_backend import plot_event_probability_with_ci
from constants_interface import TIME_GRID, FEATURE_ORDER_BM_PTAU


MODEL_PATH = "models/xgb_cox_model.joblib"


# ======================================
# LOAD MODEL
# ======================================
@st.cache_resource
def load_model(model_path, bootstrap_dir=None):
    return XGB_Model_Backend(
        model_path=model_path,
        bootstrap_dir=bootstrap_dir
    )

model_backend = load_model(MODEL_PATH)

@st.cache_resource
def get_shap_explainer(_model_backend):
    return shap.TreeExplainer(_model_backend.model)
def compute_shap(explainer, X: pd.DataFrame):
    shap_values = explainer(X)
    return shap_values[0]

shap_explainer = get_shap_explainer(model_backend)


# ======================================
# SESSION STATE
# ======================================
if "results" not in st.session_state:
    st.session_state.results = None


# ======================================
# HELPERS
# ======================================
def empty_to_none(x):
    """Convert empty string → None"""
    if x is None:
        return None
    if isinstance(x, str) and x.strip() == "":
        return None
    return x


def to_float(x):
    """Safe float conversion: None → np.nan"""
    if x is None:
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def to_int_bool(x):
    return int(bool(x))


# ======================================
# INPUT WIDGET (MISSINGNESS-SAFE)
# ======================================
def missing_number_input(label, key):
    raw = st.text_input(
        label=label,
        key=key,
        placeholder="Leave blank if unknown"
    )

    raw = empty_to_none(raw)

    return to_float(raw)


# ======================================
# UI
# ======================================
st.title("Calibrated Survival Risk Prediction")
st.caption(
    "Predict the individualized risk of developing cognitive impairment over the next decade."
    "This prediction tool is a proof-of-concept and not meant for clinical application at this stage."
    )

with st.form("inputs"):
    st.caption(
    "Predictions support missing values. "
    "Empty fields are treated as missing (NaN) internally."
    )

    ptau = missing_number_input("Harmonized pTau217", "ptau")
    age = missing_number_input("Age (years)", "age")
    mmse = missing_number_input("MMSE", "mmse")
    bp_sys = missing_number_input("Systolic BP", "bp_sys")
    bp_dia = missing_number_input("Diastolic BP", "bp_dia")
    edu = missing_number_input("Education", "edu")
    bmi = missing_number_input("BMI", "bmi")

    hyp = st.checkbox("Hypertension ever", key="hyp")
    sex = st.selectbox("Sex", ["Female", "Male"], key="sex")
    rw = st.checkbox("White", key="rw")
    rb = st.checkbox("Black or African American", key="rb")
    ra = st.checkbox("Asian", key="ra")

    submit = st.form_submit_button("Predict")


# ======================================
# FEATURE SET
# ======================================
numeric_cols = [
    "ptau217_harm",
    "ptau217_Age",
    "BP_Systolic",
    "BP_Diastolic",
    "Education",
    "BMI",
    "MMSE"
]


# ======================================
# PREDICTION
# ======================================
if submit:

    # ---------- model row (SAFE CASTING LAYER) ----------
    row = {
        "ptau217_harm": to_float(ptau),
        "ptau217_Age": to_float(age),
        "BP_Systolic": to_float(bp_sys),
        "BP_Diastolic": to_float(bp_dia),
        "Education": to_float(edu),
        "BMI": to_float(bmi),
        "Hypertension_ever": to_int_bool(hyp),
        "MMSE": to_float(mmse),
        "Sex": int(sex == "Female"),
        "Race_White": int(rw),
        "Race_Black or African American": int(rb),
        "Race_Asian": int(ra),
    }

    X = pd.DataFrame([row])[FEATURE_ORDER_BM_PTAU]
    X[numeric_cols] = X[numeric_cols].astype(float)

    # ---------- prediction ----------
    predicted_risk = model_backend.predict_for_single_patient(
        X, TIME_GRID, return_CI=True
    )

    # ---------- SHAP ----------
    shap_single = compute_shap(shap_explainer, X)
    shap_eta = np.asarray(shap_single.values).squeeze()
    base_eta = float(shap_single.base_values)

    hazard_ratios = np.exp(shap_eta)

    shap_hr_df = pd.DataFrame({
        "Feature": shap_single.feature_names,
        "Hazard Multiplier": hazard_ratios,
    }).sort_values(
        by="Hazard Multiplier",
        key=lambda s: np.abs(np.log(s)),
        ascending=False
    )

    patient_eta = base_eta + np.sum(shap_eta)
    population_rr = np.exp(base_eta)
    patient_rr = np.exp(patient_eta)

    # ---------- STORE ----------
    st.session_state.results = {
        "predicted_risk": predicted_risk,
        "X": X,
        "shap_single": shap_single,
        "shap_hr_df": shap_hr_df,
        "risk_5y": predicted_risk.loc[5, "event_probability"],
        "risk_10y": predicted_risk.loc[10, "event_probability"],
        "rr_ratio": patient_rr / population_rr,
    }


# ======================================
# OUTPUT
# ======================================
results = st.session_state.results

if results is not None:

    predicted_risk = results["predicted_risk"]

    st.metric("5-Year Absolute Risk", f"{results['risk_5y']:.2%}")
    st.metric("10-Year Absolute Risk", f"{results['risk_10y']:.2%}")
    st.metric("Relative Risk vs Population", f"{results['rr_ratio']:.2f}×")

    # ---------- RISK CURVE ----------
    st.subheader("Risk Trajectory")

    X = results["X"]

    X_low_ptau = X.copy()
    X_low_ptau["ptau217_harm"] *= 0.9

    X_lowest_ptau = X.copy()
    X_lowest_ptau["ptau217_harm"] *= 0.75

    predicted_lowest = model_backend.predict_for_single_patient(
        X_lowest_ptau, TIME_GRID, return_CI=True
    )

    fig = plot_event_probability_with_ci(
        predicted_risk,
        title="Predicted Event Probability Over Time",
        vertical_lines_at_x=[5, 10],
        breslow_estimator=model_backend.breslow_estimator,
        plot_reduced_ptau={
            "25% reduced pTau": predicted_lowest
        }
    )

    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(predicted_risk.style.format("{:.2%}"))

    # ---------- SHAP ----------
    st.subheader("Feature Effects on Hazard (Model-Based)")
    st.dataframe(results["shap_hr_df"].style.format({"Hazard Multiplier": "{:.2f}×"}))

    st.subheader("SHAP Waterfall")

    fig_shap = plt.figure()
    shap.plots.waterfall(results["shap_single"], show=False, max_display=10)
    plt.xlabel("Contribution to log-hazard")
    st.pyplot(fig_shap)
    plt.close(fig_shap)