"""HDP SYSTEM - Heart Disease Prediction (Streamlit app).

Run from the project root:  streamlit run app.py
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data import FEATURES
from src.db import load_history, save_prediction
from src.predict import (
    MODEL_PATH,
    feature_importance,
    load_metrics,
    load_model,
    predict_risk,
    suggestions,
)

st.set_page_config(
    page_title="HDP SYSTEM · Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
)

# ---------------------------------------------------------------- styling
st.markdown(
    """
<style>
.hero {padding: 1.5rem 1.8rem; border-radius: 18px; margin-bottom: 1.2rem;
       background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 55%, #9f1239 100%);}
.hero h1 {font-size: 2rem; margin: 0 0 .3rem 0; color: #ffffff;}
.hero p {margin: 0; color: rgba(255,255,255,.88);}
.badge {display: inline-block; padding: .4rem 1rem; border-radius: 999px;
        font-weight: 600; margin: .2rem 0 .6rem 0;}
</style>
<div class="hero">
  <h1>❤️ HDP SYSTEM</h1>
  <p>Heart Disease Prediction · Logistic Regression, Random Forest &amp; XGBoost · Streamlit</p>
</div>
""",
    unsafe_allow_html=True,
)

RISK_STYLE = {  # band -> (background, text colour)
    "Low": ("#2e7d32", "#ffffff"),
    "Moderate": ("#f9a825", "#1f1f1f"),
    "High": ("#c62828", "#ffffff"),
}

# ---------------------------------------------------------------- sample patients
DEFAULTS = {
    "age": 54, "sex": 1, "cp": 0, "trestbps": 130, "chol": 240, "fbs": False,
    "restecg": 0, "thalach": 150, "exang": False, "oldpeak": 1.0, "slope": 1,
    "ca": 0, "thal": 2,
}
SAMPLE_PATIENTS = {
    "Custom (defaults)": DEFAULTS,
    "Sample: low-risk patient": {
        "age": 45, "sex": 1, "cp": 0, "trestbps": 118, "chol": 190, "fbs": False,
        "restecg": 0, "thalach": 170, "exang": True, "oldpeak": 0.0, "slope": 1,
        "ca": 2, "thal": 3,
    },
    "Sample: high-risk patient": {
        "age": 60, "sex": 0, "cp": 1, "trestbps": 150, "chol": 280, "fbs": False,
        "restecg": 1, "thalach": 110, "exang": False, "oldpeak": 0.5, "slope": 2,
        "ca": 0, "thal": 2,
    },
}

for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)
st.session_state.setdefault("sample", "Custom (defaults)")


def apply_sample():
    for key, value in SAMPLE_PATIENTS[st.session_state["sample"]].items():
        st.session_state[key] = value


# ---------------------------------------------------------------- model
@st.cache_resource
def get_model():
    return load_model()


if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `python -m src.train` first, then refresh this page.")
    st.stop()

model = get_model()
metrics = load_metrics()
best_name = metrics["best_model"] if metrics else "Trained model"

with st.sidebar:
    st.markdown("### Model")
    st.write(f"**{best_name}**")
    if metrics:
        test = metrics["test"][best_name]
        c1, c2 = st.columns(2)
        c1.metric("Recall", f"{test['recall']:.0%}", help="Share of sick patients correctly flagged on unseen test data")
        c2.metric("ROC-AUC", f"{test['roc_auc']:.2f}")
    st.divider()
    st.caption("Educational demo only. Not a medical diagnosis.")

tab_predict, tab_perf, tab_history, tab_about = st.tabs(
    ["🩺 Predict", "📊 Model performance", "🗂️ History", "ℹ️ About"]
)

# ---------------------------------------------------------------- predict tab
with tab_predict:
    st.selectbox("Load a sample patient", options=list(SAMPLE_PATIENTS), key="sample", on_change=apply_sample)
    st.caption("Change any input and the risk updates instantly. Press **Save to history** to log a result.")

    form_col, result_col = st.columns([1.1, 1], gap="large")

    with form_col:
        st.markdown("#### Demographics & vitals")
        d1, d2 = st.columns(2)
        with d1:
            st.slider("Age", 18, 100, key="age")
            st.radio("Sex", options=[1, 0], format_func=lambda v: "Male" if v == 1 else "Female",
                     horizontal=True, key="sex")
            st.slider("Resting blood pressure (mm Hg)", 80, 220, key="trestbps")
        with d2:
            st.slider("Cholesterol (mg/dl)", 100, 600, key="chol")
            st.slider("Max heart rate achieved", 60, 220, key="thalach")
            st.selectbox("Chest pain type (code 0-3)", options=[0, 1, 2, 3], key="cp")

        st.markdown("#### ECG & exercise test")
        e1, e2 = st.columns(2)
        with e1:
            st.toggle("Fasting blood sugar > 120 mg/dl", key="fbs")
            st.selectbox("Resting ECG (code 0-2)", options=[0, 1, 2], key="restecg")
            st.toggle("Exercise-induced angina", key="exang")
        with e2:
            st.slider("ST depression (oldpeak)", 0.0, 7.0, step=0.1, key="oldpeak")
            st.selectbox("ST slope at peak exercise (code 0-2)", options=[0, 1, 2], key="slope")

        st.markdown("#### Imaging")
        i1, i2 = st.columns(2)
        with i1:
            st.selectbox("Major vessels (0-4)", options=[0, 1, 2, 3, 4], key="ca")
        with i2:
            st.selectbox("Thal (code 0-3)", options=[0, 1, 2, 3], key="thal")

    # Collect the current inputs in the order the model expects.
    inputs = {}
    for feature in FEATURES:
        value = st.session_state[feature]
        inputs[feature] = float(value) if feature == "oldpeak" else int(value)

    probability, level = predict_risk(model, inputs)
    band = level.split()[0]  # "Low", "Moderate" or "High"
    bg, fg = RISK_STYLE[band]

    with result_col:
        with st.container(border=True):
            st.markdown("#### Risk assessment")
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%", "valueformat": ".1f"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": bg},
                    "steps": [
                        {"range": [0, 30], "color": "rgba(46,125,50,0.18)"},
                        {"range": [30, 60], "color": "rgba(249,168,37,0.25)"},
                        {"range": [60, 100], "color": "rgba(198,40,40,0.18)"},
                    ],
                },
            ))
            gauge.update_layout(height=250, margin=dict(l=30, r=30, t=20, b=0))
            st.plotly_chart(gauge, width="stretch")
            st.markdown(
                f'<span class="badge" style="background:{bg};color:{fg}">{level}</span>',
                unsafe_allow_html=True,
            )
            m1, m2 = st.columns(2)
            m1.metric("Estimated probability", f"{probability:.1%}")
            m2.metric("Model", best_name)

        with st.container(border=True):
            st.markdown("#### Personalized suggestions")
            for tip in suggestions(inputs):
                st.markdown(f"- {tip}")

        if st.button("💾 Save to history", type="primary", width="stretch"):
            try:
                save_prediction(inputs, probability, level)
                st.toast("Prediction saved to history", icon="✅")
            except Exception as exc:  # saving must never crash the page
                st.error(f"Could not save the prediction: {exc}")

    st.caption(
        "Educational demo only. This is not a medical diagnosis. "
        "Please consult a doctor for any health concern."
    )

# ---------------------------------------------------------------- performance tab
with tab_perf:
    if metrics is None:
        st.info("No metrics report found. Run `python -m src.train` to generate it.")
    else:
        test = metrics["test"][best_name]
        cm = test["confusion_matrix"]

        st.markdown(f"#### Best model: {best_name}")
        st.caption("Scores below use the held-out test set (20% of the data), which the model never trained on.")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy", f"{test['accuracy']:.1%}")
        k2.metric("Recall (sick patients caught)", f"{test['recall']:.1%}")
        k3.metric("ROC-AUC", f"{test['roc_auc']:.3f}")
        k4.metric("Missed cases (false negatives)", cm["fn"])

        rows = []
        for name, cv in metrics["cross_validation"].items():
            t = metrics["test"][name]
            rows.append({
                "Model": name,
                "CV F2": cv["f2"],
                "CV recall": cv["recall"],
                "Test accuracy": t["accuracy"],
                "Test recall": t["recall"],
                "Test ROC-AUC": t["roc_auc"],
                "Missed (FN)": t["confusion_matrix"]["fn"],
            })
        comparison = pd.DataFrame(rows).round(3)
        st.dataframe(comparison, hide_index=True, width="stretch")

        compare_chart = go.Figure()
        for col in ["Test accuracy", "Test recall", "Test ROC-AUC"]:
            compare_chart.add_bar(name=col, x=comparison["Model"], y=comparison[col])
        compare_chart.update_layout(
            barmode="group", height=340, yaxis=dict(range=[0, 1]),
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(compare_chart, width="stretch")

    left, right = st.columns(2)
    with left:
        if metrics is not None:
            st.markdown("**Confusion matrix (test set)**")
            z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
            heat = go.Figure(go.Heatmap(
                z=z,
                x=["Predicted: no", "Predicted: yes"],
                y=["Actual: no", "Actual: yes"],
                text=z,
                texttemplate="%{text}",
                textfont={"size": 18},
                colorscale="Blues",
                showscale=False,
            ))
            heat.update_yaxes(autorange="reversed")
            heat.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(heat, width="stretch")
    with right:
        st.markdown("**What the model relies on most**")
        importance = feature_importance(model)
        if importance is None:
            st.caption("This model type does not expose feature importance.")
        else:
            bars = go.Figure(go.Bar(
                x=importance["importance"], y=importance["feature"],
                orientation="h", marker_color="#1e3a8a",
            ))
            bars.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Importance")
            st.plotly_chart(bars, width="stretch")

# ---------------------------------------------------------------- history tab
with tab_history:
    records = load_history()
    if not records:
        st.info("No saved predictions yet. Use **Save to history** on the Predict tab.")
    else:
        history = pd.DataFrame(records)
        h1, h2, h3 = st.columns(3)
        h1.metric("Saved predictions", len(history))
        h2.metric("High-risk results", int(history["risk_level"].str.startswith("High").sum()))
        h3.metric("Average probability", f"{history['probability'].mean():.1%}")
        st.dataframe(history, hide_index=True, width="stretch")
        st.download_button(
            "⬇️ Download CSV",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="hdp_predictions.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------- about tab
with tab_about:
    st.markdown(
        """
#### What this app does
Estimates the probability that a patient has heart disease from 13 clinical measurements, and
shows simple, general suggestions based on the values entered.

#### How the model was built
1. **Data:** UCI Heart Disease dataset, 303 patient records.
2. **Cleaning:** missing values and duplicate rows are removed, and data types are checked.
3. **Training:** Logistic Regression (baseline), Random Forest and XGBoost.
4. **Evaluation:** 5-fold cross-validation scored with F2 (recall counts more than precision),
   a held-out test set, and a confusion matrix.
5. **Serving:** the best model is served through this app, and every saved prediction is stored with SQLAlchemy.

#### Disclaimer
This is an educational demonstration. It is **not** a medical device and must not be used for
diagnosis or treatment decisions.
"""
    )
