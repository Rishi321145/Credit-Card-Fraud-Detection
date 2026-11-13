import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Analytics | Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "credit_fraud_model.pkl"
DATA_CANDIDATES = [
    ROOT / "data" / "archive" / "creditcard.csv",
    ROOT / "data" / "creditcard.csv",
]

# ------------------------------------------------------------
# Dark fintech styling
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background:
          radial-gradient(circle at 10% 0%, rgba(28, 67, 104, .20), transparent 28%),
          #080d17;
        color: #edf3ff;
    }
    [data-testid="stSidebar"] {
        background: #0b1220;
        border-right: 1px solid #1d293b;
    }
    [data-testid="stHeader"] { background: rgba(8,13,23,0.85); }
    h1, h2, h3 { color: #f4f7ff !important; }
    p, label, span { color: #c3cde0; }
    .hero {
        padding: 22px 24px;
        border: 1px solid #24334a;
        border-radius: 18px;
        background: linear-gradient(120deg, #111d31, #0d1524 70%);
        margin-bottom: 18px;
    }
    .hero-kicker {
        color: #36d6a3;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-size: 11px;
        font-weight: 800;
    }
    .hero-title {
        color: #f6f8ff;
        font-size: 28px;
        font-weight: 800;
        margin: 5px 0 4px 0;
    }
    .hero-sub { color: #9eacc3; font-size: 13px; }
    .metric-card {
        background: linear-gradient(145deg, #121d2e, #0e1726);
        border: 1px solid #24334a;
        border-radius: 15px;
        padding: 16px 17px;
        min-height: 105px;
    }
    .metric-label {
        color: #92a2ba;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .7px;
    }
    .metric-value {
        color: #f7f9ff;
        font-size: 25px;
        font-weight: 800;
        margin-top: 8px;
    }
    .metric-note { color: #71839e; font-size: 11px; margin-top: 3px; }
    .section-card {
        background: #0e1726;
        border: 1px solid #223149;
        border-radius: 16px;
        padding: 17px;
    }
    .muted { color: #91a1b8; font-size: 12px; }
    .success-box {
        border: 1px solid rgba(46, 213, 155, .4);
        background: rgba(46, 213, 155, .09);
        border-radius: 14px;
        padding: 17px;
    }
    .danger-box {
        border: 1px solid rgba(255, 77, 109, .45);
        background: rgba(255, 77, 109, .09);
        border-radius: 14px;
        padding: 17px;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #24334a;
        border-radius: 12px;
        overflow: hidden;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid #2c8e82;
        background: linear-gradient(90deg, #087e78, #0d9b83);
        color: white;
        font-weight: 700;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: #42e0b4;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Load model and data
# ------------------------------------------------------------
@st.cache_resource
def load_bundle(path_string):
    return joblib.load(path_string)

@st.cache_data
def load_dataset(path_string):
    return pd.read_csv(path_string)

def find_dataset():
    for candidate in DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    return None

def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def get_model_objects(bundle):
    # Supports the saved dictionary bundle and a plain estimator as fallback.
    if isinstance(bundle, dict):
        model = bundle.get("model")
        scaler = bundle.get("scaler")
        features = bundle.get("features")
        scale_columns = bundle.get("scale_columns", [])
        threshold = float(bundle.get("threshold", 0.50))
    else:
        model, scaler, features, scale_columns, threshold = bundle, None, None, [], 0.50
    if model is None:
        raise ValueError("Model bundle does not contain a 'model' entry.")
    if not features:
        raise ValueError("Model bundle does not contain the feature list.")
    return model, scaler, list(features), list(scale_columns), threshold

def prepare_input_frame(frame, features, scale_columns, scaler_object=None):
    data = frame.copy()
    if "Log_Amount" in features and "Log_Amount" not in data.columns:
        if "Amount" not in data.columns:
            raise ValueError("CSV must contain an Amount column to calculate Log_Amount.")
        data["Log_Amount"] = np.log1p(data["Amount"].clip(lower=0))

    missing = [col for col in features if col not in data.columns]
    if missing:
        raise ValueError(
            "CSV is missing required model features: " + ", ".join(missing)
        )

    data = data[features].copy()
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if data.isna().any().any():
        bad_cols = data.columns[data.isna().any()].tolist()
        raise ValueError("Missing or non-numeric values in: " + ", ".join(bad_cols))

    # The scaler was fitted on the selected scaled columns during training.
    if scale_columns and scaler_object is not None:
        cols = [c for c in scale_columns if c in data.columns]
        if cols:
            data.loc[:, cols] = scaler_object.transform(data[cols])
    return data

# Load saved model
if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: `{MODEL_PATH}`. "
        "Place credit_fraud_model.pkl inside the project's models folder."
    )
    st.stop()

try:
    bundle = load_bundle(str(MODEL_PATH))
    model, scaler_object, feature_names, scale_columns, saved_threshold = get_model_objects(bundle)
except Exception as exc:
    st.error(f"Could not load the model bundle: {exc}")
    st.stop()

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ Fraud Analytics")
    st.caption("Credit Card Transaction Intelligence")
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["Overview", "Single Prediction", "Bulk Prediction", "PhonePe Test Mode", "Model Performance"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown("**MODEL STATUS**")
    st.success("Model loaded")
    st.caption(f"Random Forest • Threshold {saved_threshold:.2f}")
    st.caption("Local application")

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Credit Card Security • Machine Learning</div>
      <div class="hero-title">Fraud Analytics Dashboard</div>
      <div class="hero-sub">Monitor transactions, identify suspicious activity, and review model performance.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

dataset_path = find_dataset()
dataset = None
if dataset_path:
    try:
        dataset = load_dataset(str(dataset_path))
    except Exception:
        dataset = None

# ------------------------------------------------------------
# Overview
# ------------------------------------------------------------
if page == "Overview":
    st.subheader("Overview")
    if dataset is not None and "Class" in dataset.columns:
        total = len(dataset)
        fraud_count = int((dataset["Class"] == 1).sum())
        genuine_count = int((dataset["Class"] == 0).sum())
        fraud_rate = (fraud_count / total * 100) if total else 0
        source_note = "Loaded from local creditcard.csv"
    else:
        total = fraud_count = genuine_count = 0
        fraud_rate = 0
        source_note = "Upload a dataset in Bulk Prediction to view live transaction analytics"

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Total transactions", f"{total:,}" if total else "—", source_note)
    with c2: metric_card("Fraud transactions", f"{fraud_count:,}" if total else "—", "Class 1")
    with c3: metric_card("Genuine transactions", f"{genuine_count:,}" if total else "—", "Class 0")
    with c4: metric_card("Fraud rate", f"{fraud_rate:.3f}%" if total else "—", "Based on available dataset")

    left, right = st.columns([1.65, 1])
    with left:
        st.markdown("### Transaction class distribution")
        if dataset is not None and "Class" in dataset.columns:
            counts = dataset["Class"].map({0: "Genuine", 1: "Fraud"}).fillna(dataset["Class"].astype(str)).value_counts().reset_index()
            counts.columns = ["Transaction type", "Count"]
            fig = px.bar(
                counts, x="Transaction type", y="Count", color="Transaction type",
                color_discrete_map={"Genuine": "#1dbb91", "Fraud": "#ff5574"},
                template="plotly_dark",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False, margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Place your dataset at `data/archive/creditcard.csv` to show the real class distribution here.")
    with right:
        st.markdown("### Model snapshot")
        st.markdown(
            """
            <div class="section-card">
              <div class="metric-label">Selected model</div>
              <div style="font-size:22px;font-weight:800;color:#f5f7ff;margin:7px 0;">Random Forest</div>
              <div class="muted">Saved model bundle</div>
              <hr style="border-color:#26364e;">
              <div style="display:flex;justify-content:space-between;margin:9px 0;"><span>Accuracy</span><b>99.95%</b></div>
              <div style="display:flex;justify-content:space-between;margin:9px 0;"><span>Precision</span><b>91.03%</b></div>
              <div style="display:flex;justify-content:space-between;margin:9px 0;"><span>Recall</span><b>74.74%</b></div>
              <div style="display:flex;justify-content:space-between;margin:9px 0;"><span>F1-score</span><b>82.08%</b></div>
              <div style="display:flex;justify-content:space-between;margin:9px 0;"><span>PR-AUC</span><b>81.84%</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Metrics are from the held-out test evaluation recorded during model development.")

    st.markdown("### Model workflow")
    st.write("Dataset → preprocessing → Random Forest → fraud probability → threshold-based prediction")
    st.caption("The model expects the original 30 transaction features plus the derived Log_Amount feature.")

# ------------------------------------------------------------
# Single Prediction
# ------------------------------------------------------------
elif page == "Single Prediction":
    st.subheader("Single transaction prediction")
    st.info(
        "The dataset's V1–V28 columns are anonymised PCA features. "
        "For a meaningful prediction, enter the real feature values from a transaction processed in the same way as the training dataset."
    )
    with st.form("single_prediction_form"):
        st.markdown("#### Transaction details")
        a, b = st.columns(2)
        with a:
            amount = st.number_input("Transaction amount", min_value=0.0, value=100.0, step=10.0)
        with b:
            time_value = st.number_input("Time (seconds from dataset reference)", min_value=0.0, value=0.0, step=1.0)

        values = {"Amount": amount, "Time": time_value, "Log_Amount": float(np.log1p(amount))}
        with st.expander("Advanced features (V1–V28)", expanded=False):
            st.caption("Enter the anonymised PCA values. Defaults are zero and are only placeholders.")
            cols = st.columns(4)
            for i in range(1, 29):
                col_name = f"V{i}"
                with cols[(i - 1) % 4]:
                    values[col_name] = st.number_input(col_name, value=0.0, format="%.6f", key=f"single_{col_name}")

        submitted = st.form_submit_button("Analyse transaction", use_container_width=True)

    if submitted:
        try:
            row = pd.DataFrame([{feature: values.get(feature, 0.0) for feature in feature_names}])
            for col in feature_names:
                if col not in row.columns:
                    raise ValueError(f"Missing required feature: {col}")
            # Apply saved scaler only to the columns used during training.
            if scale_columns and scaler_object is not None:
                row.loc[:, scale_columns] = scaler_object.transform(row[scale_columns])
            probability = float(model.predict_proba(row[feature_names])[:, 1][0])
            prediction = int(probability >= saved_threshold)
            st.markdown("---")
            if prediction == 1:
                st.markdown(
                    f'<div class="danger-box"><h3>⚠️ Potential fraud detected</h3><p>Fraud probability: <b>{probability:.2%}</b></p><p>Threshold: {saved_threshold:.2f}</p></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="success-box"><h3>✓ Transaction classified as genuine</h3><p>Fraud probability: <b>{probability:.2%}</b></p><p>Threshold: {saved_threshold:.2f}</p></div>',
                    unsafe_allow_html=True,
                )
        except Exception as exc:
            st.error(f"Prediction could not be completed: {exc}")

# ------------------------------------------------------------
# Bulk Prediction
# ------------------------------------------------------------
elif page == "Bulk Prediction":
    st.subheader("Bulk transaction screening")
    st.write("Upload a CSV containing the model's input features. `Class` is optional and is not used for prediction.")
    st.caption("Required: Time, V1–V28, Amount. Log_Amount is calculated automatically if absent.")
    uploaded = st.file_uploader("Upload transaction CSV", type=["csv"])
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
            input_frame = raw.drop(columns=["Class"], errors="ignore")
            if "Log_Amount" in feature_names and "Log_Amount" not in input_frame.columns:
                input_frame["Log_Amount"] = np.log1p(pd.to_numeric(input_frame["Amount"], errors="coerce").clip(lower=0))
            missing = [col for col in feature_names if col not in input_frame.columns]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                prepared = input_frame[feature_names].copy()
                for col in prepared.columns:
                    prepared[col] = pd.to_numeric(prepared[col], errors="coerce")
                if prepared.isna().any().any():
                    st.error("CSV contains missing or non-numeric feature values. Please clean the file and upload again.")
                else:
                    if scale_columns and scaler_object is not None:
                        prepared.loc[:, scale_columns] = scaler_object.transform(prepared[scale_columns])
                    probabilities = model.predict_proba(prepared)[:, 1]
                    predictions = (probabilities >= saved_threshold).astype(int)

                    results = raw.copy()
                    results["Fraud_Probability"] = probabilities
                    results["Prediction"] = np.where(predictions == 1, "Fraud", "Genuine")
                    results["Risk_Level"] = np.select(
                        [probabilities >= max(saved_threshold, 0.80), probabilities >= saved_threshold],
                        ["High", "Review"],
                        default="Low",
                    )

                    total_rows = len(results)
                    fraud_rows = int((predictions == 1).sum())
                    x1, x2, x3 = st.columns(3)
                    with x1: metric_card("Rows analysed", f"{total_rows:,}")
                    with x2: metric_card("Potential fraud", f"{fraud_rows:,}")
                    with x3: metric_card("Genuine", f"{total_rows - fraud_rows:,}")

                    st.markdown("### Prediction results")
                    st.dataframe(results.head(500), use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download all predictions as CSV",
                        data=results.to_csv(index=False).encode("utf-8"),
                        file_name="credit_fraud_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                    chart_data = results["Prediction"].value_counts().rename_axis("Type").reset_index(name="Count")
                    fig = px.pie(
                        chart_data, names="Type", values="Count", hole=0.62,
                        color="Type",
                        color_discrete_map={"Genuine": "#1dbb91", "Fraud": "#ff5574"},
                        template="plotly_dark",
                    )
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=20, b=10))
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not process this CSV: {exc}")


# ------------------------------------------------------------
# PhonePe Test Mode
# ------------------------------------------------------------
elif page == "PhonePe Test Mode":
    st.subheader("PhonePe Test Mode")
    st.write(
        "Upload your privacy-safe PhonePe CSV to analyse transaction patterns. "
        "This page is separate from the Credit Card Random Forest model."
    )
    st.warning(
        "The credit-card model cannot validly classify PhonePe transactions: "
        "it requires Time, V1–V28 and Amount features, while PhonePe statements "
        "do not contain the anonymised PCA features V1–V28."
    )

    phonepe_file = st.file_uploader(
        "Upload PhonePe transaction CSV",
        type=["csv"],
        key="phonepe_test_mode_upload"
    )

    if phonepe_file is not None:
        try:
            phonepe = pd.read_csv(phonepe_file)
            required = {"date", "time", "transaction_category", "type", "amount_inr"}
            missing = sorted(required - set(phonepe.columns))

            if missing:
                st.error("Missing PhonePe CSV columns: " + ", ".join(missing))
            else:
                phonepe["date"] = pd.to_datetime(phonepe["date"], errors="coerce")
                phonepe["amount_inr"] = pd.to_numeric(phonepe["amount_inr"], errors="coerce")
                phonepe["type"] = phonepe["type"].astype(str).str.upper().str.strip()
                phonepe = phonepe.dropna(subset=["date", "amount_inr"])
                phonepe = phonepe[phonepe["type"].isin(["DEBIT", "CREDIT"])].copy()

                if phonepe.empty:
                    st.info("No valid PhonePe transactions found in this CSV.")
                else:
                    total_txns = len(phonepe)
                    debit = phonepe[phonepe["type"] == "DEBIT"]
                    credit = phonepe[phonepe["type"] == "CREDIT"]

                    k1, k2, k3, k4 = st.columns(4)
                    with k1:
                        metric_card("Total transactions", f"{total_txns:,}")
                    with k2:
                        metric_card("Total money sent", f"₹{debit['amount_inr'].sum():,.2f}")
                    with k3:
                        metric_card("Total money received", f"₹{credit['amount_inr'].sum():,.2f}")
                    with k4:
                        metric_card("Average transaction", f"₹{phonepe['amount_inr'].mean():,.2f}")

                    st.markdown("### Transaction overview")
                    left, right = st.columns(2)

                    with left:
                        counts = (
                            phonepe["type"].value_counts()
                            .rename_axis("Transaction type")
                            .reset_index(name="Count")
                        )
                        fig = px.pie(
                            counts, names="Transaction type", values="Count",
                            hole=0.58, template="plotly_dark",
                            color="Transaction type",
                            color_discrete_map={"DEBIT": "#ff5574", "CREDIT": "#1dbb91"}
                        )
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            margin=dict(t=15, b=10)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with right:
                        fig = px.histogram(
                            phonepe, x="amount_inr", color="type", nbins=20,
                            template="plotly_dark",
                            labels={"amount_inr": "Amount (₹)", "type": "Transaction type"}
                        )
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(t=15, b=10)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Simple amount outlier review using the IQR rule.
                    q1 = phonepe["amount_inr"].quantile(0.25)
                    q3 = phonepe["amount_inr"].quantile(0.75)
                    iqr = q3 - q1
                    upper_bound = q3 + 1.5 * iqr
                    phonepe["Review_Flag"] = np.where(
                        phonepe["amount_inr"] > upper_bound,
                        "Review: unusually high amount",
                        "No amount outlier"
                    )

                    flagged = phonepe[phonepe["amount_inr"] > upper_bound]
                    st.markdown("### Transactions for manual review")
                    st.caption(
                        f"IQR amount boundary: ₹{upper_bound:,.2f}. "
                        "This identifies statistical outliers only; it does not prove fraud."
                    )
                    if flagged.empty:
                        st.success("No unusually high amount outliers found by this rule.")
                    else:
                        st.warning(f"{len(flagged)} transaction(s) are above the IQR boundary.")
                        st.dataframe(
                            flagged[[
                                "date", "time", "transaction_category",
                                "type", "amount_inr", "Review_Flag"
                            ]],
                            use_container_width=True,
                            hide_index=True
                        )

                    st.markdown("### All PhonePe transactions")
                    st.dataframe(
                        phonepe[[
                            "date", "time", "transaction_category", "type", "amount_inr"
                        ]],
                        use_container_width=True,
                        hide_index=True
                    )

                    st.download_button(
                        "Download analysed PhonePe CSV",
                        data=phonepe.to_csv(index=False).encode("utf-8"),
                        file_name="phonepe_analysis_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        except Exception as exc:
            st.error(f"Could not analyse PhonePe CSV: {exc}")

    st.markdown("---")
    st.markdown("### Credit-card model compatibility")
    st.caption(
        "No Credit Card Fraud prediction is generated on this page. "
        "The model requires the original training feature schema (Time, V1–V28, Amount, Log_Amount)."
    )

# ------------------------------------------------------------
# Model Performance
# ------------------------------------------------------------
elif page == "Model Performance":
    st.subheader("Model performance")
    st.caption("Held-out test metrics from the completed model evaluation.")
    metric_cols = st.columns(5)
    values = [
        ("Accuracy", "99.95%"),
        ("Precision", "91.03%"),
        ("Recall", "74.74%"),
        ("F1-score", "82.08%"),
        ("PR-AUC", "81.84%"),
    ]
    for col, (label, value) in zip(metric_cols, values):
        with col:
            metric_card(label, value)

    st.markdown("### Classification performance")
    performance = pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1-score", "PR-AUC"],
        "Score": [0.9103, 0.7474, 0.8208, 0.8184],
    })
    fig = px.bar(
        performance, x="Metric", y="Score", text=performance["Score"].map(lambda x: f"{x:.2%}"),
        color="Metric", template="plotly_dark",
    )
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Confusion matrix — held-out test set")
    cm = np.array([[56645, 6], [25, 70]])
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted genuine", "Predicted fraud"],
        y=["Actual genuine", "Actual fraud"],
        text=cm,
        texttemplate="%{text}",
        colorscale=[[0, "#10243a"], [0.5, "#087e78"], [1, "#31d6a3"]],
        showscale=False,
    ))
    fig_cm.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_cm, use_container_width=True)
    st.warning(
        "This is a decision-support demonstration, not a production payment-blocking system. "
        "Review flagged transactions before taking action."
    )

st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#687991;font-size:11px;padding:8px;">Credit Card Fraud Detection • Machine Learning Project • Streamlit</div>',
    unsafe_allow_html=True,
)
