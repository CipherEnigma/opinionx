import streamlit as st
import pickle
import re
import string
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.express as px

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="OpinionX — Sentiment Intelligence",
    page_icon="🔍",
    layout="wide"
)

# ─── Load Models ───────────────────────────────────────────
@st.cache_resource
def load_models():
    base = "outputs/models"
    tfidf = pickle.load(open(f"{base}/tfidf_vectorizer.pkl", "rb"))
    lr = pickle.load(open(f"{base}/logistic_regression.pkl", "rb"))
    svm = pickle.load(open(f"{base}/linear_svm.pkl", "rb"))
    return tfidf, lr, svm


@st.cache_data
def load_results():
    return pd.read_csv("outputs/results.csv")


@st.cache_data
def load_sample_data():
    return pd.read_csv("data/sample/sample_5k.csv")


# ─── Helper Functions ──────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_trust(text):
    words = text.split()
    score = 1.0

    if len(words) < 5:
        score -= 0.35
    elif len(words) < 10:
        score -= 0.20
    elif len(words) < 20:
        score -= 0.10

    caps = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if caps > 0.4:
        score -= 0.15

    uniq = len(set(words)) / max(len(words), 1)
    if uniq < 0.5:
        score -= 0.15

    if text.count("!") > 3:
        score -= 0.10

    return round(max(0.0, min(1.0, score)), 2)

def predict(text, model_name, tfidf, svm, lr):
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])

    if model_name == "Linear SVM":
        pred = svm.predict(vec)[0]
        conf = min(abs(svm.decision_function(vec)[0]) / 3, 1.0)
    else:
        pred = lr.predict(vec)[0]
        conf = float(max(lr.predict_proba(vec)[0]))

    return {
        "sentiment": "Positive" if pred == 1 else "Negative",
        "label": int(pred),
        "confidence": round(conf, 3),
        "trust_score": compute_trust(text),
        "word_count": len(text.split())
    }


# ─── Title ─────────────────────────────────────────────────
st.title("OpinionX — E-Commerce Review Intelligence")
st.markdown("*Sentiment analysis + trust scoring for Amazon Electronics reviews*")
st.divider()

# ─── Load Data ─────────────────────────────────────────────
try:
    tfidf, lr, svm = load_models()
    results_df = load_results()
    sample_df = load_sample_data()

    if "trust_score" not in sample_df.columns:
        sample_df["trust_score"] = sample_df["review_body"].astype(str).apply(compute_trust)

    models_loaded = True

except Exception as e:
    st.error(f"Could not load files: {e}")
    models_loaded = False


# ─── Main App ──────────────────────────────────────────────
if models_loaded:

    tab1, tab2, tab3 = st.tabs([
        "Analyze Review",
        "Model Performance",
        "Dataset Insights"
    ])

    # ======================================================
    # TAB 1
    # ======================================================
    with tab1:

        st.subheader("Paste any product review")

        col1, col2 = st.columns([3, 1])

        with col1:
            review_text = st.text_area(
                "Review text",
                height=160,
                placeholder="These headphones are fabulous..."
            )

        with col2:
            model_choice = st.selectbox(
                "Model",
                ["Linear SVM", "Logistic Regression"]
            )

            st.markdown("**Quick examples:**")

            if st.button("Positive example"):
                review_text = "Absolutely love this product. Amazing quality and battery life."

            if st.button("Negative example"):
                review_text = "Stopped working after one week. Waste of money."

        if st.button("Analyze", type="primary", use_container_width=True):

            if not review_text.strip():
                st.warning("Please enter review text.")
            else:
                result = predict(review_text, model_choice, tfidf, svm, lr)

                c1, c2, c3, c4 = st.columns(4)

                c1.metric("Sentiment", result["sentiment"])
                c2.metric("Confidence", f"{result['confidence']*100:.1f}%")
                c3.metric("Trust Score", f"{result['trust_score']*100:.0f}%")
                c4.metric("Word Count", result["word_count"])

                st.divider()

                col_a, col_b = st.columns(2)

                with col_a:
                    fig1 = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result["confidence"] * 100,
                        title={"text": "Confidence"},
                        gauge={"axis": {"range": [0, 100]}}
                    ))
                    st.plotly_chart(fig1, use_container_width=True)

                with col_b:
                    fig2 = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result["trust_score"] * 100,
                        title={"text": "Trust Score"},
                        gauge={"axis": {"range": [0, 100]}}
                    ))
                    st.plotly_chart(fig2, use_container_width=True)

    # ======================================================
    # TAB 2
    # ======================================================
    with tab2:

        st.subheader("Model Performance")

        st.dataframe(results_df, use_container_width=True)

        fig = px.bar(
            results_df.melt(
                id_vars="Model",
                value_vars=["Accuracy", "Precision", "Recall", "F1"]
            ),
            x="variable",
            y="value",
            color="Model",
            barmode="group",
            title="Performance Comparison"
        )

        fig.update_layout(yaxis_range=[0.85, 1.0])

        st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # TAB 3
    # ======================================================
    with tab3:

        st.subheader("Dataset Insights")

        m1, m2, m3 = st.columns(3)

        m1.metric("Total Reviews", f"{len(sample_df):,}")
        m2.metric("Positive", f"{(sample_df['label']==1).sum():,}")
        m3.metric("Negative", f"{(sample_df['label']==0).sum():,}")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            sent = sample_df["sentiment"].value_counts()

            fig_sent = px.pie(
                values=sent.values,
                names=sent.index,
                title="Sentiment Distribution"
            )

            st.plotly_chart(fig_sent, use_container_width=True)

        with col2:
            fig_trust = px.histogram(
                sample_df,
                x="trust_score",
                nbins=20,
                title="Trust Score Distribution"
            )

            st.plotly_chart(fig_trust, use_container_width=True)

        fig_trust = px.histogram(
            sample_df,
            x="trust_score",
            nbins=10,
            color="sentiment",
            barmode="overlay",
            title="Trust Score Distribution"
        )

        st.plotly_chart(fig_trust, use_container_width=True)