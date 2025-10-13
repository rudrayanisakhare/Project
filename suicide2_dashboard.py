# streamlit_suicide_detection_app_auto.py
"""
Dataset-agnostic Streamlit app for Suicidal Ideation Detection (TF-IDF + Logistic Regression)
- Automatically detects text column
- No NLTK dependency
- Save file and run: streamlit run streamlit_suicide_detection_app_auto.py
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.exceptions import NotFittedError

# -------------------------
# Configuration / constants
# -------------------------
TEXT_COL = "usertext"  # default, will auto-detect if missing
LABEL_COL = "label"
STOPWORDS_SET = set(STOPWORDS)

MODEL_DIR = "models"
VECT_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "logreg_grid.joblib")
os.makedirs(MODEL_DIR, exist_ok=True)


# -------------------------
# Helper functions
# -------------------------
def categorize_risk(text, label=None):
    """Simple regex-derived risk labels if no label column exists."""
    if label == 0:
        return "Non-Suicidal"
    if text is None:
        text = ""
    t = str(text).lower()
    if re.search(r"\b(suicide|kill myself|end my life|die by|die|overdose|hang|cut)\b", t):
        return "Critical Risk"
    if re.search(r"\b(depressed|hopeless|worthless|don(?:'t|’t) want to live|give up)\b", t):
        return "High Risk"
    if re.search(r"\b(anxious|sad|lonely|struggling|tired|empty|stress)\b", t):
        return "Moderate Risk"
    return "Low Risk"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    text = re.sub(r"#[A-Za-z0-9_]+", "", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_text_column(df):
    """Detect text column automatically (highest avg string length)."""
    string_cols = df.select_dtypes(include=["object"]).columns
    if len(string_cols) == 0:
        raise ValueError("No string columns found in CSV.")
    col_lengths = df[string_cols].fillna("").applymap(lambda x: len(str(x)))
    avg_lengths = col_lengths.mean()
    text_col = avg_lengths.idxmax()
    return text_col


def prepare_dataframe(df_raw):
    df = df_raw.copy()

    global TEXT_COL
    if TEXT_COL not in df.columns:
        TEXT_COL = detect_text_column(df)
        st.warning(f"No 'usertext' column found. Using '{TEXT_COL}' as text column.")

    df[TEXT_COL] = df[TEXT_COL].astype(str).fillna("")

    # derive risk_level
    if "risk_level" not in df.columns:
        if LABEL_COL in df.columns:
            df["risk_level"] = df.apply(lambda r: categorize_risk(r[TEXT_COL], r.get(LABEL_COL, None)), axis=1)
        else:
            df["risk_level"] = df[TEXT_COL].apply(lambda x: categorize_risk(x, None))

    df["clean_text"] = df[TEXT_COL].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    return df


def build_vectorizer(texts, max_features=5000, ngram_range=(1, 2)):
    vec = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=ngram_range)
    vec.fit(texts)
    return vec


def train_grid(X_train, y_train):
    lr = LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear')
    param_grid = {"C": [0.1, 1, 10]}
    grid = GridSearchCV(lr, param_grid, cv=3, scoring="f1_weighted", n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Suicidal Ideation Detection (Auto)", layout="wide")
st.title("Suicidal Ideation Detection — Auto Text Column Detection")

with st.sidebar:
    st.header("Options / Upload")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    max_features = st.slider("TF-IDF max_features", min_value=1000, max_value=20000, value=5000, step=500)
    ngram_min = st.selectbox("TF-IDF ngram min", [1, 2], index=0)
    ngram_max = st.selectbox("TF-IDF ngram max", [1, 2], index=1)
    test_size = st.slider("Test set proportion", 0.05, 0.5, 0.2, step=0.05)
    retrain_button = st.button("Train / Retrain model")
    st.markdown("---")
    st.markdown("This version automatically detects the text column. Labels are derived if missing.")

if uploaded_file is None:
    st.info("Please upload a CSV file to proceed.")
    st.stop()

# Load CSV
try:
    df_raw = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

# Prepare data
try:
    df = prepare_dataframe(df_raw)
except Exception as e:
    st.error(f"Data preparation failed: {e}")
    st.stop()

st.subheader("Dataset preview")
st.dataframe(df.head())

# Risk distribution
st.subheader("Risk-level distribution")
counts = df["risk_level"].value_counts()
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(counts.index, counts.values)
ax.set_ylabel("Count")
ax.set_xlabel("Risk Level")
plt.xticks(rotation=20)
st.pyplot(fig)

# Wordcloud
st.subheader("Word Cloud (corpus)")
corpus_text = " ".join(df["clean_text"].values)
wc = WordCloud(width=1000, height=400, background_color="white", stopwords=STOPWORDS_SET).generate(corpus_text)
fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
ax_wc.imshow(wc, interpolation="bilinear")
ax_wc.axis("off")
st.pyplot(fig_wc)

# Vectorizer
st.subheader("Model: TF-IDF + Logistic Regression")
try:
    vectorizer = None
    if os.path.exists(VECT_PATH) and not retrain_button:
        vectorizer = joblib.load(VECT_PATH)
        st.info("Loaded cached vectorizer.")
    else:
        vectorizer = build_vectorizer(df["clean_text"], max_features=max_features, ngram_range=(ngram_min, ngram_max))
        joblib.dump(vectorizer, VECT_PATH)
        st.success("Built and cached vectorizer.")
except Exception as e:
    st.error(f"Vectorizer error: {e}")
    st.stop()

# Prepare X and y
label_order = sorted(df["risk_level"].unique())
label_map = {label: idx for idx, label in enumerate(label_order)}
inv_label_map = {v: k for k, v in label_map.items()}
y = df["risk_level"].map(label_map)
X = vectorizer.transform(df["clean_text"])

# Split & train
if len(np.unique(y)) == 1:
    st.warning("Only one class present. Model will use trivial predictor.")
    trivial_class_idx = int(y.iloc[0])
    model = None
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    model = None
    if os.path.exists(MODEL_PATH) and not retrain_button:
        try:
            model = joblib.load(MODEL_PATH)
            st.success("Loaded cached model.")
        except Exception:
            st.warning("Cached model exists but failed to load; will retrain.")

    if model is None:
        with st.spinner("Training model..."):
            try:
                model = train_grid(X_train, y_train)
            except Exception as e:
                st.error(f"Training failed: {e}")
                st.stop()
            joblib.dump(model, MODEL_PATH)
            st.success(f"Training complete (best params: {model.best_params_})")

# Evaluation
if model is not None and len(np.unique(y)) > 1:
    st.subheader("Evaluation on test set")
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=label_order, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        im = ax_cm.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax_cm.set_xticks(np.arange(len(label_order)))
        ax_cm.set_yticks(np.arange(len(label_order)))
        ax_cm.set_xticklabels(label_order, rotation=45, ha="right")
        ax_cm.set_yticklabels(label_order)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                           color="white" if cm[i, j] > cm.max() / 2 else "black")
        st.pyplot(fig_cm)
    except Exception as e:
        st.warning(f"Could not evaluate model: {e}")

# Top features
st.subheader("Top features per class")
try:
    if model is not None and hasattr(model, "best_estimator_"):
        clf = model.best_estimator_
    elif model is not None:
        clf = model
    else:
        clf = None

    if clf is not None and hasattr(clf, "coef_"):
        try:
            feature_names = vectorizer.get_feature_names_out()
        except Exception:
            feature_names = vectorizer.get_feature_names()
        coefs = clf.coef_
        topn = 10
        for class_idx in range(coefs.shape[0]):
            class_name = inv_label_map.get(class_idx, str(class_idx))
            st.markdown(f"**{class_name}**")
            idxs = np.argsort(coefs[class_idx])[-topn:][::-1]
            top_feats = [(feature_names[i], float(coefs[class_idx, i])) for i in idxs]
            df_top = pd.DataFrame(top_feats, columns=["feature", "coef"])
            st.table(df_top)
    else:
        st.info("No linear model coefficients available.")
except Exception as e:
    st.warning(f"Could not compute top features: {e}")

# Predict user input
st.subheader("Predict risk level for input text")
user_input = st.text_area("Paste a post/comment here:", height=150)
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Enter some text first.")
    else:
        text_clean = clean_text(user_input)
        x_vec = vectorizer.transform([text_clean])
        if len(np.unique(y)) == 1:
            pred_label = inv_label_map[trivial_class_idx]
            st.info(f"Only one class — predicted: **{pred_label}**")
        else:
            try:
                pred_idx = int(model.predict(x_vec)[0])
                pred_label = inv_label_map[pred_idx]
                st.success(f"Predicted risk level: **{pred_label}**")
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(x_vec)[0]
                    prob_df = pd.DataFrame({"class": [inv_label_map[i] for i in range(len(probs))], "prob": probs})
                    st.table(prob_df.sort_values("prob", ascending=False))
            except NotFittedError:
                st.error("Model not fitted. Please retrain.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# High-risk posts feed
st.subheader("High & Critical Risk Posts (recent)")
if "posts_shown" not in st.session_state:
    st.session_state.posts_shown = 10
high_df = df[df["risk_level"].isin(["High Risk", "Critical Risk"])]
for _, row in high_df.tail(st.session_state.posts_shown).iterrows():
    if row["risk_level"] == "Critical Risk":
        st.error(f"🚨 **{row['risk_level']}**\n\n{row[TEXT_COL]}")
    else:
        st.warning(f"⚠️ **{row['risk_level']}**\n\n{row[TEXT_COL]}")

if st.button("Load More Posts"):
    st.session_state.posts_shown += 10
    st.experimental_rerun()

st.markdown("---")
st.info("TF-IDF + Logistic Regression demo. Upgrade to transformer models for higher accuracy.")
