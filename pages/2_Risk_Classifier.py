# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Lightweight deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(layout="wide", page_title="Universal Text Risk Classifier (DL1)")
st.title("Risk Classifier")

# Optional default local paths (if you want to test quickly)
DEFAULT_LOCAL_PATHS = [
    "/mnt/data/ed9c10e4-c738-41a1-b508-fb51ceec4374.png",
    "/mnt/data/99d2c679-1558-417b-a51f-5c71524c2db3.png"
]
# Replace above with a real CSV path for local quick test if desired.

# ---------------- Helpers ----------------
def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"http\S+|www\S+", " ", s)
    s = re.sub(r"@\w+", " ", s)
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Very broad regex-based risk mapper (captures many variants)
def categorize_risk(text):
    if not isinstance(text, str):
        return "Low Risk"
    t = text.lower()

    critical = r"""(suicide|kill myself|end my life|take my life|commit suicide|suicidal|goodbye forever|
                   jump off|hang myself|slit my wrists|cut deep|overdose|take pills|pill overdose|
                   i will die|i want to die|i will kill myself|kill me|die tonight|die today|
                   shoot myself|gun to my head|bridge jump|train tracks|crash my car on purpose)"""

    high = r"""(want to die|wish i was dead|want to end|thinking of ending|thinking about ending|
                self harm|self-harm|hurt myself|cutting|cut myself|burn myself|overdosing|
                i can't go on|can't go on|give up on life|tired of living|i may kill myself|
                want to disappear|want to disappear|wish to die|kill me please|kill me)"""

    moderate = r"""(depress|depression|anxiety|panic attack|hopeless|worthless|overwhelmed|
                    i'm done|im tired|i'm tired|broken inside|emotional pain|trauma|
                    suicidal thoughts|dark thoughts|not okay|sad life|feeling empty|hurt)"""

    low = r"""(sad|stress|stressed|lonely|bad day|upset|down|tired|frustrated|angry|crying|confused)"""

    if re.search(critical, t, re.IGNORECASE | re.VERBOSE):
        return "Critical Risk"
    if re.search(high, t, re.IGNORECASE | re.VERBOSE):
        return "High Risk"
    if re.search(moderate, t, re.IGNORECASE | re.VERBOSE):
        return "Moderate Risk"
    if re.search(low, t, re.IGNORECASE | re.VERBOSE):
        return "Low Risk"
    return "Low Risk"

# probability -> risk label (used when we have probability)
def prob_to_risk_label(p):
    if p >= 0.80:
        return "Critical Risk"
    if p >= 0.60:
        return "High Risk"
    if p >= 0.40:
        return "Moderate Risk"
    return "Low Risk"

# ---------------- Input ----------------
st.sidebar.header("Input / Options")
use_local = st.sidebar.checkbox("Use default local file (if available)", value=False)
local_choice = None
if use_local:
    local_choice = st.sidebar.selectbox("Pick a default path (replace with real CSV path for testing)", DEFAULT_LOCAL_PATHS)

uploaded = st.file_uploader("Upload CSV file (text-heavy). Labeled or unlabeled - both supported.", type=["csv"])
data_path = None
if use_local and local_choice and os.path.exists(local_choice):
    data_path = local_choice
elif uploaded is not None:
    data_path = uploaded

if data_path is None:
    st.info("Upload a CSV to begin, or enable local file option (with a real CSV path).")
    st.stop()

# load dataframe
try:
    if isinstance(data_path, str):
        df_raw = pd.read_csv(data_path)
    else:
        df_raw = pd.read_csv(data_path)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

st.write("### Data preview")
st.dataframe(df_raw.head(50))

# ---------------- Detect text column ----------------
obj_cols = df_raw.select_dtypes(include=["object"]).columns.tolist()
if len(obj_cols) == 0:
    st.error("No object/text columns detected. This tool expects text data.")
    st.stop()

avg_lens = {c: df_raw[c].astype(str).map(len).mean() for c in obj_cols}
default_text_col = max(avg_lens.items(), key=lambda x: x[1])[0]

st.sidebar.markdown("**Detected text column** (largest average length)")
text_col = st.sidebar.selectbox("Text column (override)", options=[default_text_col] + obj_cols, index=0)

st.write(f"Using text column: **{text_col}**")

# detect existing label column (optional)
def detect_label_column(df):
    keywords = ['label','target','class','category','risk','y','sentiment']
    for c in df.columns:
        if str(c).lower() in keywords:
            return c
    # heuristic: object column with small unique ratio
    ratios = {c: df[c].nunique()/max(1, len(df)) for c in df.columns}
    cand = [c for c,r in ratios.items() if r <= 0.2 and df[c].dtype == 'object']
    return cand[0] if cand else None

detected_label = detect_label_column(df_raw)
label_col = st.sidebar.selectbox("Detected label column (override). Choose None for unlabeled", [None] + list(df_raw.columns), index=0 if detected_label is None else list(df_raw.columns).index(detected_label)+1)
if label_col is None:
    st.write("No user label selected — will use regex to create unified labels.")
else:
    st.write(f"User label column selected: **{label_col}**")

# ---------------- Build unified risk labels ----------------
df = df_raw.copy()
df["_text_clean"] = df[text_col].astype(str).map(clean_text)

# If user provided label column, attempt to map it to our four buckets using regex on label or mapping dictionary
# Broad mapping dictionary (can be extended)
label_map = {
    "suicidal": "Critical Risk", "attempt": "Critical Risk", "suicide": "Critical Risk",
    "ideation": "High Risk", "self-harm": "High Risk", "self harm": "High Risk",
    "severe": "High Risk", "moderate": "Moderate Risk", "mild": "Low Risk",
    "no risk": "Low Risk", "none": "Low Risk", "normal": "Low Risk", "support": "Low Risk"
}

def map_existing_label(lbl, text):
    if pd.isna(lbl):
        return categorize_risk(text)
    s = str(lbl).lower()
    # direct mapping
    for k,v in label_map.items():
        if k in s:
            return v
    # fallback to regex on label string
    r = categorize_risk(s)
    if r:
        return r
    # fallback to regex on text
    return categorize_risk(text)

if label_col and label_col in df.columns:
    df["unified_label"] = df.apply(lambda r: map_existing_label(r[label_col], r["_text_clean"]), axis=1)
else:
    # no user labels; use text regex to create labels
    df["unified_label"] = df["_text_clean"].map(categorize_risk)

st.write("### Unified label counts (Critical/High/Moderate/Low)")
st.write(df["unified_label"].value_counts())

# ---------------- Features: TF-IDF for classical models ----------------
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_tfidf = tfidf.fit_transform(df["_text_clean"].fillna(""))

label_encoder = LabelEncoder()
y_all = label_encoder.fit_transform(df["unified_label"].astype(str))

# Split
test_size = st.sidebar.slider("Test set proportion", 0.1, 0.5, 0.25)
if len(np.unique(y_all)) == 1:
    st.warning("Only one unified_label present — metrics will be trivial.")

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_tfidf, y_all, df.index, test_size=test_size, random_state=42,
    stratify=y_all if len(np.unique(y_all)) > 1 else None
)

# ---------------- Train classical models ----------------
st.write("## Training classical models (LogisticRegression, RandomForest, SVM, KNN)")
classical_models = {
    "LR": LogisticRegression(max_iter=400),
    "RF": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier()
}

metrics = {}
preds_full = {}
probs_full = {}

for name, model in classical_models.items():
    st.write(f"Training {name}...")
    try:
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)

        # probabilities for full set if supported
        if hasattr(model, "predict_proba"):
            full_probs = model.predict_proba(X_tfidf)
            probs_full[name] = np.max(full_probs, axis=1)
        else:
            probs_full[name] = np.zeros(X_tfidf.shape[0])

        acc = accuracy_score(y_test, y_pred_test)
        prec = precision_score(y_test, y_pred_test, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred_test, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred_test, average="macro", zero_division=0)
        metrics[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}

        # full predictions (decoded)
        full_pred = model.predict(X_tfidf)
        preds_full[name] = label_encoder.inverse_transform(full_pred)
        st.success(f"{name} done — Acc={acc:.3f}, F1={f1:.3f}")
    except Exception as e:
        st.error(f"{name} failed: {e}")
        metrics[name] = {"Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1": 0.0}
        preds_full[name] = np.array([None] * df.shape[0])
        probs_full[name] = np.zeros(df.shape[0])

# ---------------- Deep learning (LSTM & CNN - DL1) ----------------
st.write("## Training DL models (LSTM & CNN)")
MAX_WORDS = 20000
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df["_text_clean"].tolist())
sequences = tokenizer.texts_to_sequences(df["_text_clean"].tolist())
MAXLEN = min(300, max(len(s) for s in sequences)) if sequences else 100
padded_all = pad_sequences(sequences, maxlen=MAXLEN, padding="post", truncating="post")

# split padded
padded_train = padded_all[idx_train]
padded_test = padded_all[idx_test]

n_classes = len(label_encoder.classes_)
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=n_classes)

EMBED_DIM = 64

def build_lstm(max_words, embed_dim, maxlen, n_classes):
    model = Sequential([
        Embedding(input_dim=min(max_words, len(tokenizer.word_index)+1), output_dim=embed_dim, input_length=maxlen),
        LSTM(64),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn(max_words, embed_dim, maxlen, n_classes):
    model = Sequential([
        Embedding(input_dim=min(max_words, len(tokenizer.word_index)+1), output_dim=embed_dim, input_length=maxlen),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# LSTM
try:
    lstm = build_lstm(MAX_WORDS, EMBED_DIM, MAXLEN, n_classes)
    st.write("Training LSTM (few epochs)...")
    lstm.fit(padded_train, y_train_cat, validation_split=0.1, epochs=3, batch_size=32, callbacks=[EarlyStopping(monitor='val_loss', patience=2)], verbose=0)
    lstm_test_preds = np.argmax(lstm.predict(padded_test), axis=1)
    acc = accuracy_score(y_test, lstm_test_preds)
    prec = precision_score(y_test, lstm_test_preds, average='macro', zero_division=0)
    rec = recall_score(y_test, lstm_test_preds, average='macro', zero_division=0)
    f1 = f1_score(y_test, lstm_test_preds, average='macro', zero_division=0)
    metrics['LSTM'] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
    lstm_proba_full = lstm.predict(padded_all)
    preds_full['LSTM'] = label_encoder.inverse_transform(np.argmax(lstm_proba_full, axis=1))
    probs_full['LSTM'] = np.max(lstm_proba_full, axis=1)
    st.success(f"LSTM done — Acc={acc:.3f}, F1={f1:.3f}")
except Exception as e:
    st.error(f"LSTM failed: {e}")
    metrics['LSTM'] = {"Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1": 0.0}
    preds_full['LSTM'] = np.array([None]*df.shape[0])
    probs_full['LSTM'] = np.zeros(df.shape[0])

# CNN
try:
    cnn = build_cnn(MAX_WORDS, EMBED_DIM, MAXLEN, n_classes)
    st.write("Training CNN (few epochs)...")
    cnn.fit(padded_train, y_train_cat, validation_split=0.1, epochs=3, batch_size=32, callbacks=[EarlyStopping(monitor='val_loss', patience=2)], verbose=0)
    cnn_test_preds = np.argmax(cnn.predict(padded_test), axis=1)
    acc = accuracy_score(y_test, cnn_test_preds)
    prec = precision_score(y_test, cnn_test_preds, average='macro', zero_division=0)
    rec = recall_score(y_test, cnn_test_preds, average='macro', zero_division=0)
    f1 = f1_score(y_test, cnn_test_preds, average='macro', zero_division=0)
    metrics['CNN'] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
    cnn_proba_full = cnn.predict(padded_all)
    preds_full['CNN'] = label_encoder.inverse_transform(np.argmax(cnn_proba_full, axis=1))
    probs_full['CNN'] = np.max(cnn_proba_full, axis=1)
    st.success(f"CNN done — Acc={acc:.3f}, F1={f1:.3f}")
except Exception as e:
    st.error(f"CNN failed: {e}")
    metrics['CNN'] = {"Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1": 0.0}
    preds_full['CNN'] = np.array([None]*df.shape[0])
    probs_full['CNN'] = np.zeros(df.shape[0])

# ------------------ Metrics table ------------------
metrics_df = pd.DataFrame(metrics).T.fillna(0)[["Accuracy","Precision","Recall","F1"]]
st.write("### Model comparison (Accuracy / Precision / Recall / F1)")
st.dataframe(metrics_df.style.format("{:.3f}"))

# choose best model by F1 (macro)
best_model = metrics_df["F1"].idxmax()
st.success(f"Selected best model (by F1 macro): {best_model}")

# ------------------ Build final output CSV ------------------
out = df_raw.copy()
out[text_col] = out[text_col].astype(str)
out["unified_label"] = df["unified_label"]
out["true_label"] = df.get(label_col, "").astype(str) if label_col else ""

# attach per-model cols
for m in ['LR','RF','SVM','KNN','LSTM','CNN']:
    out[m] = preds_full.get(m, np.array([""] * len(out)))

out["final_result"] = out[best_model].astype(str)

# Risk_Level based on best model probabilities if available
best_probs = probs_full.get(best_model, np.zeros(len(out)))
out["best_prob"] = best_probs
out["Risk_Level"] = out["best_prob"].apply(lambda p: prob_to_risk_label(p) if pd.notna(p) and p>0 else None)

# fallback mapping from final_result string if needed
def fallback(lbl):
    if not isinstance(lbl, str): return "Moderate Risk"
    s = lbl.lower()
    if any(k in s for k in ["crit","attempt","kill","suicid","die"]): return "Critical Risk"
    if any(k in s for k in ["ideat","want to die","hurt myself","self-harm","cut"]): return "High Risk"
    if any(k in s for k in ["depress","anxiety","sad","hopeless","lonely"]): return "Moderate Risk"
    return "Low Risk"

out["Risk_Level"] = out.apply(lambda r: r["Risk_Level"] if pd.notna(r["Risk_Level"]) and r["Risk_Level"]!="" else fallback(r["final_result"]), axis=1)
out["Risk_Level"] = out["Risk_Level"].astype(str) + ""

if "Risk_Level" in out.columns:
    dist = out["Risk_Level"].value_counts()
    fig, ax = plt.subplots(figsize=(1.5, 1.5))  # very small figure
    ax.pie(dist.values, labels=dist.index, autopct="%1.0f%%", textprops={"fontsize":5})
    ax.set_title("Risk Level Distribution", fontsize=7)
    st.pyplot(fig)
else:
    st.warning("Risk_Level not available")

# Reorder final columns
cols_final = [text_col, "true_label", "unified_label", "LR","RF","SVM","KNN","LSTM","CNN","final_result","Risk_Level"]
cols_final = [c for c in cols_final if c in out.columns]
out_final = out[cols_final]

st.write("### Sample final predictions")
st.dataframe(out_final.head(200))

# Downloads
csv_all = out_final.to_csv(index=False).encode("utf-8")
st.download_button("Download FULL predictions CSV", csv_all, "predictions_full.csv", "text/csv")

mask = out_final["Risk_Level"].str.contains("Critical") | out_final["Risk_Level"].str.contains("High")
out_ch = out_final[mask]
csv_ch = out_ch.to_csv(index=False).encode("utf-8")
st.download_button("Download Critical+High CSV", csv_ch, "critical_high_posts.csv", "text/csv")

st.success("Processing complete!")