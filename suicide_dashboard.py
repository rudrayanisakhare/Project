import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import re

file_path = r"C:\Users\dell\Downloads\suicide1\suicidal_ideation_reddit_annotated.csv"
df = pd.read_csv(file_path)

TEXT_COL = "usertext"
LABEL_COL = "label"

def categorize_risk(text, label):
    if label == 0:
        return "Non-Suicidal"
    text = str(text).lower()
    if re.search(r"(suicide|kill myself|end my life|die|overdose|hang|cut)", text):
        return "Critical Risk"
    elif re.search(r"(depressed|hopeless|worthless|don[’']t want to live|give up)", text):
        return "High Risk"
    elif re.search(r"(anxious|sad|lonely|struggling|tired|empty|stress)", text):
        return "Moderate Risk"
    else:
        return "Low Risk"

df["risk_level"] = df.apply(lambda row: categorize_risk(row[TEXT_COL], row[LABEL_COL]), axis=1)

st.set_page_config(page_title="Suicidal Ideation Dashboard", layout="wide")
st.title("🧠 Suicidal Ideation Analysis Dashboard")

st.sidebar.header("🔍 Filters")
selected_label = st.sidebar.multiselect(
    "Select Risk Levels",
    options=df["risk_level"].unique(),
    default=df["risk_level"].unique()
)
filtered_df = df[df["risk_level"].isin(selected_label)]

col1, col2, col3 = st.columns(3)
col1.metric("📊 Total Posts", len(df))
col2.metric("⚠️ At-Risk Posts", len(df[df["risk_level"] != "Non-Suicidal"]))
avg_risk = (df["risk_level"] != "Non-Suicidal").mean() * 100
col3.metric("🔥 % At Risk", f"{avg_risk:.1f}%")

st.subheader(" Risk Level Distribution")
risk_counts = filtered_df["risk_level"].value_counts().reset_index()
risk_counts.columns = ["Risk Level", "Count"]

fig_pie = px.pie(
    risk_counts,
    names="Risk Level",
    values="Count",
    color="Risk Level",
    title="Risk Distribution (Pie)",
    hole=0.4
)
st.plotly_chart(fig_pie, use_container_width=True)

fig_bar = px.bar(
    risk_counts,
    x="Risk Level",
    y="Count",
    color="Risk Level",
    title="Risk Distribution (Bar)",
    text_auto=True
)
st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("💬 Keyword Frequency")
text = " ".join(filtered_df[TEXT_COL].astype(str).values)
wc = WordCloud(width=1000, height=500, background_color="white").generate(text)
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

st.subheader("📝 High-Risk & Critical Posts")

if "posts_shown" not in st.session_state:
    st.session_state.posts_shown = 10

high_risk = df[df["risk_level"].isin(["High Risk", "Critical Risk"])].tail(st.session_state.posts_shown)

for _, row in high_risk.iterrows():
    if row["risk_level"] == "Critical Risk":
        st.error(f"🚨 **{row['risk_level']}**\n\n{row[TEXT_COL]}")
    else:
        st.warning(f"⚠️ **{row['risk_level']}**\n\n{row[TEXT_COL]}")

if st.button("Load More Posts"):
    st.session_state.posts_shown += 10
    st.rerun()
