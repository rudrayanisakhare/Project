import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Suicide in India",
    layout="wide"
)

# ---------------------------------------------------------------
# CUSTOM STYLING (Red-Purple-Green Theme)
# ---------------------------------------------------------------
page_bg = """
<style>
body {
    background: linear-gradient(135deg, #FFE6E6 0%, #F0E5FF 100%);
    font-family: 'Arial', sans-serif;
}

/* PAGE TITLES */
.title {
    color: #9B59B6;
    font-size: 45px;
    font-weight: 900;
    text-align: center;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}
.sub {
    color: #BB33AA;
    font-size: 22px;
    text-align: center;
}

/* BOX STYLING */
.box {
    background-color: rgba(255, 255, 255, 0.9);
    padding: 20px;
    border-radius: 15px;
    border: 2px solid #C8B6FF;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

/* DISCLAIMER */
.disclaimer {
    color: #D00000;
    font-size: 16px;
    text-align: center;
    font-weight: bold;
    background-color: rgba(255, 255, 255, 0.8);
    padding: 10px;
    border-radius: 10px;
}

/* METRIC CARDS */
.metric-card {
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    color: white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.metric-red { background-color: #FF4C4C; }
.metric-green { background-color: #28B463; }
.metric-purple { background: linear-gradient(135deg, #9B59B6, #BB33AA); color: white; }
.metric-lavender { background-color: #E6E6FA; color: black; }

/* MOBILE VIEW */
@media (max-width: 768px) {
    .title { font-size: 30px; }
    .metric-card { padding: 10px; }
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---------------------------------------------------------------
# DATA LOADING (Hardcoded for Demo)
# ---------------------------------------------------------------
state_data = pd.DataFrame({
    "State": ["Kerala", "Chhattisgarh", "Telangana", "Tamil Nadu", "Karnataka"],
    "Suicide Rate (per 100k)": [28.5, 28.2, 26.2, 22.2, 21.2]
})

age_data = pd.DataFrame({
    "Age Group": ["18-30", "30-45", "45-60", "60+"],
    "% of Suicides": [34, 32, 19, 8]
})

cause_data = pd.DataFrame({
    "Reason": [
        "Family Problems", "Illness (physical + mental)", "Drug/Alcohol Addiction",
        "Marriage-related Issues", "Love Affairs", "Bankruptcy/Financial Problems",
        "Unemployment", "Others"
    ],
    "Percent": [31.9, 19.0, 7.0, 5.3, 4.7, 3.8, 1.8, 26.5]
})

gender_data = pd.DataFrame({
    "Gender": ["Male", "Female"],
    "Percent": [72.8, 27.2]
})

trend_data = pd.DataFrame({
    "Year": [2020, 2021, 2022, 2023],
    "Total Suicides": [153000, 164000, 171000, 171418]
})

# ---------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------
st.markdown("<h1 class='title'>Suicide in India (NCRB 2023)</h1>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# METRICS SECTION
# ---------------------------------------------------------------
metrics = {
    "Total suicides (2023)": 171418,
    "Change from 2022 (%)": 0.3,
    "National rate (per 100k)": 12.4,
    "Male (%)": 72.8,
    "Female (%)": 27.2
}

col1, col2, col3, col4 = st.columns(4)
col1.markdown(f"<div class='metric-card metric-red'><h3>Total suicides</h3><h2>{metrics['Total suicides (2023)']:,}</h2></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-card metric-green'><h3>Change from 2022</h3><h2>{metrics['Change from 2022 (%)']}%</h2></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-card metric-purple'><h3>Rate (per 100k)</h3><h2>{metrics['National rate (per 100k)']}</h2></div>", unsafe_allow_html=True)
col4.markdown(f"<div class='metric-card metric-lavender'><h3>Male : Female</h3><h2>{metrics['Male (%)']} : {metrics['Female (%)']}</h2></div>", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------
# LAYOUT (Left: charts / Right: tables)
# ---------------------------------------------------------------
left, right = st.columns((2, 1))

# Red-Purple-Green Palette
color_palette = ["#FF4C4C", "#9B59B6", "#28B463", "#FF9999", "#BB33AA", "#58D68D", "#DDA0DD", "#2ECC71"]

# LEFT SIDE VISUALS
with left:
    st.subheader("State-wise suicide rates")
    fig = px.bar(
        state_data, x="State", y="Suicide Rate (per 100k)",
        title="Suicide Rate by State (per 100,000)",
        text="Suicide Rate (per 100k)",
        color="State",
        color_discrete_sequence=color_palette
    )
    fig.update_traces(textposition="inside", textfont_color="black", hovertemplate='%{x}: %{y}')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Suicide Trends Over Time")
    fig = px.line(
        trend_data, x="Year", y="Total Suicides",
        markers=True, title="Total Suicides in India (2020–2023)",
        color_discrete_sequence=["#FF4C4C"]
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Age-wise Distribution")
    fig = px.pie(
        age_data, values="% of Suicides", names="Age Group",
        title="Suicides by Age Group (2023)",
        color_discrete_sequence=color_palette[:4],
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Reasons for Suicide (2023)")
    fig = px.pie(
        cause_data, values="Percent", names="Reason",
        title="Leading Causes (2023)",
        color_discrete_sequence=color_palette,
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)

# RIGHT SIDE VISUALS
with right:
    st.subheader("Gender Distribution")
    fig = px.bar(
        gender_data, x="Gender", y="Percent", text="Percent",
        title="Male vs Female (%)",
        color="Gender",
        color_discrete_sequence=color_palette[:2]
    )
    fig.update_traces(textposition="inside", textfont_color="black", hovertemplate='%{x}: %{y}%')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Data Tables")
    st.markdown("**State-wise Data**")
    st.dataframe(state_data)

    st.markdown("**Age-wise Distribution**")
    st.dataframe(age_data)

    st.markdown("**Causes**")
    st.dataframe(cause_data)

# ---------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------
st.markdown("---")
st.markdown("*Data sourced from NCRB 2023. Visit the official NCRB portal for detailed reports.*")
