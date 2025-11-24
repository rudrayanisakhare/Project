import streamlit as st

# Custom CSS for red-blue themed cards
st.markdown("""
<style>
.card {
    background: linear-gradient(135deg, #ff4d4d, #4d79ff); /* red to blue gradient */
    color: white;
    border-radius: 12px;
    padding: 20px;
    margin: 10px;
    box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    transition: transform 0.2s, box-shadow 0.2s;
}
.card:hover {
    transform: scale(1.03);
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
}
.card h3 {
    margin-bottom: 8px;
    font-size: 20px;
}
.card p {
    margin: 4px 0;
    font-size: 14px;
}
.card a {
    color: #ffe066;
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

# Data for Indian resources
helplines = [
    {"name": "AASRA", "description": "24/7 confidential support for emotional distress and suicide prevention in Mumbai.", "contact": "9820466726", "website": "https://aasra.info/"},
    {"name": "Vandrevala Foundation", "description": "24/7 helpline for mental health, depression, and suicide prevention across India.", "contact": "1860-266-2345", "website": "https://www.vandrevalafoundation.com/"},
]

ngos = [
    {"name": "Mind India", "description": "NGO providing mental health awareness, counseling, and support programs.", "contact": "info@mindindia.org", "website": "https://mindindia.org/"},
    {"name": "Tata Institute of Social Sciences (TISS)", "description": "Offers mental health research, counseling, and community programs.", "contact": "022-25525000", "website": "https://www.tiss.edu/"},
]

government_resources = [
    {"name": "Ministry of Health and Family Welfare (MoHFW)", "description": "Government portal for mental health policies, schemes, and national programs.", "contact": "1800-11-0031", "website": "https://www.mohfw.gov.in/"},
    {"name": "NIMHANS", "description": "Premier government institute for mental health research, treatment, and training in Bengaluru.", "contact": "080-26995001", "website": "https://nimhans.ac.in/"},
]

# Streamlit UI
st.title("Mental Health Resources")
st.write("**Disclaimer:** This page provides information on helplines, NGOs, and government resources for mental health support in India. It is not a substitute for professional medical advice. If you're in crisis, contact a local emergency service (e.g., 112) or helpline immediately.")

# Function to display cards
def display_cards(resources, title):
    st.subheader(title)
    cols = st.columns(2)
    for i, resource in enumerate(resources):
        with cols[i % 2]:
            with st.container():
                st.markdown(f"""
                <div class="card">
                    <h3>{resource['name']}</h3>
                    <p><strong>Description:</strong> {resource['description']}</p>
                    <p><strong>Contact:</strong> {resource['contact']}</p>
                    <p><a href="{resource['website']}" target="_blank">Visit Website</a></p>
                </div>
                """, unsafe_allow_html=True)

# Display sections
display_cards(helplines, "📞 Helplines")
display_cards(ngos, "🤝 NGOs and Organizations")
display_cards(government_resources, "🏛️ Government Resources")
