import streamlit as st

# Custom CSS for red-blue-green theme
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #2196f3, #4caf50);  /* Blue to green gradient */
    font-family: 'Arial', sans-serif;
    color: white;
}
.header {
    text-align: center;
    background: linear-gradient(90deg, #2196f3, #f44336);  /* Blue to red gradient */
    color: white;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}
.conversation-area {
    margin-bottom: 20px;
}
.user-text {
    color: #b19cd9;  /* Dark lavender for user text */
    font-style: italic;
    margin-bottom: 10px;
}
.bot-text {
    color: #2196f3;  /* Blue for bot text */
    font-weight: bold;
    margin-bottom: 10px;
}
.input-section {
    margin-bottom: 20px;
}
.button {
    background: linear-gradient(90deg, #f44336, #2196f3);  /* Red to blue gradient */
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
    transition: transform 0.2s;
}
.button:hover {
    transform: scale(1.05);
}
.disclaimer {
    background: #fff3cd;
    border: 1px solid #f44336;  /* Red border */
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 20px;
    color: black;  /* Black font for disclaimer */
}
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Function to generate friendly bot responses
def get_response(user_input):
    user_input = user_input.lower()
    if any(x in user_input for x in ['sad', 'depressed', 'down', 'unhappy']):
        return "Hey there, friend! 😔 I'm here for you. Let's chat about something that lifts your mood!"
    elif any(x in user_input for x in ['anxious', 'worried', 'stress', 'panic', 'nervous']):
        return "Take a deep breath 🌬️. You're doing great by reaching out. Let's talk about it calmly."
    elif any(x in user_input for x in ['happy', 'good', 'joy', 'excited', 'great']):
        return "Yay! 😊 That's awesome to hear! What's the best part of your day?"
    elif any(x in user_input for x in ['help', 'suicide', 'crisis', 'end it']):
        return "You matter! 🚨 Reach out immediately to a helpline: AASRA (9820466726) or Vandrevala (1860-266-2345)."
    else:
        return "Hi there! 💬 Thanks for sharing. I'm here to chat about anything. What's on your mind?"

# Header
st.markdown("""
<div class="header">
    <h1>Mental Health Chatbot</h1>
</div>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <strong>Disclaimer:</strong> This is a general support tool. Not a substitute for professional advice. If in crisis, contact a qualified professional or hotline immediately.
</div>
""", unsafe_allow_html=True)

# Conversation Area
st.subheader("💬 Let's Chat!")
if not st.session_state.messages:
    st.write("No messages yet. Start chatting below! 😊")
else:
    for message in st.session_state.messages:
        role = message.get('role', 'Unknown')
        content = message.get('content', '')
        if role == 'You':
            st.markdown(f'<div class="user-text"><strong>You:</strong> {content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-text"><strong>Bot:</strong> {content}</div>', unsafe_allow_html=True)

# Input Section
st.subheader("✍️ Share Your Thoughts")
user_input = st.text_input("Type your message here:", key="input", placeholder="How are you feeling today?", label_visibility="collapsed")

if st.button("Send Message", key="send"):
    if user_input.strip():
        st.session_state.messages.append({"role": "You", "content": user_input})
        response = get_response(user_input)
        st.session_state.messages.append({"role": "Bot", "content": response})
        st.rerun()
    else:
        st.warning("Please enter a message.")

# Footer
st.write("---")
st.write("🌟 For more resources, visit our [Mental Health Resources Hub for India](https://your-link-here) or check out [AASRA](https://aasra.info/). Take care! 💙")
