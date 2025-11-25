import streamlit as st
import time

# Custom CSS for improved conversation UI
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #2196f3, #4caf50); 
    font-family: 'Arial', sans-serif;
    color: white;
}
.header {
    text-align: center;
    background: linear-gradient(90deg, #2196f3, #f44336);
    color: white;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}
.disclaimer {
    background: #fff3cd;
    border: 1px solid #f44336;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 20px;
    color: black;
}
.conversation-area {
    max-height: 400px;
    overflow-y: auto;
    padding: 10px;
    margin-bottom: 20px;
    border-radius: 15px;
    background: rgba(255, 255, 255, 0.1);
}
.user-text, .bot-text {
    padding: 12px 15px;
    border-radius: 15px;
    margin-bottom: 10px;
    max-width: 70%;
    word-wrap: break-word;
}
.user-text {
    background: rgba(177, 156, 217, 0.6);
    color: #fff;
    align-self: flex-end;
}
.bot-text {
    background: rgba(33, 150, 243, 0.8);
    color: #fff;
    align-self: flex-start;
}
.input-section {
    display: flex;
    gap: 10px;
}
.button {
    background: linear-gradient(90deg, #f44336, #2196f3);
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []


# Bot response function (without emojis)
def get_response(user_input):
    user_input = user_input.lower()
    if any(x in user_input for x in ['sad', 'depressed', 'down', 'unhappy']):
        return "I'm here for you. Let's chat about something that can help improve your mood."
    elif any(x in user_input for x in ['anxious', 'worried', 'stress', 'panic', 'nervous']):
        return "Take a deep breath. You're doing great by reaching out. Let's talk about it calmly."
    elif any(x in user_input for x in ['happy', 'good', 'joy', 'excited', 'great']):
        return "That's wonderful to hear! What's the best part of your day?"
    elif any(x in user_input for x in ['help', 'suicide', 'crisis', 'end it']):
        return "You matter. Reach out immediately to a helpline: AASRA (9820466726) or Vandrevala (1860-266-2345)."
    else:
        return "Thanks for sharing. I'm here to chat about anything. What's on your mind?"


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

# Conversation area
st.subheader("Let's Chat")
conv_container = st.container()
with conv_container:
    if not st.session_state.messages:
        st.write("No messages yet. Start chatting below.")
    else:
        for message in st.session_state.messages:
            role = message['role']
            content = message['content']
            if role == 'You':
                st.markdown(f'<div class="user-text"><strong>You:</strong> {content}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-text"><strong>Bot:</strong> {content}</div>', unsafe_allow_html=True)

# Input section with form for Enter key submission
st.subheader("Share Your Thoughts")
with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_input("Type your message here:", placeholder="How are you feeling today?")
    submit_button = st.form_submit_button(label='Send Message')

    if submit_button and user_input.strip():
        st.session_state.messages.append({"role": "You", "content": user_input})
        placeholder = st.empty()
        placeholder.markdown('<div class="bot-text"><strong>Bot:</strong> typing...</div>', unsafe_allow_html=True)
        time.sleep(1)
        response = get_response(user_input)
        st.session_state.messages.append({"role": "Bot", "content": response})
        placeholder.empty()
        st.experimental_rerun()

# Footer with resources
st.write("---")
st.write(
    "For more resources, visit our [Mental Health Resources Hub for India](https://your-link-here) or check out [AASRA](https://aasra.info/). Take care.")
