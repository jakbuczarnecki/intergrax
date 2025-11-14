# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import streamlit as st
from sidebar import display_sidebar
from chat_interface import display_chat_interface

st.title("intergrax RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

display_sidebar()

display_chat_interface()