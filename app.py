import streamlit as st
from src.chain import ask

st.set_page_config(
    page_title="MTG Oracle",
    page_icon="🃏",
    layout="wide"
)

st.markdown("""
    <style>
    .title-text {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #8B0000, #4B0082);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title-text">MTG Oracle</p>', unsafe_allow_html=True)
st.caption("RAG + Fine-tuned phi-2 · 76,704 rulings · Powered by Scryfall")

st.divider()

col_info, col_chat = st.columns([1, 3])

with col_info:
    st.markdown("### About")
    st.markdown("""
    MTG Oracle combines:
    - **RAG** — searches 76,704 official rulings from Scryfall
    - **Fine-tuned phi-2** — responds in the style of an expert MTG judge
    """)
    st.divider()
    st.markdown("### Try asking")
    st.markdown("""
    - What is the ruling for Snapcaster Mage regarding flashback?
    - How does the Splinter Twin combo work?
    - Evaluate Blood Moon for Modern
    - How does Tarmogoyf's power work?
    """)

with col_chat:
    st.markdown("### Ask MTG Oracle")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask about any MTG ruling or card evaluation...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Consulting the Oracle..."):
                answer = ask(question)
            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})