"""
app.py — Streamlit interface for MTG Oracle
 
Entry point for the MTG Oracle RAG system. Provides a chat interface where
users can ask natural language questions about Magic: The Gathering rulings
and card evaluations.
 
Run with: streamlit run app.py
"""
 
import streamlit as st
from src.chain import ask  # imports the full RAG pipeline
 
# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MTG Oracle",
    page_icon="🃏",
    layout="wide"   # two-column layout: sidebar info + main chat
)
 
# ── Custom styling ─────────────────────────────────────────────────────────────
# Streamlit supports CSS injection via markdown with unsafe_allow_html=True
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
 
# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="title-text">MTG Oracle</p>', unsafe_allow_html=True)
st.caption("RAG + Fine-tuned phi-2 · 76,704 official rulings · Powered by Scryfall")
 
# ── Disclaimer ─────────────────────────────────────────────────────────────────
st.info(
    "⚠️ This system uses official Scryfall rulings as its knowledge base. "
    "Responses are grounded in real MTG rules data, but always verify "
    "with a certified judge for tournament play."
)
 
st.divider()
 
# ── Two-column layout ──────────────────────────────────────────────────────────
col_info, col_chat = st.columns([1, 3])
 
# ── Left column: system information ───────────────────────────────────────────
with col_info:
    st.markdown("### How it works")
    st.markdown("""
    1. Your question is converted to a **vector embedding**
    2. **ChromaDB** finds the 5 most relevant rulings (semantic search)
    3. A **fine-tuned phi-2** model generates a grounded response
    """)
 
    st.divider()
 
    st.markdown("### System info")
    st.markdown("""
    - **Model**: phi-2 + LoRA adapter
    - **Base model**: microsoft/phi-2 (2.8B params)
    - **Fine-tuned on**: 4,750 MTG Q&A pairs
    - **Knowledge base**: 76,704 Scryfall rulings
    - **Embeddings**: all-MiniLM-L6-v2
    """)
 
    st.divider()
 
    st.markdown("### Try asking")
    st.markdown("""
    - What is the ruling for Snapcaster Mage regarding flashback?
    - How does the Splinter Twin combo work?
    - Evaluate Blood Moon for Modern
    - How does Tarmogoyf's power work?
    - What happens when a creature with persist dies?
    """)
 
# ── Right column: chat interface ───────────────────────────────────────────────
with col_chat:
    st.markdown("### Ask MTG Oracle")
 
    # Session state persists data across Streamlit reruns
    # Without it, the message history would reset on every interaction
    if "messages" not in st.session_state:
        st.session_state.messages = []
 
    # Render existing conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
 
    # Chat input — st.chat_input returns None if empty, string if submitted
    question = st.chat_input("Ask about any MTG ruling or card evaluation...")
 
    if question:
        # Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
 
        # Generate and display the assistant's response
        # The RAG pipeline runs here: ChromaDB retrieval → prompt construction → model inference
        with st.chat_message("assistant"):
            with st.spinner("Consulting the Oracle..."):
                answer = ask(question)  # full RAG pipeline call
            st.write(answer)
 
        # Add assistant response to history for multi-turn conversation
        st.session_state.messages.append({"role": "assistant", "content": answer})