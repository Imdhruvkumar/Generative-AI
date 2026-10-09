
from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from transformers import pipeline
from transformers.utils import logging
from langchain_core.prompts import PromptTemplate

logging.set_verbosity_error()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Paragraph Summarizer",
    page_icon="📝",
    layout="centered"
)

# ---------------- FRONTEND DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #f1f5f9;
}

.block-container {
    max-width: 850px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

.header {
    text-align: center;
    padding: 25px 10px;
}

.header h1 {
    color: #38bdf8;
    font-size: 40px;
    font-weight: 800;
    margin-bottom: 8px;
}

.header p {
    color: #94a3b8;
    font-size: 16px;
}

div[data-testid="stTextArea"] textarea {
    background: #172033;
    color: #f8fafc;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 15px;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: #38bdf8;
    box-shadow: 0 0 0 1px #38bdf8;
}

.stButton > button {
    background: #0284c7;
    color: white;
    border: none;
    border-radius: 10px;
    height: 48px;
    font-size: 17px;
    font-weight: 700;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #0369a1;
    color: white;
    border: none;
}

.result-card {
    background: #172033;
    border: 1px solid #334155;
    border-left: 4px solid #38bdf8;
    border-radius: 12px;
    padding: 20px;
    line-height: 1.8;
    color: #e2e8f0;
    margin-top: 10px;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="header">
    <h1>📝 AI Summarizer</h1>
    <p>Turn long paragraphs into short, clear summaries.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- PROMPT TEMPLATE ----------------
prompt_template = PromptTemplate.from_template(
    """You are a helpful AI assistant.
Summarize the following paragraph in simple and clear English.
Keep the summary short and include only the main ideas.

Paragraph: {paragraph}

Summary:"""
)

# ---------------- LOAD LOCAL MODEL ----------------
@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model="./models/Qwen2.5-1.5B-Instruct"
    )

# ---------------- INPUT SECTION ----------------
st.markdown("### Enter your paragraph")

paragraph = st.text_area(
    "Paragraph",
    placeholder="Paste or type your paragraph here...",
    height=220,
    label_visibility="collapsed"
)

st.caption("Paste your text above and click the button to summarize it.")

# ---------------- SUMMARIZE BUTTON ----------------
if st.button("✨ Generate Summary", use_container_width=True):

    if not paragraph.strip():
        st.warning("Please enter a paragraph first.")

    else:
        with st.spinner("Your AI is generating a summary..."):

            try:
                pipe = load_model()

                final_prompt = prompt_template.format(
                    paragraph=paragraph
                )

                result = pipe(
                    final_prompt,
                    max_new_tokens=150,
                    return_full_text=False,
                    do_sample=False
                )

                summary = result[0]["generated_text"].strip()

                # Remove an accidental repeated summary label
                if summary.lower().startswith("summary:"):
                    summary = summary[len("summary:"):].strip()

                st.markdown("### 📄 Your Summary")

                st.markdown(
                    f'<div class="result-card">{summary}</div>',
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"Something went wrong: {e}")

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    Powered by your local Qwen2.5 model · No API key required
</div>
""", unsafe_allow_html=True)