
import html
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(
    page_title="SmolLM Studio",
    page_icon="✳️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = "./models/smollm2-360m"
MODEL_NAME = "SmolLM2-360M-Instruct"

# --------------------------------------------------
# Apple-inspired iOS design
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --page: #f5f5f7;
    --surface: rgba(255,255,255,0.88);
    --text: #1d1d1f;
    --muted: #777780;
    --blue: #007aff;
    --border: #e5e5ea;
}

.stApp {
    background: var(--page);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Inter", sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 8rem;
}

/* Top navigation */
.topbar {
    display: flex;
    align-items: center;
    gap: 13px;
    padding: 8px 0 22px;
}

.app-icon {
    width: 48px;
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 15px;
    color: white;
    font-size: 25px;
    background: linear-gradient(145deg, #52a8ff, #0066e8);
    box-shadow: 0 5px 14px rgba(0,122,255,.22);
}

.app-title {
    color: var(--text);
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.7px;
}

.app-subtitle {
    color: var(--muted);
    font-size: 12px;
    margin-top: 3px;
}

/* Welcome panel */
.welcome {
    background: linear-gradient(145deg, #ffffff, #eef4ff);
    border: 1px solid #e4eaf5;
    border-radius: 26px;
    padding: 27px;
    margin: 8px 0 24px;
    box-shadow: 0 6px 24px rgba(0,0,0,.035);
}

.welcome h2 {
    font-size: 28px;
    letter-spacing: -1px;
    line-height: 1.2;
    color: #1d1d1f;
    margin: 0 0 10px;
}

.welcome p {
    color: #666675;
    font-size: 15px;
    line-height: 1.6;
    margin: 0;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: white;
    border: 1px solid #e8e8ed;
    border-radius: 21px;
    padding: 17px 19px;
    margin-bottom: 14px;
    box-shadow: 0 3px 12px rgba(0,0,0,.025);
    animation: messageIn .38s ease-out both;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    color: #242428;
    font-size: 15px;
    line-height: 1.75;
}

[data-testid="stChatMessage"] code {
    color: #b4235a;
    background: #f4f4f8;
    border-radius: 5px;
}

[data-testid="stChatMessage"] pre {
    border-radius: 13px;
    border: 1px solid #e5e5ea;
}

/* Composer: readable text and clear focus */
[data-testid="stChatInput"] {
    background: #ffffff;
    border: 1px solid #d8d8df;
    border-radius: 23px;
    box-shadow: 0 8px 30px rgba(0,0,0,.09);
    padding: 5px;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #1684ff;
    box-shadow: 0 0 0 3px rgba(0,122,255,.12),
                0 8px 30px rgba(0,0,0,.08);
}

[data-testid="stChatInput"] textarea {
    color: #1d1d1f !important;
    -webkit-text-fill-color: #1d1d1f !important;
    caret-color: #007aff !important;
    font-family: -apple-system, BlinkMacSystemFont, "Inter", sans-serif !important;
    font-size: 16px !important;
    line-height: 1.55 !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #8e8e93 !important;
    -webkit-text-fill-color: #8e8e93 !important;
    opacity: 1 !important;
}

[data-testid="stChatInput"] button {
    background: #007aff !important;
    border-radius: 50% !important;
    color: white !important;
    border: none !important;
    transition: transform .2s ease, background .2s ease;
}

[data-testid="stChatInput"] button:hover {
    background: #0066d6 !important;
    transform: scale(1.06);
}

/* Starter prompt buttons */
.stButton > button {
    width: 100%;
    min-height: 54px;
    border-radius: 16px;
    border: 1px solid #e1e1e8;
    background: white;
    color: #242428;
    font-size: 14px;
    transition: all .2s ease;
}

.stButton > button:hover {
    border-color: #80baff;
    background: #f0f7ff;
    color: #0066d6;
    transform: translateY(-2px);
}

/* Animated thinking indicator */
.thinking {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 16px 18px;
    margin: 8px 0 16px;
    width: fit-content;
    border-radius: 18px;
    border: 1px solid #e6e6ec;
    background: white;
    color: #6e6e78;
    font-size: 13px;
    animation: messageIn .3s ease-out both;
}

.dot {
    width: 7px;
    height: 7px;
    background: #007aff;
    border-radius: 50%;
    animation: dotPulse 1s infinite ease-in-out;
}

.dot:nth-child(2) { animation-delay: .15s; }
.dot:nth-child(3) { animation-delay: .3s; }

.answer-reveal {
    animation: messageIn .45s ease-out both;
}

.status-pill {
    display: inline-block;
    padding: 6px 11px;
    border-radius: 20px;
    background: #e7f3e9;
    color: #247a39;
    font-size: 12px;
    font-weight: 600;
}

.footer {
    text-align: center;
    color: #92929a;
    font-size: 12px;
    padding: 20px 0 4px;
}

@keyframes messageIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes dotPulse {
    0%, 60%, 100% { opacity: .3; transform: translateY(0); }
    30% { opacity: 1; transform: translateY(-4px); }
}

/* Smaller screens */
@media (max-width: 640px) {
    .block-container {
        padding: 1rem 1rem 7rem;
    }
    .welcome {
        padding: 21px;
        border-radius: 21px;
    }
    .welcome h2 {
        font-size: 24px;
    }
    [data-testid="stChatMessage"] {
        padding: 13px;
    }
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Load local model
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
    )
    model.to(device)
    model.eval()

    return tokenizer, model, device


# --------------------------------------------------
# Session state
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("""
<div class="topbar">
    <div class="app-icon">✳</div>
    <div>
        <div class="app-title">SmolLM Studio</div>
        <div class="app-subtitle">Your personal AI · Running locally on your Mac</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<span class="status-pill">● Local model ready</span>',
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Sidebar settings
# --------------------------------------------------
with st.sidebar:
    st.markdown("## Settings")
    st.caption("Customize your local AI")

    max_tokens = st.slider(
        "Response length",
        min_value=32,
        max_value=512,
        value=160,
        step=32,
    )

    temperature = st.slider(
        "Creativity",
        min_value=0.1,
        max_value=1.5,
        value=0.7,
        step=0.1,
    )

    top_p = st.slider(
        "Response diversity",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.05,
    )

    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.rerun()

    st.caption("Model: SmolLM2-360M-Instruct")


# --------------------------------------------------
# Welcome screen
# --------------------------------------------------
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome">
        <h2>What can I help you with?</h2>
        <p>Ask a question, explore an idea, learn something new,
        or get help with your code. Start a conversation below.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### Try asking")

    prompts = [
        "Explain AI in simple terms",
        "Help me write Python code",
        "Give me creative project ideas",
    ]

    cols = st.columns(3)

    for index, suggestion in enumerate(prompts):
        with cols[index]:
            if st.button(suggestion, key=f"suggestion_{index}"):
                st.session_state.pending_prompt = suggestion
                st.rerun()


# --------------------------------------------------
# Conversation history
# --------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# Chat composer
# --------------------------------------------------
typed_prompt = st.chat_input(
    "Message SmolLM2...",
    key="chat_composer",
)

prompt = typed_prompt

if not prompt and st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None


# --------------------------------------------------
# Generate a response
# --------------------------------------------------
if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()

        thinking_placeholder.markdown("""
        <div class="thinking">
            <span>Thinking</span>
            <span class="dot"></span>
            <span class="dot"></span>
            <span class="dot"></span>
        </div>
        """, unsafe_allow_html=True)

        try:
            tokenizer, model, device = load_model()

            # Keep the most recent conversation context
            history = st.session_state.messages[-8:]

            formatted_messages = [
                {
                    "role": item["role"],
                    "content": item["content"],
                }
                for item in history
            ]

            chat_prompt = tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(
                chat_prompt,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )

            new_tokens = output[0][
                inputs["input_ids"].shape[1]:
            ]

            answer = tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            ).strip()

            if not answer:
                answer = "I couldn't generate a response. Please try again."

            thinking_placeholder.empty()

            st.markdown(
                '<div class="answer-reveal"></div>',
                unsafe_allow_html=True,
            )
            st.markdown(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

        except Exception as error:
            thinking_placeholder.empty()
            st.error(f"Unable to generate a response: {error}")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("""
<div class="footer">
    SmolLM Studio · Private local AI · Powered by your Mac
</div>
""", unsafe_allow_html=True)
