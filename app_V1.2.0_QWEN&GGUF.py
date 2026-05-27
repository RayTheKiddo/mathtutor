import os
import re
from io import BytesIO
import base64

import streamlit as st
from PIL import Image

from huggingface_hub import InferenceClient
from llama_cpp import Llama


# =========================================================
# Streamlit Config
# =========================================================

st.set_page_config(
    page_title="Math OCR Tutor",
    page_icon="📐",
    layout="centered"
)

st.title("📐 AI Math OCR Tutor")

st.caption(
    "Upload handwritten math image → OCR → Edit OCR → Math Solver"
)


# =========================================================
# CPU / Runtime Optimization
# =========================================================

try:
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
except Exception:
    pass


# =========================================================
# CONFIG
# =========================================================

# OCR: Qwen image-to-text API
OCR_MODEL = "Qwen/Qwen3.6-27B:featherless-ai"

# Local GGUF model for math solving
MATH_MODEL_REPO = "RayLLLLL/magpie-math-q4-gguf"
MATH_MODEL_FILE = "magpie-q4.gguf"

MAX_NEW_TOKENS = 1024


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("OCR Model")
    st.code(OCR_MODEL)

    st.subheader("Math Solver")
    st.code(f"{MATH_MODEL_REPO}\n{MATH_MODEL_FILE}")

    st.divider()

    user_prompt = st.text_area(
        "Tutor Instruction",
        value=(
            "Please explain the math problem "
            "step-by-step and give the final answer."
        ),
        height=150
    )

    st.divider()

    st.markdown(
        """
### Usage

1. Input your HF Token
2. Upload handwritten math image
3. Qwen OCR recognizes math
4. Edit OCR result manually
5. Send corrected text to the local Math Solver
"""
    )


# =========================================================
# HF TOKEN INPUT
# =========================================================

st.subheader("🔑 Hugging Face Token")

hf_token = st.text_input(
    "Enter your Hugging Face Token",
    type="password",
    placeholder="hf_xxxxxxxxxxxxxxxxx"
)

if not hf_token:
    st.warning("Please input your Hugging Face token first.")
    st.stop()


# =========================================================
# Clients
# =========================================================

# Only used for Qwen OCR
client = InferenceClient(api_key=hf_token)


# =========================================================
# Utility Functions
# =========================================================

def clean_ocr_text(text: str) -> str:
    text = str(text).strip()

    text = text.replace("\\displaystyle", "")

    # Remove fenced code blocks if the model returns them
    text = re.sub(r"^```(?:markdown|md|text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    # Remove common leading wrappers, but keep actual math content
    text = re.sub(r"^\s*(OCR\s*Result\s*:|Recognized\s*Text\s*:)\s*", "", text, flags=re.IGNORECASE)

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def image_bytes_to_data_url(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


# =========================================================
# QWEN OCR
# =========================================================

def run_qwen_ocr(uploaded_bytes: bytes, mime_type: str) -> str:
    image_url = image_bytes_to_data_url(uploaded_bytes, mime_type)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict OCR engine for mathematical images.\n"
                "Return ONLY the recognized content.\n"
                "Do NOT explain.\n"
                "Do NOT summarize.\n"
                "Do NOT add any extra words.\n"
                "Do NOT add prefixes like 'OCR Result:'.\n"
                "Use Markdown for text structure.\n"
                "Use LaTeX for formulas.\n"
                "Preserve line breaks and layout as much as possible."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Recognize all mathematical content in this image. "
                        "Return only the OCR result."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
            ]
        }
    ]

    completion = client.chat.completions.create(
        model=OCR_MODEL,
        messages=messages,
        max_tokens=2048,
        temperature=0.0,
        top_p=1.0,
        presence_penalty=0.0,
        extra_body={
            "top_k": 20,
        },
    )

    result = completion.choices[0].message.content
    return clean_ocr_text(result)


# =========================================================
# Load Local GGUF Model
# =========================================================

@st.cache_resource
def load_math_model():
    llm = Llama.from_pretrained(
        repo_id=MATH_MODEL_REPO,
        filename=MATH_MODEL_FILE,
        n_ctx=4096,
        n_threads=max(1, (os.cpu_count() or 1)),
        verbose=False,
    )
    return llm


# =========================================================
# Math Solver
# =========================================================

def ask_math_model(llm, ocr_text: str, user_prompt: str) -> str:
    prompt = f"""You are a professional math tutor.

Math problem:
{ocr_text}

Instruction:
{user_prompt}

Requirements:
1. Solve step-by-step
2. Correct obvious OCR mistakes
3. Use Markdown formatting
4. Use $$...$$ for display math
5. Use $...$ for inline math
6. Be clear and concise
"""

    completion = llm.create_completion(
        prompt=prompt,
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.2,
        top_p=0.95,
        repeat_penalty=1.1,
        stop=["</s>"]
    )

    return completion["choices"][0]["text"].strip()


# =========================================================
# Upload Config
# =========================================================

MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

ALLOWED_IMAGE_TYPES = [
    "png",
    "jpg",
    "jpeg",
    "webp",
    "bmp",
    "tiff",
    "tif",
    "gif"
]


# =========================================================
# Upload Image
# =========================================================

uploaded_file = st.file_uploader(
    "📤 Upload handwritten math image",
    type=ALLOWED_IMAGE_TYPES,
    help=f"Maximum file size: {MAX_FILE_SIZE_MB} MB"
)


# =========================================================
# Validate Upload
# =========================================================

if uploaded_file is not None:
    if uploaded_file.size > MAX_FILE_SIZE_BYTES:
        st.error(
            f"""
File too large.

Maximum allowed size:
{MAX_FILE_SIZE_MB} MB

Current file size:
{uploaded_file.size / (1024 * 1024):.2f} MB
"""
        )
        st.stop()

    uploaded_bytes = uploaded_file.getvalue()
    image = Image.open(BytesIO(uploaded_bytes))


# =========================================================
# Main Pipeline
# =========================================================

if uploaded_file is not None:
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # =====================================================
    # OCR BUTTON
    # =====================================================

    if st.button(
        "🚀 Run OCR",
        use_container_width=True
    ):
        with st.spinner("Running Qwen OCR..."):
            try:
                ocr_result = run_qwen_ocr(
                    uploaded_bytes,
                    uploaded_file.type
                )
                st.session_state["ocr_result"] = ocr_result
            except Exception as e:
                st.error(f"OCR failed:\n\n{e}")
                st.stop()

    # =====================================================
    # OCR EDITOR
    # =====================================================

    if "ocr_result" in st.session_state:
        st.subheader("📄 OCR Result (Editable)")

        edited_ocr = st.text_area(
            "You can manually edit OCR result before sending to the Math Solver",
            value=st.session_state["ocr_result"],
            height=250
        )

        st.session_state["edited_ocr"] = edited_ocr

        st.download_button(
            label="Download OCR Result",
            data=edited_ocr.encode("utf-8"),
            file_name="ocr_result.txt",
            mime="text/plain"
        )

        # =================================================
        # SEND TO LOCAL MATH SOLVER
        # =================================================

        if st.button(
            "🧠 Solve Math Problem",
            use_container_width=True
        ):
            with st.spinner("Loading local GGUF model..."):
                try:
                    llm = load_math_model()
                except Exception as e:
                    st.error(f"Failed to load local math model:\n\n{e}")
                    st.stop()

            with st.spinner("Solving..."):
                try:
                    answer = ask_math_model(
                        llm,
                        st.session_state["edited_ocr"],
                        user_prompt
                    )
                except Exception as e:
                    st.error(f"Math model failed:\n\n{e}")
                    st.stop()

            st.subheader("🧠 AI Tutor Response")
            st.markdown(answer, unsafe_allow_html=True)

            combined = f"""
OCR RESULT
==========

{st.session_state["edited_ocr"]}


AI TUTOR RESPONSE
=================

{answer}
"""

            st.download_button(
                label="Download Full Result",
                data=combined.encode("utf-8"),
                file_name="math_tutor_result.txt",
                mime="text/plain"
            )

else:
    st.info("Upload a handwritten math image to begin.")
