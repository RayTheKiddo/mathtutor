import re
import streamlit as st
from PIL import Image

import torch

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)

from huggingface_hub import InferenceClient


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
    "Upload handwritten math image → OCR → Edit OCR → DeepSeek explanation"
)


# =========================================================
# CPU Optimization
# =========================================================

torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except:
    pass


# =========================================================
# CONFIG
# =========================================================

# 改成你自己的 fine-tuned model
MODEL_NAME = "microsoft/trocr-small-handwritten"

DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V4-Pro"

MAX_NEW_TOKENS = 128


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.subheader("OCR Model")

    st.code(MODEL_NAME)

    st.subheader("LLM")

    st.code(DEEPSEEK_MODEL)

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
3. OCR recognizes math
4. Edit OCR result manually
5. Send corrected text to DeepSeek
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

    st.warning(
        "Please input your Hugging Face token first."
    )

    st.stop()


# =========================================================
# DeepSeek Client
# =========================================================

client = InferenceClient(
    provider="novita",
    api_key=hf_token
)


# =========================================================
# Utility Functions
# =========================================================

def clean_ocr_text(text):

    text = str(text)

    text = text.replace("\\displaystyle", "")
    text = text.replace("$", "")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================================================
# Load OCR Model
# =========================================================

@st.cache_resource
def load_ocr_model():

    processor = TrOCRProcessor.from_pretrained(
        MODEL_NAME
    )

    model = VisionEncoderDecoderModel.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=True
    )

    model.to("cpu")

    model.eval()

    return processor, model


# =========================================================
# OCR
# =========================================================

def run_ocr(image, processor, model):

    image = image.convert("RGB")

    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values

    pixel_values = pixel_values.to("cpu")

    with torch.no_grad():

        generated_ids = model.generate(

            pixel_values,

            max_new_tokens=MAX_NEW_TOKENS,

            num_beams=1,

            do_sample=False
        )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    text = clean_ocr_text(text)

    return text


# =========================================================
# DeepSeek
# =========================================================

def ask_deepseek(ocr_text, user_prompt):

    messages = [

        {
            "role": "system",

            "content": (
                "You are a professional math tutor."
            )
        },

        {
            "role": "user",

            "content": f"""
OCR recognized this math content:

{ocr_text}

User instruction:

{user_prompt}

Please:
1. Explain the math clearly
2. Solve step-by-step if possible
3. Correct obvious OCR mistakes
4. Keep explanations educational
"""
        }
    ]

    completion = client.chat.completions.create(

        model=DEEPSEEK_MODEL,

        messages=messages,

        temperature=0.2,

        max_tokens=512
    )

    return completion.choices[0].message.content


# =========================================================
# Upload Image
# =========================================================

uploaded_file = st.file_uploader(
    "📤 Upload handwritten math image",
    type=["png", "jpg", "jpeg", "webp"]
)


# =========================================================
# Main Pipeline
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

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

        with st.spinner("Loading OCR model..."):

            processor, model = load_ocr_model()

        with st.spinner("Running OCR..."):

            try:

                ocr_result = run_ocr(
                    image,
                    processor,
                    model
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

            "You can manually edit OCR result before sending to DeepSeek",

            value=st.session_state["ocr_result"],

            height=200
        )

        # update session state
        st.session_state["edited_ocr"] = edited_ocr

        st.download_button(

            label="Download OCR Result",

            data=edited_ocr.encode("utf-8"),

            file_name="ocr_result.txt",

            mime="text/plain"
        )

        # =================================================
        # SEND TO DEEPSEEK
        # =================================================

        if st.button(
            "🧠 Send to DeepSeek",
            use_container_width=True
        ):

            with st.spinner("Asking DeepSeek..."):

                try:

                    answer = ask_deepseek(
                        st.session_state["edited_ocr"],
                        user_prompt
                    )

                except Exception as e:

                    st.error(
                        f"DeepSeek API failed:\n\n{e}"
                    )

                    st.stop()

            st.subheader("🧠 AI Tutor Response")

            st.write(answer)

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

    st.info(
        "Upload a handwritten math image to begin."
    )
