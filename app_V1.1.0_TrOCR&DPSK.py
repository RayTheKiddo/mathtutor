import os
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
    "Upload handwritten math image → TrOCR OCR → DeepSeek explanation"
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

# 改成你自己的 HuggingFace fine-tuned model
MODEL_NAME = "microsoft/trocr-small-handwritten"

# DeepSeek via HuggingFace Inference Providers
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V4-Pro:novita"

# generation settings
MAX_NEW_TOKENS = 128


# =========================================================
# Load Secrets
# =========================================================

HF_TOKEN = None

try:
    HF_TOKEN = st.secrets["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"]
except:
    HF_TOKEN = os.getenv("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")

if HF_TOKEN is None:
    st.error(
        "HF_TOKEN not found. "
        "Please configure Streamlit secrets."
    )
    st.stop()


# =========================================================
# DeepSeek Client
# =========================================================

client = InferenceClient(
    provider="novita",
    api_key=HF_TOKEN
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
# OCR Inference
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
# DeepSeek Tutor
# =========================================================

def ask_deepseek(ocr_text, user_prompt):

    messages = [

        {
            "role": "system",
            "content": (
                "You are a professional math tutor. "
                "You help students understand OCR'd math expressions."
            )
        },

        {
            "role": "user",
            "content": f"""
OCR recognized this math content:

{ocr_text}

User request:

{user_prompt}

Please:
1. Correct OCR mistakes if obvious
2. Explain the math clearly
3. Solve step-by-step if possible
4. Keep the explanation concise but educational
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
# Sidebar
# =========================================================

with st.sidebar:

    st.header("Settings")

    st.write("OCR Model:")
    st.code(MODEL_NAME)

    st.write("LLM:")
    st.code(DEEPSEEK_MODEL)

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

1. Upload handwritten math image
2. Run OCR
3. DeepSeek explains the result
"""
    )


# =========================================================
# Main UI
# =========================================================

uploaded_file = st.file_uploader(
    "Upload handwritten math image",
    type=["png", "jpg", "jpeg", "webp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:
        ocr_button = st.button(
            "Run OCR",
            use_container_width=True
        )

    with col2:
        tutor_button = st.button(
            "Run OCR + AI Tutor",
            use_container_width=True
        )

    if ocr_button or tutor_button:

        # =================================================
        # Load model
        # =================================================

        with st.spinner("Loading OCR model..."):

            processor, model = load_ocr_model()

        # =================================================
        # OCR
        # =================================================

        with st.spinner("Running OCR..."):

            try:

                ocr_text = run_ocr(
                    image,
                    processor,
                    model
                )

            except Exception as e:

                st.error(f"OCR failed:\n\n{e}")

                st.stop()

        # =================================================
        # Display OCR
        # =================================================

        st.subheader("📄 OCR Result")

        st.code(ocr_text)

        st.download_button(
            label="Download OCR Result",

            data=ocr_text.encode("utf-8"),

            file_name="ocr_result.txt",

            mime="text/plain"
        )

        # =================================================
        # DeepSeek
        # =================================================

        if tutor_button:

            with st.spinner("Asking DeepSeek..."):

                try:

                    answer = ask_deepseek(
                        ocr_text,
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

{ocr_text}


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
