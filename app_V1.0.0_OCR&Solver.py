import streamlit as st
from pix2text import Pix2Text
from PIL import Image

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel


# =========================
# 页面配置
# =========================

st.set_page_config(
    page_title="数学导师",
    layout="centered"
)


# =========================
# OCR模型
# =========================

@st.cache_resource
def load_ocr_model():

    p2t = Pix2Text.from_config()

    return p2t


# =========================
# 数学模型
# =========================

@st.cache_resource
def load_math_model():

    BASE_MODEL = "Qwen/Qwen3-0.6B-Base"

    ADAPTER_MODEL = "RayLLLLL/magpie-math-finetuned-lora-v3"

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True
    )

    # base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    # 加载LoRA
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_MODEL
    )

    model.eval()

    return tokenizer, model


# =========================
# 数学解题
# =========================

def solve_math_problem(question):

    tokenizer, model = load_math_model()

    prompt = f"""
You are a professional math tutor.

Solve the following math problem step by step.

Problem:
{question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return response


# =========================
# 主界面
# =========================

def main():

    st.title("📷 数学导师：拍照识题")

    st.write(
        "上传数学题图片，OCR识别后，可调用 Magpie 数学模型自动解题。"
    )

    uploaded_file = st.file_uploader(
        "选择图片",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="上传的题目",
            use_column_width=True
        )

        # OCR
        with st.spinner("正在识别题目..."):

            p2t = load_ocr_model()

            recognized_text = p2t.recognize(
                image,
                file_type='text_formula',
                return_text=True
            )

        # OCR结果
        st.subheader("OCR识别结果")

        edited_text = st.text_area(
            "可手动编辑修正",
            recognized_text,
            height=200
        )

        # 解题按钮
        if st.button("确认题目并开始解答"):

            st.session_state['math_question'] = edited_text

            st.success("题目已保存！")

            st.subheader("当前题目")

            st.write(edited_text)

            # 解题
            with st.spinner("Magpie 数学模型正在解题中..."):

                answer = solve_math_problem(
                    edited_text
                )

            st.subheader("模型解答")

            st.write(answer)

        # 原始OCR
        with st.expander("查看原始OCR输出"):

            st.code(recognized_text)


# =========================
# 启动
# =========================

if __name__ == "__main__":

    main()
