import streamlit as st
from pix2text import Pix2Text
from PIL import Image

from llama_cpp import Llama
from huggingface_hub import hf_hub_download


# =========================
# 页面配置
# =========================

st.set_page_config(
    page_title="数学导师",
    layout="centered"
)


# =========================
# OCR 模型
# =========================

@st.cache_resource
def load_ocr_model():

    p2t = Pix2Text.from_config()

    return p2t


# =========================
# 下载 GGUF
# =========================

@st.cache_resource
def download_model():

    model_path = hf_hub_download(
        repo_id="RayLLLLL/magpie-math-q4-gguf",
        filename="magpie-q4.gguf"
    )

    return model_path


# =========================
# 加载 GGUF
# =========================

@st.cache_resource
def load_math_model():

    model_path = download_model()

    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=4,
        verbose=False
    )

    return llm


# =========================
# 数学解题
# =========================

def solve_math_problem(question):

    llm = load_math_model()

    prompt = f"""
You are a professional math tutor.

Solve the following math problem step by step.

Problem:
{question}

Answer:
"""

    output = llm(
        prompt,
        max_tokens=256,
        temperature=0.3,
        stop=["Problem:"]
    )

    answer = output["choices"][0]["text"]

    return answer


# =========================
# 主界面
# =========================

def main():

    st.title("📷 数学导师：拍照识题")

    st.write(
        "上传数学题图片，OCR识别后，自动进行数学解题。"
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

            st.success("题目已保存！")

            st.subheader("当前题目")

            st.write(edited_text)

            # 解题
            with st.spinner("数学模型正在解题中..."):

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
