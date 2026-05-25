import streamlit as st
from transformers import pipeline
from PIL import Image
import io
import torch
import streamlit as st
from transformers import pipeline
from PIL import Image

# 设置页面
st.set_page_config(page_title="数学导师 - OCR识别", layout="centered")

# 缓存 OCR pipeline
@st.cache_resource
def load_ocr():
    # 使用 TrOCR 基础版（印刷体）；如果内存小可换 "microsoft/trocr-small-printed"
    return pipeline("image-to-text", model="microsoft/trocr-base-printed")

def main():
    st.title("📷 数学导师：拍照识题")
    st.write("上传一张数学题照片，系统将自动识别题目文本。")

    uploaded_file = st.file_uploader("选择图片", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的题目", use_column_width=True)

        with st.spinner("正在识别题目..."):
            ocr = load_ocr()
            # pipeline 直接接收 PIL Image
            result = ocr(image)
            recognized_text = result[0]['generated_text']

        st.subheader("识别结果")
        edited_text = st.text_area("可手动编辑修正", recognized_text, height=150)

        if st.button("确认题目"):
            st.session_state['math_question'] = edited_text
            st.success("题目已保存！接下来可以使用解题模块。")
            st.write("当前题目：", edited_text)

        with st.expander("查看原始OCR输出"):
            st.code(recognized_text)

if __name__ == "__main__":
    main()
