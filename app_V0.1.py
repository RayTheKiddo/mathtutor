import streamlit as st
from transformers import pipeline
from PIL import Image
import io
import torch
from transformers import AutoModelForCausalLM

torch.compiler.set_stance("force_eager")

# 设置页面标题
st.set_page_config(page_title="数学导师 - OCR识别", layout="centered")

# 缓存 OCR 模型，避免重复加载
@st.cache_resource
def load_ocr_model():
    # 如果你遇到内存问题，把模型换成 "microsoft/trocr-small-printed"
    return AutoModelForCausalLM.from_pretrained(
    "tiiuae/Falcon-OCR",
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map=cpu
    )

def main():
    st.title("📷 数学导师：拍照识题")
    st.write("上传一张数学题照片，系统将自动识别题目文本。")
    
    # 1. 上传图片
    uploaded_file = st.file_uploader("选择图片", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # 显示上传的图片
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的题目", use_column_width=True)
        
        # 2. 调用OCR
        with st.spinner("正在识别题目..."):
            ocr = load_ocr_model()
            result = ocr.generate(image, compile=False)
            recognized_text = result[0]['generated_text']
        
        # 3. 显示识别结果，并允许用户手动修正
        st.subheader("识别结果")
        edited_text = st.text_area("可手动编辑修正", recognized_text, height=150)
        
        # 4. 将结果存入 session_state，供后续模块使用
        if st.button("确认题目"):
            st.session_state['math_question'] = edited_text
            st.success("题目已保存！接下来可以使用解题模块。")
            st.write("当前题目：", edited_text)
        
        # 可选：展示原始识别文本（调试用）
        with st.expander("查看原始OCR输出"):
            st.code(recognized_text)

if __name__ == "__main__":
    main()
