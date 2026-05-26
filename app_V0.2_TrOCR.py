import streamlit as st
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

st.set_page_config(page_title="数学导师 - OCR识别", layout="centered")

@st.cache_resource
def load_ocr_model():
    processor = TrOCRProcessor.from_pretrained("AbteeXAILab/lumynax-ocr-trocr-large-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("AbteeXAILab/lumynax-ocr-trocr-large-handwritten")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

def main():
    st.title("📷 数学导师：拍照识题")
    st.write("上传一张数学题照片，系统将自动识别题目文本。")

    uploaded_file = st.file_uploader("选择图片", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="上传的题目", use_column_width=True)

        with st.spinner("正在识别题目..."):
            processor, model, device = load_ocr_model()
            # 直接调用 processor，它会自动将图片转为模型需要的 pixel_values
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)
            generated_ids = model.generate(pixel_values)
            recognized_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

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
