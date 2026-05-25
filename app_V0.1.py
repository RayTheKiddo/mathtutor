import streamlit as st
from transformers import pipeline
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import tempfile

torch.compiler.set_stance("force_eager")

# 设置页面标题
st.set_page_config(page_title="数学导师 - OCR识别", layout="centered")

# 缓存 OCR 模型，避免重复加载
@st.cache_resource
def load_ocr_model():
    MODEL_PATH = "zai-org/GLM-OCR"
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )

    class GLMOCRWrapper:
        def __init__(self, processor, model):
            self.processor = processor
            self.model = model

        def generate(self, image, compile=False):
            # 将 PIL Image 保存为临时文件，GLM-OCR 需要文件路径或 URL
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image.save(tmp.name)
                tmp_path = tmp.name

            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "url": tmp_path}
                        ],
                    }
                ]
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.model.device)
                inputs.pop("token_type_ids", None)
                generated_ids = self.model.generate(**inputs, max_new_tokens=8192)
                output_text = self.processor.decode(
                    generated_ids[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=False
                )
                return [{"generated_text": output_text}]
            finally:
                os.unlink(tmp_path)

    return GLMOCRWrapper(processor, model)

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
