import streamlit as st
from pix2text import Pix2Text
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==================== 页面配置 ====================
st.set_page_config(page_title="数学导师 - OCR识别与解题", layout="centered")

# ==================== 第一步：OCR 模块（完全保留，未修改）====================
@st.cache_resource
def load_ocr_model():
    p2t = Pix2Text.from_config()
    return p2t

def ocr_section():
    st.title("📷 数学导师：拍照识题")
    st.write("上传一张数学题照片，系统将自动识别题目文本。")

    uploaded_file = st.file_uploader("选择图片", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="上传的题目", use_column_width=True)

        with st.spinner("正在识别题目..."):
            p2t = load_ocr_model()
            # 识别图片中的文字/公式
            recognized_text = p2t.recognize(image, file_type='text_formula', return_text=True)

        st.subheader("识别结果")
        edited_text = st.text_area("可手动编辑修正", recognized_text, height=150)

        if st.button("确认题目"):
            st.session_state['math_question'] = edited_text
            st.success("题目已保存！接下来可以使用解题模块。")
            st.write("当前题目：", edited_text)

        with st.expander("查看原始OCR输出"):
            st.code(recognized_text)

# ==================== 第二步：数学解题模块（baseline model）====================
@st.cache_resource
def load_math_model():
    """加载 baseline 模型和分词器，适配 CPU 环境"""
    model_name = "prithivMLmods/Magpie-Qwen-CortexDual-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 注意：Streamlit Cloud 只有 CPU，需强制使用 CPU，并开启低内存模式
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,      # CPU 推荐 float32
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    return tokenizer, model

def get_math_solution(question_text, tokenizer, model):
    """根据用户问题生成解答（使用 chat template）"""
    messages = [
        {"role": "user", "content": question_text}
    ]
    # 使用模型自带的 chat template 构造输入
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)  # model.device 是 'cpu'
    
    # 生成回答
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,      # 确定性输出，节约资源
        num_beams=1
    )
    # 解码时只保留新生成的部分（去掉输入 prompt）
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return answer

def math_solver_section():
    st.markdown("---")
    st.header("🧠 数学解题助手")
    
    # 如果 OCR 模块已经保存了题目，则作为默认值填入
    default_question = st.session_state.get('math_question', '')
    user_question = st.text_area(
        "输入数学题目（可手写或使用上面识别的结果）",
        value=default_question,
        height=120
    )
    
    if st.button("🔍 获取 AI 解答", key="solve_btn"):
        if not user_question.strip():
            st.warning("请输入题目")
        else:
            with st.spinner("AI 正在思考中（首次加载模型较慢，请稍候）..."):
                tokenizer, model = load_math_model()
                answer = get_math_solution(user_question, tokenizer, model)
            st.subheader("📖 解答")
            st.write(answer)
            
            # 可选：将解答保存到 session_state，供后续错题本使用
            st.session_state['last_answer'] = answer

# ==================== 主程序 ====================
def main():
    ocr_section()           # 第一步：OCR 识别
    math_solver_section()   # 第二步：数学解题

if __name__ == "__main__":
    main()
