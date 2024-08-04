from src.viT5.ultils import *
import streamlit as st

@st.cache_resource()
def load_model_vit5():
    tokenizer = AutoTokenizer.from_pretrained('VietAI/vit5-base')
    model = AutoModelForSeq2SeqLM.from_pretrained("D:\\Nam3\\NLP\\deloy_app\\checkpoint\\vit5\\checkpoint-12570")
    return model, tokenizer