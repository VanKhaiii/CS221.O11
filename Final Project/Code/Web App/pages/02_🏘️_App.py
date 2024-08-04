import numpy as np
import re
import streamlit as st

from src.phobert.model_predict import *
from src.phobert.utils import *
from src.phobert.config import *
from src.phobert.load_model import load_model_phobert

from PIL import Image

from src.biLSTM_CRF.load_model_bi import load_model_bilstm
from src.biLSTM_CRF.biLSTM_predict import predict_bilstm

from src.viT5.load_model_vit5 import load_model_vit5
from src.viT5.ultils import *
from src.viT5.model_predict_vit5 import predict_vit5

from src.ultils import rdrsegmenter

st.set_page_config(
  page_title='Named Entity Recognition for Vietnameese Covid-19', 
  page_icon='https://cdn-icons-png.flaticon.com/512/8090/8090669.png', 
  layout="centered")
st.image(image=Image.open('D:\\Nam3\\NLP\\deloy_app\\1200x675.jpg'), caption='Named Entity Recognition')

# # ======================  Load model  ================
# with st.spinner("Loading Model....(50%)"):
#     model_bilstm = load_model_bilstm()
    
with st.spinner("Loading Model....(50%)"):
    model_phobert, tokenizer_phobert = load_model_phobert()
    
with st.spinner("Loading Model....(100%)"):
    model_vit5, tokenizer_vit5 = load_model_vit5()
    

model_name = st.sidebar.selectbox('Choose model?', ['PhoBert', 'viT5'])
realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Driver
sentence = st.text_input(label='')
sentences = []
sentences.append(sentence)

if st.button('Analyze'):
    if model_name == 'PhoBert':
        all_entities, texts = predict_phobert(sentences, model_phobert, tokenizer_phobert, rdrsegmenter)
        st.write('*Ner Tags:*')
        st.write('*Sentence:*', sentence)
        for idx, (entities, text) in enumerate(zip(all_entities, texts)):
            st.write(f'===== Text {idx} ====')
            for entity in entities:
                 st.write(f"{' '.join(text[entity[1]: entity[2] + 1])} ==> {entity[0]}")
    # if model_name == 'biLSTM':
    #     res, texts = predict_bilstm(sentences, model_bilstm, rdrsegmenter)
    #     def show(pred, text):
    #         text = text.split()
    #         entites = get_entities(pred)
    #         if len(entites) != 0:
    #             for entity in entites:
    #                 st.write(f"{' '.join(text[entity[1]: entity[2] + 1])} => {entity[0]}")
    #         else:
    #             st.write("No Tags Founds!")
    #     show(res[0], texts[0])
    if model_name == 'viT5':
        entities = predict_vit5(sentences, model_vit5, tokenizer_vit5, rdrsegmenter)
        entity_list = [entity.strip() for entity in entities.split(';')]
        st.write('*Ner Tags:*')
        st.write('*Sentence:*', sentence)
        for entity in entity_list:
            st.write(entity)
                