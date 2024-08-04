import streamlit as st
# from src.biLSTM_CRF.utils import *
from src.biLSTM_CRF.utils import load_variable, create_model_crf
import os
model_dir = "D:\\Nam3\\NLP\\deloy_app\\checkpoint\\bilstm\\model\\biLSTM_CRF.ckpt"

word_embedding_matrix = load_variable("D:\\Nam3\\NLP\\deloy_app\\checkpoint\\bilstm\\word_embedding_matrix.pkl")
@st.cache_resource()
def load_model_bilstm():
    nb_words = 4517
    embed_size=100
    word_max_length=200
    model_load = create_model_crf(word_embedding_matrix, nb_words , embed_size, word_max_length)
    model_load.load_weights(model_dir)
    return model_load