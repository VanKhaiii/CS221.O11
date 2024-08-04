import streamlit as st
from src.phobert.config import *
from src.phobert.utils import *

config = RobertaConfig.from_pretrained(model_name, finetuning_task=token_level)
slot_label_lst = get_slot_labels('D:\\Nam3\\NLP\\deloy_app\\checkpoint\\slot_labels.txt')

#allow_output_mutation=True
@st.cache_resource()
def load_model_phobert():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = NERecognizer.from_pretrained(
        model_dir,
        config=config,
        slot_label_lst=slot_label_lst,
        ignore_index=ignore_index
    )

    model.to(device)
    return model, tokenizer
    