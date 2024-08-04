from src.biLSTM_CRF.utils import FastText, load_variable
import py_vncorenlp
from seqeval.metrics.sequence_labeling import get_entities
import numpy as np

def get_tags(sequences, tag_index):
    sequence_tags = []
    for sequence in sequences:
        sequence_tag = []
        for categorical in sequence:
            sequence_tag.append(tag_index.get(np.argmax(categorical)))
        sequence_tags.append(sequence_tag)
    return sequence_tags

def predict(model, tag_tokenizer, sent):
    tag_index = tag_tokenizer.word_index
    tag_size = len(tag_index) + 1
    pred = model.predict(sent)
    sequence_tags = get_tags(pred, {i: t for t, i in tag_index.items()})
    for idx, each in enumerate(sequence_tags):
        try:
           idx_cut = each.index(None)
        except:
           idx_cut = len(each) + 1
        sequence_tags[idx] = each[:idx_cut]
    return sequence_tags


def predict_bilstm(texts, model, word_segmenter):
    texts = [' '.join(word_segmenter.word_segment(text)) for text in texts]
    text_raw = texts
    
    texts = FastText(texts)
    word_tag_tokenizer = load_variable("D:\\Nam3\\NLP\\deloy_app\\checkpoint\\bilstm\\word_tag_tokenizer.pkl")
    res = predict(model, word_tag_tokenizer, texts)
    return res, text_raw 
