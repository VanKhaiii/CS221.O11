from src.phobert.config import * 
from src.phobert.utils import *
import py_vncorenlp
from seqeval.metrics.sequence_labeling import get_entities
from vncorenlp import VnCoreNLP

# py_vncorenlp.download_model(save_dir='D:\\Nam3\\NLP\\VnCoreNLP')
# rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='D:\\Nam3\\NLP\\VnCoreNLP')

def predict_phobert(texts, model, tokenizer, word_segmenter):
    # try: 
    #     texts = [' '.join(rdrsegmenter.word_segment(text)) for text in texts]
    # except:
    #     rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='D:\\Nam3\\NLP\\VnCoreNLP')
    #     texts = [' '.join(rdrsegmenter.word_segment(text)) for text in texts]    
    
    texts = [' '.join(word_segmenter.word_segment(text)) for text in texts]
    texts = [text.split() for text in texts]
    inputs = preprocess_texts(texts, tokenizer=tokenizer, ignore_index=ignore_index, max_seq_len=max_seq_len)
    
    all_preds = infer(model, device, inputs, 2)
    all_entities = [get_entities(pred) for pred in all_preds]
    
    return all_entities, texts
    