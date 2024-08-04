from src.viT5.ultils import *

def predict_vit5(texts, model, tokenizer, word_segmenter):
    texts = [' '.join(word_segmenter.word_segment(text)) for text in texts]
    
    encoding = tokenizer(texts, return_tensors="pt", max_length=256)
    input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=256,
    )
    labels = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return labels