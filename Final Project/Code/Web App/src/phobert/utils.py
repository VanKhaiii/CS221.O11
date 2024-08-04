import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, RobertaConfig

import numpy as np
from tqdm.auto import tqdm, trange


class SlotClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_slot_labels,
        dropout_rate=0.0,
    ):
        super(SlotClassifier, self).__init__()
        self.num_slot_labels = num_slot_labels
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)
        
class NERecognizer(RobertaPreTrainedModel):
    def __init__(self, config, slot_label_lst, ignore_index):
        super(NERecognizer, self).__init__(config)
        self.num_slot_labels = len(slot_label_lst)
        self.roberta = RobertaModel(config)  # Load pretrained bert

        self.ignore_index = ignore_index

        self.slot_classifier = SlotClassifier(
            config.hidden_size,
            self.num_slot_labels,
            dropout_rate=0.1,
        )

    def forward(self, input_ids, attention_mask, slot_labels_ids=None):
        outputs = self.roberta(
            input_ids, attention_mask=attention_mask
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]


        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                active_labels = slot_labels_ids.view(-1)[active_loss]
                slot_loss = slot_loss_fct(active_logits, active_labels)
            else:
                slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += slot_loss

        outputs = ((slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs


def convert_examples_to_features_for_demo(
    texts,
    max_seq_len,
    tokenizer,
    pad_token_label_id=-100,
    cls_token_segment_id=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    sw_max_len = 0
    features = []

    for txt_index, text in enumerate(texts):
        # Tokenize word by word (for NER)
        tokens = []
        val_pos_list = []

        for word in text:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            val_pos_list.extend([True] + [False] * (len(word_tokens) - 1))
            sw_max_len = max(sw_max_len, len(tokens))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[: (max_seq_len - special_tokens_count)]
            val_pos_list = val_pos_list[: (max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        val_pos_list += [False]

        # Add [CLS] token
        tokens = [cls_token] + tokens
        val_pos_list = [False] + val_pos_list

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        val_pos_list = val_pos_list + ([False] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len
        )
        assert len(val_pos_list) == max_seq_len, "Error with valid position list length {} vs {}".format(
            len(val_pos_list), max_seq_len
        )

        feature = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'val_pos_list': val_pos_list
        }
        features.append(feature)

    print(f'Max length after splitting into subwords: {sw_max_len}')

    return features

def preprocess_texts(texts, tokenizer, ignore_index, max_seq_len):
    # Load data from json file
    # logger.info("Creating features from dataset file at %s", args.data_dir)
    # examples = processor.get_examples(args)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = ignore_index
    features = convert_examples_to_features_for_demo(
        texts, max_seq_len, tokenizer, pad_token_label_id=pad_token_label_id
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)
    all_val_pos_list = torch.tensor([f['val_pos_list'] for f in features], dtype=torch.bool)
    dataset = TensorDataset(
        all_input_ids, all_attention_mask, all_val_pos_list
    )
    return dataset
    
def get_slot_labels(slot_label_path):
    return [
        label.strip()
        for label in open(slot_label_path, "r", encoding="utf-8")
    ]

slot_label_lst = get_slot_labels('D:\\Nam3\\NLP\\deloy_app\\checkpoint\\slot_labels.txt')

def infer(
    model,
    device,
    dataset,
    batch_size=32,
):
  dataloader = DataLoader(dataset, batch_size=batch_size)
  print("***** Running inference *****")
  print("  Num examples = ", len(dataset))
  print("  Batch size = ", batch_size)

  slot_preds = None
  masks = None
  all_val_pos_list = None
  model.eval()

  for batch in tqdm(dataloader, desc="Inference"):
      val_pos_list = batch[2]
      batch = tuple(t.to(device) for t in batch[:2])
      with torch.no_grad():
          inputs = {
              "input_ids": batch[0],
              "attention_mask": batch[1]
          }
          outputs = model(**inputs)
          _, (slot_logits) = outputs[:2]


      # Slot prediction
      if slot_preds is None:
          slot_preds = slot_logits.detach().cpu().numpy()
          masks = inputs["attention_mask"].detach().cpu().numpy()
          all_val_pos_list = val_pos_list.detach().cpu().numpy()
      else:
          slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)

          masks = np.append(
            masks, inputs["attention_mask"].detach().cpu().numpy(), axis=0
          )

          all_val_pos_list = np.append(
              all_val_pos_list, val_pos_list.detach().cpu().numpy(), axis=0
          )

  # Slot result
  slot_preds = np.argmax(slot_preds, axis=2)
  slot_label_map = {i: label for i, label in enumerate(slot_label_lst)}
  slot_preds_list = [[] for _ in range(slot_preds.shape[0])]

  for i in range(all_val_pos_list.shape[0]):
      for j in range(all_val_pos_list.shape[1]):
          if all_val_pos_list[i, j]:
              slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])
  return slot_preds_list