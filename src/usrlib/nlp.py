import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertConfig, BertModel, BertForSequenceClassification

label_type = ['fear', 'sadness', 'surprise', 'joy', 'love', 'anger']
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
vocab = tokenizer.get_vocab() # a dictionary

def convert_input_to_torch_format(text):
    encoded_dict = tokenizer.encode_plus(
                        ' '.join(text.split()[:100]),    # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 102,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )

    return encoded_dict['input_ids'], encoded_dict['attention_mask']

def get_emocon_for_input(text):
    cls_model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = len(label_type), # The number of output labels.
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )
    input_ids, attention_mask = convert_input_to_torch_format(text)
    output = cls_model(input_ids, attention_mask=attention_mask)

    emocon = np.argmax(output)
    return emocon