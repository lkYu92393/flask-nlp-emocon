import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig, BertModel, BertForSequenceClassification

label_type = ['sadness', 'anger', 'love', 'surprise', 'fear', 'joy']
html_hexcode = {
    "sadness": "&#128534;",
    "anger": "&#128545;",
    "love":"&#128525;",
    "surprise": "&#128558;", 
    "fear": "&#128561;",
    "joy": "&#128513;",
    }
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
vocab = tokenizer.get_vocab() # a dictionary
path = os.path.dirname(__file__) + "/"

try:
    cls_model = BertForSequenceClassification.from_pretrained(path)
except:
    cls_model = None

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
    if cls_model == None:
        return "Missing File."
    input_ids, attention_mask = convert_input_to_torch_format(text)
    output = cls_model(input_ids, attention_mask=attention_mask)

    emocon = html_hexcode[label_type[np.argmax(output.logits.detach().numpy())]]
    return emocon


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def test_accuracy():

    # read val_data
    val_df = pd.read_csv(path + '/val_data.txt'  , delimiter=';', names=["Text", "Category"])
    val_df["id"]   = [i for i in range(0, val_df.shape[0])]
    val_df = val_df[["id","Text","Category"]]
    label_text = val_df.Category.to_list()
    label2id = {label: id for id, label in enumerate(label_type)}
    labels = [label2id[label] for label in label_text]
    labels = torch.tensor(labels)
    print(label_type)
    print('Labels:', labels)
    print("Labels shape:", labels.shape)

    # convert val_data to torch tensor dataset
    input_ids, attention_masks = [], []
    for sent in val_df.Text:
        encoded_dict = tokenizer.encode_plus(
                            ' '.join(sent.split()[:100]),    # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 102,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    batch_size = 4
    validation_dataloader = DataLoader(
                dataset, # The validation samples.
                sampler = SequentialSampler(dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )

    # create model using fine tuned model
    cls_model = BertForSequenceClassification.from_pretrained(path)
    
    cls_model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    
    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        b_input_ids = batch[0].to('cpu')
        b_input_mask = batch[1].to('cpu')
        b_labels = batch[2].to('cpu')
        
        with torch.no_grad():


            outputs = cls_model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask,
                            labels=b_labels
                           )
            loss, logits = outputs.loss, outputs.logits
            print(loss)
        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    return str(avg_val_accuracy)