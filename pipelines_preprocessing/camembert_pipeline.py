# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 13:58:39 2022

@author: admin
"""

import os
os.system("git lfs install")
os.system("git clone https://huggingface.co/oscarfossey/job_classification")
os.system("pip install pickle")
os.system("pip install spacy")
os.system("pip install keras")
os.system("spacy download fr_core_news_sm")
os.system("pip install transformers")
os.system("pip install sentencepiece")

import torch
import numpy as np
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from keras.preprocessing.sequence import pad_sequences


global camembert_tokenizer, MAX_LEN, DEVICE, labels, camembert_model
camembert_tokenizer = CamembertTokenizer.from_pretrained('camembert-base', do_lower_case=True)
MAX_LEN = 200
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'N', 'M']
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if DEVICE.type != 'cpu':
  print(torch.cuda.get_device_name(DEVICE))
camembert_model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=14) #for our 14 categories A to N
camembert_model.load_state_dict(torch.load("/content/job_classification/model_camembert_unbalancedv2.pth"))
if DEVICE.type != 'cpu':
  camembert_model.to(DEVICE)


def preprocessing_camembert(texts_array):
  """This functions takes an array of strings and return the inputs and the masks of the camemebert model"""

  texts_list = list(texts_array.flatten())
  input_ids  = [camembert_tokenizer.encode(sent,add_special_tokens=True,max_length=MAX_LEN,truncation=True) for sent in texts_list]
  input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

  attention_masks = []
  for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]  
    attention_masks.append(seq_mask)

  return (input_ids, attention_masks)


def predict_camembert(texts_array, step = 100):
  "Predicting the outputs step by step trough all the inputs"
  
  inputs, masks = preprocessing_camembert(texts_array)
  inputs = torch.tensor(inputs)
  masks = torch.tensor(masks)
  camembert_model.eval()
  predictions = []
  i = 0
  while i < len(inputs) :
    pred = []      
    if DEVICE.type != 'cpu':
      local_inputs = inputs[i:min(i + step, len(inputs))].to(DEVICE)
      local_masks = masks[i:min(i + step, len(masks))].to(DEVICE)    
    else: 
      local_inputs = inputs[i:min(i + step, len(inputs))]
      local_masks = masks[i:min(i + step, len(masks))]
    with torch.no_grad():
      outputs =  camembert_model(local_inputs, token_type_ids = None, attention_mask = local_masks)
      logits = outputs[0]
    if DEVICE.type != 'cpu':
      logits = logits.detach().cpu().numpy()
    pred.extend(np.argmax(logits, axis=1).flatten())
    predictions.extend(pred)
    i = min(i + step, len(inputs))
  
  predictions = [labels[i] for i in predictions]
      
  return predictions