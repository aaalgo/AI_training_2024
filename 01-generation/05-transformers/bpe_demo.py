#!/usr/bin/env python3
import sys
import torch
from transformers import AutoTokenizer

MODEL_PATH = '../models/hf/Meta-Llama-3.1-8B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

n = len(tokenizer)
all_codes = torch.range(0, n).long()[:, None]
VOCAB = tokenizer.batch_decode(all_codes, skip_special_tokens=True, clean_up_tokenization_spaces=False)

PROMPT = 'Byte pair encoding (also known as digram coding) is an algorithm, first described in 1994 by Philip Gage for encoding strings of text into tabular form for use in downstream modeling.'

input_ids = tokenizer.encode(PROMPT)

tokens = [VOCAB[i] for i in input_ids]

print('|'.join (tokens))

