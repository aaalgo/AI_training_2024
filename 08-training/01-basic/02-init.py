#!/usr/bin/env python3
import sys
#import gpus
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoConfig
)
from transformers.models.llama.modeling_llama import LlamaForCausalLM

CONFIG_PATH = 'models/config'
SAVE_PATH = 'models/init'

# This 4bit quantization will allow loading Llama3.1 8B
# model using ~6GB GPU memory
tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH)
llamaConfig = AutoConfig.from_pretrained(CONFIG_PATH)

llamaConfig.bos_token_id = tokenizer.bos_token_id
llamaConfig.eos_token_id = tokenizer.eos_token_id
llamaConfig.vocab_size = tokenizer.vocab_size

llm = LlamaForCausalLM(llamaConfig)

SKIPPED = set()

@torch.no_grad()
def initialize_weights (module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    #elif isinstance(module, nn.Conv2d):
    #    nn.init.xavier_normal_(module.weight)
    #    if module.bias is not None:
    #        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Embedding):
        nn.init.xavier_normal_(module.weight)
    else:
        t = type(module)
        if not t in SKIPPED:
            print("Skipping", t)
            SKIPPED.add(t)

print("initializing...")
llm.apply(initialize_weights)
print("saving...")

llm.save_pretrained(SAVE_PATH)

