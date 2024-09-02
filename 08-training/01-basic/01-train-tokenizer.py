#!/usr/bin/env python3
import sys
import os
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
import config

os.makedirs("models/init", exist_ok=True)

tokenizer = ByteLevelBPETokenizer()

dataset = load_dataset(config.DATASET, split="train")


def train_iterator (dataset, column_name="text"):
    for item in dataset:
        yield item[column_name]

tokenizer.train_from_iterator(train_iterator(dataset),
                vocab_size=config.VOCAB_SIZE,
                min_frequency=2,
                special_tokens=config.SPECIAL_TOKENS)

tokenizer.save_model("models/init")

