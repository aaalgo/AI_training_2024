#!/usr/bin/env python3
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer
)
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from datasets import load_dataset
import config

class Collator (DataCollatorWithPadding):
    def __init__ (self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)

    def __call__ (self, *kargs, **kwargs):
        batch = super().__call__(*kargs, **kwargs)
        batch['labels'] = batch['input_ids']
        return batch

def main ():

    parser = HfArgumentParser((TrainingArguments, ))
    training_args, = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained("models/init")

    def tokenization(example):
        return tokenizer(example["text"])

    dataset = load_dataset(config.DATASET)

    train_dataset = dataset['train'].take(10000)
    eval_dataset = dataset['validation'].take(100)

    print("Tokenizing")
    train_dataset = train_dataset.map(tokenization, batched=True)
    eval_dataset = eval_dataset.map(tokenization, batched=True)

    collator = Collator(tokenizer=tokenizer)

    model = AutoModelForCausalLM.from_pretrained("models/init")
    model.to('cuda')

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    tokenizer.save_pretrained(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)


if __name__ == '__main__':
    main()
