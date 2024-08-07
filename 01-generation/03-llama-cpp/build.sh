#!/bin/bash

if [ ! -d llama.cpp ]
then
    git clone https://github.com/ggerganov/llama.cpp
fi

cd llama.cpp

# build CPU version
make clean
make -j8
cp llama-cli ../llama-cli
cp llama-llava-cli ../llama-llava-cli
cp llama-server ../llama-server


# build GPU version
make clean
make -j8 GGML_CUDA=1
cp llama-cli ../llama-cli-gpu
cp llama-llava-cli ../llama-llava-cli-gpu
cp llama-server ../llama-server-gpu

