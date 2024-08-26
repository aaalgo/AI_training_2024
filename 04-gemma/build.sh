#!/bin/bash

if [ ! -d llama.cpp ]
then
    git clone --depth 1 https://github.com/google/gemma.cpp
fi

pushd gemma.cpp/build
cmake ..
make -j 8 gemma
popd
cp gemma.cpp/build/gemma ./

