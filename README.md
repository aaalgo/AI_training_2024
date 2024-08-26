2024 AAA AI Training
====================

Suggested Learning Order:

- 01-openai:  all notebooks.
    This is the OpenAI GPT API.

- 03-llama-cpp: all notebooks and scripts.
    Llama.cpp is significant in that it's the first project that made
    CPU inference possible.  You can run these if you have a
    laptop/desktop with >= 16GB memory.  The Python library is
    largely compatible to OpenAI and has a grammar API that can be very 
    handy.

- 05-transformers: all notebooks and scripts.
    This is the primary API researchers use.  All new techniques are first
    implemented here.  For local training or finetune this is the only
    choice.

- 06-vLLM: all notebooks.
    vLLM is the goto framework for local model deployment.  It is very
    easy to use and very efficient.  If you developed your method with
    transformers and then port to vLLM you might see change of accuracy.

Google's Gemini and Gemma can be safely skipped.

