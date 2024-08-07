#!/usr/bin/env python3

# llama-cpp-python installation -- pip install did not work
#
# git clone https://github.com/abetlen/llama-cpp-python.git
# cd llama-cpp-python
# git checkout 5e39a854631c0a84
# git submodule update --init --recursive
# make build.cuda

# Differences between OpenAI and Llama-cpp-python
# 1. Tool specification have minor differences.  Copy & Paste won't work.
# 2. OpenAI response are objects (resp.choices)
#    Llama-cpp-python responses are python dicts (resp['choices'])

import sys
import random
import json
from llama_cpp import Llama

MODEL_PATH = '../models/gguf/Meta-Llama-3-8B-Instruct.Q4_0.gguf'
MAX = 10

LLAMA_CONTEXT = 1024

class User:
    def __init__ (self):
        self.reset()

    def reset (self):
        self.secret = random.randint(0, MAX)

    def guess (self, guess):
        if guess > self.secret:
            return "Too big!"
        elif guess < self.secret:
            return "Too small!"
        else:
            return "Bingo!"

TOOLS = [
    { "type": "function",
      "function": {
          "name": "reset",
          "parameters": {
              "type": "object",
              "title": "reset",
              "properties": {
              },
              "required": []
          }
      }
    },
    { "type": "function",
      "function": {
          "name": "guess",
          "description": "Guess a number.",
          "parameters": {
              "type": "object",
              "title": "guess",
              "properties": {
                  "n": { "title": "n", "type": "integer"}
              },
              "required": ["n"]
          }
      }
    }
]


SYSTEM_PROMPT = ("You are only allowed to respond by using the provided tool.")
PROMPT = ("Let's play the number guessing game.\n"
    f"I'll pick a secret number between 0 and {MAX},"
    "and you are to make guess attempts until you successfully find the correct number.\n"
    "To each of your guess I'll respond by 'Too big!', 'Too small!' or 'Bingo!'\n"
    "Let's start!\n\n")

class Game:
    def __init__ (self):
        self.AI = Llama(model_path=MODEL_PATH,
                        chat_format="chatml-function-calling",
                        n_ctx=LLAMA_CONTEXT,
                        verbose=False)
        self.user = User()
        self.offset = 0
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT}
          ]

    def showMessages (self):
        while self.offset < len(self.messages):
            msg = self.messages[self.offset]
            print("%s: %s" % (msg['role'], msg['content']))
            self.offset += 1

    def testStep (self):
        print("YYY")
        resp = self.AI.create_chat_completion(tools=TOOLS, messages=self.messages, tool_choice='auto')
        print(resp)

    def step (self):
        resp = self.AI.create_chat_completion(tools=TOOLS, messages=self.messages, tool_choice='auto')
        choice = resp['choices'][0]
        for tool_call in choice['message']['tool_calls']:
            assert tool_call['type'] == 'function'
            function = tool_call['function']
            name = function['name']
            arguments = json.loads(function['arguments'])

            if name == 'reset':
                self.user.reset()
                self.messages.append({'role': 'assistant', 'content': 'reset'})
                self.messages.append({'role': 'user', 'content': 'Please guess!'})
            elif name == 'guess':
                n = arguments['n']
                r = self.user.guess(n)
                self.messages.append({'role': 'assistant', 'content': str(n)})
                self.messages.append({'role': 'user', 'content': r})
            else:
                assert False, "AI is not following protocol."

    def finish (self):
        self.messages.append({'role': 'user', 'content': 'We are going to stop here.  How many rounds have we played?'})
        resp = self.AI.create_chat_completion(messages=self.messages)
        self.messages.append(resp['choices'][0]['message'])
        self.messages.append({'role': 'user', 'content': 'How many wrong guesses did you make in total?'})
        resp = self.AI.create_chat_completion(messages=self.messages)
        self.messages.append(resp['choices'][0]['message'])

game = Game()

for i in range(10):
    game.step()

game.finish()
print()
print()
game.showMessages()

