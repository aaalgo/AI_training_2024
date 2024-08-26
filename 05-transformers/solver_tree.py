#!/usr/bin/env python3

# You are not supposed to run this script
# It is a snapshot of my solution to the AI math Olympliad
# when the code was not so complicated.

# The way how the verifier model was trained is interesting, too.
# https://huggingface.co/peiyi9979/math-shepherd-mistral-7b-rl
# https://huggingface.co/datasets/peiyi9979/Math-Shepherd/viewer/default/train?row=0

import os
import subprocess as sp
if True:
    # Automatically select 1 GPU
    found = []
    for l in sp.check_output('nvidia-smi  --query-gpu=index,memory.used --format=csv | tail -n +2', shell=True).decode('ascii').split('\n'):
        l = l.strip().split(',')
        if len(l) != 2:
            continue
        if l[1] == ' 0 MiB':
            found.append(int(l[0]))
    assert len(found) >= 1
    print(f"Found GPUS: {found}, using {found[0]}.")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(found[0])
import gc
import traceback
from time import time
from types import SimpleNamespace
from collections import defaultdict
from queue import PriorityQueue
import logging
import pandas as pd
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, LogitsProcessor, LogitsProcessorList, BitsAndBytesConfig
import colorful as cf

good_token = '+'
bad_token = '-'
step_tag = 'ки'

def load_vocaburary (tokenizer):
    # One innovation in early days of transformers is BPE,
    # that is, an algo to map a string of variable length to a token.
    # The vocaburary of today's LLM is the mapping between  range(N) and a list of strings

    # This function loads the vocaburary into a list L.  
    # L[i] is the string corresponding to i-th token.
    # So L[i] is equal to the outcome of tokenizer.decode(i)  (except for tensor shapes)
    n = len(tokenizer)
    all_codes = torch.range(0, n).long()[:, None]
    return tokenizer.batch_decode(all_codes, skip_special_tokens=True, clean_up_tokenization_spaces=False)

# ================ Logits Magics ================

def make_index_tensor (s):
    # convert a set of token IDs
    # into a tensor used for indexing
    s = list(s)
    s.sort()
    return torch.Tensor(s).long()

# After inferencing each new token, we have the chance to modify logits before
# searching for sentence is done.  A lot of control can be done by logits magics.

class SetNextToken (LogitsProcessor):
    # force generate the token given next
    def __init__(self, token):
        self.token = token

    def __call__(self, input_ids, logits):
        logits += -math.inf         # add -infinite will make anything -infinite
        logits[:, self.token] = 1   # -infinite will suppress the token
        return logits

class DetectEoS:
    # If previous token ends with '.', force next token to be end of sentence.

    def __init__(self, vocaburary, eos_token, num_beams):
        super().__init__()
        blackList = set()
        for token, span in enumerate(vocaburary):
            # I call each entry in vocaburary a span
            if len(span) == 0:
                continue
            if span[-1] == '.':     # whatever ends with '.' should be added to blacklist
                blackList.add(token)

        self.blackList = blackList
        self.eos_token = eos_token
        self.num_beams = num_beams
        self.mask = None

    def __call__(self, input_ids: torch.LongTensor, logits: torch.FloatTensor):
        # input_ids:  B * L,   batch * length of sentence (so far generated)
        # logits   :  B * N,   batch * length vocaburary, logits of next token

        assert len(input_ids.shape) == 2
        assert len(logits.shape) == 2
        n = input_ids.shape[0]
        assert n == logits.shape[0]

        if self.mask is None:
            mask = torch.full_like(logits[0, :], -math.inf)
            # mask is [-inf] * len(vocaburary)
            mask[self.eos_token] = 0
            self.mask = mask

        for i in range(n):
            if input_ids[i, -1].item() in self.blackList:
                logits[i, :] += self.mask

        return logits

class NormalizeText (LogitsProcessor):

    def __init__(self, voc, latex=False):
        blackList = set() 
        for token, span in enumerate(voc):
            if not latex:
                # disable some of the signature latex symbols
                if '$' in span:
                    blackList.add(token)
                    continue
                if '{' in span:
                    blackList.add(token)
                    continue
                if '}' in span:
                    blackList.add(token)
                    continue
                if '\\' in span:
                    blackList.add(token)
                    continue
            if '\n' in span:
                # there should be no new line in one reasoning step
                blackList.add(token)
                continue
            off = span.find('.')
            if off >= 0:
                # if '.' appears, it must be at the end of sentence, therefore
                # at the end of span
                if off + 1 < len(span):
                    blackList.add(token)
        self.blackList = make_index_tensor(blackList)

    def __call__(self, input_ids, logits):
        logits[:, self.blackList] = -math.inf
        return logits


DIGITS = '0123456789'
def test_answer_token_helper (span):
    i = 0
    while i < len(span):
        if span[i] == '.':
            return i + 1 == len(span)
        elif span[i] in DIGITS:
            i += 1
            continue
        else:
            return False
    return True

class ExtractAnswer (LogitsProcessor):
    def __init__(self, voc, eos_token_id):
        self.offset = None
        # before each invocation of "generate",
        # offset must be set to the length of history
        begin  = set()    # beginning of answer, must start with a digit
        end    = set()    # this token ends with '.'
                          # which marks the end of sentence
        middle = set()    # includes begin & end

        for token, span in enumerate(voc):
            if len(span) == 0:
                continue
            if not test_answer_token_helper(span):
                continue
            if span[-1] == '.':
                end.add(token)
            if span[0] in DIGITS:
                begin.add(token)
            middle.add(token)
        self.begin = make_index_tensor(begin)
        self.end = end      # end is a set
        self.middle = make_index_tensor(middle)
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids, logits):
        assert not self.offset is None
        mask = torch.full_like(logits, -math.inf)

        n = input_ids.shape[0]
        assert n == logits.shape[0]

        off = input_ids.shape[1] - self.offset
        if off == 0:
            mask[:, self.begin] = 0
        else:
            mask[:, self.middle] = 0
            for i in range(n):
                if input_ids[i, -1].item() in self.end:
                    # if already ended, allow only EOS
                    mask[i, :] += -math.inf
                    mask[i, self.eos_token_id] = 0

        return logits + mask

# ================ End of Logits Magics ================

class ModelWrapper:
    def __init__ (self, model_path, quant=False, flash=False):
        quantization_config = None
        if quant:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )        
        attn_implementation = None
        if flash:
            attn_implementation = 'flash_attention_2'

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_path,
                torch_dtype=torch.bfloat16,
                device_map='auto',
                quantization_config=quantization_config,
                attn_implementation=attn_implementation,
                use_cache=True)
        try:
            model.generation_config = GenerationConfig.from_pretrained(model_path)
        except:
            model.generation_config = GenerationConfig()

        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        model.eval()
        self.tokenizer = tokenizer
        self.model = model

class State:
    # We use state to manage incremental generation
    # A state is made up of
    #       - Past generated text
    #       - Prompt that has not been processed by model
    #       - Past_key_values: cache for fast incremental generation

    # A state supports the following operations

    #   1. Prompt
    #       # generate a new state without changing the old state
    #       new_state = state.addPrompt("some text")    
    #       # or modify old state
    #       state.addPrompt("some text", inplace=True)

    #   2. Generation
    #       # create a new state, doesn't change old state
    #       new_state, text = state.generate()
    #       # the new state has prompt and newly generate text appended to history
    #
    #   3. Branching
    #       _, texts = state.generate(choices=5)
    #
    #       Returns texts, a list of choices strings.
    #       This does not generate new state

    #   4. Infer one next token (for verifying model)
    #
    #       logits = state.logits()

    def __init__ (self, wrapper):
        self.wrapper = wrapper
        self.input_ids = None           # cumulative
        self.past_key_values = None
        self.full_text = ""             # the text of input_ids
        self.prompt = ""

    def save_past_key_values (self, past_key_values):
        if past_key_values is None:
            self.past_key_values = None
        else:
            self.past_key_values = tuple([
                (k.cpu(), v.cpu())
                for k, v in past_key_values
                ])

    def cuda_past_key_values (self):
        if self.past_key_values is None:
            return None
        device = self.wrapper.model.device
        return tuple([
            (k.to(device), v.to(device))
            for k, v in self.past_key_values
            ])

    def logits (self):
        assert len(self.prompt) == 0
        return self.wrapper.model.forward(self.input_ids, past_key_values=self.cuda_past_key_values(), use_cache=True, return_dict=True)[0]

    def addPrompt (self, text, inplace=False):
        # we can keep adding to prompt
        if len(text) == 0:
            return self
        if inplace:
            self.prompt += text
            return
        # non-inplace, create a new state
        state = State(self.wrapper)
        state.past_key_values = self.past_key_values
        state.input_ids = self.input_ids
        state.full_text = self.full_text
        state.prompt = self.prompt + text
        return state

    def consumePrompt (self):
        if len(self.prompt) == 0:
            return self

        input_ids = self.input_ids
        prompt_ids = self.wrapper.tokenizer.encode(self.prompt, return_tensors='pt')
        prompt_ids = prompt_ids.to(self.wrapper.model.device)
        if input_ids is None:
            input_ids = prompt_ids
        else:
            input_ids = torch.cat([input_ids, prompt_ids], dim=-1)

        last_token = input_ids[0, -1]
        logits = LogitsProcessorList([SetNextToken(last_token)])

        output = self.wrapper.model.generate(input_ids[:, :-1], max_new_tokens=1, past_key_values=self.cuda_past_key_values(), logits_processor=logits, use_cache=True, return_dict_in_generate=True)
        output_ids = output[0]
        assert input_ids[:, -1] == output_ids[:, -1]

        state = State(self.wrapper)
        state.save_past_key_values(output.past_key_values)
        state.input_ids = input_ids
        state.full_text = self.full_text + self.prompt
        return state

    def generate (self, logits=None, choices=1, max_new_tokens = 100):
        # wrapper: ModelWrapper with .model and .tokenizer

        input_ids = self.input_ids

        if len(self.prompt) > 0:
            prompt_ids = self.wrapper.tokenizer.encode(self.prompt, return_tensors='pt')
            prompt_ids = prompt_ids.to(self.wrapper.model.device)
            if input_ids is None:
                input_ids = prompt_ids
            else:
                input_ids = torch.cat([input_ids, prompt_ids], dim=-1)

        assert not input_ids is None

        num_beams = choices
        length_penalty = None
        diversity_penalty = None
        past_key_values = self.cuda_past_key_values()

        if num_beams > 1:
            # prepare for beam search, need to repeat past key-values to match shape
            length_penalty = -5.0
            diversity_penalty = 1.0
            if not past_key_values is None:
                past_key_values =tuple([
                        (
                            k.repeat(num_beams, 1, 1, 1),
                            v.repeat(num_beams, 1, 1, 1)
                        )
                        for k, v in past_key_values
                        ])

        output = self.wrapper.model.generate(input_ids, max_new_tokens=max_new_tokens, past_key_values=past_key_values, use_cache=True, logits_processor=logits, return_dict_in_generate=True, num_return_sequences=choices, num_beams=num_beams, length_penalty=length_penalty, num_beam_groups=num_beams, diversity_penalty=diversity_penalty)

        output_ids = output[0][:, input_ids.shape[1]:]
        texts = self.wrapper.tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        if choices == 1:
            state = State(self.wrapper)
            state.input_ids = input_ids
            state.save_past_key_values(output.past_key_values)
            state.full_text = self.full_text + texts[0]
            torch.cuda.empty_cache()
            gc.collect()
            return state, texts[0]
        else:
            # multiple choices, return text only
            torch.cuda.empty_cache()
            gc.collect()
            return None, texts

# ================ Tree Model ================

# A tree node represents any text that has so far generated.
# If so far the reasoning has reached an answer, the node is marked "closed"
# and will not expand.
# Otherwise the node can be expanded to multiple next step nodes.

# All node ends with:
#   - Description of problem, before 1st step.
#   - Any step, with \n

class Node:
    def __init__ (self):
        self.reason = None
        self.verify = None
        self.children = None    # if children is [], then inference is already done
        self.answer = None      # if answer is not None, then we shouldn't expand
        self.choices = None     # which the childrens are based on
        self.original = None
        self.depth = 0
        self.score = 0
    
    def __lt__ (self, other):
        return self.score > other.score

class Solver:
    def __init__ (self, args):
        self.primer = ""
        if args.primer is not None:
            with open(args.primer, 'r') as f:
                self.primer = f.read()
        self.temperature = args.temperature

        self.reasoner = ModelWrapper(args.model, args.quant, args.flash)
        self.choices = args.choices
        self.fanout = args.fanout
        self.max_answers = args.max_answers
        self.max_depth = args.max_depth
        self.max_step_length = args.max_step_length
        voc = load_vocaburary(self.reasoner.tokenizer)
        eos_token_id = self.reasoner.model.generation_config.eos_token_id

        detectEoS = DetectEoS(voc, eos_token_id, self.choices)
        normalizeText = NormalizeText(voc)
        extractAnswer = ExtractAnswer(voc, eos_token_id)

        self.logits_reason = LogitsProcessorList([normalizeText, detectEoS])
        self.logits_answer = LogitsProcessorList([extractAnswer])
        self.extractAnswer = extractAnswer
                            # need to update input length before generation

        self.verifier = ModelWrapper(args.verifier, args.quant, args.flash)
        self.candidate_tokens = self.verifier.tokenizer.encode(f"{good_token} {bad_token}")[1:] # [648, 387]

    def makeRoot (self, problem):
        root = Node()
        root.reason = State(self.reasoner)
        root.reason.addPrompt(self.primer + problem + '\n\n', inplace=True)
        root.verify = State(self.verifier)
        root.verify.addPrompt(self.primer + problem + '\n\n', inplace=True)
        return root

    def expand (self, node, incremental = False):
        assert not node.reason is None
        assert not node.verify is None
        assert node.answer is None

        if node.children is None:
            node.children = []
        else:
            assert incremental

        new_depth = node.depth + 1

        node.reason.addPrompt(f'{new_depth}. ', inplace=True)
        node.reason = node.reason.consumePrompt()

        _, node.choices = node.reason.generate(logits=self.logits_reason, choices=self.choices, max_new_tokens=self.max_step_length)
        too_long = ''

        out = []

        print("{")
        for text in node.choices:
            if len(text) == 0:
                print("\t_____: %s" % text)
                continue
            if text[-1] != '.': # might be too long
                too_long = text
                print("\t_____: %s" % text)
                continue
            child = Node()
            child.reason = node.reason.addPrompt("%s\n" % text)
            child.verify = node.verify.addPrompt("Step %d. %s ки" % (new_depth, text))
            child.verify = child.verify.consumePrompt()
            logits = child.verify.logits()
            logits = logits[0, -1, self.candidate_tokens]
            child.depth = new_depth
            child.score = logits.softmax(dim=-1)[0].item()
            child.verify.addPrompt("\n", inplace=True)
            node.children.append(child)
            out.append(child)
            print("\t%.3f: %s" % (child.score, text))
        print("}")
        return out

    def checkAnswer (self, node):
        FORCE_ANSWER = "Therefore, the answer to this problem is "
        assert len(node.reason.prompt) > 0
        if (node.depth >= self.max_depth) or ('answer' in node.reason.prompt):
            branch = node.reason.addPrompt("")
            branch.prompt = FORCE_ANSWER
            branch = branch.consumePrompt()
            self.extractAnswer.offset = branch.input_ids.shape[-1]
            branch, guess = branch.generate(logits=self.logits_answer)
            assert guess[-1] == '.'
            node.answer = int(guess[:-1])
            del branch

    # This is the main solving algo
    def solve (self, problem, answer):

        root = self.makeRoot(problem)

        Q = PriorityQueue()
        Q.put(root)

        vote = defaultdict(lambda: [])
        n_votes = 0
        max_vote = 0

        steps = 0

        while not Q.empty():
            if (n_votes >= self.max_answers) and (max_vote > 1):
                break
            node = Q.get()
            steps += 1
            print(f'{cf.blue}{steps}/{Q.qsize()}{cf.cyan} {node.score:.3f}{cf.yellow} {"-" * 32} {answer} {cf.reset}')
            print(node.reason.full_text + node.reason.prompt)
            self.checkAnswer(node)
            if not node.answer is None:
                if answer == node.answer:
                    print(f"{cf.green}{node.answer}{cf.reset}")
                else:
                    print(f"{cf.red}{node.answer} != {answer}{cf.reset}")
                a = vote[node.answer]
                a.append(node.score)
                max_vote = max(max_vote, len(a))
                n_votes += 1
                print("----")
                for k, v in vote.items():
                    if k == answer:
                        print(f"{cf.green}{k}{cf.reset} {len(v)}:{sum(v):.3f}")
                    else:
                        print(f"{cf.red}{k}{cf.reset} {len(v)}:{sum(v):.3f}")
                continue
            if node.depth >= self.max_depth:
                continue
            children = self.expand(node)
            children.sort()
            added = 0
            for i, node in enumerate(children):
                dup = False
                for j in range(i):
                    # use better test
                    if node.reason.prompt == children[j].reason.prompt:
                        dup = True
                        break
                if dup:
                    continue
                if added >= self.fanout:
                    break
                added += 1
                Q.put(node)
        a = []
        for k, v in vote.items():
            a.append((sum(v), k))
        a.sort(reverse=True)
        print(a)
        return a[0][1]


def clean_amie (problem, answer):
    if isinstance(answer, str):
        answer = int(answer.replace(',', ''))
    #q = problem.find('?')
    #if q >= 0:
    #    problem = problem[q+1]
    q = problem.find(r'$\textbf{(A')
    if q >= 0:
        problem = problem[:q]
    return problem, answer

if __name__ == '__main__':
    import pandas as pd
    import argparse

    HOME = os.path.dirname(__file__)

    logging.basicConfig(level=logging.INFO)

    DEFAULT_MODEL = os.path.join(HOME, 'models', 'llama3_hint.model')
    DEFAULT_VERIFY = os.path.join(HOME, 'models', 'math-shepherd-mistral-7b-prm')

    parser = argparse.ArgumentParser()
    #parser.add_argument("-i", "--input", default='amie200', type=str)
    parser.add_argument("-i", "--input", default='kaggle', type=str)
    parser.add_argument("-c", "--choices", default=10, type=int)
    parser.add_argument("-f", "--fanout", default=2, type=int)
    parser.add_argument("--max_answers", default=10, type=int)
    parser.add_argument("-d", "--max_depth", default=16, type=int)
    parser.add_argument("-l", "--max_step_length", default=150, type=int)
    parser.add_argument("--quant", action='store_true')
    parser.add_argument("--flash", action='store_true')
    parser.add_argument("-t", "--temperature", default=0.0, type=float)
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, type=str)
    parser.add_argument("-v", "--verifier", default=DEFAULT_VERIFY, type=str)
    parser.add_argument("-p", "--primer", default=None, type=str)
    parser.add_argument("--id", default=None, type=str)
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(HOME, 'data', args.input, 'data.csv'))

    solver = Solver(args)
    total = 0
    correct = 0
    total_time = 0
    for _, row in df.iterrows():
        if not args.id is None:
            if row['id'] != args.id:
                continue
        problem = row['problem']
        answer = row['answer']
        problem, answer = clean_amie(problem, answer)
        # https://github.com/timofurrer/colorful
        # print('{c.bold}{c.lightCoral_on_white}Hello World{c.reset}'.format(c=cf))
        print(f"""{cf.blue}==== {cf.green}{correct} / {total}{cf.blue} at {cf.cyan}{total_time/(total+0.001):.2f}{cf.blue}s/problem, next problem {cf.cyan}{row['id']}{cf.blue}: {cf.yellow}{answer}{cf.reset}""")
        tic = time()
        try:
            guess = solver.solve(problem, answer)
        except KeyboardInterrupt:
            break
        except:
            raise
            #traceback.print_exc()
            #guess = 0
            pass
        toc = time()
        total_time += (toc - tic)
        total += 1
        if answer == guess:
            correct += 1
    print(f"{correct} / {total}")

