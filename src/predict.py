import argparse
import backoff
import json
import openai
import os
import pathlib
import random
import torch
import transformers
import tqdm
import numpy as np
import httpx

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

TOKENS = json.load(open('tokens.json', 'r'))
TKN = TOKENS['HF']
OAIK = TOKENS['OAI']
TGIK = TOKENS['TG']
MK = TOKENS['MK']

class Tom:

    system = """You are TheoryOfMindGPT, an expert language model at using your theory-of-mind capabilities to predict the beliefs and actions of others in human conversations. You will be given a potentially unfinished conversation between two speakers. Put yourself in the mindset of the speakers and try to reason about the requested conversation outcome. Use the keyword "CERTAINTY" to report your prediction for the outcome of interest. Report your answer on a scale from 1 to 10 with 1 indicating "not likely at all" and 10 indicating "almost certainly". For example, "CERTAINTY = 7"."""

    def prompt(x, demographics=False):
        p = f'In the following conversation segment, {x["context"]}.'
        if demographics:
            p += '\n\nThe demographics of Speaker 0 are:\n'
            p += '\n'.join(f'-{k}: {v}' for k,v in x['demographics']['Speaker 0'].items())
            p += '\n\nThe demographics of Speaker 1 are:\n'
            p += '\n'.join(f'-{k}: {v}' for k,v in x['demographics']['Speaker 1'].items())
        p += f'\n\n[SEGMENT START]\n{x["input"]} ...\n[SEGMENT END]'
        p += '\n\nNow, fast-forward to the end of the conversation. '
        p += f"How certain {x['question']}? Let's think step by step, but keep your answer concise (less than 100 words)."
        return p

def get_tokenizer(tokenizer_arg):
    if 'openai' in tokenizer_arg or 'together' in tokenizer_arg or 'mistralapi' in tokenizer_arg:
        # use gpt2 tokenizer as a placeholder for batching ops (tokenization is inverted before sending to the api)
        return transformers.AutoTokenizer.from_pretrained('gpt2', padding_side='left', token=TKN)
    else:
        t = transformers.AutoTokenizer.from_pretrained(tokenizer_arg, padding_side='left', token=TKN)
        if t.chat_template is None:
            print('Replacing chat template.')
            t.chat_template = "{% for message in messages %}{{ 'Instruct: ' + message['content'] + '\nOutput:'}}{% endfor %}"
            print('Example:', t.apply_chat_template([{'role' : 'user', "content" : "example content"}], tokenize=False))
        return t

# https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/17
def to_tokens_and_logprobs(model, tokenizer, input_texts):
    input_ids = tokenizer(input_texts, padding=True, return_tensors="pt").input_ids
    outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()

    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

    batch = []
    for input_sentence, input_probs in zip(input_ids, gen_probs):
        text_sequence = []
        for token, p in zip(input_sentence, input_probs):
            if token not in tokenizer.all_special_ids:
                text_sequence.append((tokenizer.decode(token), p.item()))
        batch.append(text_sequence)
    return batch

def batch_to_device(batch, args):
    res = {}
    for k, v in batch.items():
        res[k] = v.to(args.device)
    return res
    
@backoff.on_exception(backoff.fibo, openai.RateLimitError)
def completions_with_backoff(client, mistral=False, **kwargs):
    if mistral:
        kwargs['messages'] = [ChatMessage(**x) for x in kwargs['messages']]
        # sampling not supported for mistral api
        if 'n' in kwargs: del kwargs['n']
        if 'logprobs' in kwargs: del kwargs['logprobs']
        return client.chat(**kwargs)
    else:
        return client.chat.completions.create(**kwargs)

def call_chat_gpt(client, prompt, args):

    system = Tom.system
    prompt = prompt.split(system)[-1]
    
    # print(system)
    # print('p', prompt)

    messages = [
        {"role": "system", "content" : system},
        {"role": "user", "content": prompt}
    ]

    completion = completions_with_backoff(
        client,
        model='/'.join(args.model.split('/')[1:]),
        temperature=args.temp,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        n=args.mc_samples,
        logprobs=True,
        messages=messages,
        mistral=('mistralapi/' in args.model)
    )

    text = [choice.message.content
        for choice in completion.choices]
    
    if 'together' in args.model:
        probs = [
            list(zip(choice.logprobs.tokens, choice.logprobs.token_logprobs)) 
            for choice in completion.choices
        ]
    elif 'mistralapi' in args.model:
        probs = None
    else:
        probs = [choice.logprobs.content
            for choice in completion.choices]
        probs = [[(pi.token, pi.logprob) for pi in p] for p in probs]

    return text, probs

def safe_call_chat_gpt(client, prompt, args):
    try:
        return call_chat_gpt(client, prompt, args)
    except openai.OpenAIError as e:
        # Handle all OpenAI API errors
        print(f"Error: {e}")
    return '**API_Error_encountered**', []

class OpenAIModelWrapper:
    # to make sure inference setup is consistent with other models

    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
        self.dtype = None

        timeout = httpx.Timeout(15.0, read=5.0, write=10.0, connect=3.0)
        if 'together/' in args.model:
            self.client = openai.OpenAI(
                api_key=TGIK,
                base_url="https://api.together.xyz/v1",
                timeout=timeout
            )
        elif 'mistralapi/' in args.model:
            self.client = MistralClient(api_key=MK)
        else:
            self.client = openai.OpenAI(api_key=OAIK, timeout=timeout)
    
    def __call__(self, *args, **kwargs):
        raise AttributeError('OpenAI Wrapper can only be used for generation.')
    
    def generate(self, input_ids, return_dict_in_generate=False, **kwargs):
        x = [self.tokenizer.decode(i, skip_special_tokens=True) for i in input_ids]
        sample = []
        probs = []
        for p in x:
            s, lp = safe_call_chat_gpt(self.client, p, self.args)
            # # api is slow enough to print and watch if you want
            # print(p)
            # print('--------')
            # print(s)
            # print('+++++++++++++++')
            # # end print statements
            sample.append(s)
            probs.append(lp)
        seqs = [[p.tolist() + self.tokenizer(' ')['input_ids'] + self.tokenizer(si)['input_ids'] 
            for si in s] for s, p in zip(sample, input_ids)]
        if return_dict_in_generate:
            return {'sequences': seqs, 'logprobs': probs}
        else:
            return seqs
    
    def to(self, *args, **kwargs):
        return self
    
    def eval(self):
        return self

class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, tokenizer):
        fname = os.path.join('data', args.data)
        with open(fname + '.jsonl', 'r') as f:
            data = [json.loads(line) for line in f.readlines()]
            n = int(args.frac * len(data))
            random.seed(args.seed)
            self.data = random.sample(data, n)
        self.task = Tom
        self.demographics = args.demographics
        self.tokenizer = tokenizer
    
    def get_source(self, index):
        return self.data[index]

    def __getitem__(self, index):
    
        data = self.data[index]
        chat = []

        temp = self.tokenizer(data['input'])['input_ids']
        if len(temp) > args.max_size:
            temp = temp[:args.max_size]
            data['input'] = self.tokenizer.decode(temp)
            # print(data['input'])
        
        # data["input"] = "<test>"
        # print(self.task.prompt(data, self.demographics))
        # exit()

        chat.append({'role': 'system', 'content': self.task.system})
        chat.append({'role' : 'user', 'content' : self.task.prompt(data, self.demographics)})

        inputs = self.tokenizer.apply_chat_template(chat, tokenize=False)
        # print('chat temp:', inputs)
        # exit()
        inputs = self.tokenizer(inputs)
        inputs['len'] = len(inputs['input_ids'])
        # print('len', inputs['len'])
        inputs['index'] = index
        return inputs

    def __len__(self,):
        return len(self.data)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str)
    parser.add_argument('--data', type=str)
    parser.add_argument('--demographics', type=int, default=0)
    parser.add_argument('--temp', default=1, type=float)
    parser.add_argument('--top_p', default=1, type=float)
    parser.add_argument('--frac', default=1, type=float)
    parser.add_argument('--oom', default=0, type=float)
    parser.add_argument('--seed', default=1, type=float)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--max_size', type=int, default=5096)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--load_in_4bit', type=int, default=1)
    parser.add_argument('--hidden_states', type=int, default=0)
    parser.add_argument('--mc_samples', type=int, default=1)

    args = parser.parse_args()

    if args.temp > 0:
        args.output = f'{args.model.split("/")[-1]}-{args.data}-d={args.demographics}'
    else:
        mname = args.model.split("/")[-1]
        mname += '-notemp'
        args.output = f'{mname}-{args.data}-d={args.demographics}'
    
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = get_tokenizer(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataloader = torch.utils.data.DataLoader(
        Dataset(args, tokenizer) if not args.oom \
            else sorted(Dataset(args, tokenizer), key=lambda x: -x['len']), 
        shuffle=False, 
        collate_fn=transformers.DataCollatorWithPadding(
            tokenizer,
            return_tensors='pt'
        ), 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )

    if 'openai' in args.model or 'together' in args.model or 'mistralapi' in args.model:
        model = OpenAIModelWrapper(tokenizer, args)
    else:
        if args.load_in_4bit and args.device != 'cpu':

            model = transformers.AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.float16,
                quantization_config=transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_type='nf4'),
                    trust_remote_code=True, token=TKN)
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, token=TKN)

    outputs = []
    tensors = dict()
        
    for inputs in tqdm.tqdm(dataloader, total=len(dataloader)):
        
        inputs = batch_to_device(inputs, args)
        
        if args.temp == 0:
            output = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True,
                output_hidden_states=args.hidden_states,
                do_sample=False
            )
        else:
            output = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                use_cache=True,
                return_dict_in_generate=True,
                output_hidden_states=args.hidden_states,
                temperature=args.temp,
                top_p=args.top_p,
                num_return_sequences=args.mc_samples
            )
        
        if args.oom:
            print('Passed at len:', inputs['len'])
            exit()

        try:
            b = output['sequences'].size(0) // args.mc_samples
            output['sequences'] = output['sequences'].view(b, args.mc_samples, -1)
        except AttributeError:
            pass

        # write everything after the prompt
        response = [[tokenizer.decode(s[i][l:], skip_special_tokens=True) for i in range(len(s))] 
            for s, l in zip(output['sequences'], inputs['len'])]
        
        if 'logprobs' not in output:
            logprobs = [to_tokens_and_logprobs(model, tokenizer, input_texts=r) for r in response]
        else:
            logprobs = output['logprobs']

        completion = [[tokenizer.decode(s[i], skip_special_tokens=True) for i in range(len(s))]
            for s, l in zip(output['sequences'], inputs['len'])]
        
        # print(output['logprobs'])
        # print(len(output['logprobs']), len(output['logprobs'][0]))
        # exit()
        
        # print(len(response), len(response[0]))

        # concat and save last hidden states at each step as tensor for space efficiency:
        # generation steps x layers x batch x seq_len x hdim
        # Example at first step:
        # > tuple 256 x tuple 13 X tensor(1, 152, 1024)
        # Uses cache, so example at second step:
        # tuple 256 x tuple 13 X tensor(1, 1, 1024)
        if args.hidden_states:
            raise NotImplementedError('Not Implemented for MC sampling...')
            hidden_states = torch.cat([output['hidden_states'][i][-1] for i in range(len(output['hidden_states']))], dim=1)
            # print(inputs['len'], output['hidden_states'][0][-1].size(), hidden_states.size())
            # > tensor([153]) torch.Size([1, 153, 1024]) torch.Size([1, 408, 1024])
            
            # pool mean over sequence: https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently
            for i, h in enumerate(output['hidden_states']):
                if i == 0:
                    input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(h[-1].size()).float()
                    sum_embeddings = torch.sum(h[-1] * input_mask_expanded, 1)
                    sum_mask = input_mask_expanded.sum(1)
                    sum_mask = torch.clamp(sum_mask, min=1e-9)
                else:
                    sum_embeddings += h[-1].squeeze(1)
                    sum_mask += 1.
            mean_embeddings = sum_embeddings / sum_mask

        for i, r, c, lp in zip(inputs['index'].tolist(), response, completion, logprobs):
            output = dataloader.dataset.get_source(i)
            output['response'] = r
            output['logprobs'] = lp
            output['completion'] = c
            output['model'] = args.model
            output['data'] = args.data
            output['demographics'] = args.demographics
            output['max_new_tokens'] = args.max_new_tokens
            output['temp'] = args.temp
            output['top_p'] = args.top_p
            output['tensor'] = i
            outputs.append(output)
        
        if args.hidden_states:
            raise NotImplementedError('Not Implemented for MC sampling...')
            for i, h in zip(inputs['index'].tolist(), mean_embeddings):
                tensors[str(i)] = h.cpu().numpy()
    
    pathlib.Path('outputs/').mkdir(exist_ok=True)
    if args.hidden_states:
        # be careful this can blow up
        np.savez_compressed(f'outputs/{args.output}.npz', **tensors)
    with open(f'outputs/{args.output}.jsonl', 'w') as out:
        for o in outputs:
            out.write(json.dumps(o) + '\n')
