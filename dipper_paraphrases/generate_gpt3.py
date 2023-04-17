import argparse
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import tqdm
import json
import os
import torch
import random

from utils import get_openai_response, form_partitions, get_chatgpt_qa_response, get_chatgpt_completion_response

nltk.download('punkt')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="lfqa-data/inputs.jsonl")
parser.add_argument('--output_dir', default="lfqa-data")
parser.add_argument('--model', default="davinci_003")
parser.add_argument('--num_samples', default=1, type=int)
parser.add_argument('--max_new_tokens', default=300, type=int)
parser.add_argument('--top_k', default=None, type=int)
parser.add_argument('--top_p', default=None, type=float)
parser.add_argument('--typical_p', default=None, type=float)
parser.add_argument('--num_shards', default=1, type=int)
parser.add_argument('--local_rank', default=0, type=int)
args = parser.parse_args()

with open(args.dataset, "r") as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]

output_file = f"{args.output_dir}/gpt3_{args.model}_300_len.jsonl"

random.seed(43)
device = "cuda" if torch.cuda.is_available() else "cpu"


if os.path.exists(output_file):
    with open(output_file, "r") as f:
        num_curr_outputs = len(f.read().strip().split("\n"))
else:
    num_curr_outputs = 0

print("Skipping {} instances".format(num_curr_outputs))
data = data[num_curr_outputs:]

if args.num_shards > 1:
    partitions = form_partitions(data, args.num_shards)
    data = partitions[args.local_rank]
    output_file = f'{output_file}.shard_{args.local_rank}'

outputs = []

if args.model == "davinci_003":
    openai_fn = get_openai_response
elif args.model == "chatgpt" and "lfqa" in args.dataset:
    openai_fn = get_chatgpt_qa_response
elif args.model == "chatgpt" and "lfqa" not in args.dataset:
    openai_fn = get_chatgpt_completion_response
else:
    raise NotImplementedError

for idx, dd in tqdm.tqdm(enumerate(data), total=len(data)):
    if "gold_completion" in dd:
        prefix = dd['prefix']
        prefix = "Answer the following question in 200-250 words.\n" + prefix
        gold_completion = dd['gold_completion']
    else:
        gold_sequence = dd['prefix'] + " " + dd['targets'][0]
        gold_sents = sent_tokenize(gold_sequence)
        # use the first 2 sentences as prefix
        prefix = " ".join(gold_sents[:2])
        gold_completion = " ".join(gold_sents[2:])

    gen_text = openai_fn(prefix, max_tokens=args.max_new_tokens)

    outputs.append(json.dumps({
        "prefix": prefix,
        "gold_completion": gold_completion,
        "gen_completion": gen_text
    }))

    if (idx + 1) % 100 == 0:
        with open(output_file, "a") as f:
            f.write("\n".join(outputs) + "\n")
        outputs = []

with open(output_file, "a") as f:
    f.write("\n".join(outputs) + "\n")
    outputs = []
