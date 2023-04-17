import argparse
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import tqdm
import time
import json
import torch
import os
import random

from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from utils import form_partitions, postprocess, WatermarkLogitsWarper

nltk.download('punkt')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="lfqa-data/inputs.jsonl")
parser.add_argument('--output_dir', default="lfqa-data")
parser.add_argument('--model_size', default="13b")
parser.add_argument('--num_instances', default=3000, type=int)
parser.add_argument('--num_samples', default=1, type=int)
parser.add_argument('--max_new_tokens', default=300, type=int)
parser.add_argument('--top_k', default=None, type=int)
parser.add_argument('--top_p', default=0.9, type=float)
parser.add_argument('--typical_p', default=None, type=float)
parser.add_argument('--num_shards', default=1, type=int)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--strength', default=2.0, type=float)
parser.add_argument('--watermark_fraction', default=0.5, type=float)
parser.add_argument('--debug', action="store_true")
args = parser.parse_args()

with open(args.dataset, "r") as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]

start = time.time()
tokenizer = AutoTokenizer.from_pretrained(f"facebook/opt-{args.model_size}", torch_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(f"facebook/opt-{args.model_size}")
model.cuda()
model.eval()

print(f"Loading model took {time.time() - start} seconds")

if args.top_p is None:
    output_file = f"{args.output_dir}/opt_{args.model_size}_strength_{args.strength}_frac_{args.watermark_fraction}_{args.max_new_tokens}_len.jsonl"
else:
    output_file = f"{args.output_dir}/opt_{args.model_size}_strength_{args.strength}_frac_{args.watermark_fraction}_{args.max_new_tokens}_len_top_p_{args.top_p}.jsonl"

watermark_processor = LogitsProcessorList([WatermarkLogitsWarper(fraction=args.watermark_fraction, strength=args.strength, debug=args.debug)])

random.seed(43)
device = "cuda" if torch.cuda.is_available() else "cpu"

outputs = []

if args.num_shards > 1:
    partitions = form_partitions(data, args.num_shards)
    data = partitions[args.local_rank]
    output_file = f'{output_file}.shard_{args.local_rank}'

if os.path.exists(output_file):
    with open(output_file, "r") as f:
        num_curr_outputs = len(f.read().strip().split("\n"))
else:
    num_curr_outputs = 0

print("Skipping {} instances".format(num_curr_outputs))

for idx, dd in tqdm.tqdm(enumerate(data), total=min(len(data), args.num_instances)):
    if idx < num_curr_outputs:
        continue
    if len(outputs) >= args.num_instances:
        break

    if "gold_completion" in dd:
        prefix = dd['prefix']
        gold_completion = dd['gold_completion']
    else:
        gold_sequence = dd['prefix'] + " " + dd['targets'][0]
        gold_sents = sent_tokenize(gold_sequence)
        # use the first 2 sentences as prefix
        prefix = " ".join(gold_sents[:2])
        gold_completion = " ".join(gold_sents[2:])

    batch = tokenizer(prefix, truncation=True, padding="longest", return_tensors="pt", max_length=1024 - args.max_new_tokens).to(device)
    num_tokens = len(batch['input_ids'][0])

    with torch.inference_mode():
        generation = model.generate(**batch,
            logits_processor=watermark_processor,
            do_sample=True,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            typical_p=args.typical_p,
            top_p=args.top_p,
            num_return_sequences=args.num_samples)
        gen_text = postprocess(generation['sequences'][:, num_tokens:], tokenizer)

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
