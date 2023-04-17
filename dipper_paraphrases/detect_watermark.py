import argparse
import json
import functools
import os
import pickle
import numpy as np
import tqdm
import torch
from functools import partial
from pathlib import Path

from transformers import AutoTokenizer
from utils import watermark_detect, load_shared_args, get_roc, print_tpr_target, print_accuracies, do_sim_stuff, load_sim_stuff, get_longest_answer


parser = argparse.ArgumentParser()
load_shared_args(parser)
parser.add_argument('--threshold', default=4.0, type=float)
parser.add_argument('--total_tokens', default=200, type=int)
parser.add_argument('--paraphrase_no_exist_behavior', default='skip', type=str)
args = parser.parse_args()

model_size, watermark_fraction = Path(args.output_file).stem.split("_")[1], float(Path(args.output_file).stem.split("_")[5])

if "/gpt2" in args.output_file:
    tokenizer = AutoTokenizer.from_pretrained(f"gpt2-{model_size}", torch_dtype=torch.float16)
elif "/opt" in args.output_file:
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/opt-{model_size}", torch_dtype=torch.float16)

# read args.output_file
with open(args.output_file, "r") as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]

if os.path.exists(args.detector_cache):
    with open(args.detector_cache, "r") as f:
        cache = json.load(f)
    # save a copy of cache as a backup
    with open(args.detector_cache + ".bak", "w") as f:
        json.dump(cache, f)
else:
    cache = {}

sim_gold, sim_pp0, sim_cache, sim_fn = load_sim_stuff(args)

if "/opt" in args.output_file:
    vocab_size = 50272
else:
    vocab_size = tokenizer.vocab_size

detect_fn = functools.partial(watermark_detect, watermark_fraction=watermark_fraction, vocab_size=vocab_size)
acc_gen = []
acc_gold = []
acc_pp0 = []

num_paraphrase_pts = len([x for x in data if "paraphrase_outputs" in x])

# iterate over data and tokenize each instance
for idx, dd in tqdm.tqdm(enumerate(data), total=num_paraphrase_pts):
    # tokenize prefix
    if "paraphrase_outputs" not in dd and args.paraphrase_no_exist_behavior == "skip":
        continue
    elif "paraphrase_outputs" not in dd and args.paraphrase_no_exist_behavior == "use_gen":
        dd['paraphrase_outputs'] = {args.paraphrase_type: {'output': dd['gen_completion']}}

    prefix = dd['prefix']
    last_prefix_token = tokenizer(prefix, add_special_tokens=True)["input_ids"][-1]
    assert not isinstance(dd['gen_completion'], str)
    gen_tokens = [last_prefix_token] + tokenizer(dd['gen_completion'], add_special_tokens=False)["input_ids"][0]

    if "lfqa" in args.output_file:
        gold_tokens = get_longest_answer(dd['prefix'])
    else:
        gold_tokens = dd['gold_completion']
    gold_tokens = [last_prefix_token] + tokenizer(gold_tokens, add_special_tokens=False)["input_ids"]

    if "paraphrase_outputs" in dd:
        pp0_tokens = [last_prefix_token] + tokenizer(dd['paraphrase_outputs'][args.paraphrase_type]['output'][0], add_special_tokens=False)["input_ids"]
        pp_len_tokens = [last_prefix_token] + tokenizer(dd['paraphrase_outputs']['lex_40_order_100']['output'][0], add_special_tokens=False)["input_ids"]

    total_tokens = min(len(gen_tokens), len(gold_tokens), len(pp_len_tokens))

    if total_tokens < args.total_tokens:
        continue

    do_sim_stuff(tokenizer.decode(gen_tokens[1:]), tokenizer.decode(gold_tokens[1:]), tokenizer.decode(pp0_tokens[1:]), sim_cache, sim_fn, args, sim_gold, sim_pp0)

    gen_tokens = gen_tokens[:total_tokens]
    gold_tokens = gold_tokens[:total_tokens]
    pp0_tokens = pp0_tokens[:total_tokens]

    gen_z, cache1 = detect_fn(gen_tokens, cache)
    gold_z, cache2 = detect_fn(gold_tokens, cache)
    pp0_z, cache3 = detect_fn(pp0_tokens, cache)

    acc_gen.append(gen_z)
    acc_gold.append(gold_z)
    acc_pp0.append(pp0_z)

    # uncomment below to get accuracies at the currently set value of --threshold
    # print_accuracies(acc_gen, acc_gold, acc_pp0, sim_gold, sim_pp0, args)

    if cache1 or cache2 or cache3:
        # save cache
        with open(args.watermark_cache, "w") as f:
            json.dump(cache, f)

stats = get_roc(acc_gold, acc_gen)
stats2 = get_roc(acc_gold, acc_pp0)

print_tpr_target(stats[0], stats[1], "generation", args.target_fpr)
print_tpr_target(stats2[0], stats2[1], "paraphrase", args.target_fpr)
if sim_fn is not None:
    print(f"Sim between paraphrase and generation = {np.mean(sim_pp0)*100:.1f}%")

with open("roc_plots/watermark.pkl", 'wb') as f:
    pickle.dump((stats, stats2), f)
