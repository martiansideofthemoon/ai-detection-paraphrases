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
parser.add_argument('--output_file', default="watermark_outputs/gpt2_xl_strength_2.0_frac_0.5_300_len_top_p_0.9.jsonl_pp_pp")
parser.add_argument('--watermark_cache', default="watermark_outputs/watermark_cache.json")
parser.add_argument('--threshold', default=4.0, type=float)
parser.add_argument('--total_tokens', default=200, type=int)
parser.add_argument('--paraphrase_no_exist_behavior', default='skip', type=str)
args = parser.parse_args()

model_size, watermark_fraction = Path(args.output_file).stem.split("_")[:2], float(Path(args.output_file).stem.split("_")[5])
if args.base_model:
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, torch_dtype=torch.float16)
else:
    tokenizer = AutoTokenizer.from_pretrained("-".join(model_size))

# read args.output_file
with open(args.output_file, "r") as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]

if os.path.exists(args.watermark_cache):
    with open(args.watermark_cache, "r") as f:
        cache = json.load(f)
    # save a copy of cache as a backup
    with open(args.watermark_cache + ".bak", "w") as f:
        json.dump(cache, f)
else:
    cache = {}

sim_gold, sim_pp0, sim_cache, sim_fn = load_sim_stuff(args)

if args.base_model.startswith("facebook/opt"):
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

    print_accuracies(acc_gen, acc_gold, acc_pp0, sim_gold, sim_pp0, args)

    if cache1 or cache2 or cache3:
        # save cache
        with open(args.watermark_cache, "w") as f:
            json.dump(cache, f)

    # if sim_cache1 or sim_cache2 or sim_cache3:
    #     # save cache
    #     with open(args.sim_cache, "wb") as f:
    #         pickle.dump(sim_cache, f)

stats = get_roc(acc_gold, acc_gen)
stats2 = get_roc(acc_gold, acc_pp0)

print_tpr_target(stats[0], stats[1], "gen", args.target_fpr, acc_gold)
print_tpr_target(stats2[0], stats2[1], "pp0", args.target_fpr, acc_gold)

# with open("detect-plots/watermark.pkl", 'wb') as f:
#     pickle.dump((plot1, plot2), f)
