import argparse
import json
import pickle
from functools import partial
import os
import tqdm
from utils import openai_detect, get_roc, print_tpr_target, load_sim_stuff, do_sim_stuff, print_accuracies, load_shared_args, get_longest_answer
import numpy as np


parser = argparse.ArgumentParser()
load_shared_args(parser)
parser.add_argument('--output_file', default="watermark_outputs/gpt3_300_len.jsonl_pp")
parser.add_argument('--threshold', default=0.98, type=float)
parser.add_argument('--detector_cache', default="watermark_outputs/detector_cache.json")
parser.add_argument('--total_chars', default=1000, type=float)
args = parser.parse_args()

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

acc_gen = []
acc_gold = []
acc_pp0 = []

num_paraphrase_pts = len([x for x in data if "paraphrase_outputs" in x])


sim_gold, sim_pp0, sim_cache, sim_fn = load_sim_stuff(args)

# iterate over data and tokenize each instance
for idx, dd in tqdm.tqdm(enumerate(data), total=num_paraphrase_pts):
    # tokenize prefix
    if "paraphrase_outputs" not in dd:
        continue
    if isinstance(dd['gen_completion'], str):
        gen_tokens = dd['gen_completion']
    else:
        gen_tokens = dd['gen_completion'][0]

    pp_target_tokens = dd['paraphrase_outputs']['lex_40_order_100']['output'][0]
    pp0_tokens = dd['paraphrase_outputs'][args.paraphrase_type]['output'][0]

    # if len(pp_target_tokens) < 1000 and len(gen_tokens) > 1000:
    #     import pdb; pdb.set_trace()
    #     pass
    if "lfqa" in args.output_file:
        gold_tokens = get_longest_answer(dd['prefix'])
    else:
        gold_tokens = dd['gold_completion']

    if len(gen_tokens) < args.total_chars or len(pp_target_tokens) < args.total_chars or len(gold_tokens) < args.total_chars:
        continue

    min_len = min(len(gen_tokens), len(gold_tokens), len(pp_target_tokens))
    do_sim_stuff(gen_tokens, gold_tokens, pp0_tokens, sim_cache, sim_fn, args, sim_gold, sim_pp0)

    gold_tokens = gold_tokens[:min_len]
    pp0_tokens = pp0_tokens[:min_len]
    gen_tokens = gen_tokens[:min_len]

    gen_prob, cache1 = openai_detect(gen_tokens, cache)
    gold_prob, cache2 = openai_detect(gold_tokens, cache)
    pp0_prob, cache3 = openai_detect(pp0_tokens, cache)

    acc_gen.append(gen_prob)
    acc_gold.append(gold_prob)
    acc_pp0.append(pp0_prob)

    print_accuracies(acc_gen, acc_gold, acc_pp0, sim_gold, sim_pp0, args)

    # write cache
    if cache1 or cache2 or cache3:
        with open(args.detector_cache, "w") as f:
            json.dump(cache, f)

stats = get_roc(acc_gold, acc_gen)
stats2 = get_roc(acc_gold, acc_pp0)

print_tpr_target(stats[0], stats[1], "gen", args.target_fpr, acc_gold)
print_tpr_target(stats2[0], stats2[1], "pp0", args.target_fpr, acc_gold)

# with open("detect-plots/openai.pkl", 'wb') as f:
#     pickle.dump((stats, stats2), f)
