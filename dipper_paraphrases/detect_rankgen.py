import argparse
import json
import pickle
from functools import partial
import os
import tqdm
from utils import rankgen_detect, get_roc, print_tpr_target, print_accuracies, load_shared_args, do_sim_stuff, load_sim_stuff
from rankgen import RankGenEncoder


parser = argparse.ArgumentParser()
load_shared_args(parser)
parser.add_argument('--output_file', default="watermark_outputs/gpt3_davinci_003_300_len.jsonl_pp")
parser.add_argument('--threshold', default=-7.0, type=float)
parser.add_argument('--rankgen_cache', default="watermark_outputs/rankgen_cache.json")
parser.add_argument('--min_words', default=50, type=float)
args = parser.parse_args()

# read args.output_file
with open(args.output_file, "r") as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]

if os.path.exists(args.rankgen_cache):
    with open(args.rankgen_cache, "r") as f:
        cache = json.load(f)
    # save a copy of cache as a backup
    with open(args.rankgen_cache + ".bak", "w") as f:
        json.dump(cache, f)
else:
    cache = {}

acc_gen = []
acc_gold = []
acc_pp0 = []
num_paraphrase_pts = len([x for x in data if "paraphrase_outputs" in x])

# load rankgen model
rankgen_encoder = RankGenEncoder("kalpeshk2011/rankgen-t5-xl-all")

sim_gold, sim_pp0, sim_cache, sim_fn = load_sim_stuff(args)

# iterate over data and tokenize each instance
for idx, dd in tqdm.tqdm(enumerate(data), total=num_paraphrase_pts):
    # tokenize prefix
    if "paraphrase_outputs" not in dd:
        continue
    if isinstance(dd['gen_completion'], str):
        gen_tokens = dd['gen_completion'].split()
    else:
        gen_tokens = dd['gen_completion'][0].split()
    gold_tokens = dd['gold_completion'].split()

    pp_target_tokens = dd['paraphrase_outputs']['lex_40_order_100']['output'][0].split()
    pp0_tokens = dd['paraphrase_outputs'][args.paraphrase_type]['output'][0].split()

    if len(gen_tokens) < args.min_words or len(pp_target_tokens) < args.min_words or len(gold_tokens) < args.min_words:
        continue

    min_len = min(len(gen_tokens), len(gold_tokens), len(pp_target_tokens))

    do_sim_stuff(" ".join(gen_tokens), " ".join(gold_tokens), " ".join(pp0_tokens), sim_cache, sim_fn, args, sim_gold, sim_pp0)

    gold_tokens = " ".join(gold_tokens[:min_len])
    pp0_tokens = " ".join(pp0_tokens[:min_len])
    gen_tokens = " ".join(gen_tokens[:min_len])


    gen_prob, cache1 = rankgen_detect(gen_tokens, dd['prefix'], cache, rankgen_encoder)
    gold_prob, cache2 = rankgen_detect(gold_tokens, dd['prefix'], cache, rankgen_encoder)
    pp0_prob, cache3 = rankgen_detect(pp0_tokens, dd['prefix'], cache, rankgen_encoder)

    acc_gen.append(gen_prob)
    acc_gold.append(gold_prob)
    acc_pp0.append(pp0_prob)

    do_sim_stuff(gen_tokens, gold_tokens, pp0_tokens, sim_cache, sim_fn, args, sim_gold, sim_pp0)

    print_accuracies(acc_gen, acc_gold, acc_pp0, sim_gold, sim_pp0, args)

    # write cache
    if cache1 or cache2 or cache3:
        with open(args.rankgen_cache, "w") as f:
            json.dump(cache, f)

stats = get_roc(acc_gold, acc_gen)
stats2 = get_roc(acc_gold, acc_pp0)

print_tpr_target(stats[0], stats[1], "gen", args.target_fpr, acc_gold)
print_tpr_target(stats2[0], stats2[1], "pp0", args.target_fpr, acc_gold)

# with open("detect-plots/openai.pkl", 'wb') as f:
#     pickle.dump((stats, stats2), f)
