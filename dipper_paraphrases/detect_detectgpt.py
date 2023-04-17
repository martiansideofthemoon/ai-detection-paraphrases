import argparse
import json
import tqdm
import functools
import os
import numpy as np
import pickle

from utils import load_shared_args, detectgpt_detect, get_roc, form_partitions, print_tpr_target, load_sim_stuff, do_sim_stuff, print_accuracies, get_longest_answer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM


parser = argparse.ArgumentParser()
load_shared_args(parser)
parser.add_argument('--threshold', default=3, type=float)
parser.add_argument('--mask_model', default="t5-3b", type=str)
parser.add_argument('--min_words', default=50, type=float)
parser.add_argument('--skip_prompt', action='store_true')
args = parser.parse_args()

if args.base_model is not None and args.base_model not in ["none", "None", "text-davinci-003"]:
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    model.eval()
    model.cuda()
elif args.base_model == "text-davinci-003":
    model = "text-davinci-003"
    tokenizer = None
else:
    model = None
    tokenizer = None

print(f"Loading mask model of type {args.mask_model}...")
if args.mask_model is not None and args.mask_model not in ["none", "None"]:
    mask_tokenizer = AutoTokenizer.from_pretrained(args.mask_model)
    mask_model = AutoModelForSeq2SeqLM.from_pretrained(args.mask_model)
    mask_model.eval()
    mask_model.cuda()
else:
    mask_model = None
    mask_tokenizer = None

detect_fn = functools.partial(detectgpt_detect, mask_model=mask_model, mask_tokenizer=mask_tokenizer, base_model=model, base_tokenizer=tokenizer)

# read args.output_file
with open(args.output_file, "r") as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]

if args.num_shards > 1:
    partitions = form_partitions(data, args.num_shards)
    data = partitions[args.local_rank]
    args.detector_cache = f'{args.detector_cache}.shard_{args.local_rank}'

acc_gen = []
acc_gold = []
acc_pp0 = []

sim_gold, sim_pp0, sim_cache, sim_fn = load_sim_stuff(args)

num_paraphrase_pts = len([x for x in data if "paraphrase_outputs" in x])

if os.path.exists(args.detector_cache):
    with open(args.detector_cache, "r") as f:
        cache = json.load(f)
    # save a copy of cache as a backup
    with open(args.detector_cache + ".bak", "w") as f:
        json.dump(cache, f)
else:
    cache = {}
print(f"Cache contains {len(cache)} entries.")

# iterate over data and tokenize each instance
for idx, dd in tqdm.tqdm(enumerate(data), total=num_paraphrase_pts):
    # tokenize prefix
    if "paraphrase_outputs" not in dd:
        continue
    if isinstance(dd['gen_completion'], str):
        gen_tokens = dd['gen_completion'].split()
    else:
        gen_tokens = dd['gen_completion'][0].split()

    if "lfqa" in args.output_file:
        gold_tokens = get_longest_answer(dd['prefix']).split()
    else:
        gold_tokens = dd['gold_completion'].split()

    pp0_tokens = dd['paraphrase_outputs'][args.paraphrase_type]['output'][0].split()
    pp_target_tokens = dd['paraphrase_outputs']['lex_40_order_100']['output'][0].split()

    if len(gen_tokens) < args.min_words or len(pp_target_tokens) < args.min_words or len(gold_tokens) < args.min_words:
        continue
    min_len = min(len(gen_tokens), len(gold_tokens), len(pp_target_tokens))

    do_sim_stuff(" ".join(gen_tokens), " ".join(gold_tokens), " ".join(pp0_tokens), sim_cache, sim_fn, args, sim_gold, sim_pp0)

    gold_tokens = " ".join(gold_tokens[:min_len])
    gen_tokens = " ".join(gen_tokens[:min_len])
    pp0_tokens = " ".join(pp0_tokens[:min_len])

    if not args.skip_prompt:
        gold_tokens = dd['prefix'].strip() + " " + gold_tokens
        gen_tokens = dd['prefix'].strip() + " " + gen_tokens
        pp0_tokens = dd['prefix'].strip() + " " + pp0_tokens

    try:
        gen_z, cache1 = detect_fn(gen_tokens, cache)
        gold_z, cache2 = detect_fn(gold_tokens, cache)
        pp0_z, cache3 = detect_fn(pp0_tokens, cache)
    except:
        continue

    acc_gen.append(gen_z)
    acc_gold.append(gold_z)
    acc_pp0.append(pp0_z)

    # uncomment below to get accuracies at the currently set value of --threshold
    # print_accuracies(acc_gen, acc_gold, acc_pp0, sim_gold, sim_pp0, args)

    if cache1 or cache2 or cache3:
        # save cache
        with open(args.detectgpt_cache, "w") as f:
            json.dump(cache, f)


stats = get_roc(acc_gold, acc_gen)
stats2 = get_roc(acc_gold, acc_pp0)
print_tpr_target(stats[0], stats[1], "generation", args.target_fpr)
print_tpr_target(stats2[0], stats2[1], "paraphrase", args.target_fpr)

with open("roc_plots/detectgpt.pkl", 'wb') as f:
    pickle.dump((stats, stats2), f)