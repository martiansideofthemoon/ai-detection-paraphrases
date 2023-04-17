import argparse
import json
import glob
import numpy as np
import tqdm
import os
import torch
import random
from utils import get_roc, print_tpr_target, f1_score
from retriv import SearchEngine
from retrieval_utils import load_index_and_candidates


parser = argparse.ArgumentParser()
parser.add_argument('--data_files', default="rankgen_train/pg19_256_128_sent_boundary_with_gen_negative/*.tsv")
parser.add_argument('--sim_threshold', default=0.7, type=float)
parser.add_argument('--target_fpr', default=0.01, type=float)
parser.add_argument('--total_tokens', default=60, type=int)
parser.add_argument('--memory_limit', default=4000000, type=int)
parser.add_argument('--index_path_prefix', default="rankgen_train/pg19_retriv", type=str)
parser.add_argument('--num_generations', default=15000000, type=int)
parser.add_argument('--paraphrase_type', default='lex_40_order_100', type=str)
parser.add_argument('--candidates_file', default='candidates_pg19.jsonl_pp.inp', type=str)
args = parser.parse_args()


all_files = glob.glob(args.data_files)
all_files.sort()
random.seed(43)
random.shuffle(all_files)

# read candidates file
if args.candidates_file is not None and os.path.exists(args.candidates_file):
    with open(args.candidates_file, "r") as f:
        cands = [json.loads(x) for x in f.read().strip("\n").split("\n")]
    cands_dict = {x['prefix']: x for x in cands}
    # assert len(cands_dict) == len(cands)
else:
    cands_dict = {}

# index_generations(all_files)
se, cands = load_index_and_candidates(cands_dict)

# iterate over cands and get similarity scores
human_detect = []
human_same_prefix = []

paraphrase_detect = []
paraphrase_same_prefix = []

for cand in tqdm.tqdm(cands):
    human_ret = se.search(cand["human"])[0]['text']
    paraphrase_ret = se.search(cand["paraphrase"])[0]['text']

    human_detect.append(f1_score(human_ret, cand["human"])[2])
    paraphrase_detect.append(f1_score(paraphrase_ret, cand["paraphrase"])[2])

    # human_same_prefix.append(cand["idx"] == argmax_sim_score[0].item())
    # paraphrase_same_prefix.append(cand["idx"] == argmax_sim_score[1].item())

    if len(human_detect) % 20 == 0 or len(human_detect) == len(cands):
        # print(f"Human detect ({len(human_detect)} instances): {np.mean([x > args.sim_threshold for x in human_detect]) * 100:.3f}, {np.mean(human_detect_remove_prompt) * 100:.3f} remove same prompt, {np.mean(human_same_prefix) * 100:.1f} argmax same prefix")
        # print(f"Paraphrase detect ({len(paraphrase_detect)} instances): {np.mean([x > args.sim_threshold for x in paraphrase_detect]) * 100:.3f}, {np.mean(paraphrase_same_prefix) * 100:.1f} argmax same prefix\n")

        stats = get_roc(human_detect, paraphrase_detect)
        print_tpr_target(stats[0], stats[1], "paraphrase", args.target_fpr, human_detect)
