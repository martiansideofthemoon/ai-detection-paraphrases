import argparse
import json
import glob
import numpy as np
import tqdm
import os
import torch
from functools import partial
import random
from utils import get_roc, print_tpr_target
from dipper_paraphrases.sim.models import load_model
from dipper_paraphrases.sim.embed_sentences import embed_all


parser = argparse.ArgumentParser()
parser.add_argument('--data_files', default="rankgen_train/wiki_256_128_sent_boundary_with_gen_negative/*.tsv")
parser.add_argument('--sim_threshold', default=0.7, type=float)
parser.add_argument('--target_fpr', default=0.01, type=float)
parser.add_argument('--total_tokens', default=60, type=int)
parser.add_argument('--memory_limit', default=4000000, type=int)
parser.add_argument('--index_path_prefix', default="rankgen_train/wiki_index", type=str)
parser.add_argument('--num_generations', default=10000000000000, type=int)
parser.add_argument('--paraphrase_type', default='lex_40_order_100', type=str)
parser.add_argument('--candidates_file', default='candidates_wiki.jsonl_pp', type=str)
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


# load paraphrase model
sim_model = load_model("dipper_paraphrases/sim/paraphrase-at-scale/model.para.lc.100.pt")
sim_model.eval()
embedder = partial(embed_all, model=sim_model)


def index_and_build_candidates(all_files):
    shard_num = 0
    global_counter = -1
    # iterate over data and tokenize each instance
    data = []
    cands = []

    for fnum, f in tqdm.tqdm(enumerate(all_files), total=len(all_files)):
        if global_counter + 1 >= args.num_generations:
            break

        with open(f, "r") as f:
            data.extend([x.split("\t") for x in f.read().strip("\n").split("\n")])
        # keep reading until we have enough data to fill the memory limit
        if fnum < len(all_files) - 1 and len(data) < args.memory_limit:
            continue
        random.shuffle(data)

        gens_list = []
        for dd in data:
            gen_tokens = dd[3].split()
            if len(gen_tokens) < args.total_tokens:
                continue
            global_counter += 1
            gens_list.append(" ".join(gen_tokens[:args.total_tokens]))

            if len(cands_dict) == 0 and len(cands) < 30000 and len(dd[1].split()) > args.total_tokens:
                gold_tokens = dd[1].split()
                cands.append({
                    "prefix": dd[0],
                    "human": " ".join(gold_tokens[:args.total_tokens]),
                    "gen_completion": [" ".join(gen_tokens)],
                    "idx": global_counter
                })
            elif dd[0] in cands_dict and " ".join(gen_tokens) == cands_dict[dd[0]]['gen_completion'][0] and " ".join(dd[1].split()[:args.total_tokens]) == cands_dict[dd[0]]['human']:
                cands_dict[dd[0]]['idx'] = global_counter
                cands.append(cands_dict[dd[0]])

        data = []
        if not os.path.exists(f"{args.index_path_prefix}_{args.total_tokens}_tokens_{shard_num}.npy"):
            print(f"Indexing gens: part {shard_num}")
            vectors = embedder(sentences=gens_list)
            # gen_vecs.append(vectors)
            # save the vectors
            np.save(f"{args.index_path_prefix}_{args.total_tokens}_tokens_{shard_num}.npy", vectors)
        shard_num += 1

    # write the candidates to a file
    with open(args.candidates_file + ".inp", "w") as f:
        f.write("\n".join([json.dumps(x) for x in cands]) + "\n")

def load_index_and_candidates(cands_dict):
    # load the index first
    global_counter = 0
    shard_num = 0
    gen_vecs = []
    while os.path.exists(f"{args.index_path_prefix}_{args.total_tokens}_tokens_{shard_num}.npy"):
        if global_counter >= args.num_generations:
            break
        print(f"Loading gens: part {shard_num}")
        curr_tensor = torch.Tensor(np.load(f"{args.index_path_prefix}_{args.total_tokens}_tokens_{shard_num}.npy"))
        gen_vecs.append(curr_tensor)

        global_counter += len(gen_vecs[-1])
        shard_num += 1
    print(f"Finished loading {global_counter} gens")
    gen_vecs = torch.cat(gen_vecs, dim=0)
    gen_vecs = gen_vecs[:args.num_generations]

    cands = []
    # load the candidates
    for val in cands_dict.values():
        pp0_tokens = val['paraphrase_outputs'][args.paraphrase_type]['output'][0].split()
        gold_tokens = val['human'].split()
        if len(pp0_tokens) >= args.total_tokens and len(gold_tokens) >= args.total_tokens:
            cands.append({
                "human": " ".join(gold_tokens[:args.total_tokens]),
                "paraphrase": " ".join(pp0_tokens[:args.total_tokens]),
                "idx": val["idx"]
            })
    # filter candidates with idx more than global_counter
    cands = [x for x in cands if x['idx'] <= args.num_generations]
    return gen_vecs, cands

# index_and_build_candidates(all_files)
# import pdb; pdb.set_trace()
# pass
gen_vecs, cands = load_index_and_candidates(cands_dict)
print("Number of gens: ", len(gen_vecs))
print("Number of candidates: ", len(cands))

# iterate over cands and get similarity scores
human_detect = []
human_detect_remove_prompt = []
human_same_prefix = []

paraphrase_detect = []
paraphrase_same_prefix = []
# gen_vec_norms = np.linalg.norm(gen_vecs, axis=1, keepdims=True).T
# gen_vecs = gen_vecs.T
gen_vec_norms = torch.norm(gen_vecs, dim=1, keepdim=True).t()
gen_vecs = gen_vecs.t()

for cand in tqdm.tqdm(cands):
    # cand_vecs = embedder(sentences=[cand["human"]], disable=True)
    cand_vecs = embedder(sentences=[cand["human"], cand["paraphrase"]], disable=True)
    cand_vecs = torch.Tensor(cand_vecs)
    # get similarity scores
    sim_matrix = torch.matmul(cand_vecs, gen_vecs)
    norm_matrix = torch.matmul(torch.norm(cand_vecs, dim=1, keepdim=True), gen_vec_norms)
    sim_scores = sim_matrix / norm_matrix
    # get the max similarity score
    max_sim_score = torch.max(sim_scores, dim=1).values
    argmax_sim_score = torch.argmax(sim_scores, dim=1)
    top_two = torch.topk(sim_scores[0], k=2)

    if top_two.indices[0].item() == cand["idx"]:
        human_detect_remove_prompt.append(top_two.values[1].item() > args.sim_threshold)
    else:
        human_detect_remove_prompt.append(top_two.values[0].item() > args.sim_threshold)

    human_detect.append(max_sim_score[0].item())
    paraphrase_detect.append(max_sim_score[1].item())

    human_same_prefix.append(cand["idx"] == argmax_sim_score[0].item())
    paraphrase_same_prefix.append(cand["idx"] == argmax_sim_score[1].item())

    if len(human_detect) % 20 == 0 or len(human_detect) == len(cands):
        print(f"Human detect ({len(human_detect)} instances): {np.mean([x > args.sim_threshold for x in human_detect]) * 100:.3f}, {np.mean(human_detect_remove_prompt) * 100:.3f} remove same prompt, {np.mean(human_same_prefix) * 100:.1f} argmax same prefix")
        print(f"Paraphrase detect ({len(paraphrase_detect)} instances): {np.mean([x > args.sim_threshold for x in paraphrase_detect]) * 100:.3f}, {np.mean(paraphrase_same_prefix) * 100:.1f} argmax same prefix\n")

        stats = get_roc(human_detect, paraphrase_detect)
        print_tpr_target(stats[0], stats[1], "paraphrase", args.target_fpr, human_detect)
