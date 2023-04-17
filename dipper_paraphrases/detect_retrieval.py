import argparse
import json
import nltk
import numpy as np
import tqdm
from functools import partial
from retriv import SearchEngine
import pickle

from transformers import AutoTokenizer
from utils import print_tpr_target, get_roc, f1_score, load_shared_args
from dipper_paraphrases.sim.models import load_model
from dipper_paraphrases.sim.embed_sentences import embed_all

nltk.download('punkt')


parser = argparse.ArgumentParser()
load_shared_args(parser)
parser.add_argument('--threshold', default=0.75, type=float)
parser.add_argument('--total_tokens', default=50, type=int)
parser.add_argument('--technique', default='bm25', type=str)
parser.add_argument('--retrieval_corpus', default='single', type=str)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.base_model)

if args.retrieval_corpus == "pooled":
    folder = "open-generation-data" if "open-generation-data" in args.output_file else "lfqa-data"
    corpora_files = [
        f"{folder}/gpt2_xl_strength_0.0_frac_0.5_300_len_top_p_0.9.jsonl_pp",
        f"{folder}/opt_13b_strength_0.0_frac_0.5_300_len_top_p_0.9.jsonl_pp",
        f"{folder}/gpt3_davinci_003_300_len.jsonl_pp",
    ]
    assert args.output_file in corpora_files
else:
    corpora_files = [args.output_file]

# load SIM model
sim_model = load_model("dipper_paraphrases/sim/model.para.lc.100.pt")
sim_model.eval()
embedder = partial(embed_all, model=sim_model)

gens_list = []
cands = []
truncate_tokens = 10000 #args.total_tokens

for op_file in corpora_files:
    # read args.output_file
    with open(op_file, "r") as f:
        data = [json.loads(x) for x in f.read().strip().split("\n")]

    # iterate over data and tokenize each instance
    for idx, dd in tqdm.tqdm(enumerate(data), total=len(data)):
        if isinstance(dd['gen_completion'], str):
            gen_tokens = dd['gen_completion']
        else:
            gen_tokens = dd['gen_completion'][0]
        gen_tokens = gen_tokens.split()
        gold_tokens = dd['gold_completion'].split()

        if len(gen_tokens) <= args.total_tokens:
            continue

        if "paraphrase_outputs" not in dd:
            continue

        pp0_tokens = dd['paraphrase_outputs'][args.paraphrase_type]['output'][0].split()
        pp_target_tokens = dd['paraphrase_outputs']['lex_40_order_100']['output'][0].split()
        min_len = min(len(gold_tokens), len(pp_target_tokens), len(gen_tokens))
        gens_list.append(" ".join(gen_tokens[:min_len]))

        if len(gold_tokens) >= args.total_tokens and len(pp_target_tokens) >= args.total_tokens and op_file == args.output_file:
            cands.append({
                "generation": " ".join(gen_tokens[:min_len]),
                "human": " ".join(gold_tokens[:min_len]),
                "paraphrase": " ".join(pp0_tokens[:min_len]),
                "idx": idx
            })

print("Number of gens: ", len(gens_list))
print("Number of candidates: ", len(cands))

# index the cand_gens
if args.technique == "sim":
    gen_vecs = embedder(sentences=gens_list, disable=True)
elif args.technique == "bm25":
    collection = [{"text": x, "id": f"doc_{i}"} for i, x in enumerate(gens_list)]
    se = SearchEngine(f"index-{args.output_file.split('/')[1]}")
    se.index(collection)

# iterate over cands and get similarity scores
human_detect = []
paraphrase_detect = []
generation_detect = []

for cand in tqdm.tqdm(cands):
    if args.technique == "sim":
        cand_vecs = embedder(sentences=[cand["human"], cand["paraphrase"], cand["generation"]], disable=True)
        # get similarity scores
        sim_matrix = np.matmul(cand_vecs, gen_vecs.T)
        norm_matrix = np.linalg.norm(cand_vecs, axis=1, keepdims=True) * np.linalg.norm(gen_vecs, axis=1, keepdims=True).T
        sim_scores = sim_matrix / norm_matrix

        max_sim_score = np.max(sim_scores, axis=1)
        human_detect.append(max_sim_score[0])
        paraphrase_detect.append(max_sim_score[1])
        generation_detect.append(max_sim_score[2])

    elif args.technique == "bm25":
        res1 = se.search(cand["human"])[0]
        try:
            res2 = se.search(cand["paraphrase"])[0]
        except:
            res2 = {"text": ""}
        res3 = se.search(cand["generation"])[0]
        human_detect.append(
            f1_score(cand["human"], res1['text'])[2]
        )
        paraphrase_detect.append(
            f1_score(cand["paraphrase"], res2['text'])[2]
        )
        generation_detect.append(
            f1_score(cand["generation"], res3['text'])[2]
        )


plot1 = get_roc(human_detect, paraphrase_detect)
print_tpr_target(plot1[0], plot1[1], "paraphrase", args.target_fpr)
plot2 = get_roc(human_detect, generation_detect)
print_tpr_target(plot2[0], plot2[1], "generation", args.target_fpr)

with open("roc_plots/s2_sim.pkl", 'wb') as f:
    pickle.dump((plot1, plot1), f)
