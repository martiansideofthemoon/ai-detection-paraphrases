import argparse
import json
import nltk
import numpy as np
import tqdm
from functools import partial
from pathlib import Path
from retriv import SearchEngine

from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
from utils import print_tpr_target, get_roc, f1_score, normalize_answer
from dipper_paraphrases.sim.models import load_model
from dipper_paraphrases.sim.embed_sentences import embed_all, similarity

nltk.download('punkt')


parser = argparse.ArgumentParser()
parser.add_argument('--output_file', default="watermark_outputs/gpt2_xl_strength_0.0_frac_0.5_300_len_top_p_0.9.jsonl_pp")
parser.add_argument('--threshold', default=0.75, type=float)
parser.add_argument('--target_fpr', default=0.01, type=float)
parser.add_argument('--total_tokens', default=200, type=int)
parser.add_argument('--paraphrase_type', default='lex_40_order_100', type=str)
parser.add_argument('--technique', default='sim', type=str)
parser.add_argument('--base_model', default='gpt2-xl', type=str)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.base_model)

corpora_files = [
    "watermark_outputs/gpt2_xl_strength_0.0_frac_0.5_300_len_top_p_0.9.jsonl_pp",
    "watermark_outputs/opt_13b_strength_0.0_frac_0.5_300_len_top_p_0.9.jsonl_pp",
    "watermark_outputs/gpt3_davinci_003_300_len.jsonl_pp",
]
corpora_files = [args.output_file]

# load paraphrase model
paraphrase_model = load_model("dipper_paraphrases/sim/paraphrase-at-scale/model.para.lc.100.pt")
paraphrase_model.eval()
embedder = partial(embed_all, model=paraphrase_model)

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
        gen_tokens = tokenizer(gen_tokens, add_special_tokens=False)["input_ids"]
        gold_tokens = tokenizer(dd['gold_completion'], add_special_tokens=False)["input_ids"]

        if len(gen_tokens) <= 10:
            continue

        gens_list.append(tokenizer.decode(gen_tokens[:truncate_tokens]))
        if len(gen_tokens) < args.total_tokens:
            continue
        if "paraphrase_outputs" not in dd:
            continue
        pp0_tokens = tokenizer(dd['paraphrase_outputs'][args.paraphrase_type]['output'][0], add_special_tokens=False)["input_ids"]
        if len(gold_tokens) >= args.total_tokens and len(pp0_tokens) >= args.total_tokens and op_file == args.output_file:
            cands.append({
                "generation": tokenizer.decode(gen_tokens[:truncate_tokens]),
                "human": tokenizer.decode(gold_tokens[:truncate_tokens]),
                "paraphrase": tokenizer.decode(pp0_tokens[:truncate_tokens]),
                "idx": idx
            })

print("Number of gens: ", len(gens_list))
print("Number of candidates: ", len(cands))

# index the cand_gens
if args.technique == "sim":
    gen_vecs = embedder(sentences=gens_list, disable=True)
elif args.technique == "bm25":
    bm25 = BM25Okapi([normalize_answer(x).split() for x in gens_list])
elif args.technique in ["bm25_retriv", "bm25_retriev_score_unigram"]:
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

    elif args.technique == "unigram":
        sim_scores = np.array([
            [f1_score(cand["human"], gen)[2] for gen in gens_list],
            [f1_score(cand["paraphrase"], gen)[2] for gen in gens_list],
            [f1_score(cand["generation"], gen)[2] for gen in gens_list]
        ])
        max_sim_score = np.max(sim_scores, axis=1)
        human_detect.append(max_sim_score[0])
        paraphrase_detect.append(max_sim_score[1])
        generation_detect.append(max_sim_score[2])

    elif args.technique == "bm25":
        res1 = bm25.get_scores(normalize_answer(cand["human"]).split())
        res2 = bm25.get_scores(normalize_answer(cand["paraphrase"]).split())
        res3 = bm25.get_scores(normalize_answer(cand["generation"]).split())
        human_detect.append(np.max(res1))
        paraphrase_detect.append(np.max(res2))
        generation_detect.append(np.max(res3))

    elif args.technique == "bm25_retriv":
        res1 = se.search(cand["human"])[0]['score']
        res2 = se.search(cand["paraphrase"])[0]['score']
        res3 = se.search(cand["generation"])[0]['score']
        human_detect.append(res1)
        paraphrase_detect.append(res2)
        generation_detect.append(res3)

    elif args.technique == "bm25_retriev_score_unigram":
        res1 = se.search(cand["human"])[0]
        res2 = se.search(cand["paraphrase"])[0]
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

    # print("Human detect: ", np.mean([x > args.threshold for x in human_detect]))
    # print("Paraphrase detect: ", np.mean([x > args.threshold for x in paraphrase_detect]))
    # print("Generation detect: ", np.mean([x > args.threshold for x in generation_detect]))
import pdb; pdb.set_trace()
pass
plot1 = get_roc(human_detect, paraphrase_detect)
print_tpr_target(plot1[0], plot1[1], "pp0", args.target_fpr, human_detect)
plot2 = get_roc(human_detect, generation_detect)
print_tpr_target(plot2[0], plot2[1], "gen", args.target_fpr, human_detect)

# with open("detect-plots/s2.pkl", 'wb') as f:
#     pickle.dump((plot1, plot1), f)
