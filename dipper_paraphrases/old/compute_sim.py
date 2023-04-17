import argparse
import json
import numpy as np
import tqdm
from utils import load_shared_args, do_sim_stuff, load_sim_stuff


parser = argparse.ArgumentParser()
load_shared_args(parser)
parser.add_argument('--output_file', default="watermark_outputs/gpt3_davinci_003_300_len.jsonl_pp")
parser.add_argument('--min_words', default=50, type=int)
args = parser.parse_args()

# read args.output_file
with open(args.output_file, "r") as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]

sim_gold, sim_pp0, sim_cache, sim_fn = load_sim_stuff(args)

# iterate over data and tokenize each instance
for idx, dd in tqdm.tqdm(enumerate(data)):
    # tokenize prefix
    if "paraphrase_outputs" not in dd:
        continue

    if isinstance(dd['gen_completion'], str):
        gen_tokens = dd['gen_completion'].split()
    else:
        gen_tokens = dd['gen_completion'][0].split()
    gold_tokens = dd['gold_completion'].split()

    pp0_tokens = dd['paraphrase_outputs'][args.paraphrase_type]['output'][0].split()

    if len(gen_tokens) < args.min_words or len(pp0_tokens) < args.min_words or len(gold_tokens) < args.min_words:
        continue

    do_sim_stuff(" ".join(gen_tokens), " ".join(gold_tokens), " ".join(pp0_tokens), sim_cache, sim_fn, args, sim_gold, sim_pp0)

print("sim_gold: ", np.mean(sim_gold))
print("sim_pp0: ", np.mean(sim_pp0))
