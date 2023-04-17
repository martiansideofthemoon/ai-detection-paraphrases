import argparse
import json
import pickle
import mauve
import os

parser = argparse.ArgumentParser()
parser.add_argument('--output_file', default="watermark_outputs/gpt2_xl_strength_2.0_frac_0.5.jsonl_pp")
parser.add_argument('--num_instances', default=1000000, type=int)

args = parser.parse_args()

# read args.output_file
with open(args.output_file, "r") as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]

data = data[:args.num_instances]

all_gen = []
all_human = []

# iterate over data and tokenize each instance
for idx, dd in enumerate(data):
    # tokenize prefix
    # if "paraphrase_outputs" not in dd:
    #     continue
    prefix = dd['prefix']
    assert not isinstance(dd['gen_completion'], str)
    gen_tokens = dd['gen_completion'][0]
    gold_tokens = dd['gold_completion']
    num_gen_words = len(gen_tokens.split())
    num_gold_words = len(gold_tokens.split())
    min_tokens = min(num_gen_words, num_gold_words)

    # keep only min_tokens
    gen_tokens = " ".join(gen_tokens.split()[:min_tokens])
    gold_tokens = " ".join(gold_tokens.split()[:min_tokens])

    all_gen.append(prefix.strip() + " " + gen_tokens.strip())
    all_human.append(prefix.strip() + " " + gold_tokens.strip())

if os.path.exists(f"{args.output_file}.mauve"):
    print("Loading existing Mauve file...")
    with open(f"{args.output_file}.mauve", "rb") as f:
        mauve1 = pickle.load(f)
else:
    mauve1 = mauve.compute_mauve(p_text=all_gen, q_text=all_human, device_id=0, max_text_length=768, verbose=False)
    with open(f"{args.output_file}.mauve", "wb") as f:
        pickle.dump(mauve1, f)

print(mauve1.mauve)

