import argparse
import json
import nltk
import numpy as np
from transformers import AutoTokenizer
from utils import get_green_list, get_z, Bcolors

nltk.download('punkt')


parser = argparse.ArgumentParser()
parser.add_argument('--output_file', default="watermark_outputs/example_generate.jsonl_pp")
parser.add_argument('--threshold', default=4.0, type=float)
parser.add_argument('--sim_threshold', default=0.75, type=float)
parser.add_argument('--paraphrase_type', default='lex_40_order_100', type=str)
args = parser.parse_args()

watermark_fraction = 0.5
tokenizer = AutoTokenizer.from_pretrained(f"gpt2-xl")

# read args.output_file
with open(args.output_file, "r") as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")][0]

paraphrase = "There were never any reports of them mixing with people. It is believed they live in an unspoiled environment surrounded by mountains and protected by a thick clump of wattle. The herd has a regal look to it, with the magic, rainbow-colored coat and golden feathers. Some of them are said to be capable of speaking many languages. They eat deer and goats, because they are the descendants of those animals that sprang from fierce, dangerous animals and have horns long enough to \"eat\" these animals."
original = "They have never been known to mingle with humans. Today, it is believed these unicorns live in an unspoilt environment which is surrounded by mountains. Its edge is protected by a thick wattle of wattle trees, giving it a majestic appearance. Along with their so-called miracle of multicolored' coat, their golden coloured feather makes them look like mirages. Some of them are rumored to be capable of speaking a large amount of different languages. They feed on elk and goats as they were selected from those animals that possess a fierceness to them, and can \"eat\" them with their long horns."
prefix = data['prefix']
last_prefix_token = tokenizer(prefix, add_special_tokens=True)["input_ids"][-1]
gen_tokens = [last_prefix_token] + tokenizer(original, add_special_tokens=False)["input_ids"]
if "paraphrase_outputs" in data:
    pp0_tokens = [last_prefix_token] + tokenizer(paraphrase, add_special_tokens=False)["input_ids"]
    pp0_green_tokens = 0

gen_green_tokens = 0
annotated_output = ""
annotated_output_pp = ""

for i in range(1, len(gen_tokens)):
    gen_green_list = get_green_list([gen_tokens[i - 1]], watermark_fraction, tokenizer.vocab_size)[0]

    if gen_green_list[gen_tokens[i]]:
        annotated_output += f"<green>{tokenizer.decode([gen_tokens[i]])}</>"
        gen_green_tokens += 1
    else:
        annotated_output += tokenizer.decode([gen_tokens[i]])

if "paraphrase_outputs" in data:
    for i in range(1, len(pp0_tokens)):
        pp0_green_list = get_green_list([pp0_tokens[i - 1]], watermark_fraction, tokenizer.vocab_size)[0]

        if pp0_green_list[pp0_tokens[i]]:
            annotated_output_pp += f"<green>{tokenizer.decode([pp0_tokens[i]])}</>"
            pp0_green_tokens += 1
        else:
            annotated_output_pp += tokenizer.decode([pp0_tokens[i]])

print(Bcolors.postprocess(annotated_output))
print("\n")
print(Bcolors.postprocess(annotated_output_pp))

import pdb; pdb.set_trace()
pass
gen_z = get_z(gen_green_tokens, len(gen_tokens) - 1, watermark_fraction)
print(gen_z)

if "paraphrase_outputs" in data:
    pp0_z = get_z(pp0_green_tokens, len(pp0_tokens) - 1, watermark_fraction)
    acc_pp0.append(pp0_z > args.threshold)

    print(f"PP0 ({len(acc_pp0)} instances) Acc: {np.mean(acc_pp0):.4f}, sim: {np.mean(sim_pp0):.4f}, joint: {np.mean([not x and y for x, y in zip(acc_pp0, sim_pp0)])}")

print(f"Gold ({len(acc_gold)} instances) Acc: {np.mean(acc_gold):.4f}, sim: {np.mean(sim_gold):.4f}, joint: {np.mean([not x and y for x, y in zip(acc_gold, sim_gold)])}")
print(f"Gen acc: {np.mean(acc_gen):.4f}, {len(acc_gen)} instances")
