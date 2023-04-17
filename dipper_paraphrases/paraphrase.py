import argparse
import json
import nltk
import time
import os
import tqdm

from nltk.tokenize import sent_tokenize

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

from utils import form_partitions

nltk.download('punkt')


parser = argparse.ArgumentParser()
parser.add_argument('--output_file', default="lfqa-data/gpt2_xl_strength_2.0_frac_0.5_300_len_top_p_0.9.jsonl")
parser.add_argument('--model', type=str, default='kalpeshk2011/dipper-paraphraser-xxl', help='data to paraphrase')
parser.add_argument("--refresh", action='store_true', help="Renew the data")
parser.add_argument('--num_shards', type=int, default=1, help='data to paraphrase')
parser.add_argument('--local_rank', type=int, default=0, help='data to paraphrase')
parser.add_argument('--sent_interval', type=int, default=3, help='data to paraphrase')
args = parser.parse_args()

# read args.output_file
with open(args.output_file, "r") as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]

if "no-ctx" in args.model:
    args.output_file = args.output_file + "_no_ctx"

if args.sent_interval == 1:
    args.output_file = args.output_file + "_sent"

if args.num_shards > 1:
    partitions = form_partitions(data, args.num_shards)
    data = partitions[args.local_rank]
    output_file = f'{args.output_file}_pp.shard_{args.local_rank}'
else:
    output_file = args.output_file + "_pp"

if os.path.exists(output_file):
    with open(output_file, "r") as f:
        num_output_points = len([json.loads(x) for x in f.read().strip().split("\n")])
else:
    num_output_points = 0
print(f"Skipping {num_output_points} points")

time1 = time.time()
tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
model = T5ForConditionalGeneration.from_pretrained(args.model)
print("Model loaded in ", time.time() - time1)
model.cuda()
model.eval()

# iterate over data and tokenize each instance
for idx, dd in tqdm.tqdm(enumerate(data), total=len(data)):
    if idx < num_output_points:
        continue
    # tokenize prefix
    if args.refresh or "paraphrase_outputs" not in dd:
        paraphrase_outputs = {}
        for lex, order in [(40, 40), (40, 100), (60, 100), (80, 100)]:
            if isinstance(dd['gen_completion'], str):
                input_gen = dd['gen_completion'].strip()
            else:
                input_gen = dd['gen_completion'][0].strip()

            # remove spurious newlines
            input_gen = " ".join(input_gen.split())
            sentences = sent_tokenize(input_gen)
            prefix = " ".join(dd['prefix'].replace("\n", " ").split())
            output_text = ""
            final_input_text = ""

            for sent_idx in range(0, len(sentences), args.sent_interval):
                curr_sent_window = " ".join(sentences[sent_idx:sent_idx + args.sent_interval])
                if "no-ctx" in args.model:
                    final_input_text = f"lexical = {lex}, order = {order} <sent> {curr_sent_window} </sent>"
                else:
                    final_input_text = f"lexical = {lex}, order = {order} {prefix} <sent> {curr_sent_window} </sent>"

                if idx == 0 and lex == 40 and order == 40:
                    print(final_input_text)

                final_input = tokenizer([final_input_text], return_tensors="pt")
                final_input = {k: v.cuda() for k, v in final_input.items()}

                with torch.inference_mode():
                    outputs = model.generate(**final_input, do_sample=True, top_p=0.75, top_k=None, max_length=512)
                outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                prefix += " " + outputs[0]
                output_text += " " + outputs[0]

            paraphrase_outputs[f"lex_{lex}_order_{order}"] = {
                "final_input": final_input_text,
                "output": [output_text.strip()],
                "lex": lex,
                "order": order
            }
        dd["paraphrase_outputs"] = paraphrase_outputs
    with open(output_file, "a") as f:
        f.write(json.dumps(dd) + "\n")
