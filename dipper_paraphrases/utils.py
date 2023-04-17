import json
import numpy as np
import hashlib
import os
import functools
import sys
import numpy as np
import openai
import re
import string
import collections as cll
import time
import requests
import pickle
import torch
from transformers import LogitsWarper
from dipper_paraphrases.detect_gpt.run import get_perturbation_results
from sklearn.metrics import roc_curve, auc

from dipper_paraphrases.sim.models import load_model
from dipper_paraphrases.sim.embed_sentences import embed_all, similarity


GPTZERO_API_KEYS = [os.getenv(f"GPTZERO_API_KEY{i}") for i in range(1, 11)]
gptzero_idx = 0

lfqa_database = None
with open("lfqa-data/inputs.jsonl", "r") as f:
    lfqa_database = [json.loads(x) for x in f.read().strip().split("\n")]
    lfqa_database = {dd["prefix"]: dd for dd in lfqa_database}

def get_longest_answer(prefix):
    if "Answer the following question in 200-250 words.\n" in prefix:
        prefix = prefix.split("Answer the following question in 200-250 words.\n")[1]
    assert prefix in lfqa_database
    dd = lfqa_database[prefix]
    longest_answer = None
    longest_answer_len = 0
    for _, ans_set in dd["comments"].items():
        for ans in ans_set:
            if len(ans) > longest_answer_len:
                longest_answer = ans
                longest_answer_len = len(ans)
    return longest_answer.replace("@@@@@@", "\n").strip()

def load_shared_args(parser):
    parser.add_argument('--output_file', default="lfqa-data/gpt2_xl_strength_0.0_frac_0.5_300_len_top_p_0.9.jsonl_pp")
    parser.add_argument('--base_model', default="gpt2-xl", type=str)
    parser.add_argument('--detector_cache', default="lfqa-data/openai_cache.json")
    parser.add_argument('--num_shards', default=1, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--sim_threshold', default=0.75, type=float)
    parser.add_argument('--target_fpr', default=0.01, type=float)
    parser.add_argument('--sim_cache', default="lfqa-data/sim_cache.pkl")
    parser.add_argument('--paraphrase_type', default='lex_40_order_100', type=str)

def print_accuracies(acc_gen, acc_gold, acc_pp0, sim_gold, sim_pp0, args):
    print(f"PP0 ({len(acc_pp0)} instances) Acc: {np.mean([x > args.threshold for x in acc_pp0]):.4f}, sim: {np.mean(sim_pp0):.4f}, joint: {np.mean([x < args.threshold and y for x, y in zip(acc_pp0, sim_pp0)])}")
    print(f"Gold ({len(acc_gold)} instances) Acc: {np.mean([x > args.threshold for x in acc_gold]):.4f}, sim: {np.mean(sim_gold):.4f}, joint: {np.mean([x < args.threshold and y for x, y in zip(acc_gold, sim_gold)])}")
    print(f"Gen acc: {sum([x > args.threshold for x in acc_gen]) / len(acc_gen):.4f}, {len(acc_gen)} instances")

def do_sim_stuff(gen_tokens, gold_tokens, pp0_tokens, sim_cache, sim_fn, args, sim_gold, sim_pp0):
    if sim_fn is None:
        sim_gold.append(False)
        sim_pp0.append(False)
        return
    gen_vec, _ = sim_fn(gen_tokens, sim_cache)
    gold_vec, _ = sim_fn(gold_tokens, sim_cache)
    pp0_vec, _ = sim_fn(pp0_tokens, sim_cache)
    sim_gold.append(similarity(gen_vec, gold_vec) > args.sim_threshold)
    sim_pp0.append(similarity(gen_vec, pp0_vec) > args.sim_threshold)

def load_sim_stuff(args):
    sim_gold = []
    sim_pp0 = []

    if os.path.exists(args.sim_cache):
        with open(args.sim_cache, "rb") as f:
            sim_cache = pickle.load(f)
        # save a copy of cache as a backup
        with open(args.sim_cache + ".bak", "wb") as f:
            pickle.dump(sim_cache, f)
    else:
        sim_cache = {}

    # load paraphrase model
    if os.path.exists("dipper_paraphrases/sim/model.para.lc.100.pt"):
        paraphrase_model = load_model("dipper_paraphrases/sim/model.para.lc.100.pt")
        paraphrase_model.eval()
        embedder = functools.partial(embed_all, model=paraphrase_model, disable=True)
        sim_fn = functools.partial(get_sim_vectors, embedder=embedder)
    else:
        sim_fn = None
    return sim_gold, sim_pp0, sim_cache, sim_fn

def get_sim_vectors(sequence, cache, embedder):
    cache_updated = False
    if sequence in cache:
        gen_vec = cache[sequence]
    else:
        gen_vec = embedder(sentences=[sequence])[0]
        cache[sequence] = gen_vec
        cache_updated = True
    return gen_vec, cache_updated

def print_tpr_target(fpr, tpr, name, target_fpr):
    indices = None
    for i in range(len(fpr)):
        if fpr[i] >= target_fpr:
            if i == 0:
                indices = [i]
            else:
                indices = [i-1, i]
            break

    if indices is None:
        print(f"{name} TPR at {target_fpr*100}% FPR: {tpr[-1]}. FPR is too high.")
    else:
        tpr_values = [tpr[i] for i in indices]
        print(f"{name} TPR at {target_fpr*100}% FPR: {np.mean(tpr_values) * 100:5.1f}%")

def get_roc(human_scores, machine_scores, max_fpr=1.0):
    fpr, tpr, _ = roc_curve([0] * len(human_scores) + [1] * len(machine_scores), human_scores + machine_scores)
    fpr_auc = [x for x in fpr if x <= max_fpr]
    tpr_auc = tpr[:len(fpr_auc)]
    roc_auc = auc(fpr_auc, tpr_auc)
    return fpr.tolist(), tpr.tolist(), float(roc_auc), float(roc_auc) * (1.0 / max_fpr)

def retrieval_sim_detect(generation, cache, embedder, all_vecs, index_path):
    cache_updated = False
    key = index_path + "---" + generation
    if key not in cache:
        cand_vecs = torch.tensor(
            embedder(sentences=[generation], disable=True)
        )
        # get similarity scores
        sim_matrix = torch.matmul(cand_vecs, all_vecs.T)
        norm_matrix = torch.norm(cand_vecs, dim=1, keepdim=True) * torch.norm(all_vecs, dim=1, keepdim=True).T
        sim_scores = sim_matrix / norm_matrix

        score = torch.max(sim_scores).item()

        cache[key] = score
        cache_updated = True
    else:
        score = cache[key]

    return score, cache_updated

def retrieval_bm25_detect(generation, cache, se, index_path):
    cache_updated = False
    key = index_path + "---" + generation
    if key not in cache:
        res1 = se.search(generation)
        if len(res1) == 0:
            score = 0.0
        else:
            score = f1_score(generation, res1[0]['text'])[2]
        cache[key] = score
        cache_updated = True
    else:
        score = cache[key]

    return score, cache_updated

def rankgen_detect(generation, prefix, cache, encoder):
    cache_updated = False
    key = prefix + "---" + generation
    if key not in cache:
        prefix_vector = encoder.encode(prefix, vectors_type="prefix")["embeddings"]
        suffix_vector = encoder.encode(generation, vectors_type="suffix")["embeddings"]
        score = (prefix_vector * suffix_vector).sum().item()
        cache[key] = score
        cache_updated = True
    else:
        score = cache[key]

    return -1 * score, cache_updated

def detectgpt_detect(generation, cache, mask_model, mask_tokenizer, base_model, base_tokenizer):
    cache_updated = False
    if generation not in cache:
        if len(mask_tokenizer.tokenize(generation)) > 510:
            gen_input = mask_tokenizer.decode(mask_tokenizer(generation)['input_ids'][:510])
        else:
            gen_input = generation
        output = get_perturbation_results(gen_input, mask_model, mask_tokenizer, base_model, base_tokenizer)
        cache[generation] = output
        cache_updated = True
    else:
        output = cache[generation]

    z_score = (output['original_ll'] - output['mean_perturbed_ll']) / output['std_perturbed_ll']
    return z_score, cache_updated

def get_openai_response(prompt: str, max_tokens = 150, temperature = 0.7, top_p = 1, n = 1, logprobs = 1, stop = None, echo = True):
    response = openai.Completion.create(engine="text-davinci-003",
                                        prompt=prompt,
                                        max_tokens=max_tokens,
                                        temperature = temperature,
                                        top_p=top_p,
                                        n=n,
                                        logprobs=logprobs,
                                        stop=stop,
                                        echo=echo)
    output = response['choices'][0]['text']
    assert output.startswith(prompt)
    gen_text = output[len(prompt):].strip()
    return gen_text

def get_chatgpt_qa_response(prompt_text, max_tokens=1000):
    messages = [{"role":"system", "content": "You are a helpful assistant that answers the question provided."},
                {"role":"user", "content": prompt_text}]
    response = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                messages = messages,
                max_tokens = max_tokens
    )
    return response['choices'][0]['message']['content']

def get_chatgpt_completion_response(prompt_text, max_tokens):
    messages = [{"role":"system", "content": "You are a helpful assistant that continues the passage from the sentences provided."},
                {"role":"user", "content": prompt_text}]
    response = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                messages = messages,
                max_tokens = max_tokens
    )
    return response['choices'][0]['message']['content']

def openai_detect(prompt, cache):
    prompt = prompt + "<|disc_score|>"
    cache_updated = False
    if prompt in cache:
        top_logprobs = cache[prompt]
    else:
        response = openai.Completion.create(engine="model-detect-v2",
                                            prompt=prompt,
                                            max_tokens=1,
                                            temperature=1,
                                            top_p=1,
                                            n=1,
                                            logprobs=5,
                                            stop="\n",
                                            stream=False)
        top_logprobs = response["choices"][0]["logprobs"]["top_logprobs"][0]
        cache[prompt] = top_logprobs
        cache_updated = True
    if "\"" in top_logprobs:
        quote_logprob = np.exp(top_logprobs["\""])
    elif "!" in top_logprobs:
        quote_logprob = 1.0 - np.exp(top_logprobs["!"])
    else:
        print("No quote or exclamation mark found in top logprobs")
        return 0.5
    return quote_logprob, cache_updated


def gptzero_detect(generation, cache):
    global gptzero_idx
    obj = {"document": generation}
    headers = {"X-Api-Key": GPTZERO_API_KEYS[gptzero_idx]}
    cache_updated = False
    if generation in cache:
        response = cache[generation]
    else:
        response = requests.post("https://api.gptzero.me/v2/predict/text", json=obj, headers=headers)
        if response.status_code != 200:
            print(response.text)
            gptzero_idx  = (gptzero_idx + 1) % len(GPTZERO_API_KEYS)
            print("Switching to GPT Zero API key ", gptzero_idx)
            headers = {"X-Api-Key": GPTZERO_API_KEYS[gptzero_idx]}
            response = requests.post("https://api.gptzero.me/v2/predict/text", json=obj, headers=headers)
        try:
            response = json.loads(response.text)['documents'][0]
        except:
            print(response.text)
            print("exiting...")
            print("Final key used was ", gptzero_idx)
            sys.exit(1)
        cache[generation] = {
            "average_generated_prob": response["average_generated_prob"],
            "completely_generated_prob": response["completely_generated_prob"],
        }
        cache_updated = True
    return response["completely_generated_prob"], cache_updated


def watermark_detect(sequence, cache, watermark_fraction, vocab_size):
    seq_key = "_".join([str(x) for x in sequence])
    cache_updated = False
    if seq_key in cache:
        z_val = cache[seq_key]
    else:
        total_tokens = len(sequence)
        green_tokens = 0
        for i in range(1, total_tokens):
            green_list = get_green_list([sequence[i - 1]], watermark_fraction, vocab_size)[0]

            if green_list[sequence[i]]:
                green_tokens += 1
        z_val = get_z(green_tokens, total_tokens - 1, watermark_fraction)
        cache[seq_key] = z_val
        cache_updated = True

    return z_val, cache_updated

def hash_fn(x):
    # solution from https://stackoverflow.com/questions/67219691/python-hash-function-that-returns-32-or-64-bits
    x = np.int64(x)
    return int.from_bytes(hashlib.sha256(x).digest()[:4], 'little')

def get_z(num_green, total, fraction):
    return (num_green - fraction * total) / np.sqrt(fraction * (1 - fraction) * total)

def get_green_list(last_token_ids, fraction, vocab_size):
    all_masks = []
    for last_token_id in last_token_ids:
        random_seed = hash_fn(last_token_id)
        rng = np.random.default_rng(random_seed)
        mask = np.full(vocab_size, False)
        mask[:int(fraction * vocab_size)] = True
        rng.shuffle(mask)
        all_masks.append(mask)
    return np.array(all_masks)

def entropy(p):
    """Calculate the entropy of a distribution p using pytorch."""
    return -torch.sum(p * torch.log(p))

def spike_entropy(p, modulus=1):
    """Calculate the spike entropy of a distribution p using pytorch."""
    return torch.sum(p / (1.0 + modulus * p))

class WatermarkLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] for watermarking distributions with green-listed tokens. Implementation of https://arxiv.org/abs/2301.10226.
    Args:
        fraction (`float`):
            The fraction of the distribution to be green-listed.
        strength (`float`):
            The strength of the green-listing. A higher value means that the green-listed tokens will have a higher logit score.
    """

    def __init__(self, fraction: float = 0.5, strength: float = 2.0, debug=False):
        self.fraction = fraction
        self.strength = strength
        self.debug = debug
        self.entropies = []
        self.spike_entropies = []
        self.mean_logit = []
        self.watermark_probability_mass = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        green_list_mask = get_green_list(input_ids[:, -1].tolist(), self.fraction, scores.shape[-1])
        green_list_mask = torch.tensor(green_list_mask, dtype=torch.float32, device=scores.device)
        watermark = self.strength * green_list_mask
        final_logits = scores + watermark
        if self.debug:
            distribution = scores.softmax(-1)
            self.entropies.append(entropy(distribution).item())
            self.spike_entropies.append(spike_entropy(distribution).item())
            self.mean_logit.append(torch.mean(scores).item())
            self.watermark_probability_mass.append(torch.sum(final_logits.softmax(-1) * green_list_mask).item())
            if len(self.entropies) % 1000 == 0:
                print(f"Vocab size: {scores.shape[-1]}")
                print(f"Entropy ({len(self.entropies)} tokens): {np.mean(self.entropies):.4f}")
                print(f"Spike Entropy: {np.mean(self.spike_entropies):.4f}")
                print(f"Mean logit: {np.mean(self.mean_logit):.4f}")
                print(f"Watermark probability mass: {np.mean(self.watermark_probability_mass):.4f}")
        return final_logits

def form_partitions(dataset, num_shards):
    p_indices = np.round(np.linspace(0, len(dataset), num_shards + 1))
    p_indices = [int(x) for x in p_indices]
    partitions = [dataset[p_indices[i]:p_indices[i + 1]] for i in range(len(p_indices) - 1)]
    assert len(partitions) == num_shards
    return partitions


def truncate(text):
    """Truncate text to the last full sentence."""
    last_punc = 0
    if "." in text:
        last_punc = max(last_punc, text.rindex("."))
    if "?" in text:
        last_punc = max(last_punc, text.rindex("?"))
    if "!" in text:
        last_punc = max(last_punc, text.rindex("!"))
    if last_punc != 0:
        text = text[:last_punc + 1]
    return text

def postprocess(outputs, tokenizer):
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @classmethod
    def postprocess(cls, input_str):
        input_str = input_str.replace("<h>", cls.HEADER)
        input_str = input_str.replace("<blue>", cls.OKBLUE)
        input_str = input_str.replace("<green>", cls.OKGREEN)
        input_str = input_str.replace("<yellow>", cls.WARNING)
        input_str = input_str.replace("<red>", cls.FAIL)
        input_str = input_str.replace("</>", cls.ENDC)
        input_str = input_str.replace("<b>", cls.BOLD)
        input_str = input_str.replace("<u>", cls.UNDERLINE)
        input_str = input_str.replace("<clean>", "")
        return input_str

def f1_score(prediction, ground_truth, gram=1, stopwords=None):
    """Calculate word level F1 score."""
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()
    prediction_tokens = [
        " ".join(prediction_tokens[i:i + gram])
        for i in range(0, len(prediction_tokens) - gram + 1)
    ]
    ground_truth_tokens = [
        " ".join(ground_truth_tokens[i:i + gram])
        for i in range(0, len(ground_truth_tokens) - gram + 1)
    ]

    if stopwords:
        prediction_tokens = [x for x in prediction_tokens if x not in stopwords]
        ground_truth_tokens = [x for x in ground_truth_tokens if x not in stopwords]

    if not prediction_tokens or not ground_truth_tokens:
        return 1.0, 1.0, 1.0, True
    common = cll.Counter(prediction_tokens) & cll.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0, False
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1, False

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
