import tqdm, json, os, random, glob
import numpy as np
from retriv import SearchEngine
import torch

MEMORY_LIMIT = 4000000

def get_index_path(args, num_generations=None):
    if num_generations is None:
        num_generations = args.num_generations

    if args.truncate_corpus:
        index_path = f"{args.index_path_prefix}_{args.total_tokens}_truncate_{num_generations // 1000000}M_gens"
    else:
        index_path = f"{args.index_path_prefix}_no_truncate_{num_generations // 1000000}M_gens"
    return index_path

def create_index(args, initial_generations=None, embedder=None):
    all_files = glob.glob(args.data_files)
    all_files.sort()
    random.seed(43)
    random.shuffle(all_files)

    shard_num = 0
    # iterate over data and tokenize each instance
    if initial_generations is not None:
        gens_list = [json.dumps({"text": x, "id": i}) for i, x in enumerate(initial_generations)]
        global_counter = len(gens_list)
    else:
        gens_list = []
        global_counter = -1

    data = []

    if args.num_generations < 10000 and args.technique == "sim":
        index_path = get_index_path(args, num_generations=15000000)
    else:
        index_path = get_index_path(args)

    if os.path.exists(f"{index_path}/all.jsonl"):
        print("Index already exists at", index_path)
        return

    os.makedirs(index_path, exist_ok=True)

    print("Creating index at", index_path)

    for fnum, f in tqdm.tqdm(enumerate(all_files), total=len(all_files)):
        if global_counter >= args.num_generations:
            break

        with open(f, "r") as f:
            data.extend([x.split("\t") for x in f.read().strip("\n").split("\n")])
        # keep reading until we have enough data to fill the memory limit
        if fnum < len(all_files) - 1 and len(data) < MEMORY_LIMIT:
            continue
        random.shuffle(data)

        for dd in data:
            gen_tokens = dd[3].split()
            if len(gen_tokens) < args.total_tokens:
                continue
            global_counter += 1

            if args.truncate_corpus:
                gen_tokens = " ".join(gen_tokens[:args.total_tokens])
            else:
                gen_tokens = " ".join(gen_tokens)

            gens_list.append(json.dumps({
                "text": gen_tokens,
                "id": global_counter
            }))

            if global_counter >= args.num_generations:
                break

        with open(f"{index_path}/all.jsonl", "a") as f:
            f.write("\n".join(gens_list) + "\n")

        if args.technique != "bm25_retriv" and not os.path.exists(f"{index_path}/{shard_num}.npy"):
            print(f"Indexing gens: part {shard_num}")
            vectors = embedder(sentences=[json.loads(x)["text"] for x in gens_list])
            # save the vectors
            np.save(f"{index_path}/{shard_num}.npy", vectors)

        gens_list = []
        data = []
        shard_num += 1

    print("Total index size:", global_counter)
    if args.technique == "bm25_retriv":
        se = SearchEngine(index_name=index_path.replace("/", "_"))
        se = se.index_file(path=f"{index_path}/all.jsonl")


def load_index(args):
    # load the index first
    if args.num_generations < 10000 and args.technique == "sim":
        index_path = get_index_path(args, num_generations=15000000)
    else:
        index_path = get_index_path(args)

    if args.technique == "bm25_retriv":
        engine = SearchEngine.load(index_name=index_path.replace("/", "_"))
        print(f"Finished loading search engine")
    else:
        # load the index first
        global_counter = 0
        shard_num = 0
        gen_vecs = []
        while os.path.exists(f"{index_path}/{shard_num}.npy"):
            if global_counter >= args.num_generations:
                break
            print(f"Loading gens: part {shard_num}")
            curr_tensor = torch.Tensor(np.load(f"{index_path}/{shard_num}.npy"))
            gen_vecs.append(curr_tensor)

            global_counter += len(gen_vecs[-1])
            shard_num += 1
        gen_vecs = torch.cat(gen_vecs, dim=0)
        engine = gen_vecs[:args.num_generations]
        print(f"Finished loading {len(engine)} gens")
    return engine, index_path

def load_candidates(cands_dict, args):
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
    return cands
