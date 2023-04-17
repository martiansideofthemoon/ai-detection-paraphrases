import argparse
import glob
import json

parser = argparse.ArgumentParser()
parser.add_argument('--input_pattern', default="openwebtext_vectors/2016-06.pkl_0_small.pkl.matches_entity_*", type=str)
parser.add_argument('--output_file', default=None, type=str)
parser.add_argument('--skip_lines', default=None, type=str)
args = parser.parse_args()

files = glob.glob(args.input_pattern)
files = [f for f in files if not f.endswith(".bak")]
file_with_ids = [(int(f.split("_")[-1]), f) for f in files]
file_with_ids.sort(key=lambda x: x[0])
print("Number of files: {}".format(len(file_with_ids)))
data = {}
mismatches = 0
for file in file_with_ids:
    with open(file[1], "r") as f:
        curr_data = json.loads(f.read())
    # merge dictionaries
    for k, v in curr_data.items():
        if k in data:
            mismatches += 1
        else:
            data[k] = v

if args.output_file is not None:
    output_file = args.output_file
else:
    output_file = ".".join(args.input_pattern.split(".")[:-1])
print(output_file)
print("Number of merged entries: {}. Mismatches: {}".format(len(data), mismatches))
with open(output_file, "w") as f:
    f.write(json.dumps(data))
