import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--input_pattern', default="openwebtext_vectors/2016-06.pkl_0_small.pkl.matches_entity_*", type=str)
parser.add_argument('--output_file', default=None, type=str)
parser.add_argument('--skip_lines', default=None, type=str)
args = parser.parse_args()

files = glob.glob(args.input_pattern)
file_with_ids = [(int(f.split("_")[-1]), f) for f in files]
file_with_ids.sort(key=lambda x: x[0])

data = ""
size = 0
for file in file_with_ids:
    with open(file[1], "r") as f:
        curr_data = f.read()
    if args.skip_lines is None:
        data += curr_data
        size += len(curr_data.strip().split("\n"))
    elif args.skip_lines == "half":
        curr_lines = curr_data.strip().split("\n")
        curr_lines = curr_lines[len(curr_lines)//2:]
        data += "\n".join(curr_lines) + "\n"

if args.output_file is not None:
    output_file = args.output_file
else:
    output_file = ".".join(args.input_pattern.split(".")[:-1])
print(output_file)
print(f"Total merged size = {size}")
with open(output_file, "w") as f:
    f.write(data)
