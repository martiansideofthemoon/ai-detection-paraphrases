## Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![arxiv](https://img.shields.io/badge/arXiv-2303.13408-b31b1b.svg)](https://arxiv.org/abs/2303.13408)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This is the official repository for our new [preprint](https://arxiv.org/pdf/2303.13408.pdf), "Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense".

### Updates

* (May 2023) The non-contextual ablated version of DIPPER (Section 6.1 of paper) is now available on the HuggingFace model hub! ([link](https://huggingface.co/kalpeshk2011/dipper-paraphraser-xxl-no-context))
* (April 2023) We have now released our paraphraser DIPPER on the HuggingFace model hub ([link](https://huggingface.co/kalpeshk2011/dipper-paraphraser-xxl))!
* (April 2023) Benchmark data, preprocessed paraphrases and scripts to reproduce the paper's experiments are now available!

### Running the paraphraser model (DIPPER)

**Requirements**

Since DIPPER is a 11B parameter model, please use a GPU with at least 40GB of memory to reproduce the experiments in the paper. Lower precision approximations or DeepSpeed optimizations may also be fine on lower memory GPUs, but we have not tested them in our experiments.

```
# required (for paraphrasing)
pip install torch transformers sklearn nltk
pip install --editable .

# optional (needed for some detection experiments)
pip install openai rankgen retriv sentencepiece
```

**Model Download**

*DIPPER from HuggingFace*

HuggingFace Model Hub links -
full model: https://huggingface.co/kalpeshk2011/dipper-paraphraser-xxl  
ablated model without context: https://huggingface.co/kalpeshk2011/dipper-paraphraser-xxl-no-context

Script: [`dipper_paraphrases/paraphrase_minimal.py`](dipper_paraphrases/paraphrase_minimal.py)

*DIPPER manual download*

Checkpoint: https://drive.google.com/file/d/1LJJ1P5X2An0kMn8WAAAJBmxBuNS-5GiK/view?usp=sharing  
To run this downloaded model, in [`dipper_paraphrases/paraphrase_minimal.py`](dipper_paraphrases/paraphrase_minimal.py), uncomment the line `dp = DipperParaphraser(model="...")` and specify your model checkpoint path.

*SIM model*

You could optionally download the SIM model from Wieting et al. 2021 for calculating semantic similarity of the paraphrased outputs. Download the two files in [this link](https://drive.google.com/drive/folders/1rOOYF3ioDD_Nm0sduhD6ZE1xdUQQKqvG?usp=sharing) and place them in [`dipper_paraphrases/sim`](dipper_paraphrases/sim).

*T5X versions*

Please see our official Google Research release here: https://github.com/google-research/google-research/tree/master/dipper

**Verify DIPPER is working**

Please run the script [`dipper_paraphrases/paraphrase_minimal.py`](dipper_paraphrases/paraphrase_minimal.py) and compare the outputs with [`sample_outputs.md`](sample_outputs.md). The greedy decoded outputs should exactly match, while the top_p samples will have some differences from the sample outputs but have higher diversity.

**(IMPORTANT) paraphraser differences from paper**

There are two minor differences between the actual model and the paper's description:

1. Our model uses `<sent> ... </sent>` tags instead of `<p> ... </p>` tags.

2. The lexical and order diversity codes used by the actual model correspond to "similarity" rather than "diversity". For a diversity of X, please use the control code value `100 - X`. In other words, L60-O60 in the paper corresponds to `lex = 40, order = 40` as the control code input to the model.

This is all documented in our minimal sample script to run DIPPER, [`dipper_paraphrases/paraphrase_minimal.py`](dipper_paraphrases/paraphrase_minimal.py), and also in footnote 6 in our [paper](https://arxiv.org/pdf/2303.13408.pdf).

### Reproducing experiments in the paper

Dataset: Download the folders `open-generation-data` and `lfqa-data` from [this Google Drive link](https://drive.google.com/drive/folders/1mPROenBB0fzLO9AX4fe71k0UYv0xt3X1?usp=share_link). Place them in your root folder. Reproducing the experiments in the paper has three steps. We have already done Step 1 and Step 2 and added preprocessed data to Google Drive link.

**Step 1: Generating text from large language models**

Use the scripts [`dipper_paraphrases/generate_gpt2.py`](dipper_paraphrases/generate_gpt3.py), or [`dipper_paraphrases/generate_gpt3.py`](dipper_paraphrases/generate_gpt3.py), or [`dipper_paraphrases/generate_opt.py`](dipper_paraphrases/generate_opt.py) as shown below,

```
# for no watermarking
python dipper_paraphrases/generate_gpt2.py --strength 0.0 --dataset lfqa-data/inputs.jsonl --output_dir lfqa-data

# for including watermarking
python dipper_paraphrases/generate_gpt2.py --strength 2.0 --dataset lfqa-data/inputs.jsonl --output_dir lfqa-data
```

You can speed this up by parallelizing it across multiple GPUs on SLURM using the code below. Please read the script before using parallelization, it will likely need modifications depending on your specific SLURM setup.

```
python dipper_paraphrases/parallel/schedule.py --command "python dipper_paraphrases/generate_gpt2.py --strength 0.0 --dataset lfqa-data/inputs.jsonl --output_dir lfqa-data" --partition gpu-preempt --num_shards 8

# after completion
python dipper_paraphrases/parallel/merge.py --input_pattern "lfqa-data/gpt2_xl_strength_2.0_frac_0.5_300_len_top_p_0.9.jsonl.shard_*"
```

**Step 2: Paraphrasing text generated by large language models**

Use the scripts [`dipper_paraphrases/paraphrase.py`](dipper_paraphrases/paraphrase.py) as shown below,

```
python dipper_paraphrases/paraphrase.py --output_file lfqa-data/gpt2_xl_strength_2.0_frac_0.5_300_len_top_p_0.9.jsonl --model kalpeshk2011/dipper-paraphraser-xxl
```

You can also parallelize this in a manner identical to Stage 1.

**Step 3: Run AI-text detectors**

Use any of the scripts to run various detectors: [`dipper_paraphrases/detect_*.py`](dipper_paraphrases) as follows. Each script caches the processed data (such as API calls) and will run a lot quicker the next time. Note that the GPTZero and OpenAI experiments need access to API keys, see [`dipper_paraphrases/utils.py`](dipper_paraphrases/utils.py) for details.

```
python dipper_paraphrases/detect_watermark.py --output_file lfqa-data/gpt2_xl_strength_2.0_frac_0.5_300_len_top_p_0.9.jsonl_pp --detector_cache lfqa-data/watermark_cache.json
python dipper_paraphrases/detect_openai.py --output_file lfqa-data/gpt2_xl_strength_0.0_frac_0.5_300_len_top_p_0.9.jsonl_pp --detector_cache lfqa-data/openai_cache.json
python dipper_paraphrases/detect_gptzero.py --output_file lfqa-data/gpt2_xl_strength_0.0_frac_0.5_300_len_top_p_0.9.jsonl_pp --detector_cache lfqa-data/gptzero_cache.json
python dipper_paraphrases/detect_detectgpt.py --base_model "facebook/opt-13b" --output_file lfqa-data/opt_13b_strength_0.0_frac_0.5_300_len_top_p_0.9.jsonl_pp --detector_cache lfqa-data/detectgpt_cache_opt.json
python dipper_paraphrases/detect_rankgen.py --output_file lfqa-data/gpt2_xl_strength_0.0_frac_0.5_300_len_top_p_0.9.jsonl_pp --detector_cache lfqa-data/rankgen_cache.json
python dipper_paraphrases/detect_retrieval.py --output_file lfqa-data/gpt2_xl_strength_0.0_frac_0.5_300_len_top_p_0.9.jsonl_pp --retrieval_corpus pooled --technique bm25
```

We recommend reporting true positive rates at a false positive rate of 1% instead of ROC curves, as discussed in the paper. This will be printed by the script. Nevertheless, the full ROC curves will be stored in `roc_plots`, use [`dipper_paraphrases/plot_roc.py`](dipper_paraphrases/plot_roc.py) to plot them.

Since DetectGPT takes a while to run, it may be helpful to shard the DetectGPT experiments using the parallel scripts of the previous two steps. Use [`dipper_paraphrases/parallel/merge_json.py`](dipper_paraphrases/parallel/merge_json.py) to merge the cache. Set `--base_model none` to ignore loading the LLM and just rely on cached results. Also, don't forget the `--base_model` flag in DetectGPT runs, see the code for more details.

For the scaled retrieval experiments, please see [`dipper_paraphrases/detect_retrieval_scale_*.py`](dipper_paraphrases). Please contact me if you want the raw data accompanying this experiment (email me at kalpesh@cs.umass.edu).

### Citation

If you found the code, model or paper useful please cite:

```
@article{krishna2023paraphrasing,
  title={Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense},
  author={Krishna, Kalpesh and Song, Yixiao and Karpinska, Marzena and Wieting, John and Iyyer, Mohit},
  journal={arXiv preprint arXiv:2303.13408},
  year={2023}
}
```
