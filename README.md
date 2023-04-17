## Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![arxiv](https://img.shields.io/badge/arXiv-2303.13408-b31b1b.svg)](https://arxiv.org/abs/2303.13408)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This is the official repository for our new [preprint](https://arxiv.org/pdf/2303.13408.pdf), "Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense". We have currently released our model checkpoint and a script to paraphrase data (see details below).

### Updates

* (April 2023) We have now released DIPPER on the HuggingFace model hub ([link](https://huggingface.co/kalpeshk2011/dipper-paraphraser-xxl))!

### Running the paraphraser model (DIPPER)

**Requirements**

Since DIPPER is a 11B parameter model, please use a GPU with at least 40GB of memory to reproduce the experiments in the paper. Lower precision approximations or DeepSpeed optimizations may also be fine on lower memory GPUs, but we have not tested them in our experiments.

```
# required
pip install torch transformers sklearn nltk
pip install --editable .
# optional
pip install openai rankgen
```

**Model Download**

(from HuggingFace)

HuggingFace Model Hub link: https://huggingface.co/kalpeshk2011/dipper-paraphraser-xxl  
Script: [`dipper_paraphrases/paraphrase_minimal.py`](dipper_paraphrases/paraphrase_minimal.py)

(manual download)

Checkpoint: https://drive.google.com/file/d/1LJJ1P5X2An0kMn8WAAAJBmxBuNS-5GiK/view?usp=sharing  
To run this downloaded model, in [`dipper_paraphrases/paraphrase_minimal.py`](dipper_paraphrases/paraphrase_minimal.py), uncomment the line `dp = DipperParaphraser(model="...")` and specify your model checkpoint path.

**Verify DIPPER is working**

Please run the script [`dipper_paraphrases/paraphrase_minimal.py`](dipper_paraphrases/paraphrase_minimal.py) and compare the outputs with [`sample_outputs.md`](sample_outputs.md). The greedy decoded outputs should exactly match, while the top_p samples will have some differences from the sample outputs but have higher diversity.

**(IMPORTANT) paraphraser differences from paper**

There are two minor differences between the actual model and the paper's description:

1. Our model uses `<sent> ... </sent>` tags instead of `<p> ... </p>` tags.

2. The lexical and order diversity codes used by the actual model correspond to "similarity" rather than "diversity". For a diversity of X, please use the control code value `100 - X`. In other words, L60-O60 in the paper corresponds to `lex = 40, order = 40` as the control code input to the model.

This is all documented in our minimal sample script to run DIPPER, [`dipper_paraphrases/paraphrase_minimal.py`](dipper_paraphrases/paraphrase_minimal.py), and also in footnote 6 in our [paper](https://arxiv.org/pdf/2303.13408.pdf).

### Reproducing experiments in the paper

Dataset: 

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
