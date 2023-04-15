## Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense

This is the official repository for our new [preprint](https://arxiv.org/pdf/2303.13408.pdf). We have currently released our model checkpoint and a script to paraphrase data (see details below). We plan to clean up the repository further and release the model checkpoint on the HuggingFace model hub in the next 2 weeks.

### Requirements

```
# required
pip install torch
pip install transformers
pip install sklearn
pip install nltk
# optional
pip install openai
pip install rankgen
```

### Running the Model

**Checkpoint**: https://drive.google.com/file/d/1LJJ1P5X2An0kMn8WAAAJBmxBuNS-5GiK/view?usp=sharing
**Script**: [`dipper_paraphrases/paraphrase.py`](dipper_paraphrases/paraphrase.py)

Please read the important note in the next section to understand the differences from the paper.

### Paraphraser notes (differences from paper)

There are two minor differences between the actual model and the paper's description:

1. Our model uses `<sent> ... </sent>` tags instead of `<p> ... </p>` tags.

2. The lexical and order diversity codes used by the actual model correspond to "similarity" rather than "diversity". For a diversity of X, please use the control code value `100 - X`. In other words, L60-O60 in the paper corresponds to `lex = 40, order = 40` as the control code input to the model.

This is all documented in our sample script to run DIPPER, [`dipper_paraphrases/paraphrase.py`](dipper_paraphrases/paraphrase.py).

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
