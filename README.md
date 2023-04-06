## Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense

This is the official repository for our new [preprint](https://arxiv.org/pdf/2303.13408.pdf).
We plan to release our pretrained paraphraser and data in the next 2-3 weeks. Please email me (kalpesh@cs.umass.edu) if you want quicker access to the model.

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
