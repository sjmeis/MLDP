# MLDP
This repository contains the code used in the work: *A Comparative Analysis of Word-Level Metric Differential Privacy: Benchmarking The Privacy-Utility Trade-off* (LREC-COLING 2024). In particular, provided is the code for five word-level MLDP mechanisms, previously unavailable publicly.

## Included Mechanisms
In the provided class code (**MLDP**), you will find five runnable mechanisms:
- *MultivariateCalibrated*: [paper](https://dl.acm.org/doi/10.1145/3336191.3371856)
- *TruncatedGumbel*: [paper](https://journals.flvc.org/FLAIRS/article/view/128463)
- *VickreyMechanism*: [paper](https://aclanthology.org/2021.privatenlp-1.2/)
- *TEM*: [paper](https://epubs.siam.org/doi/10.1137/1.9781611977653.ch99)
- *Mahalanobis*: [paper](https://aclanthology.org/2020.privatenlp-1.2/)
- *SynTF*: [paper](https://dl.acm.org/doi/10.1145/3209978.3210008)

Note that the code for SanText is not included as it is already publicly available [here](https://github.com/xiangyue9607/SanText).

## Getting Started
Getting started is as simple as importing the module provided in this repository (`MLDP.py`):

`import sys`

`sys.path.insert(0, "/path/to/MLDP.py")`

`import MLDP`

## Basic Usage (example)
For all mechanisms, you have the option to employ `faiss` ([link](https://github.com/facebookresearch/faiss)), which can most likely speed up the above mechanisms.

Basic usage for all mechanisms (*M*) besides SynTF:

`mechanism = MLDP.M(epsilon=1, use_faiss=False)`

`perturbed_word = mechanism.replace_word(orig_word)`

For SynTF, an extra step must be taken to initialize the mechanism, namely to initialize the TF-IDF vectorizer. To do this, pass in the `data` parameter, which represents a list (or other iterable) of documents. This corpus of documents can most likely be the documents which you wish to privatize.

`mechanism = MLDP.SynTF(epsilon=1, data=CORPUS)`

`perturbed_word = mechanism.replace_word(orig_word)`

## Embedding Model
By default, we use the `glove.840B.300d` embedding model (included in the `data` folder), which has been filtered down to a fixed vocabulary (`data/vocab.txt`). We have also included a smaller 50-d embedding model. Both included models are based on the GloVe models provided at this [link](https://nlp.stanford.edu/projects/glove/).

If you would like to change the default embedding model, please change line 28 of `MLDP.py` (global `EMBED` variable) to the correct model path. Note that the embedding model file must follow the file format as necessitated by the `gensim` library, namely with the header line: `[VOCAB SIZE] [EMBEDDING DIMENSION]`. See the included embedding files for an example.

## Get Privatizing!
With these methods, you can now explore word-level Metric Local Differential Privacy text privatization. In case of any questions or suggestions, feel free to reach out to the authors.
