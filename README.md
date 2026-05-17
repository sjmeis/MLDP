<div align="center">

  # MLDP

  [![PyPI version](https://img.shields.io/pypi/v/mldp-text.svg)](https://pypi.org/project/mldp-text/)
  [![GitHub stars](https://img.shields.io/github/stars/sjmeis/MLDP.svg?style=social)](https://github.com/sjmeis/MLDP/stargazers)
  [![License](https://img.shields.io/github/license/sjmeis/MLDP.svg)](https://github.com/sjmeis/MLDP/blob/main/LICENSE)

</div>

This repository contains the official implementation for the paper: *A Comparative Analysis of Word-Level Metric Differential Privacy: Benchmarking The Privacy-Utility Trade-off* (LREC-COLING 2024). It provides production-ready, highly optimized implementations for six word-level Metric Local Differential Privacy (MLDP) mechanisms.

## Included Mechanisms
The package implements the following MLDP text privatization strategies:

- *MultivariateCalibrated*: [paper](https://dl.acm.org/doi/10.1145/3336191.3371856)
- *TruncatedGumbel*: [paper](https://journals.flvc.org/FLAIRS/article/view/128463)
- *VickreyMechanism*: [paper](https://aclanthology.org/2021.privatenlp-1.2/)
- *TEM*: [paper](https://epubs.siam.org/doi/10.1137/1.9781611977653.ch99)
- *Mahalanobis*: [paper](https://aclanthology.org/2020.privatenlp-1.2/)
- *SynTF*: [paper](https://dl.acm.org/doi/10.1145/3209978.3210008)

Note that the code for SanText is not included as it is already publicly available [here](https://github.com/sjmeis/SANTEXT).

## Installation
Getting started is as simple as installing the package:

```bash
pip install mldp-text
```

## Basic Usage 
The package exposes a unified factory function called `get_mechanism()` to seamlessly switch between different MLDP algorithms using string IDs.

### Embedding Perturbation Mechanisms
For any of these mechanism, initialization is straightforward. By default, mechanisms look for an optimized `faiss` index to accelerate nearest-neighbor lookups:

```python
import mldp_text

# Initialize your chosen strategy
mechanism = mldp_text.get_mechanism("multivariate_calibrated", epsilon=1, use_faiss=True)

# Privatize individual words
perturbed_word = mechanism.replace_word("pizza")
print(perturbed_word)
```

### SynTF Mechanism
The `SynTF` mechanism is frequency-driven and requires a document corpus to pre-calculate and cache its reference TF-IDF matrix:

```python
import mldp_text

corpus = ["your list of reference dataset documents here", "another document sample"]

# Initialize SynTF with document data
mechanism = mldp_text.get_mechanism("syntf", epsilon=1.0, data=corpus)

perturbed_word = mechanism.replace_word("pizza")
```

## Supported Mechanisms
When using `get_mechanism(name)`, you can pass any of the following string variants for the name parameter (case-insensitive, hyphens/underscores are normalized automatically):

| MLDP Mechanism  | Allowed String IDs (name=)                      |
|------------------------|---------------------------------------------------------|
| MultivariateCalibrated | `multivariate_calibrated`|`multivariate-calibrated` |
| TruncatedGumbel        | `truncated_gumbel`|`truncated-gumbel`               |
| VickreyMechanism       | `vickrey`|`vickrey_mechanism`                        |
| TEM                    | `tem`                                                 |
| Mahalanobis            | `mahalanobis`                                         |
| SynTF                  | `syntf`                                               |


## Embedding Models
By default, the package looks for the `glove.840B.300d` embedding model pre-filtered to a fixed companion vocabulary (`data/vocab.txt`). Both assets are derived from the official [Stanford GloVe project](https://nlp.stanford.edu/projects/glove/).

### Loading Custom Embeddings
You can pass your own custom word embedding model into any mechanism. The package automatically inspects your file header beforehand to confirm it aligns with the native `gensim` format standard: `[VOCAB SIZE] [EMBEDDING DIMENSION]` (e.g., `400000 300`).

You can feed custom paths into the package in two ways:

#### Option 1: Session-Wide Override
Change the underlying fallback path before instantiating any mechanisms:

```python
import mldp_text

mldp.utils.EMBED = "/path/to/your/custom_gensim_embeddings.txt"

engine = mldp_text.get_mechanism("mahalanobis", epsilon=1.2)
```

#### Option 2: Mechanism Parameter
Pass the file path directly to the instantiation call:

```python
import mldp_text

engine = mldp_text.get_mechanism(
    "vickrey", 
    epsilon=1, 
    embed="/path/to/custom_vectors.txt"
)
```

## Get Privatizing!
With these methods, you can now explore word-level Metric Local Differential Privacy text privatization. In case of any questions or suggestions, feel free to reach out to the authors.

## Citation
If you find this work useful, please consider citing the original LREC-COLING work, which implemented and evaluated these MLDP mechanisms:

```
@inproceedings{meisenbacher-etal-2024-comparative,
    title = "A Comparative Analysis of Word-Level Metric Differential Privacy: Benchmarking the Privacy-Utility Trade-off",
    author = "Meisenbacher, Stephen  and
      Nandakumar, Nihildev  and
      Klymenko, Alexandra  and
      Matthes, Florian",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.16/",
    pages = "174--185"
}
```