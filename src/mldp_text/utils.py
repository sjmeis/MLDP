# -*- coding: utf-8 -*-
import os
import numpy as np
import scipy.stats as sct
from scipy.special import softmax
import numba as nb
import importlib.resources as pkg_resources
import gensim.models

faiss_num_list = 100
faiss_num_probe = 50

try:
    _data_dir = pkg_resources.files("mldp_text") / "data"
    _vocab_path = _data_dir / "vocab.txt"
    
    _embed_path_300d = _data_dir / "glove.840B.300d_filtered.txt"
    
    if _vocab_path.exists():
        with open(_vocab_path, 'r', encoding='utf-8') as f:
            VOCAB = set([x.strip() for x in f.readlines()])
    else:
        VOCAB = set()
        
    EMBED = str(_embed_path_300d) if _embed_path_300d.exists() else None
except Exception:
    VOCAB = set()
    EMBED = None


def validate_and_load_embeddings(embed_path: str, binary: bool = False):
    """
    Validates that the file exists and follows the gensim word2vec text header format:
    '[VOCAB SIZE] [EMBEDDING DIMENSION]'
    If valid, returns the loaded KeyedVectors object.
    """
    if not embed_path or not os.path.exists(embed_path):
        raise FileNotFoundError(f"Embedding file not found at path: {embed_path}")

    if not binary:
        try:
            with open(embed_path, 'r', encoding='utf-8', errors='ignore') as f:
                header = f.readline().strip().split()
                
            if len(header) != 2:
                raise ValueError(
                    f"Invalid header format in '{embed_path}'. Gensim text format requires "
                    f"the first line to be '[VOCAB SIZE] [EMBEDDING DIMENSION]' (e.g., '400000 300'). "
                    f"Found: '{' '.join(header)}'"
                )
                
            vocab_size, embedding_dim = int(header[0]), int(header[1])
            
        except (ValueError, IndexError) as e:
            raise ValueError(
                f"Failed to parse embedding file header. Ensure it matches the "
                f"Gensim requirement of an integer vocab size and dimension pair. Details: {e}"
            )

    print(f"Success: Validated format. Loading {embedding_dim}d embeddings ({vocab_size} words)...")
    return gensim.models.KeyedVectors.load_word2vec_format(embed_path, binary=binary, unicode_errors="ignore")

@nb.njit(fastmath=True, parallel=True)
def calc_distance(vec_1, vec_2):
    res = np.empty((vec_1.shape[0], vec_2.shape[0]), dtype=vec_1.dtype)
    for i in nb.prange(vec_1.shape[0]):
        for j in range(vec_2.shape[0]):
            res[i, j] = np.sqrt((vec_1[i, 0] - vec_2[j, 0]) ** 2 + 
                                (vec_1[i, 1] - vec_2[j, 1]) ** 2 + 
                                (vec_1[i, 2] - vec_2[j, 2]) ** 2)
    return res

def calc_probability(embed1, embed2, epsilon=2):
    distance = calc_distance(embed1, embed2)
    sim_matrix = -distance
    prob_matrix = softmax(epsilon * sim_matrix / 2, axis=1)
    return prob_matrix

def euclidean_dt(m, v):
    diff = m - v
    return np.sqrt(np.sum(np.square(diff), axis=-1))

def truncated_Poisson(mu, max_value, size):
    temp_size = size
    while True:
        temp_size *= 2
        temp = sct.poisson.rvs(mu, size=temp_size)
        truncated = temp[temp <= max_value]
        if len(truncated) >= size:
            return truncated[:size]

def truncated_Gumbel(mu, scale, max_value, size):
    temp_size = size
    while True:
        temp_size *= 2
        temp = np.random.gumbel(loc=mu, size=temp_size, scale=scale.real)
        truncated = temp[np.absolute(temp) <= max_value]
        if len(truncated) >= size:
            return truncated[:size]
        
def euclidean_laplace_rand_fn(dimensions, epsilon):
    v = np.random.multivariate_normal(mean=np.zeros(dimensions), cov=np.eye(dimensions))
    v_norm = np.linalg.norm(v) + 1e-30
    v = v / v_norm
    l = np.random.gamma(shape=dimensions, scale=1 / epsilon)
    return l * v