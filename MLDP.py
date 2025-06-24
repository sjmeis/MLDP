# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.special import lambertw, softmax
import scipy.stats as sct
from scipy.linalg import sqrtm
import nltk
nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet
import numba as nb
import math
import faiss
import random
import gensim.models
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle

import importlib_resources as impresources

faiss_num_list = 100
faiss_num_probe = 50

# Get the directory of the current script to build relative paths
_mldp_dir = os.path.dirname(os.path.abspath(__file__))

# set global (default) vocab
try:
    with open(os.path.join(_mldp_dir, "data", "vocab.txt"), 'r') as f:
        VOCAB = set([x.strip() for x in f.readlines()])
except FileNotFoundError:
    print("Warning: Default vocab file not found. Please set MLDP.VOCAB manually.")
    VOCAB = set()

# set global (default) embedding matrix
EMBED = os.path.join(_mldp_dir, "data", "glove.840B.300d_filtered.txt")
if not os.path.exists(EMBED):
    print("Warning: Default embedding file not found. Please set MLDP.EMBED manually.")
    EMBED = None

@nb.njit(fastmath=True, parallel=True)
def calc_distance(vec_1,vec_2):
    res = np.empty((vec_1.shape[0],vec_2.shape[0]),dtype=vec_1.dtype)
    for i in nb.prange(vec_1.shape[0]):
        for j in range(vec_2.shape[0]):
            res[i,j] = np.sqrt((vec_1[i,0]-vec_2[j,0])**2+(vec_1[i,1]-vec_2[j,1])**2+(vec_1[i,2]-vec_2[j,2])**2)
    return res

def calc_probability(embed1, embed2, epsilon=2):
    distance = calc_distance(embed1, embed2)
    sim_matrix = -distance
    prob_matrix = softmax(epsilon * sim_matrix / 2, axis=1)
    return prob_matrix

def euclidean_dt(m, v):
    diff = m - v
    dist = np.sum(np.square(diff), axis=-1)
    return np.sqrt(dist)

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
        temp = np.random.gumbel(loc = mu, size=temp_size, scale = scale.real)
        truncated = temp[np.absolute(temp) <= max_value]
        if len(truncated) >= size:
            return truncated[:size]
        
def euclidean_laplace_rand_fn(dimensions, epsilon):
    v = np.random.multivariate_normal(mean = np.zeros(dimensions),
                                        cov = np.eye(dimensions))
    v_norm = np.linalg.norm(v) + 1e-30
    v = v / v_norm

    l = np.random.gamma(shape = dimensions, scale = 1 / epsilon)
    return l * v


class MultivariateCalibrated:
    def __init__(self, epsilon=None, 
                 embedding_matrix=None, 
                 dim=300, 
                 use_faiss=True, 
                 vocab=VOCAB, 
                 embed=EMBED,
                 return_noise=False):
        self.vocab = vocab
        self.epsilon = epsilon
        if embedding_matrix is not None:
            self.embedding_matrix = embedding_matrix
        else:
            self.embedding_matrix = gensim.models.KeyedVectors.load_word2vec_format(embed, binary=False, unicode_errors="ignore")
        self.dim = dim
        self.use_faiss = use_faiss
        self.return_noise = return_noise

        if use_faiss:
            nlist = faiss_num_list
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            self.index.train(self.embedding_matrix.vectors)
            self.index.add(self.embedding_matrix.vectors)
            self.index.probe = faiss_num_probe

    def get_perturbed_vector(self, word_vec, n, epsilon):
        noise = np.random.multivariate_normal(np.zeros(n), np.identity(n))
        norm_noise = noise / np.linalg.norm(noise)
        N = np.random.gamma(n, 1/(epsilon)) * norm_noise
        return word_vec + N, N

    def get_nearest(self, vector):
        if self.use_faiss:
            _, I = self.index.search(np.array([vector.astype('float32')]), k=1)
            return I[0][0]
        else:
            diff = (self.embedding_matrix.vectors - vector)
            most_sim_index = np.argmin(np.linalg.norm(diff, axis=1))
            return most_sim_index

    def get_word_from_index(self, index):
        if 0 <= index < len(self.embedding_matrix.index_to_key):
            return self.embedding_matrix.index_to_key[index]
        else:
            return None

    def replace_word(self, word, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        if epsilon is None:
            raise ValueError("Epsilon must be provided either at instantiation or in the replace_word call.")
        N = None
        if word in self.embedding_matrix:
            embedding_vector = self.embedding_matrix[word]
        else:
            embedding_vector = None

        if embedding_vector is not None:
            perturbed_vector, noise = self.get_perturbed_vector(embedding_vector, self.dim, epsilon)
            sim_ind = self.get_nearest(perturbed_vector)
            new_word = self.get_word_from_index(int(sim_ind))
            if self.return_noise == True:
                if new_word is None:
                    return word, noise
                else: 
                    return new_word, noise
            else:
                if new_word is None:
                    return word
                else: 
                    return new_word
        if self.return_noise == True:
            return word, None
        else:
            return word

class TruncatedGumbel:
    def __init__(self,
                 epsilon=None, 
                 embedding_matrix=None,
                 dim=300,
                 max_inter_dist=0,
                 min_inter_dist=np.inf,
                 use_faiss=True,
                 vocab=VOCAB, 
                 embed=EMBED,
                 return_noise=False):
        self.vocab = vocab
        self.epsilon = epsilon
        if embedding_matrix is not None:
            self.embedding_matrix = embedding_matrix
        else:
            self.embedding_matrix = gensim.models.KeyedVectors.load_word2vec_format(embed, binary=False, unicode_errors="ignore")
        self.dim = dim
        self.vocab_size = len(self.vocab)
        self.use_faiss = use_faiss

        self.return_noise = return_noise

        self.max_inter_dist = max_inter_dist
        self.min_inter_dist = min_inter_dist

        if self.max_inter_dist == 0 or self.min_inter_dist == np.inf:
            start = random.sample(self.embedding_matrix.key_to_index.keys(), k=1)[0]
            index_flat = faiss.IndexFlatL2(self.dim)
            index_flat.add(np.ascontiguousarray(self.embedding_matrix.vectors).astype(np.float32))
            D, _ = index_flat.search(np.array([self.embedding_matrix[start]]).astype(np.float32), len(self.embedding_matrix))
            d = [np.sqrt(x) for x in D[0] if x != 0]
            self.min_inter_dist = np.min(d)
            self.max_inter_dist = np.max(d)

        self.a = (self.epsilon - (2*(1 + np.log(len(self.embedding_matrix))) / self.min_inter_dist)) / 3

        if self.a * self.min_inter_dist <= 0:
            self.b = 2 * self.max_inter_dist / (lambertw(2 * self.a * self.max_inter_dist).real)
        else:
            self.b = 2 * self.max_inter_dist / (np.min(np.array([lambertw(2 * self.a * self.max_inter_dist).real, np.log(self.a * self.min_inter_dist)])))

        if use_faiss:
            nlist = faiss_num_list
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            self.index.train(self.embedding_matrix.vectors)
            self.index.add(self.embedding_matrix.vectors)
            self.index.probe = faiss_num_probe
        
    def replace_word(self, word, epsilon=None):    
        if epsilon is None:
            epsilon = self.epsilon
        if epsilon is None:
            raise ValueError("Epsilon must be provided either at instantiation or in the replace_word call.")
        if word in self.embedding_matrix:
            word_embed = self.embedding_matrix[word]
        else:
            word_embed = None

        if word_embed is None: 
            return word

        # recalculate a and b for this epsilon
        a = (epsilon - (2*(1 + np.log(len(self.embedding_matrix))) / self.min_inter_dist)) / 3
        if a * self.min_inter_dist <= 0:
            b = 2 * self.max_inter_dist / (lambertw(2 * a * self.max_inter_dist).real)
        else:
            b = 2 * self.max_inter_dist / (np.min(np.array([lambertw(2 * a * self.max_inter_dist).real, np.log(a * self.min_inter_dist)])))

        k = truncated_Poisson(mu = np.log(len(self.embedding_matrix) - 1), 
                              size = 1,
                              max_value = len(self.embedding_matrix) - 1)
        k = k[0]

        k = max(0, int(k))

        if self.use_faiss:
            D, I = self.index.search(np.array([word_embed.astype('float32')]), k=k)
            idx = I[0]
            dist = D[0]
        else:
            dist = euclidean_dt(self.embedding_matrix.vectors, word_embed)
            idx = np.argsort(dist)
            idx = idx[:k]
            dist = dist[idx]

        noise = truncated_Gumbel(mu = 0, scale = b, 
                                       size=len(dist), 
                                       max_value=self.max_inter_dist)
        dist = dist + noise
        indexes = np.argsort(dist)
        if len(indexes) > 0:
            i = idx[indexes[0]]
        else:
            if self.return_noise == True:
                return word, None
            else:
                return word

        perturbed_word = self.embedding_matrix.index_to_key[i]
        if self.return_noise == True:
            return perturbed_word, noise    
        else:
            return perturbed_word

class VickreyMechanism:
    def __init__(self,
                 epsilon=None,
                 embedding_matrix=None, 
                 dim=300,
                 k = 2, t = [1, 0],
                 use_faiss=True,
                 vocab=VOCAB, 
                 embed=EMBED,
                 return_noise=False):
        self.vocab = vocab
        self.epsilon = epsilon
        if embedding_matrix is not None:
            self.embedding_matrix = embedding_matrix
        else:
            self.embedding_matrix = gensim.models.KeyedVectors.load_word2vec_format(embed, binary=False, unicode_errors="ignore")
        self.dim = dim
        self.num_perturbed = 0
        self.num_words = 0
        self.k = k
        self.t = np.asarray(t)
        self.use_faiss = use_faiss

        self.return_noise = return_noise

        if use_faiss:
            nlist = faiss_num_list
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            self.index.train(self.embedding_matrix.vectors)
            self.index.add(self.embedding_matrix.vectors)
            self.index.probe = faiss_num_probe
    
    def replace_word(self, word, epsilon=None):    
        if epsilon is None:
            epsilon = self.epsilon
        if epsilon is None:
            raise ValueError("Epsilon must be provided either at instantiation or in the replace_word call.")
        if word in self.embedding_matrix:
            word_embed = self.embedding_matrix[word]
        else:
            word_embed = None

        if word_embed is None: 
            return word

        noise = euclidean_laplace_rand_fn(dimensions=self.dim, epsilon=epsilon)
        noisy_vector = word_embed + noise

        if self.use_faiss:
            D, I = self.index.search(np.array([noisy_vector.astype('float32')]), int(self.k + 1))
            indices = I[0]
            dists = D[0]
        else:
            dists = euclidean_dt(self.embedding_matrix.vectors, noisy_vector)
            indices = np.argsort(dists)
            dists = dists[indices]

        if indices[0] == self.embedding_matrix.key_to_index[word]:
            idx = indices[1:self.k + 1]
            dist = dists[1: self.k + 1]
        else:
            idx = indices[:self.k]
            dist = dists[:self.k]

        p = -self.t * dist
        p = softmax(p)

        i = np.random.choice(idx, p = p)

        perturbed_word = self.embedding_matrix.index_to_key[i]
        if self.return_noise == True:
            return perturbed_word, noise
        else:
            return perturbed_word

class TEM:
    def __init__(self, 
                epsilon=None, 
                embedding_matrix=None, 
                dim=300,
                use_faiss=False,
                vocab=VOCAB, 
                embed=EMBED,
                return_noise=False):
        self.vocab = vocab
        self.epsilon = epsilon
        if embedding_matrix is not None:
            self.embedding_matrix = embedding_matrix
        else:
            self.embedding_matrix = gensim.models.KeyedVectors.load_word2vec_format(embed, binary=False, unicode_errors="ignore")
        self.vocab_size = len(self.vocab)
        self.dim = dim
        self.return_noise = return_noise
        self.use_faiss = use_faiss

        if self.use_faiss:
            self.index = faiss.IndexFlatL2(self.dim)
            self.index.add(np.ascontiguousarray(self.embedding_matrix.vectors).astype(np.float32))

    def replace_word(self, input_word, epsilon=None, threshold=0.5):     
        if epsilon is None:
            epsilon = self.epsilon
        if epsilon is None:
            raise ValueError("Epsilon must be provided either at instantiation or in the replace_word call.")
        if input_word in self.embedding_matrix:
            word_embed = self.embedding_matrix[input_word]
        else:
            word_embed = None

        if word_embed is None: 
            return input_word

        beta = 0.001
        threshold = round(2/epsilon * math.log(((1-beta)*len(self.embedding_matrix))/beta), 1)

        if self.use_faiss:
            # Use FAISS range_search to find neighbors efficiently
            lims, D, I = self.index.range_search(np.array([word_embed.astype('float32')]), threshold**2)
            indices = I
            dists = np.sqrt(D)
            # Create a dictionary of the words and their distances for the neighborhood
            word_euclid_dict = {self.embedding_matrix.index_to_key[i]: dist for i, dist in zip(indices, dists)}
            Lw = list(word_euclid_dict.keys())
        else:
            euclid_dists = np.linalg.norm(self.embedding_matrix.vectors - word_embed, axis=1)
            word_euclid_dict = {word:dist for word, dist in zip(self.embedding_matrix.key_to_index.keys(), euclid_dists)}
            Lw = [word for word in word_euclid_dict if word_euclid_dict[word] <= threshold]

        f = {word: -word_euclid_dict[word] for word in Lw}

        f["⊥"] = -threshold + 2 * np.log(self.vocab_size/len(Lw)) / epsilon

        noise = [np.random.gumbel(0, 2 / epsilon) for _ in f]
        f = {word: f[word] + noise[i] for i, word in enumerate(f)}
        
        privatized_word = max(f, key=f.get)

        if self.return_noise == True:
            if privatized_word == "⊥":
                new_word = np.random.choice([word for word in self.embedding_matrix.key_to_index.keys() if word not in Lw])
                return new_word, noise
            else:
                return privatized_word, noise
        else:
            if privatized_word == "⊥":
                new_word = np.random.choice([word for word in self.embedding_matrix.key_to_index.keys() if word not in Lw])
                return new_word
            else:
                return privatized_word
            
class Mahalanobis:
    def __init__(self, 
                epsilon=None, 
                embedding_matrix=None, 
                lambd=0.2, 
                dim=300, 
                use_faiss=True,
                vocab=VOCAB, 
                embed=EMBED,
                return_noise=False):
        self.vocab_dict = vocab
        self.epsilon = epsilon
        self.lambd = lambd
        self.dim = dim
        if embedding_matrix is not None:
            self.embedding_matrix = embedding_matrix
        else:
            self.embedding_matrix = gensim.models.KeyedVectors.load_word2vec_format(embed, binary=False, unicode_errors="ignore")
        self.cov_mat = np.cov(self.embedding_matrix.vectors, rowvar=False) / np.var(self.embedding_matrix.vectors)
        self.identity_mat = np.identity(self.dim)
        
        # Pre-compute the expensive matrix square root and store it
        maha_mat = self.lambd*self.cov_mat + (1-self.lambd)*self.identity_mat
        self.maha_mat_sqrt = sqrtm(maha_mat).real

        self.use_faiss = use_faiss

        self.return_noise = return_noise

        if use_faiss:
            nlist = faiss_num_list
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            self.index.train(self.embedding_matrix.vectors)
            self.index.add(self.embedding_matrix.vectors)
            self.index.probe = faiss_num_probe

    def get_perturbed_vector(self, word_vec, n, epsilon):
        noise = np.random.multivariate_normal(np.zeros(n), np.identity(n))
        norm_noise = np.divide(noise, np.linalg.norm(noise))
        # Use the pre-computed matrix square root
        Z = np.multiply(np.random.gamma(n, 1/epsilon), np.dot(self.maha_mat_sqrt, norm_noise))
        return word_vec + Z, Z

    def get_nearest(self, vector):
        if self.use_faiss:
            _, I = self.index.search(np.array([vector.astype('float32')]), k=1)
            return I[0][0]
        else:
            diff = (self.embedding_matrix.vectors - vector)
            most_sim_index = np.argmin(np.linalg.norm(diff, axis=1))

            return most_sim_index

    def get_word_from_index(self, index):
        if 0 <= index < len(self.embedding_matrix.index_to_key):
            return self.embedding_matrix.index_to_key[index]
        return None

    def replace_word(self, word, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        if epsilon is None:
            raise ValueError("Epsilon must be provided either at instantiation or in the replace_word call.")
        if word in self.embedding_matrix:
            embedding_vector = self.embedding_matrix[word]
        else:
            embedding_vector = None

        if embedding_vector is not None:
            perturbed_vector, noise = self.get_perturbed_vector(embedding_vector, self.dim, epsilon)
            sim_ind = self.get_nearest(perturbed_vector)
            new_word = self.get_word_from_index(sim_ind)
            if self.return_noise == True:
                if new_word is None:
                    return word, noise
                else:
                    return new_word, noise
            else:
                if new_word is None:
                    return word
                else:
                    return new_word
        if self.return_noise == True:
            return word, None
        else:
            return word
        
class SynTF:
    def __init__(self, epsilon=None, data=None, cache_path="syntf_cache.pkl"):
        self.epsilon = epsilon
        self.sensitivity = 1.0
        self.cache_path = cache_path

        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                self.vectorizer = cache_data['vectorizer']
                self.tfidf_matrix = cache_data['tfidf_matrix']
                self.words = cache_data['words']
                self.syn_dict = cache_data['syn_dict']
                self.syn_scores = cache_data['syn_scores']
                return
            except Exception as e:
                print(f"Could not load cache, rebuilding: {e}")

        self.entire_doc = [" ".join([doc for doc in data])]
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.get_tfidf()
        self.words = list(self.tfidf_matrix.index)
        self.syn_dict = {word:self.synonym_extractor(phrase=word) for word in self.words}
        self.syn_scores = {word:self.get_synonym_score(word) for word in list(self.syn_dict.keys())}

        try:
            with open(self.cache_path, 'wb') as f:
                cache_data = {
                    'vectorizer': self.vectorizer,
                    'tfidf_matrix': self.tfidf_matrix,
                    'words': self.words,
                    'syn_dict': self.syn_dict,
                    'syn_scores': self.syn_scores
                }
                pickle.dump(cache_data, f)
        except Exception as e:
            print(f"Could not save cache: {e}")

    def get_tfidf(self):
        vectors = self.vectorizer.fit_transform(self.entire_doc)
        tf_idf = pd.DataFrame(vectors.todense())
        tf_idf.columns = self.vectorizer.get_feature_names_out()
        tfidf_matrix = tf_idf.T
        return tfidf_matrix

    def get_synonym_score(self, word):
        score_dict = {}
        for syn in self.syn_dict[word]:
            if syn in self.tfidf_matrix.index: score_dict[syn] = self.tfidf_matrix.loc[word][0]
            else: score_dict[syn] = 0
        return score_dict

    def synonym_extractor(self, phrase):
        synonyms = set()
        for syn in wordnet.synsets(phrase):
            for l in syn.lemmas():
                synonyms.add(l.name())
        return synonyms

    def replace_word(self, word, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        if epsilon is None:
            raise ValueError("Epsilon must be provided either at instantiation or in the replace_word call.")
        if word not in self.syn_scores: return word

        scores = self.syn_scores[word]
        if not scores: return word

        probabilities = [np.exp(epsilon * score / (2 * self.sensitivity)) for score in scores.values()]

        probabilities = probabilities / np.linalg.norm(probabilities, ord=1)

        return np.random.choice(list(scores.keys()), 1, p=probabilities)[0]