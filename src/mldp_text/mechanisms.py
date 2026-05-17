import random
import numpy as np
import pandas as pd
import math
import faiss
from scipy.special import lambertw, softmax
from scipy.linalg import sqrtm
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Ensure NLTK downloader doesn't crash multi-threaded systems
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet

# Import primitives from our utility module
from .utils import (
    VOCAB, EMBED, faiss_num_list, faiss_num_probe, validate_and_load_embeddings,
    truncated_Poisson, truncated_Gumbel, euclidean_laplace_rand_fn, euclidean_dt
)

class MultivariateCalibrated:
    def __init__(self, epsilon, 
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
            self.embedding_matrix = validate_and_load_embeddings(embed, binary=False)
        self.dim = dim
        self.use_faiss = use_faiss
        self.return_noise = return_noise

        if use_faiss:
            nlist = faiss_num_list
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            self.index.train(self.embedding_matrix.vectors)
            self.index.add(self.embedding_matrix.vectors)
            self.index.nprobe = faiss_num_probe

    def get_perturbed_vector(self, word_vec, n):
        noise = np.random.multivariate_normal(np.zeros(n), np.identity(n))
        norm_noise = noise / np.linalg.norm(noise)
        N = np.random.gamma(n, 1/self.epsilon) * norm_noise
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
        found = [k for k,v in self.embedding_matrix.key_to_index.items() if v == index]
        if len(found) > 0:
            return found[0]
        else:
            return None

    def replace_word(self, word):
        N = None
        if word in self.embedding_matrix:
            embedding_vector = self.embedding_matrix[word]
        else:
            embedding_vector = None

        if embedding_vector is not None:
            perturbed_vector, noise = self.get_perturbed_vector(embedding_vector, self.dim)
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
                 epsilon, 
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
            self.embedding_matrix = validate_and_load_embeddings(embed, binary=False)
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
            self.index.nprobe = faiss_num_probe
        
    def replace_word(self, word):    
        if word in self.embedding_matrix:
            word_embed = self.embedding_matrix[word]
        else:
            word_embed = None

        if word_embed is None: 
            return word

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

        noise = truncated_Gumbel(mu = 0, scale = self.b, 
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
                 epsilon,
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
            self.embedding_matrix = validate_and_load_embeddings(embed, binary=False)
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
            self.index.nprobe = faiss_num_probe
    
    def replace_word(self, word):    
        if word in self.embedding_matrix:
            word_embed = self.embedding_matrix[word]
        else:
            word_embed = None

        if word_embed is None: 
            return word

        noise = euclidean_laplace_rand_fn(dimensions=self.dim, epsilon=self.epsilon)
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
                epsilon, 
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
            self.embedding_matrix = validate_and_load_embeddings(embed, binary=False)
        self.vocab_size = len(self.vocab)
        self.dim = dim
        self.return_noise = return_noise

    def replace_word(self, input_word, threshold=0.5):     
        if input_word in self.embedding_matrix:
            word_embed = self.embedding_matrix[input_word]
        else:
            word_embed = None

        if word_embed is None: 
            return input_word

        euclid_dists = np.linalg.norm(self.embedding_matrix.vectors - word_embed, axis=1)

        word_euclid_dict = {word:dist for word, dist in zip(self.embedding_matrix.key_to_index.keys(), euclid_dists)}

        beta = 0.001
        
        threshold = round(2/self.epsilon * math.log(((1-beta)*len(self.embedding_matrix))/beta), 1)

        Lw = [word for word in word_euclid_dict if word_euclid_dict[word] <= threshold]

        f = {word: -word_euclid_dict[word] for word in Lw}

        f["⊥"] = -threshold + 2 * np.log(self.vocab_size/len(Lw)) / self.epsilon

        noise = [np.random.gumbel(0, 2 / self.epsilon) for _ in f]
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
                epsilon, 
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
            self.embedding_matrix = validate_and_load_embeddings(embed, binary=False)
        self.cov_mat = np.cov(self.embedding_matrix.vectors, rowvar=False) / np.var(self.embedding_matrix.vectors)
        self.identity_mat = np.identity(self.dim)
        self.use_faiss = use_faiss

        self.return_noise = return_noise

        if use_faiss:
            nlist = faiss_num_list
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            self.index.train(self.embedding_matrix.vectors)
            self.index.add(self.embedding_matrix.vectors)
            self.index.nprobe = faiss_num_probe

    def get_perturbed_vector(self, word_vec, n):
        noise = np.random.multivariate_normal(np.zeros(n), np.identity(n))
        norm_noise = np.divide(noise, np.linalg.norm(noise))
        Z = np.multiply(np.random.gamma(n, 1/self.epsilon), np.dot(sqrtm(self.lambd*self.cov_mat + (1-self.lambd)*self.identity_mat), norm_noise))
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
        found = [k for k, v in self.embedding_matrix.key_to_index.items() if v == index]
        if len(found) > 0:
            return found[0]
        return None

    def replace_word(self, word):
        if word in self.embedding_matrix:
            embedding_vector = self.embedding_matrix[word]
        else:
            embedding_vector = None

        if embedding_vector is not None:
            perturbed_vector, noise = self.get_perturbed_vector(embedding_vector, self.dim)
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
    def __init__(self, epsilon, data):
        self.epsilon = epsilon
        self.sensitivity = 1.0
        self.entire_doc = [" ".join([doc for doc in data])]
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.get_tfidf()
        self.words = list(self.tfidf_matrix.index)
        self.syn_dict = {word:self.synonym_extractor(phrase=word) for word in self.words}
        self.syn_scores = {word:self.get_synonym_score(word) for word in list(self.syn_dict.keys())}

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

    def replace_word(self, word):
        if word not in self.syn_scores: return word

        scores = self.syn_scores[word]
        if not scores: return word

        probabilities = [np.exp(self.epsilon * score / (2 * self.sensitivity)) for score in scores.values()]

        probabilities = probabilities / np.linalg.norm(probabilities, ord=1)

        return np.random.choice(list(scores.keys()), 1, p=probabilities)[0]