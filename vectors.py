from typing import Dict, List, NamedTuple
from data import Document
from collections import Counter, defaultdict
import numpy as np
from numpy.linalg import norm

### Term-Document Matrix
class TermWeights(NamedTuple):
    author: float
    title: float
    keyword: float
    body_text: float

def compute_doc_freqs(docs: List[Document]):
    '''
    Computes document frequency, i.e. how many documents contain a specific word
    '''
    freq = Counter()
    for doc in docs:
        words = set()
        for sec in doc.sections():
            for word in sec:
                words.add(word)
        for word in words:
            freq[word] += 1
    return freq

def compute_tf(doc: Document, doc_freqs: Dict[str, int], weights: list, N_docs):
    vec = defaultdict(float)
    for word in doc.author:
        vec[word] += weights.author
    for word in doc.keyword:
        vec[word] += weights.keyword
    for word in doc.title:
        vec[word] += weights.title
    for word in doc.processed_body_text:
        vec[word] += weights.body_text
    return dict(vec)  # convert back to a regular dict

def compute_tfidf(doc, doc_freqs, weights, N_docs):
    tfidf = defaultdict(float)  
    tf = compute_tf(doc, doc_freqs, weights, N_docs)
    for word in tf:
        if doc_freqs[word] == 0:
            doc_freqs[word] += 0.001
        tfidf[word] = tf[word] * (np.log(N_docs) - np.log(doc_freqs[word]))
    return dict(tfidf)

def compute_boolean(doc, doc_freqs, weights, N_docs):
    vec = defaultdict(float)
    for word in doc.author:
        vec[word] = weights.author
    for word in doc.keyword:
        vec[word] = weights.keyword
    for word in doc.title:
        vec[word] = weights.title
    for word in doc.processed_body_text:
        vec[word] = weights.body_text
    return dict(vec) 

### Vector Similarity

def safe_div(num, den):
    if den == 0:
        return num
    return num/den

def dictdot(x: Dict[str, float], y: Dict[str, float]):
    '''
    Computes the dot product of vectors x and y, represented as sparse dictionaries.
    '''
    keys = list(x.keys()) if len(x) < len(y) else list(y.keys())
    return sum(x.get(key, 0) * y.get(key, 0) for key in keys)

def cosine_sim(x, y):
    '''
    Computes the cosine similarity between two sparse term vectors represented as dictionaries.
    '''
    num = dictdot(x, y)
    if num == 0:
        return 0
    return num / (norm(list(x.values())) * norm(list(y.values())))

def dice_sim(x, y):
    num = 2*dictdot(x, y)
    den = sum(list(x.values())) + sum(list(y.values()))
    return safe_div(num, den)

def jaccard_sim(x, y):
    num = dictdot(x, y)
    den = sum(list(x.values())) + sum(list(y.values())) - num
    return safe_div(num, den)

def overlap_sim(x, y):
    num = dictdot(x, y)
    den = min(sum(list(x.values())), sum(list(y.values())))
    return safe_div(num, den)