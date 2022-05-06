from copyreg import pickle
from genericpath import exists
from typing import Dict, List, NamedTuple
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import re
import os
import pickle

class Document(NamedTuple):
    doc_id: int
    author: List[str]
    title: List[str]
    keyword: List[str]
    body_text: List[str]
    processed_body_text: List[str]

    def sections(self):
        return [self.author, self.title, self.keyword, self.processed_body_text]

    def __repr__(self):
        return (f"doc_id: {self.doc_id}\n" +
            f"  author: {self.author}\n" +
            f"  title: {self.title}\n" +
            f"  keyword: {self.keyword}\n" +
            f"  body_text: {self.body_text}\n" +
            f"  processed_body_text: {self.processed_body_text}")


def read_stopwords(file):
    with open(file) as f:
        return set([x.strip() for x in f.readlines()])

def read_rels(file):
    '''
    Reads the file of related documents and returns a dictionary of query id -> list of related documents
    '''
    rels = {}
    with open(file) as f:
        for line in f:
            qid, rel = line.strip().split()
            qid = int(qid)
            rel = int(rel)
            if qid not in rels:
                rels[qid] = []
            rels[qid].append(rel)
    return rels

def read_docs(path, MAX_I=-1):
    '''
    Reads the corpus into a list of Documents
    '''
    if os.path.isdir(path):  
        dir_files = os.listdir(path) 
        max_doc_n = len(dir_files)
        if MAX_I < 0:
            MAX_I = max_doc_n
        return [ pickle.load(open(os.path.join(path, f'{i}.pkl'),  'rb')) for i in range(1,MAX_I+1) ]
      
    docs = [defaultdict(list)]  # empty 0 index
    category = ''
    with open(path) as f:
        i = 0
        for line in f:
            line = line.strip()
            if line.startswith('.I'):
                i = int(line[3:])
                docs.append(defaultdict(list))
            elif re.match(r'\.\w', line):
                category = line[1]
            elif line != '':
                for word in word_tokenize(line):
                    docs[i][category].append(word.lower())
            if i == MAX_I:
                break
    return [Document(i + 1, d['A'], d['T'], d['K'], d['W'], d['W'])
        for i, d in enumerate(docs[1:])]
    


def format_query(query, kw, auths):
    '''
    Formats a string query to a list of Documents
    '''
    body_text = []
    for word in word_tokenize(query):
        body_text.append(word.lower())

    keywords = []
    for word in word_tokenize(kw):
        keywords.append(word.lower())

    authors = []
    for word in word_tokenize(auths):
        authors.append(word.lower())


    return [ Document(1, authors, [], keywords, body_text, body_text) ]

stopwords = read_stopwords('common_words')
stemmer = PorterStemmer()

def stem_doc(doc: Document):
    return Document(doc.doc_id, doc.author, doc.title, 
                    doc.keyword, doc.body_text, 
                    [stemmer.stem(word) for word in doc.processed_body_text])

def stem_docs(docs: List[Document]):
    return [stem_doc(doc) for doc in tqdm(docs, desc='Stemming', disable=(len(docs)<500))]

def remove_stopwords_doc(doc: Document):
    return Document(doc.doc_id, doc.author, doc.title, 
                    doc.keyword, doc.body_text, 
                    [word for word in doc.processed_body_text if word not in stopwords])

def remove_stopwords(docs: List[Document]):
    return [remove_stopwords_doc(doc) for doc in tqdm(docs, desc='Stopwords', disable=(len(docs)<500))]

def process_docs_and_queries(docs, queries, stem, removestop, stopwords):
    processed_docs = docs
    processed_queries = queries
    if removestop:
        processed_docs = remove_stopwords(processed_docs)
        processed_queries = remove_stopwords(processed_queries)
    if stem:
        processed_docs = stem_docs(processed_docs)
        processed_queries = stem_docs(processed_queries)
    return processed_docs, processed_queries

def process_docs(docs, stem, removestop, stopwords):
    processed_docs = docs
    if removestop:
        processed_docs = remove_stopwords(processed_docs)
    if stem:
        processed_docs = stem_docs(processed_docs)
    return processed_docs

if __name__ == '__main__':
    docs = read_docs('tobacco_data/tobacco.raw', MAX_I=1000)
    stopwords = read_stopwords('common_words')

    pd = process_docs(docs, True, True, stopwords)