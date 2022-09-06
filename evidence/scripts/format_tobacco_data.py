import argparse
import logging
import time
import pandas as pd
from tqdm import tqdm, trange
import os
import pickle
from data import read_docs, process_docs, read_stopwords, Document, word_tokenize
import re
from vectors import compute_doc_freqs
from collections import Counter, defaultdict

def main():
    # i = 0
    infile = 'tobacco_data/metadata.csv'
    outfile = open('tobacco_data/tobacco.raw', 'w')
    df = pd.read_csv(infile)
    print(df.head())
    print(df.columns)
    for i, row in tqdm(df.iterrows(), ncols=0, total=len(df)):
        body_text = row['abstract'].replace('\n', ' ')
        line = f".I {i+1}\n.T \n{row['title']} \n.W \n{body_text} \n.A \n{row['source_x']} \n"
        outfile.write(line)
    outfile.close()

def preprocess_docs():
    infile = 'tobacco_data/metadata.csv'
    df = pd.read_csv(infile)
    df = df.fillna('')

    print(df.head())
    print(df.columns
    )
    stopwords = read_stopwords('common_words')
    stem, removestop = True, True

    all_processed_docs = []

    for i, row in tqdm(df.iterrows(), ncols=0, total=len(df)):
        data_path = f'processed_docs/full/{i+1}.pkl'
        # if not os.path.exists(data_path):
        body_text = row['abstract'].replace('\n', ' ')
        body_text = [w.lower() for w in word_tokenize(body_text)]
        doc = Document(i+1, word_tokenize(row['source_x'].lower()), word_tokenize(row['title'].lower()), [], body_text, body_text)
        all_processed_docs += process_docs([doc], stem, removestop, stopwords)

        with open(data_path, 'wb') as out_docs:
            pickle.dump(all_processed_docs[-1], out_docs)

    doc_freqs = compute_doc_freqs(all_processed_docs)
    with open('processed_docs/full/freqs.pkl', 'wb') as out_freqs:
        pickle.dump(doc_freqs, out_freqs)

def preprocess2():
    df = pd.read_csv('tobacco_data/annotated_data_100.v2.tsv', sep='\t', nrows=100)
    df = df.fillna('')
    # tobacco.raw
    outfile = open('tobacco_data/tobacco_test.v2.raw', 'w')
    print(df.head())
    print(df.columns)
    for i, row in tqdm(df.iterrows(), ncols=0, total=len(df)):
        body_text = row['text'].replace('\n', ' ')
        line = f".I {i+1}\n.T \n{row['title']} \n.W \n{body_text} \n.A \n{row['api']} \n"
        outfile.write(line)
    outfile.close()
    # query.raw
    raw_queries = df.columns[5:]
    query_ids = {}
    query_file = open('tobacco_data/query.v2.raw', 'w')
    for i, q in enumerate(raw_queries, start=1):
        query_ids[q] = i
        query_file.write(f'\n\n.I {i}\n.W\n {q}')
    query_file.close()

    # query.evidence 
    evidence_file = open('tobacco_data/query.v2.evidence', 'w')
    for doc_idx, row in tqdm(df.iterrows(), ncols=0, total=len(df)):
        doc_id = doc_idx + 1
        doc_text =  word_tokenize(row['text'].lower())
        for col in raw_queries:
            if not row[col] == "":
                # get token idxs
                answer_text = word_tokenize(row[col].lower())
                for idx in range(len(doc_text) - len(answer_text) + 1):
                    if doc_text[idx : idx + len(answer_text)] == answer_text:
                        start, end = idx, idx + len(answer_text)
                line = f'{query_ids[col]} {doc_id} {start} {end}\n'
                evidence_file.write(line)
    evidence_file.close()     

    # preprocess docs
    stopwords = read_stopwords('common_words')
    stem, removestop = True, True

    all_processed_docs = []

    for i, row in tqdm(df.iterrows(), ncols=0, total=len(df)):
        data_path = f'processed_docs/test.v2/{i+1}.pkl'
        # if not os.path.exists(data_path):
        body_text = row['text'].replace('\n', ' ')
        body_text = [w.lower() for w in word_tokenize(body_text)]
        doc = Document(i+1, word_tokenize(row['api'].lower()), word_tokenize(row['title'].lower()), [], body_text, body_text)
        all_processed_docs += process_docs([doc], stem, removestop, stopwords)

        with open(data_path, 'wb') as out_docs:
            pickle.dump(all_processed_docs[-1], out_docs)

    doc_freqs = compute_doc_freqs(all_processed_docs)
    with open('processed_docs/test.v2.freqs.pkl', 'wb') as out_freqs:
        pickle.dump(doc_freqs, out_freqs)

    
    print()
if __name__=="__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()
    # main()
    # preprocess_docs()
    preprocess2()
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')