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


if __name__=="__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()
    # main()
    preprocess_docs()
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')