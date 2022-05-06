import itertools
import logging
from typing import Dict
import numpy as np
from pprint import pprint
from tqdm import tqdm
import argparse
import time
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from rank_bm25 import BM25Okapi
import torch
import pickle

# local imports
from data import read_stopwords, read_rels, read_docs, format_query, process_docs, process_docs_and_queries
from vectors import TermWeights
from vectors import compute_tf, compute_boolean, compute_doc_freqs, compute_tfidf
from vectors import cosine_sim, dice_sim, jaccard_sim, overlap_sim
from eval import precision_at, mean_precision1, mean_precision2, norm_precision, norm_recall


def experiment(args):
    docs = read_docs(args.docs_path, MAX_I=args.max_docs)
    queries = read_docs(args.queries_path)
    rels = read_rels(args.rels_path)
    stopwords = read_stopwords('common_words')

    global N_docs
    N_docs = len(docs)

    term_funcs = {
        'tf': compute_tf,
        'tfidf': compute_tfidf, # DEFAULT
        'boolean': compute_boolean
    }

    sim_funcs = {
        'cosine': cosine_sim, # DEFAULT
        'dice': dice_sim,
        'jaccard': jaccard_sim,
        'overlap': overlap_sim
    }
    search_funcs = {
       "search_basic": search_basic,
       "search_bm25": search_bm25,
    }

    permutations = [
        term_funcs,
        search_funcs,
        [
            False, 
            True # DEFAULT
        ],  # stem
        [
            True, # DEFAULT
            False, 
        ],  # remove stopwords
        sim_funcs,
        [
            TermWeights(author=1, title=1, keyword=1, body_text=1),
            TermWeights(author=3, title=3, keyword=4, body_text=1), # DEFAULT
            TermWeights(author=1, title=1, keyword=1, body_text=4),
        ]
    ]

    print('term', 'search', 'stem', 'removestop', 'sim', 'termweights', 'p_0.25', 'p_0.5', 'p_0.75', 'p_1.0', 'p_mean1', 'p_mean2', 'r_norm', 'p_norm', sep='\t')

    # This loop goes through all permutations. You might want to test with specific permutations first
    for term, search, stem, removestop, sim, term_weights in itertools.product(*permutations):
        processed_docs, processed_queries = process_docs_and_queries(docs, queries, stem, removestop, stopwords)
        doc_freqs = compute_doc_freqs(processed_docs)
        doc_vectors = [term_funcs[term](doc, doc_freqs, term_weights, N_docs) for doc in processed_docs]
        metrics = []

        # Process queries
        for query in processed_queries:
            query_vec = term_funcs[term](query, doc_freqs, term_weights, N_docs)
            results = search_funcs[search](doc_vectors, query_vec, sim_funcs[sim])
          
            rel = rels[query.doc_id]

            metrics.append([
                precision_at(0.25, results, rel),
                precision_at(0.5, results, rel),
                precision_at(0.75, results, rel),
                precision_at(1.0, results, rel),
                mean_precision1(results, rel),
                mean_precision2(results, rel),
                norm_recall(results, rel),
                norm_precision(results, rel)
            ])

        averages = [f'{np.mean([metric[i] for metric in metrics]):.4f}'
            for i in range(len(metrics[0]))]


        print(term, stem, removestop, sim, ','.join(map(str, term_weights)), *averages, sep='\t')



def get_query():
    raw_query = ''
    while len(raw_query) < 5:
        raw_query = input('Query: ')
        if raw_query == 'quit' or raw_query =='q':
            return 0
        elif len(raw_query) < 5:
            print(f'Invalid input: {repr(raw_query)}')
        else:
            keywords = input('Keywords: ')
            authors = input('Authors/API: ')

    return format_query(raw_query, keywords, authors)

def highlight_text(query, body_text, tokenizer, model):
    question, text = ' '.join(query.body_text), ' '.join(body_text)
    inputs = tokenizer(question, text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()
    predict_answer_tokens = inputs.input_ids[0, answer_start_index - 5 : answer_end_index + 5 + 1 ]
    answer = tokenizer.decode(predict_answer_tokens)
    return (answer_start_index, answer_end_index), answer

def answer_single_query(tokenizer, model, search_func,
                        query, docs, doc_freqs, doc_vectors, 
                        sim_func, term_func, term_weights):    
    # Process queries
    query_vec = term_func(query, doc_freqs, term_weights, N_docs)
    results = search_func(docs, query, doc_vectors, query_vec, sim_func)

    for i in range(20):
        doc_id = results[i]
        _, evidence = highlight_text(query, docs[doc_id-1].body_text, tokenizer, model)
        line =f'\t{i+1}) {results[i]} {" ".join(docs[doc_id-1].title)}\n\t{evidence}'
        print(line)

def interactive(args):
    docs = read_docs(args.docs_path, MAX_I=args.max_docs)
    global N_docs
    N_docs = len(docs)
    
    stopwords = read_stopwords('common_words')
    stem, removestop, sim_func, term_func = True, True, cosine_sim, compute_tfidf
    term_weights = TermWeights(author=3, title=3, keyword=4, body_text=1) 

    processed_docs = process_docs(docs, stem, removestop, stopwords)
    doc_freqs = compute_doc_freqs(processed_docs)
    doc_vectors = [term_func(doc, doc_freqs, term_weights) for doc in processed_docs]
    tokenizer = AutoTokenizer.from_pretrained(args.qa_model)
    model = AutoModelForQuestionAnswering.from_pretrained(args.qa_model)

    query = get_query()
    while query:
        query = process_docs(query, stem, removestop, stopwords)[0]
        answer_single_query(tokenizer, model, search_bm25, query, docs, doc_freqs, doc_vectors, sim_func, term_func, term_weights)
        query = get_query()

def command_line_query(args):
    docs = read_docs(args.docs_path, MAX_I=args.max_docs)
    global N_docs
    N_docs = len(docs)

    stem, removestop, sim_func, term_func = False, False, cosine_sim, compute_tfidf
    term_weights = TermWeights(author=3, title=3, keyword=4, body_text=1) 
    
    stopwords = read_stopwords('common_words')
    if not args.freqs_path:
        docs = process_docs(docs, stem, removestop, stopwords)
        doc_freqs = compute_doc_freqs(docs)
    else:
        print(f'Loading {args.freqs_path}')
        doc_freqs = pickle.load(open(args.freqs_path, 'rb'))
    doc_vectors = [term_func(doc, doc_freqs, term_weights, N_docs) for doc in docs]


    query = format_query(args.query, args.keywords, args.authors)
    query = process_docs(query, stem, removestop, stopwords)[0]
    tokenizer = AutoTokenizer.from_pretrained(args.qa_model)
    model = AutoModelForQuestionAnswering.from_pretrained(args.qa_model)
    answer_single_query(tokenizer, model, search_bm25, query, 
                        docs, doc_freqs, doc_vectors, 
                        sim_func, term_func, term_weights)

'''
Returns: List of doc ids
'''
def search_basic(docs, query, doc_vectors, query_vec, sim):
    results_with_score = [(doc_id + 1, sim(query_vec, doc_vec))
                    for doc_id, doc_vec in enumerate(doc_vectors)]
    results_with_score = sorted(results_with_score, key=lambda x: -x[1])
    results = [x[0] for x in results_with_score]
    return results

def search_bm25(docs, query, doc_vectors, query_vec, sim, n=20):
    tokenized_corpus = [d.processed_body_text for d in docs]
    tokenized_query = query.processed_body_text
    bm25 = BM25Okapi(tokenized_corpus)
    doc_scores = bm25.get_scores(tokenized_query)
    ind = np.argpartition(doc_scores, -n)[-n:]
    return ind[np.argsort(doc_scores[ind])]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs_path', '-dp', dest='docs_path', default='tobacco_data/tobacco_test.raw')
    parser.add_argument('--freqs_path', '-fp', dest='freqs_path', default='')
    parser.add_argument('--max_docs', '-md', dest='max_docs', default=-1, type=int)
    parser.add_argument('--qa_model', '-qa', dest='qa_model', default='abhijithneilabraham/longformer_covid_qa')
    # Experiment mode 
    parser.add_argument('--queries_path', '-qp', dest='queries_path', default='tobacco_data/query.raw')
    parser.add_argument('--rels_path', '-rp', dest='rels_path', default='tobacco_data/query.rels')
    # Interactive mode
    parser.add_argument('--interactive', '-i', dest='interactive', default=False, action='store_true')
    # Command line mode
    parser.add_argument('--query', '-q', dest='query')
    parser.add_argument('--keywords', '-kw', dest='keywords', default='')
    parser.add_argument('--authors', '--api', '-a', dest='authors', default='')

    args = parser.parse_args()
    print(args)
    start = time.time()
    if args.interactive:
        interactive(args)
    elif args.query:
        command_line_query(args)
    else:
        experiment(args)
    end = time.time()
    print(f'Time to run = {end-start} secs')