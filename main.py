import itertools
import logging
from typing import Dict
import numpy as np
from pprint import pprint
from pygments import highlight
from tqdm import tqdm
import argparse
import time
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from rank_bm25 import BM25Okapi
import torch
import pickle
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

# local imports
from data import read_highlights, read_stopwords, read_docs, format_query, process_docs, process_docs_and_queries
from vectors import TermWeights
from vectors import compute_tf, compute_boolean, compute_doc_freqs, compute_tfidf
from vectors import cosine_sim, dice_sim, jaccard_sim, overlap_sim
from eval import precision_at, mean_precision1, mean_precision2, norm_precision, norm_recall, match_highlighting, eval_highlighting


def good_combos(permutations):
    good_permuatations = []
    for term, search, qa_model, stem, removestop, sim, term_weights in itertools.product(*permutations):
        if search == "search_bm25":
            term, sim, term_weights  = 'na', 'na', 'na'
            good_permuatations.append([term, search, qa_model, stem, removestop, sim, term_weights])
        else:
            good_permuatations.append([term, search, qa_model, stem, removestop, sim, term_weights])
    df = pd.DataFrame(good_permuatations)
    df = df.drop_duplicates()
    return df.values.tolist()


def experiment(args):
    docs = read_docs(args.docs_path, MAX_I=args.max_docs)
    queries = read_docs(args.queries_path)
    rels = read_highlights(args.highlights_path)
    # stopwords = read_stopwords(args.stopwords)
    

    global N_docs
    N_docs = len(docs)

    term_funcs = {
        # 'tf': compute_tf,
        'tfidf': compute_tfidf, # DEFAULT
        # 'boolean': compute_boolean
    }

    sim_funcs = {
        'cosine': cosine_sim, # DEFAULT
        # 'dice': dice_sim,
        # 'jaccard': jaccard_sim,
        # 'overlap': overlap_sim
    }
    search_funcs = {
       "search_basic": search_basic,
       "search_bm25": search_bm25,
    }
    qa_models = {
       "bert-finetuned-squad": "huggingface-course/bert-finetuned-squad",
       "roberta-large-squad2": "navteca/roberta-large-squad2",
       "bigbird-roberta-natural-questions": "vasudevgupta/bigbird-roberta-natural-questions",
       "bert-base-newsqa": "mirbostani/bert-base-uncased-finetuned-newsqa",
       "longformer-base-squadv2": "mrm8488/longformer-base-4096-finetuned-squadv2"
    }

    permutations = [
        term_funcs,
        search_funcs,
        qa_models,
        [
            # False, 
            True # DEFAULT
        ],  # stem
        [
            True, # DEFAULT
            # False, 
        ],  # remove stopwords
        sim_funcs,
        [
            # TermWeights(author=1, title=1, keyword=1, body_text=1),
            TermWeights(author=3, title=3, keyword=4, body_text=1), # DEFAULT
            # TermWeights(author=1, title=1, keyword=1, body_text=4),
        ]
    ]
    good_permutations = good_combos(permutations)

    outfile = open(args.results_file, 'w')
    line = '\t'.join(['term', 'search', 'qa_model', 'stem', 'removestop', 'sim', 'termweights', 
                    'p_0.25', 'p_0.5', 'p_0.75', 'p_1.0', 
                    'p_mean1', 'p_mean2', 'r_norm', 'p_norm', 
                    'p_1_strict', 'r_3_strict', 'mrr_strict', 
                    'p_1_soft', 'r_3_soft', 'mrr_soft'])
    outfile.write(line + '\n')

    # This loop goes through all permutations. You might want to test with specific permutations first
    for term, search, qam, stem, removestop, sim, term_weights in tqdm(good_permutations, ncols=0):
        try:
            question_answerer = pipeline("question-answering", model=qa_models[qam], device=device)
            processed_docs, processed_queries = process_docs_and_queries(docs, queries, stem, removestop, stopwords)
            if search == "search_basic":
                doc_freqs = compute_doc_freqs(processed_docs)
                doc_vectors = [term_funcs[term](doc, doc_freqs, term_weights, N_docs) for doc in processed_docs]
            else:
                doc_freqs, doc_vectors = None, None
            metrics = []

            # Process queries
            for query in processed_queries:
                if search == "search_basic":
                    query_vec = term_funcs[term](query, doc_freqs, term_weights, N_docs)
                else:
                    query_vec = None
                results = search_funcs[search](docs, query, doc_vectors, 
                                                query_vec, 
                                                sim_funcs[sim] if sim != 'na' else None,
                                                n=N_docs)
            
                rel = rels[query.doc_id]['rels']
                highlighted_tokens = highlight_docs(query, results, docs, question_answerer)
                true_highlighted = rels[query.doc_id]['highlights']
                strict_highlights = match_highlighting(results, rel, highlighted_tokens, true_highlighted, strict=True)
                soft_highlights = match_highlighting(results, rel, highlighted_tokens, true_highlighted, strict=False)
                precision_at_1_strict, recall_at_3_strict, rr_strict = eval_highlighting(results, strict_highlights)
                precision_at_1_soft, recall_at_3_soft, rr_soft = eval_highlighting(results, soft_highlights)
                metrics.append([
                        precision_at(0.25, results, rel),
                        precision_at(0.5, results, rel),
                        precision_at(0.75, results, rel),
                        precision_at(1.0, results, rel),
                        mean_precision1(results, rel),
                        mean_precision2(results, rel),
                        norm_recall(results, rel),
                        norm_precision(results, rel),
                        precision_at_1_strict, recall_at_3_strict, rr_strict,
                        precision_at_1_soft, recall_at_3_soft, rr_soft
                    ])
            
            averages = [f'{np.mean([metric[i] for metric in metrics]):.4f}'
                for i in range(len(metrics[0]))]


            line = '\t'.join([term, search, qam, str(stem), str(removestop), sim, ','.join(map(str, term_weights)), *averages])
            outfile.write(line + '\n')

        except Exception as e:
            line = '\t'.join([term, search, qam, str(stem), str(removestop), sim, ','.join(map(str, term_weights))])
            print(f'{e} {line}')

    outfile.close()

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
            api = input('API: ')

    return format_query(raw_query, keywords, api)


def highlight_text(query, body_text, question_answerer):
    question, joined_body_text = ' '.join(query.body_text), ' '.join(body_text)
    answer = question_answerer(question=question, context=joined_body_text)

    token_idx = 0
    answer_start_index, answer_end_index = -1, -1
    char_start_idx, char_end_idx = answer['start'], answer['end']
    for char_idx, char in enumerate(joined_body_text):
        if char == ' ':
            token_idx += 1
        if char_idx == char_start_idx:
            answer_start_index = token_idx
        if char_idx == char_end_idx:
            answer_end_index = token_idx

    # predicted end character is last character
    if answer_end_index == -1:
        answer_end_index = token_idx

    predict_answer_tokens = ' '.join(body_text[max(0,answer_start_index-10) : answer_end_index + 10 ])   
    return (answer_start_index, answer_end_index), predict_answer_tokens

def highlight_docs(query, retreived, docs, question_answerer):
    highlights = []
    for ret_id in retreived:
        doc = docs[ret_id-1]
        # if doc.doc_id != ret_id:
        #     print(f"WARNING: doc.doc_id {doc.doc_id} != ret_id {ret_id}")
        answer_idxs, _ = highlight_text(query, doc.body_text, question_answerer)
        highlights.append(answer_idxs)
    return highlights

def answer_single_query(question_answerer, search_func,
                        query, docs, doc_freqs, doc_vectors, 
                        sim_func, term_func, term_weights):    
    # Process queries
    query_vec = term_func(query, doc_freqs, term_weights, N_docs)
    results = search_func(docs, query, doc_vectors, query_vec, sim_func)
    print(f'Query: {" ".join(query.body_text)}')
    for i in range(20):
        doc_id = results[i]
        _, evidence = highlight_text(query, docs[doc_id-1].body_text, question_answerer)
        line =f'{i+1}) {" ".join(docs[doc_id-1].title)}\n\t{evidence}'
        print(line)

def interactive(args):
    docs = read_docs(args.docs_path, MAX_I=args.max_docs)
    global N_docs
    N_docs = len(docs)
    
    # stopwords = read_stopwords(args.stopwords)
    stem, removestop, sim_func, term_func = True, True, cosine_sim, compute_tfidf
    term_weights = TermWeights(author=3, title=3, keyword=4, body_text=1) 

    processed_docs = process_docs(docs, stem, removestop, stopwords)
    doc_freqs = compute_doc_freqs(processed_docs)
    doc_vectors = [term_func(doc, doc_freqs, term_weights, N_docs) for doc in processed_docs]
    question_answerer = pipeline("question-answering", model=args.qa_model, device=device)

    query = get_query()
    while query:
        query = process_docs(query, stem, removestop, stopwords)[0]
        answer_single_query(question_answerer, search_bm25, query, docs, doc_freqs, doc_vectors, sim_func, term_func, term_weights)
        query = get_query()

def command_line_query(args):
    docs = read_docs(args.docs_path, MAX_I=args.max_docs)
    global N_docs
    N_docs = len(docs)

    stem, removestop, sim_func, term_func = False, False, cosine_sim, compute_tfidf
    term_weights = TermWeights(author=3, title=3, keyword=4, body_text=1) 
    
    # stopwords = read_stopwords(args.stopwords)
    if not args.freqs_path:
        docs = process_docs(docs, stem, removestop, stopwords)
        doc_freqs = compute_doc_freqs(docs)
    else:
        print(f'Loading {args.freqs_path}')
        doc_freqs = pickle.load(open(args.freqs_path, 'rb'))
    doc_vectors = [term_func(doc, doc_freqs, term_weights, N_docs) for doc in docs]


    query = format_query(args.query, args.keywords, args.api)
    query = process_docs(query, stem, removestop, stopwords)[0]

    question_answerer = pipeline("question-answering", model=args.qa_model, device=device)
    answer_single_query(question_answerer, search_bm25, query, 
                        docs, doc_freqs, doc_vectors,
                        sim_func, term_func, term_weights)

'''
Returns: List of doc ids
'''
def search_basic(docs, query, doc_vectors, query_vec, sim, n=None):
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
    return list(ind[np.argsort(doc_scores[ind])])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs_path', '-dp', dest='docs_path', 
                        default='tobacco_data/tobacco_test.v2.raw',
                        help='Path to documents. Accepts either a text file or directory of preprocessed docs. Default: tobacco_data/tobacco_test.v2.raw')
    parser.add_argument('--freqs_path', '-fp', dest='freqs_path', 
                        default='processed_docs/test.v2.freqs.pkl',
                        help='Path to precomputed doc frequencies. Default: processed_docs/test.v2.freqs.pkl')
    parser.add_argument('--stopwords', dest='stopwords', default='common_words')
    parser.add_argument('--max_docs', '-md', dest='max_docs', default=-1, type=int,
                        help='Max documents to read in. Default: -1 (read all documents)')
    parser.add_argument('--qa_model', '-qa', dest='qa_model', default='huggingface-course/bert-finetuned-squad',
                        help='QA model to load. Default:huggingface-course/bert-finetuned-squad')
    parser.add_argument('--device', '-d', dest='device', default=-1, type=int,
                        help='GPU device id. Default: -1 (CPU)')
    # Experiment mode 
    parser.add_argument('--queries_path', '-qp', dest='queries_path', default='tobacco_data/query.v2.raw',
                        help='Path to queries. Default: tobacco_data/query.v2.raw')
    parser.add_argument('--highlights_path', '-hp', dest='highlights_path', default='tobacco_data/query.v2.evidence',
                        help='Path to highlight annotations. Default: tobacco_data/query.v2.evidence')
    parser.add_argument('--results_file', '-r', dest='results_file', default='results.tsv',
                        help='Path to output results. Default: results.tsv')
    # Interactive mode
    parser.add_argument('--interactive', '-i', dest='interactive', default=False, action='store_true',
                        help='Flag for interactive mode.')
    # Command line mode
    parser.add_argument('--query', '-q', dest='query',
                        help='Query input for command line mode.')
    parser.add_argument('--keywords', '-kw', dest='keywords', default='', 
                        help='Optional keyword parameter for command line mode')
    parser.add_argument('--api', '-a', dest='api', default='', 
                        help='Optional api parameter for command line mode')

    args = parser.parse_args()
    stopwords = read_stopwords(args.stopwords)
    device = args.device if (torch.cuda.is_available() and args.device > -1) else -1
    start = time.time()
    if args.interactive:
        interactive(args)
    elif args.query:
        command_line_query(args)
    else:
        experiment(args)
    end = time.time()
    print(f'Time to run = {end-start} secs')