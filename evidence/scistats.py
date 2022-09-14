'''
Script to parse scifact data
'''
import argparse
import logging
import time
import json
import yaml
from tqdm import tqdm
import numpy as np
import pandas as pd
import itertools
from scipy.sparse import csr_matrix
# from evidence import merge_labels_with_corpus
'''
{"id": 3, 
"evidence": 
    {"14717500": 
        {"sentences": [2, 5, 7], 
        "label": "NOT_ENOUGH_INFO"}
    }
}
'''
def create_co_occurences_matrix(labels, documents):
    word_to_id = dict(zip(labels, range(len(labels))))
    documents_as_ids = [np.sort([word_to_id[w] for w in doc if w in word_to_id]).astype('uint32') for doc in documents]
    row_ind, col_ind = zip(*itertools.chain(*[[(i, w) for w in doc] for i, doc in enumerate(documents_as_ids)]))
    data = np.ones(len(row_ind), dtype='uint32')  # use unsigned int for better memory utilization
    max_word_id = max(itertools.chain(*documents_as_ids)) + 1
    docs_words_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(len(documents_as_ids), max_word_id))  # efficient arithmetic operations with CSR * CSR
    labels_cooc_matrix = docs_words_matrix.T * docs_words_matrix  # multiplying docs_words_matrix with its transpose matrix would generate the co-occurences matrix
    labels_cooc_matrix.setdiag(0)
    print(f"labels_cooc_matrix:\n{labels_cooc_matrix.todense()}")
    return labels_cooc_matrix, word_to_id 

def merge_all_data(config, split):
    labels = [json.loads(line) for line in open(config['scifact']['labels'][split])]
    claims = {}
    for line in open(config['scifact']['claims'][split]):
        line = json.loads(line)
        claims[line['id']] = line['claim']
    corpus = {}
    for line in open(config['scifact']['corpus']):
        line = json.loads(line)
        corpus[str(line['doc_id'])] = {
            'title': line['title'], 
            'abstract': line['abstract'], 
            'structured': line['structured']
        }
    label_to_int = lambda l: 0 if l=='NOT_ENOUGH_INFO' else 1
    examples = [] 
    
    cols = ['claim_id', 'claim_text', 'doc_sent_id', 'doc_id', 'sent_idx', 'sent', 'label_str', 'label_int']
    for ex in tqdm(labels):
        claim_id = ex['id']
        claim_text = claims[claim_id]
        for doc_id in ex['evidence'].keys():
            label_str = ex['evidence'][doc_id]['label']
            for sent_idx in ex['evidence'][doc_id]['sentences']:
                doc_sent_id = f'{doc_id}-{sent_idx}'
                sent = corpus[doc_id]['abstract'][sent_idx]
                if label_str == 'CONTRADICT' or label_str=='SUPPORT':
                    label_int = 1
                else:
                    label_int = 0
                examples.append(
                    [claim_id, claim_text, doc_sent_id, doc_id, sent_idx, sent, label_str, label_int]
                )
    df = pd.DataFrame(examples, columns=cols)
    df['drop'] = False
    for doc_sent_id in df['doc_sent_id'].unique():
        tmp = df[df['doc_sent_id']==doc_sent_id]
        claim_chosen = False
        if 1 in tmp['label_int'].to_list():
            for i, row in tmp.iterrows():
                if row['label_int'] == 0:
                    df.at[i, 'drop'] = True
                elif not claim_chosen:
                    claim_chosen = True
                else:
                    df.at[i, 'drop'] = True
        else:
            for i, row in tmp.iterrows():
                if claim_chosen:
                    df.at[i, 'drop'] = True
                else:
                    claim_chosen = True

    return df


def main(args):
    config = yaml.safe_load(open(args.data_config).read())
    train_all = merge_all_data(config, 'train')
    dev_all = merge_all_data(config, 'dev')
    
    train_all.to_csv(config['scifact']['merged']['train'], index=False)
    dev_all.to_csv(config['scifact']['merged']['dev'], index=False)

    train = train_all[train_all['drop']==False]
    dev = dev_all[dev_all['drop']==False]
    
    # Confirm all duplicates have been dropped
    assert len(train.groupby(['doc_id', 'sent_idx'], as_index=False).size()['size'].value_counts()) == 1
    assert len(dev.groupby(['doc_id', 'sent_idx'], as_index=False).size()['size'].value_counts()) == 1
    
    # Count number of dropped sentences
    print(f'Num dropped sent in train: {len(train_all) - len(train)} / {len(train_all)}')
    print(f'Num dropped sent in dev: {len(dev_all) - len(dev)} / {len(dev_all)}')
    
    # Print distributions
    print('Train label dist')
    print(train['label_int'].value_counts())
    print()
    print('Dev label dist')
    print(dev['label_int'].value_counts())
    print()
    
    # Text dist
    len_sent = lambda sent: len(sent.split())
    train['len_sent'] = train['sent'].apply(len_sent)
    dev['len_sent'] = dev['sent'].apply(len_sent)
    print(f'Train avg len sent: {train["len_sent"].mean()}')
    print(f'Train min len sent: {train["len_sent"].min()}')
    print(f'Train max len sent: {train["len_sent"].max()}')
    print()
    print(f'Dev avg len sent: {dev["len_sent"].mean()}')
    print(f'Dev min len sent: {dev["len_sent"].min()}')
    print(f'Dev max len sent: {dev["len_sent"].max()}')
    print()

    # Figure out overlap for repeated sents
    allowed_words = ['SUPPORT', 'CONTRADICT', 'NOT_ENOUGH_INFO']
    train_docs, dev_docs = [], []
    for doc_sent_id in train_all['doc_sent_id'].unique():
        tmp = train_all[train_all['doc_sent_id']==doc_sent_id]
        labels = tmp['label_str'].to_list()
        train_docs.append(labels)
    for doc_sent_id in dev_all['doc_sent_id'].unique():
        tmp = dev_all[dev_all['doc_sent_id']==doc_sent_id]
        labels = tmp['label_str'].to_list()
        dev_docs.append(labels)
    print('Train co-occ matrix')
    create_co_occurences_matrix(allowed_words, train_docs)
    print('Dev co-occ matrix')
    create_co_occurences_matrix(allowed_words, dev_docs)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-config', default='./scifact.yaml')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()
    main(args)
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')