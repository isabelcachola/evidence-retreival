from typing import Dict, List, NamedTuple
import numpy as np

### Precision/Recall

def safe_div(num, den):
    if den == 0:
        return 0
    return num / den

def interpolate(x1, y1, x2, y2, x):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m * x + b

def get_precision(num_docs_needed, is_relevant, results):
    # Recall = 0
    if num_docs_needed == 0:
        return 1.
    # Else
    n_rel_retrieved = 0
    i = 0
    while n_rel_retrieved < num_docs_needed:
        if is_relevant[results[i]]:
            n_rel_retrieved += 1
        i += 1
    return n_rel_retrieved / i

def precision_at(recall: float, results: List[int], relevant: List[int]) -> float:
    '''
    This function should compute the precision at the specified recall level.
    If the recall level is in between two points, you should do a linear interpolation
    between the two closest points. For example, if you have 4 results
    (recall 0.25, 0.5, 0.75, and 1.0), and you need to compute recall @ 0.6, then do something like

    interpolate(0.5, prec @ 0.5, 0.75, prec @ 0.75, 0.6)

    Note that there is implicitly a point (recall=0, precision=1).

    `results` is a sorted list of document ids
    `relevant` is a list of relevant documents
    '''

    is_relevant = {}
    num_relevant = len(relevant)
    for doc_id in results:
        is_relevant[doc_id] =  1 if doc_id in relevant else 0
    levels = [i/num_relevant for i in range(0, num_relevant+1)] 

    if recall in levels:
        num_docs_needed = levels.index(recall)
        prec = get_precision(num_docs_needed, is_relevant, results)
        return prec

    else:
        # Which recall levels to interpolate
        for i in range(len(levels)-1):
            if levels[i] < recall < levels[i+1]:
                x1, x2 = levels[i], levels[i+1]
        y1 = get_precision(levels.index(x1), is_relevant, results)
        y2 = get_precision(levels.index(x2), is_relevant, results)
        return interpolate(x1, y1, x2, y2, recall)
        
    return -1


def mean_precision1(results, relevant):
    return (precision_at(0.25, results, relevant) +
        precision_at(0.5, results, relevant) +
        precision_at(0.75, results, relevant)) / 3

def mean_precision2(results, relevant):
    sum_prec_at_k = 0
    for i in range(1,11):
        sum_prec_at_k += precision_at(i/10, results, relevant)

    return sum_prec_at_k/10

def norm_recall(results, relevant):
    n_relevant = len(relevant)
    num_1 = sum([
        results.index(doc)+1 for doc in relevant
        ])
    num_2 = sum([i for i in range(1,n_relevant+1)])
    den = n_relevant * ( len(results) - n_relevant)
    return 1 - ((num_1 - num_2)/den)

def norm_precision(results, relevant):
    rel = len(relevant)
    N = len(results)
    num1 = sum([
        np.log(results.index(doc)+1) for doc in relevant
    ])
    num2 = sum([np.log(i) for i in range(1,rel+1)])
    den = N*np.log(N) - (N-rel)*np.log(N-rel) - rel*np.log(rel)
    return 1 - ( (num1-num2)/den )

def match_highlighting(results, relevant, 
                        highlighted_tokens, true_highlighted,
                        strict=True):
    retreived = []
    for ret_idx, retrieved_doc in enumerate(results):
        if retrieved_doc in relevant:
            rel_idx = relevant.index(retrieved_doc)
            y_pred = highlighted_tokens[ret_idx]
            y_true = true_highlighted[rel_idx]
            # Check if highlight is contained within the true answer
            if (y_pred[0] >= y_true[0]) and (y_pred[1] <= y_true[1]):
                retreived.append(retrieved_doc)
            # If not strict, check for any overlap
            elif not strict:
                if y_true[0] <= y_pred[0] <= y_true[1]:
                    retreived.append(retrieved_doc)
                elif y_true[0] <= y_pred[1] <= y_true[1]:
                    retreived.append(retrieved_doc)
    return retreived

# Prec @ 1, Recall @ 3, RR
def eval_highlighting(results, highlighted_results):

    if results[0] in highlighted_results:
        precision_at_1 = 1.
    else: 
        precision_at_1 = 0.
    if len(set(results[:3]).intersection(highlighted_results)) > 0:
        recall_at_3 = 1.
    else:
        recall_at_3 = 0.
    min_rank = len(results)
    for hr in highlighted_results:
        rank = results.index(hr) + 1
        if rank < min_rank:
            min_rank = rank
    rr = 1/min_rank
    return precision_at_1, recall_at_3, rr
    # relevant = set(relevant)
    # tp = len(highlighted_results.intersection(relevant))
    # prec = safe_div(tp, len(highlighted_results))
    # recall = safe_div(tp, len(relevant))
    # f1 = 2* safe_div((prec*recall), (prec+recall))
    # return prec, recall, f1

def test_precision():
    recall_ = 0.6
    # fmt: off
    results_ = [2, 1, 3, 4, 7, 5, 6, 8, 9, 10, 11, 12, 13, 14]  # Lucas numbers for debugging.
    # fmt: on
    relevant_ = [1, 14, 2, 3, 5, 8, 13]  # Fibonacci numbers for debugging.
    print(precision_at(recall_, results_, relevant_))

    print(precision_at(0, results_, relevant_))