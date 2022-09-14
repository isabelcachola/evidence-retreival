'''
Script to train evidence filter model
'''
from ast import parse
import os
import pickle 
import json
from typing import Dict
import torch
import sklearn
import argparse
import logging
import time
import yaml
from tqdm import tqdm
from pprint import pprint
import numpy as np
import transformers
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
import datasets
from typing import Dict
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding

def merge_labels_with_corpus(labels, corpus) -> Dict:
    examples = {} # (docid, sent_idx): {label: LABEL, text: TEXT, str_label: [STR_LABEL, ...]}
    for claim in tqdm(labels):
        for doc_id in claim['evidence'].keys():
            if doc_id in corpus:
                str_label = claim['evidence'][doc_id]['label']
                for sent_idx in claim['evidence'][doc_id]['sentences']:
                    if str_label == 'CONTRADICT' or str_label=='SUPPORT':
                        label = 1
                    else:
                        label = 0
                    # check if (doc_id, sent_idx) is already represented
                    if (doc_id, sent_idx) in examples:
                        if examples[(doc_id, sent_idx)]['label'] == 0:
                            # print('HERE2')
                            examples[(doc_id, sent_idx)]['label']=  label
                            examples[(doc_id, sent_idx)]['text'] = corpus[doc_id]['abstract'][sent_idx]
                            if str_label not in examples[(doc_id, sent_idx)]['str_label']:
                                examples[(doc_id, sent_idx)]['str_label'].append(str_label)
                    else:
                        examples[(doc_id, sent_idx)] = {
                            'label': label,
                            'text': corpus[doc_id]['abstract'][sent_idx],
                            'str_label': [str_label]
                        }
            else:
                logging.warning(f'{doc_id} not in corpus.')
    data = {
            'doc_id': [], 
            'sent_idx': [],
            'text': [],
            'label': [],
            'str_label': []
            }   
    for key,value in examples.items():
        data['doc_id'].append(key[0])
        data['sent_idx'].append(key[1])
        data['text'].append(value['text'])
        data['label'].append(value['label'])
        data['str_label'].append(value['str_label'])

    return data

def load_scifact(config_path) -> DatasetDict:
    config = yaml.safe_load(open(config_path).read())
    # load labels and corpus
    train_labels = [json.loads(line) for line in open(config['scifact']['train'])]
    dev_labels = [json.loads(line) for line in open(config['scifact']['dev'])]
    corpus = {}
    for line in open(config['scifact']['corpus']):
        line = json.loads(line)
        corpus[str(line['doc_id'])] = {
            'title': line['title'], 
            'abstract': line['abstract'], 
            'structured': line['structured']
        }
    # merge corpus with labels
    train = Dataset.from_dict(merge_labels_with_corpus(train_labels, corpus))
    dev = Dataset.from_dict(merge_labels_with_corpus(dev_labels, corpus))
    return  DatasetDict({"train": train, "dev": dev})

def load_imdb():
    return load_dataset("imdb")

def main(args):
    start = time.time()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    preprocess_function = lambda examples: tokenizer(examples["text"], truncation=True)

    device = torch.device(args.device)
    model.to(device)
    data = load_scifact(args.data_config)
    tokenized_data = data.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=args.outdir,
        overwrite_output_dir=True,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        weight_decay=0.01,
        logging_steps=1,
        logging_strategy='epoch',
        logging_dir='./logs',
        save_strategy='epoch'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["dev"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    if args.mode == 'train':
        print('Training')
        trainer.train()
         
    predictions = trainer.predict(tokenized_data["dev"])
    with open(os.path.join(args.outdir, 'preds.pkl'), 'wb') as pred_file:
        pickle.dump(predictions, pred_file)
    y_pred = [np.argmax(i) for i in predictions.predictions]
    print(y_pred)
    metrics = sklearn.metrics.classification_report(predictions.label_ids, y_pred, output_dict=True)
    with open(os.path.join(args.outdir, 'metrics.json'), 'w') as metrics_file:
        json.dump(metrics, metrics_file)
    pprint(metrics)

    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="distilbert-base-uncased")
    parser.add_argument('--mode', default='train', choices=['train', 'test'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--outdir', default='./results')
    parser.add_argument('--data-config', default='./scifact.yaml')
    args = parser.parse_args()


    main(args)
