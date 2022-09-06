'''
Script to train evidence filter model
'''
import os
import pickle 
import json
import torch
import sklearn
import argparse
import logging
import time
from pprint import pprint
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorWithPadding

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    print(torch.cuda.device_count())
    device = torch.device(args.device)
    model.to(device)
    imdb = load_dataset("imdb")
    preprocess_function = lambda examples: tokenizer(examples["text"], truncation=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_imdb = imdb.map(preprocess_function, batched=True)
    
    training_args = TrainingArguments(
        output_dir=args.outdir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_imdb["train"],
        eval_dataset=tokenized_imdb["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if args.mode == 'train':
        print('Training')
        trainer.train()
         
    predictions = trainer.predict(tokenized_imdb["test"])
    with open(os.path.join(args.outdir, 'preds.pkl'), 'wb') as pred_file:
        pickle.dump(predictions, pred_file)
    y_pred = [np.argmax(i) for i in predictions.predictions]
    metrics = sklearn.metrics.classification_report(predictions.label_ids, y_pred, output_dict=True)
    with open(os.path.join(args.outdir, 'metrics.json'), 'w') as metrics_file:
        json.dump(metrics, metrics_file)
    pprint(metrics)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="distilbert-base-uncased")
    parser.add_argument('--mode', default='train', choices=['train', 'test'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--outdir', default='./results')
    # parser.add_argument('--data', required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()
    main(args)
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')