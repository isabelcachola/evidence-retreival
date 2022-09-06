'''
Script 
'''
import argparse
import logging
import time

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from datasets import load_dataset

def main(ags):
    model = 'google/bigbird-roberta-large'
    qa = pipeline("question-answering", model=model, device=args.device)
    dataset = load_dataset("natural_questions")
    answer = qa(question="What is my name?", context="My name is Isabel")
    import ipdb;ipdb.set_trace()
    print()



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=-1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()
    main(args)
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')