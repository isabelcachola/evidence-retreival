from copy import deepcopy
from typing import List
import numpy as np
from tqdm import tqdm
from pygaggle.rerank.base import Query, Text, Reranker

from transformers import (AutoTokenizer, AutoConfig,
                          AutoModelForSeq2SeqLM,
                          T5ForConditionalGeneration)
import torch
from pygaggle.rerank.base import Reranker, Query, Text
from pygaggle.rerank.transformer import MonoT5
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding

__all__ = ['DummyReranker', 'EvidenceFilter']


class DummyReranker(Reranker):
    def __init__(self):
        self.safe_ln = lambda x: 0 if x <= 0 else np.log(x)


    def rescore(self, query: Query, texts: List[Text]) -> List[Text]:
        texts = deepcopy(texts)
        for text in texts:
            text.score = self.safe_ln(np.random.normal(0.5, 1))
        return texts

class SciFact:
    def __init__(self, model, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        config = AutoConfig.from_pretrained(model, num_labels=3)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, config=config).eval().to(device)

        self.LABELS = ['CONTRADICT', 'NOT_ENOUGH_INFO', 'SUPPORT']

    def encode(self, sentences, claims):
        text =  list(zip(sentences, claims))
        encoded_dict = self.tokenizer.batch_encode_plus(
            text,
            pad_to_max_length=True,
            return_tensors='pt'
        )
        if encoded_dict['input_ids'].size(1) > 512:
            encoded_dict = self.tokenizer.batch_encode_plus(
                text,
                max_length=512,
                pad_to_max_length=True,
                truncation_strategy='only_first',
                return_tensors='pt'
            )
        encoded_dict = {key: tensor.to(self.device)
                    for key, tensor in encoded_dict.items()}
        return encoded_dict

    def predict(self, query, texts):
        output = []
        with torch.no_grad():
            for text in tqdm(texts, disable=True):
                if text.title is not None:
                    evidence = ' '.join([text.title, text.text])
                else:
                    evidence = text.text
                encoded_dict = self.encode([evidence], [query.text])
                label_scores = torch.softmax(self.model(**encoded_dict)[0], dim=1)[0]
                label_index = label_scores.argmax().item()
                label_confidence = label_scores[label_index].item()
                # results= {'label': self.LABELS[label_index], 
                #         'confidence': round(label_confidence, 4),
                #         'all_scores': label_scores.tolist()}
                results = (label_scores[0] + label_scores[2]).item()
                output.append(results)
        return output

class EvidenceFilter(Reranker):
    def __init__(self, base_model, device, batch_size, evidence_model_path):
        self.safe_ln = lambda x: 0 if x <= 0 else np.log(x)
        self.base_model = base_model
        # self.evidence_model = AutoModelForSequenceClassification.from_pretrained(evidence_model_path, num_labels=2)
        self.tokenizer = AutoTokenizer.from_pretrained(evidence_model_path)

        # training_args = TrainingArguments(
        #                     output_dir=evidence_model_path,
        #                     learning_rate=2e-5,
        #                     per_device_train_batch_size=batch_size,
        #                     per_device_eval_batch_size=batch_size,
        #                     num_train_epochs=1,
        #                     weight_decay=0.01,
        #                 )
        # data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        # self.evidence_filter = Trainer(
        #     model=self.evidence_model,
        #     args=training_args,
        #     tokenizer=self.tokenizer,
        #     data_collator=data_collator,
        # )
        # self.preprocess_function = lambda example: self.tokenizer(example.text, truncation=True)
        self.evidence_filter = SciFact(evidence_model_path, device) 

    # def preprocess(self, example):
    #     return self.tokenizer(example.text, truncation=True)

    def rescore(self, query: Query, texts: List[Text]) -> List[Text]:
        texts = deepcopy(texts)
        # proc_texts = [self.preprocess(ex) for ex in texts]
        evidence_preds = self.evidence_filter.predict(query, texts)
        texts = self.base_model.rescore(query, texts)
        for i, text in tqdm(enumerate(texts)):
            # text.score += evidence_preds[0][i, 1] 
            text.score += evidence_preds[i]

        return texts

    def train(self):
        pass

    def load(self, path):
        pass
