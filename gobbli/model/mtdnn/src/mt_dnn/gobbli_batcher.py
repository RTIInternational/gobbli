# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import csv
import json
import logging
import os
import pickle as pkl
import random
import string
import sys
from shutil import copyfile

import numpy as np

import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer

UNK_ID=100
BOS_ID=101


def tokenize_sample(bert_tokenizer, sample, max_seq_len, has_labels, labels):
    """
    Run tokenization for a singel-sentence task.
    """
    X = bert_tokenizer.tokenize(sample['X'])
    if has_labels:
        y = sample['y']
    if len(X) >  max_seq_len - 3:
        X = X[:max_seq_len - 3]
    input_ids = bert_tokenizer.convert_tokens_to_ids(['[CLS]'] + X + ['[SEP]'])
    token_type_ids = [0] * ( len(X) + 2)
    tokenized = {'input_ids': input_ids, 'token_type_ids': token_type_ids}
    if has_labels:
        tokenized['y'] = labels[y]
    return tokenized


class GobbliBatchGen:
    def __init__(self, path, batch_size=32, gpu=True, labels=None,
                 has_labels=True, is_train=True, dropout_w=0.005, maxlen=128):
        self.batch_size = batch_size
        self.has_labels = has_labels
        self.gpu = gpu
        self.labels = labels
        self.is_train = is_train
        # Explicit cache dir required for some reason -- default doesn't exist in the docker
        # container, maybe?
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='/tmp')
        self.data = self.load(path, maxlen, has_labels)
        if self.is_train:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            data = [self.data[i] for i in indices]
        self.data = GobbliBatchGen.make_batches(self.data, batch_size)
        self.offset = 0
        self.dropout_w = dropout_w

    @staticmethod
    def make_batches(data, batch_size=32):
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def load(self, path, maxlen, has_labels):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample = tokenize_sample(self.bert_tokenizer, row, maxlen, has_labels, self.labels)
                if self.is_train:
                    # TODO why is this needed?
                    if len(sample['input_ids']) > maxlen:
                        continue
                data.append(sample)

        print('Loaded {} samples'.format(len(data)))
        return data

    def reset(self):
        if self.is_train:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            self.data = [self.data[i] for i in indices]
        self.offset = 0

    def __random_select__(self, arr):
        if self.dropout_w > 0:
            return [UNK_ID if random.uniform(0, 1) < self.dropout_w else e for e in arr]
        else: return arr

    def __len__(self):
        return len(self.data)

    def patch(self, v):
        v = v.cuda(async=True)
        return v

    @staticmethod
    def todevice(v, device):
        v = v.to(device)
        return v


    def __iter__(self):
        while self.offset < len(self):
            batch = self.data[self.offset]
            batch_size = len(batch)
            tok_len = max(len(x['input_ids']) for x in batch)
            input_ids = torch.LongTensor(batch_size, tok_len).fill_(0)
            token_type_ids = torch.LongTensor(batch_size, tok_len).fill_(0)
            attention_mask = torch.LongTensor(batch_size, tok_len).fill_(0)

            for i, sample in enumerate(batch):
                select_len = min(len(sample['input_ids']), tok_len)
                tok = sample['input_ids']
                if self.is_train:
                    tok = self.__random_select__(tok)
                input_ids[i, :select_len] = torch.LongTensor(tok[:select_len])
                token_type_ids[i, :select_len] = torch.LongTensor(sample['token_type_ids'][:select_len])
                attention_mask[i, :select_len] = torch.LongTensor([1] * select_len)

            batch_data = [input_ids, token_type_ids, attention_mask]

            if self.has_labels:
                labels = [sample['y'] for sample in batch]
                batch_data.append(torch.LongTensor(labels))

            if self.gpu:
                for i, item in enumerate(batch_data):
                    batch_data[i] = self.patch(item.pin_memory())

            self.offset += 1

            if self.has_labels:
                yield batch_data[0], batch_data[1], batch_data[2], batch_data[3]
            else:
                yield batch_data[0], batch_data[1], batch_data[2]
