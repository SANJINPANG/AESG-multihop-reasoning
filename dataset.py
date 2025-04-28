import json
import os
import copy
from torch.utils.data import Dataset
from dictionary import Dictionary
import torch
import sys
import numpy as np
import networkx as nx
from tqdm import tqdm
import random
from torch import sparse
import random
import torch
import os
from torch.utils.data import Dataset
from tqdm import tqdm
from copy import deepcopy

class Seq2SeqDataset(Dataset):
    def __init__(self, data_path="FB15K237/", vocab_file="FB15K237/vocab.txt", device="cpu", args=None):
        self.data_path = data_path
        self.src_file = os.path.join(data_path, "in_" + args.trainset + ".txt")
        if args.loop:
            self.tgt_file = os.path.join(data_path, "out_" + args.trainset + "_loop.txt")
        else:
            self.tgt_file = os.path.join(data_path, "out_" + args.trainset + ".txt")
        self.vocab_file = vocab_file
        self.device = device
        self.smart_filter = args.smart_filter
        self.args = args

        try:
            self.dictionary = Dictionary.load(vocab_file)
        except FileNotFoundError:
            self.dictionary = Dictionary()
            self._init_vocab()

        self.padding_idx = self.dictionary.pad()
        self.len_vocab = len(self.dictionary)

    def __len__(self):
        with open(self.src_file) as f:
            return sum(1 for _ in f)

    def _init_vocab(self):
        self.dictionary.add_symbol('LOOP')
        N = 0
        with open(self.data_path+'relation2id.txt') as fin:
            for line in fin:
                N += 1
        with open(self.data_path+'relation2id.txt') as fin:
            for line in fin:
                r, rid = line.strip().split('\t')
                rev_rid = int(rid) + N
                self.dictionary.add_symbol('R'+rid)
                self.dictionary.add_symbol('R'+str(rev_rid))
        with open(self.data_path+'entity2id.txt') as fin:
            for line in fin:
                e, eid = line.strip().split('\t')
                self.dictionary.add_symbol(eid)
        self.dictionary.save(self.vocab_file)

    def __getitem__(self, index):
        src_line, tgt_line = None, None
        with open(self.src_file) as fsrc, open(self.tgt_file) as ftgt:
            for i, (src, tgt) in enumerate(zip(fsrc, ftgt)):
                if i == index:
                    src_line = src.strip().split(' ')
                    tgt_line = tgt.strip().split(' ')
                    break

        source_id = self.dictionary.encode_line(src_line)
        target_id = self.dictionary.encode_line(tgt_line)
        l = len(target_id)
        mask = torch.ones_like(target_id)

        for i in range(0, l - 2):
            if i % 2 == 0:  # do not mask relation
                continue
            if random.random() < self.args.prob:  # randomly replace with prob
                target_id[i] = random.randint(0, self.len_vocab - 1)
                mask[i] = 0

        return {
            "id": index,
            "tgt_length": len(target_id),
            "source": source_id,
            "target": target_id,
            "mask": mask,
        }
    def get_next_valid(self):
        def initialize_value(smart_filter):
            return {} if smart_filter else []

        def read_triples(file_paths, valid_dict, smart_filter):
            for file_path in file_paths:
                with open(file_path, 'r') as f:
                    for line in tqdm(f, desc=f"Processing {file_path}"):
                        h, r, t = line.strip().split('\t')
                        hid = self.dictionary.indices[h]
                        rid = self.dictionary.indices[r]
                        tid = self.dictionary.indices[t]
                        er = rid * len(self.dictionary) + hid  # Unique key for entity-relation pair

                        if er not in valid_dict:
                            valid_dict[er] = initialize_value(smart_filter)
                        
                        if smart_filter:
                            valid_dict[er][tid] = 0
                        else:
                            valid_dict[er].append(tid)

        train_valid = {}
        eval_valid = {}
        read_triples([self.data_path + 'train_triples_rev.txt'], train_valid, self.smart_filter)
        read_triples([self.data_path + 'valid_triples_rev.txt', self.data_path + 'test_triples_rev.txt'], eval_valid, self.smart_filter)

        return train_valid, eval_valid

    def collate_fn(self, samples):
        lens = [sample["tgt_length"] for sample in samples]
        max_len = max(lens)
        bsz = len(lens)
        
        source = torch.LongTensor(bsz, 3).fill_(self.dictionary.bos())
        prev_outputs = torch.LongTensor(bsz, max_len).fill_(self.dictionary.pad())
        target = torch.LongTensor(bsz, max_len).fill_(self.dictionary.pad())
        mask = torch.zeros(bsz, max_len)

        ids = []
        for idx, sample in enumerate(samples):
            ids.append(sample["id"])
            source_ids = sample["source"]
            target_ids = sample["target"]

            source[idx, 1:] = source_ids[:2]
            prev_outputs[idx, :sample["tgt_length"]] = target_ids[:]
            target[idx, :sample["tgt_length"]] = target_ids
            mask[idx, :sample["tgt_length"]] = sample["mask"]

        return {
            "ids": torch.LongTensor(ids).to(self.device),
            "lengths": torch.LongTensor(lens).to(self.device),
            "source": source.to(self.device),
            "prev_outputs": prev_outputs.to(self.device),
            "target": target.to(self.device),
            "mask": mask.to(self.device),
        }



                
class TestDataset(Dataset):
    def __init__(self, data_path="FB15K237/", vocab_file="FB15K237/vocab.txt", device="cpu", src_file=None):

        if src_file:
            self.src_file = os.path.join(data_path, src_file)
        else:
            self.src_file = os.path.join(data_path, "valid_triples.txt")
            
        with open(self.src_file) as f:
            self.src_lines = f.readlines()

        self.vocab_file = vocab_file
        self.device = device
    
        try:
            self.dictionary = Dictionary.load(vocab_file)
        except FileNotFoundError:
            self.dictionary = Dictionary()
            self._init_vocab()
        self.padding_idx = self.dictionary.pad()
        self.len_vocab = len(self.dictionary)

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, index):
        
        src_line = self.src_lines[index].strip().split('\t')
        source_id = self.dictionary.encode_line(src_line[:2])
        target_id = self.dictionary.encode_line(src_line[2:])
        return {
            "id": index,
            "source": source_id,
            "target": target_id,
        }

    def collate_fn(self, samples):
        bsz = len(samples)
        source = torch.LongTensor(bsz, 3)
        target = torch.LongTensor(bsz, 1)

        source[:, 0].fill_(self.dictionary.bos())

        ids =  []
        for idx, sample in enumerate(samples):
            ids.append(sample["id"])
            source_ids = sample["source"]
            target_ids = sample["target"]
            source[idx, 1:] = source_ids[: -1]
            target[idx, 0] = target_ids[: -1]
        
        return {
            "ids": torch.LongTensor(ids).to(self.device),
            "source": source.to(self.device),
            "target": target.to(self.device)
        }
