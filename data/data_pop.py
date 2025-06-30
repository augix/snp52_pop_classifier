# Purpose:
# - for each person, load the genome embeddings and his/her population label

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import random
random.seed(42)
import numpy as np
import pandas as pd

# Required config attributes:

class theDataset(Dataset):
    """
    Dataset for population classification.
    input: genome embeddings
    target: population label
    """
    def __init__(self, config):
        self.config = config
        self.init_labels(config.fn_labels)
        self.init_emb(config.fn_emb)
        print('Dataset size:', self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        each item is a genome embedding and its population label.
        """
        record = self.get_record(idx)
        record = self.masking(record)
        if self.config.add_cls:
            record = self.add_cls(record)
        return record

    def init_labels(self, fn):
        print(f'init_labels: {fn}')
        df = pd.read_csv(fn, sep='\t', header=0)
        df['Population_i'] = df['Population'].astype('category').cat.codes
        self.labels = df['Population_i']
        self.size = len(self.labels)
        print(f'{len(self.labels)} labels loaded')
    
    def init_emb(self, fn_emb):
        print(f'init_emb: {fn_emb}')
        unique_person_list, global_contig_dict, emb_dict = torch.load(fn_emb)
        self.person_list = unique_person_list
        self.global_contig_dict = global_contig_dict
        self.emb_dict = emb_dict
        print(f'{len(self.person_list)} persons loaded.')
        print(f'{len(self.global_contig_dict[0])} contigs loaded for person 0.')

    def get_seq(self, person):
        contig = self.global_contig_dict[int(person)] # [seq_len]
        emb = self.emb_dict[int(person)] # [seq_len, d_emb]
        emb = emb.to(torch.float32)
        if self.config.add_cls:
            # contig = [i+self.config.n_cls for i in contig]
            contig = contig + torch.tensor(self.config.n_cls, dtype=torch.int32)
        return contig, emb

    def get_record(self, idx):
        person = self.person_list[idx]
        population = self.labels[idx]
        population = torch.tensor(population, dtype=torch.int64)

        contig, emb = self.get_seq(person)

        record = {
            'person': person,
            'population': population,
            'contig': contig,
            'emb': emb,
        }
        return record

    def masking(self, record):
        emb = record['emb']
        mask = torch.rand(emb.shape, dtype=torch.float32) < self.config.mask_fraction
        emb[mask] = 0
        record['emb'] = emb
        return record

    def add_cls(self, record):
        cls_token = torch.zeros(self.config.n_cls, dtype=torch.int32) # 0 as pos_id, value, mask
        cls_token_emb = torch.zeros(self.config.n_cls, self.config.d_emb, dtype=torch.float32)
        record['contig'] = torch.cat([cls_token, record['contig']], dim=0)
        record['emb'] = torch.cat([cls_token_emb, record['emb']], dim=0)
        return record

    def show_example(self):
        for i in [0, 1, 2, self.size-1]:
            record = self.__getitem__(i)
            print(f'record {i}: {record}')

class theDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        if stage == 'train':
            self.setup_train()
        elif stage == 'test':
            self.setup_test()
    
    def setup_train(self):
        print('data_emb.py: creating training dataset...')
        self.config.fn_emb = self.config.fn_emb_train
        self.config.fn_labels = self.config.fn_labels_train
        self.dataset_train = theDataset(self.config)

        print('data_emb.py: creating validation dataset...')
        self.config.fn_emb = self.config.fn_emb_val
        self.config.fn_labels = self.config.fn_labels_val
        self.dataset_val = theDataset(self.config)

    def setup_test(self):
        print('data_emb.py: creating test dataset...')
        self.config.fn_emb = self.config.fn_emb_val
        self.config.fn_labels = self.config.fn_labels_val
        self.dataset_test = theDataset(self.config)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.config.bs_train, shuffle=True, drop_last=True, num_workers=self.config.cpu_per_worker, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.config.bs_val, shuffle=False, drop_last=False, num_workers=self.config.cpu_per_worker, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.config.bs_test, shuffle=False, drop_last=False, num_workers=self.config.cpu_per_worker, pin_memory=True)