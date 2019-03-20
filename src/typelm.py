from Transformer.src.utils import (FuzzyLoss, CustomLRScheduler, noam_scheme, Mask, PE, DecoderInput, sigsoftmax,
                                   EncoderInput)
from Transformer.src.Transformer import Encoder
from Transformer.src.utils import ScaledDotProduct
from src.Network import accuracy

from LassyExtraction.src.utils.PostProcess import count_occurrences, freqsort

import torch
from torch import nn
from torch.nn import functional as F

from tqdm import trange

import pickle
from sklearn import manifold
from matplotlib import pyplot as plt

import csv

from torch import optim
from torch.nn.utils.rnn import pad_sequence

import numpy as np

from src import dataprep

from typing import List, Any, Sequence, Union, Callable

FloatTensor = Union[torch.cuda.FloatTensor, torch.FloatTensor]
LongTensor = Union[torch.cuda.LongTensor, torch.LongTensor]


# class AtomicTypeLM(nn.Module):
#     def __init__(self, num_classes: int, d_model: int, device: str):
#         super(AtomicTypeLM, self).__init__()
#         self.device = device
#         self.embedding_matrix_k = torch.nn.Parameter(torch.rand(num_classes, d_model, device=device) * 0.02)
#         self.embedding_matrix_k = torch.nn.Parameter(torch.rand(num_classes, d_model, device=device) * 0.02)
#         self.embedding_matrix_k = torch.nn.Parameter(torch.rand(num_classes, d_model, device=device) * 0.02)
#
#     def forward(self, x: LongTensor, mask: LongTensor):
#         k = self.embedding_matrix_k(x)
#         q = self.embedding_matrix_k(x)
#         v = self.embedding_matrix_k(x)
#         b, n, dk = k.shape
#         pe = PE(b, n, dk, dk, device=self.device)
#
#         atn = ScaledDotProduct(q + pe, k + pe, v + pe, mask)
#
#         return torch.log(sigsoftmax(atn@self.embedding_matrix_k.transpose(1, 0) + 1e-10))
#
#     def train_batch(self, batch_x: LongTensor, encoder_mask: LongTensor,
#                     criterion: Callable[[FloatTensor, LongTensor], FloatTensor], optimizer: optim.Optimizer) -> Any:
#         optimizer.zero_grad()
#         batch_p = self.forward(batch_x, encoder_mask)
#         batch_loss = criterion(batch_p[:, :-1].permute(0, 2, 1), batch_x[:, 1:])
#         batch_loss.backward()
#         optimizer.step()
#         return batch_p.detach(), batch_loss.item()
#
#     def train_epoch(self):


class TypeLM(nn.Module):
    def __init__(self, num_classes: int, num_heads: int, num_layers: int, d_intermediate: int,
                 device: str, dropout: float = 0.1, d_model: int = 300, shift: int=1,
                 activation: Callable[[FloatTensor], FloatTensor]=sigsoftmax) -> None:
        super(TypeLM, self).__init__()
        self.embedding_matrix = torch.nn.Parameter(torch.rand(num_classes, d_model, device=device) * 0.02)
        # self.embedding_matrix = torch.nn.Embedding(num_classes, d_model, padding_idx=0,
        #                                            scale_grad_by_freq=True).to(device)
        self.network = Encoder(num_layers=num_layers, num_heads=num_heads, d_model=d_model,
                               d_k=d_model//num_heads, d_v=d_model//num_heads,
                               d_intermediate=d_intermediate, dropout=dropout).to(device)
        # self.predictor = torch.nn.Linear(d_model, num_classes).to(device)
        self.dropout_rate = dropout
        self.device = device
        self.activation = activation
        self.shift = shift

    def forward(self, x: LongTensor, mask: LongTensor) -> FloatTensor:
        x_embedded = F.embedding(x, self.embedding_matrix, padding_idx=0, scale_grad_by_freq=True)
        x_embedded = F.dropout(x_embedded, p=self.dropout_rate, training=self.training)
        b, n, dk = x_embedded.shape
        pe = PE(b, n, dk, dk, device=self.device)

        decoder_input = EncoderInput(encoder_input=x_embedded + pe, mask=mask)
        decoder_output = self.network(decoder_input)
        prediction = decoder_output.encoder_input@(self.embedding_matrix.transpose(1, 0) + 1e-10)
        # prediction = self.predictor(decoder_output.encoder_input)
        return torch.log(self.activation(prediction))

    def train_batch(self, batch_x: LongTensor, encoder_mask: LongTensor,
                    criterion: Callable[[FloatTensor, LongTensor], FloatTensor], optimizer: optim.Optimizer) -> Any:
        batch_p = self.forward(batch_x, encoder_mask)
        batch_loss = criterion(batch_p[:, :-1].permute(0, 2, 1), batch_x[:, 1:])
        batch_loss.backward()
        optimizer.step()
        return batch_p.detach(), batch_loss.item()

    def train_epoch(self, X: Sequence[LongTensor], batch_size: int,
                    criterion: Callable[[FloatTensor, LongTensor], FloatTensor],
                    optimizer: optim.Optimizer, train_indices: List[int],
                    steps_taken: int=0) -> Any:
        self.train()

        permutation = np.random.permutation(train_indices)

        batch_start = 0
        loss = 0.

        BS, BTS, BW, BTW = 0, 0, 0, 0

        while batch_start < len(permutation):
            optimizer.zero_grad()
            batch_end = min(batch_size + batch_start, len(permutation))

            batch_x = [X[permutation[i]].detach() for i in range(batch_start, batch_end)]
            batch_x = pad_sequence(batch_x, batch_first=True).to(self.device)
            encoder_mask = Mask((batch_x.shape[0], batch_x.shape[1], batch_x.shape[1])).to(self.device)
            batch_p, batch_loss = self.train_batch(batch_x, encoder_mask, criterion, optimizer)
            loss += batch_loss
            steps_taken += 1
            (bs, bts), (bw, btw) = accuracy(batch_p[:, :-1].argmax(dim=-1), batch_x[:, 1:], 0)
            BS += bs
            BTS += bts
            BW += bw
            BTW += btw

            batch_start += batch_size

        return loss, BS, BTS, BW, BTW, steps_taken

    def eval_epoch(self, X: Sequence[LongTensor], batch_size: int,
                   criterion: Callable[[FloatTensor, LongTensor], FloatTensor], val_indices: List[int]) -> Any:
        self.eval()

        with torch.no_grad():

            permutation = val_indices

            batch_start = 0
            BS, BTS, BW, BTW = 0, 0, 0, 0

            loss = 0.

            while batch_start < len(permutation):
                batch_end = min(batch_size + batch_start, len(permutation))

                batch_x = [X[permutation[i]] for i in range(batch_start, batch_end)]
                batch_x = pad_sequence(batch_x, batch_first=True).to(self.device)

                encoder_mask = Mask((batch_x.shape[0], batch_x.shape[1], batch_x.shape[1])).to(self.device)

                batch_p = self.forward(batch_x, encoder_mask)
                loss += criterion(batch_p[:, :-1].permute(0, 2, 1), batch_x[:, 1:]).item()

                (bs, bts), (bw, btw) = accuracy(batch_p[:, :-1].argmax(dim=-1), batch_x[:, 1:], 0)
                BS += bs
                BTS += bts
                BW += bw
                BTW += btw

                batch_start += batch_size

            return loss, BS, BTS, BW, BTW


def atomic_do_everything(data_path='data/symbols.p'):
    # type_sequences, type_dict = DataPrep.atomic_type_language_model(data_path)
    type_sequences, type_dict = dataprep.bpe_type_language_model()

    d_model = 300
    batch_size = 128
    num_epochs = 500

    num_classes = len(type_dict) + 1
    n = TypeLM(num_classes, 4, 4, 300, 'cuda', 0.5, d_model)

    L = FuzzyLoss(torch.nn.KLDivLoss(reduction='batchmean'), num_classes, 0.5, ignore_index=0)

    o = optim.Adam(n.parameters(), lr=2e-04, betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-04)
    o = CustomLRScheduler(o, [noam_scheme], d_model=d_model, warmup_steps=4000)

    train_indices = list(range(len(type_sequences)))

    steps = 0

    with trange(num_epochs) as t:
        for i in t:
            try:
                loss, bs, bts, bw, btw, steps = n.train_epoch(type_sequences, batch_size, L, o, train_indices,
                                                              steps_taken=steps)
                train_acc = btw / bw

            except KeyboardInterrupt:
                return n

            t.set_postfix(loss=loss, accuracy=train_acc, steps=steps, lr=o.lrs[0])
    return n


def do_everything(data_path='data/XYZ_ccg.p', split_path='split_ccg.p', store_path='stored_models/type_LM_ccg.p'):
    type_sequences, type_dict = dataprep.type_language_model(data_path)

    d_model = 300
    batch_size = 256
    num_epochs = 500

    num_classes = len(type_dict) + 1
    n = TypeLM(num_classes, 4, 6, 300, 'cuda', 0.2, d_model)

    L = FuzzyLoss(torch.nn.KLDivLoss(reduction='batchmean'), num_classes, 0.1, ignore_index=0)

    o = optim.Adam(n.parameters(), lr=2e-04, betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-04)
    o = CustomLRScheduler(o, [noam_scheme], d_model=d_model, warmup_steps=4000, batch_size=batch_size * 8)

    with open(split_path, 'rb') as f:
        train_indices, val_indices, _ = pickle.load(f)

    resplit = 0.75
    val_indices = val_indices + train_indices[int(np.floor(resplit*len(train_indices))):]
    train_indices = train_indices[:int(np.ceil(resplit * len(train_indices)))]

    v_loss = None
    best_val = None
    steps = 0
    v_accuracy = None

    with trange(num_epochs) as t:
        for i in t:
            try:
                loss, bs, bts, bw, btw, steps = n.train_epoch(type_sequences, batch_size, L, o, train_indices,
                                                              steps_taken=steps)
                train_acc = btw/bw
                if i % 5 == 0 and i > 0:
                    v_loss, val_bs, val_bts, val_bw, val_btw = n.eval_epoch(type_sequences, batch_size, L, val_indices)
                    v_accuracy = val_btw/val_bw
                    if best_val is None:
                        best_val = v_loss
                    if v_loss < best_val:
                        best_val = v_loss
                        with open(store_path, 'wb') as f:
                            print('Storing at epoch {}'.format(i, v_loss))
                            torch.save(n.state_dict(), f)
            except KeyboardInterrupt:
                return n

            t.set_postfix(loss=loss, v_loss=v_loss, best_v_loss=best_val, accuracy=train_acc, v_accuracy=v_accuracy,
                          steps=steps, lr=o.lrs[0])
    return n
