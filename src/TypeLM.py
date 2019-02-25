from Transformer.src.utils import (FuzzyLoss, CustomLRScheduler, noam_scheme, Mask, PE, DecoderInput, sigsoftmax,
                                   EncoderInput)
from Transformer.src.Transformer import Encoder
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

from src import DataPrep

from typing import List, Any, Sequence, Union, Callable

FloatTensor = Union[torch.cuda.FloatTensor, torch.FloatTensor]
LongTensor = Union[torch.cuda.LongTensor, torch.LongTensor]


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
        self.device = device
        self.activation = activation
        self.shift = shift

    def forward(self, x: LongTensor, mask: LongTensor) -> FloatTensor:
        x_embedded = F.embedding(x, self.embedding_matrix, padding_idx=0, scale_grad_by_freq=True)
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

    def eval_epoch(self, X: Sequence[LongTensor], batch_size: int, val_indices: List[int]) -> Any:
        self.eval()

        with torch.no_grad():

            permutation = val_indices

            batch_start = 0
            BS, BTS, BW, BTW = 0, 0, 0, 0

            while batch_start < len(permutation):
                batch_end = min(batch_size + batch_start, len(permutation))

                batch_x = [X[permutation[i]] for i in range(batch_start, batch_end)]
                batch_x = pad_sequence(batch_x, batch_first=True).to(self.device)

                encoder_mask = Mask((batch_x.shape[0], batch_x.shape[1], batch_x.shape[1])).to(self.device)

                batch_p = self.forward(batch_x, encoder_mask)

                (bs, bts), (bw, btw) = accuracy(batch_p[:, :-1].argmax(dim=-1), batch_x[:, 1:], 0)
                BS += bs
                BTS += bts
                BW += bw
                BTW += btw

                batch_start += batch_size

            return BS, BTS, BW, BTW


def do_everything():
    type_sequences, type_dict, occurrences = DataPrep.type_language_model()

    d_model = 1024
    batch_size = 256
    num_epochs = 500

    num_classes = len(type_dict) + 1
    n = TypeLM(num_classes, 3, 4, 1024, 'cuda', 0.15, d_model)

    L = FuzzyLoss(torch.nn.KLDivLoss(reduction='batchmean'), num_classes, 0.5, ignore_index=0)
    # L = torch.nn.NLLLoss(reduction='mean', ignore_index=0)

    o = optim.Adam(n.parameters(), lr=2e-04, betas=(0.9, 0.98), eps=1e-09)
    o = CustomLRScheduler(o, [noam_scheme], d_model=d_model, warmup_steps=4000, batch_size=batch_size * 4)

    splitpoint = int(np.floor(0.1 * len(type_sequences)))

    indices = list(range(len(type_sequences)))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[splitpoint:], indices[:splitpoint]
    val_indices = sorted(val_indices, key=lambda idx: len(type_sequences[idx]))

    best_val = None
    steps = 0

    # N = 500
    # top_N = list(map(lambda x: x[0], occurrences[:N]))
    # top_N_labels = list(map(lambda x: str(x[1]), occurrences[:N]))
    # with open('labels.tsv', 'w') as f:
    #     cw = csv.writer(f)
    #     for item in top_N_labels:
    #         cw.writerow(item)
    # exit()
    # top_N = torch.tensor(top_N, device='cuda')

    with trange(num_epochs) as t:
        for i in t:
            # e = n.embedding_matrix(top_N).detach().to('cpu').numpy()
            # e = manifold.TSNE(2, early_exaggeration=24, n_iter=10000, perplexity=30).fit_transform(e)
            # first = list(map(lambda x: x[0], e))
            # second = list(map(lambda x: x[1], e))
            # # third = list(map(lambda x: x[2], e))
            # #
            # plt.figure(figsize=(11, 11))
            # plt.scatter(first, second)
            # plt.show()
            # plt.pause(0.02)

            try:
                loss, bs, bts, bw, btw, steps = n.train_epoch(type_sequences, batch_size, L, o, train_indices,
                                                              steps_taken=steps)
                train_acc = btw/bw
                if i % 5 == 0 and i > 0:
                    bs, bts, bw, btw = n.eval_epoch(type_sequences, batch_size, val_indices)
                    if best_val is None:
                        best_val = btw/bw
                    if btw/bw > best_val:
                        best_val = btw/bw
                        with open('stored_models/type_LM.p', 'wb') as f:
                            print('Storing at epoch {}'.format(i))
                            torch.save(n.state_dict(), f)
            except KeyboardInterrupt:
                return n, train_indices

            # if btw / bw > best_val:
            #     best_val = btw / bw
            #     with open('stored_models/type_LM.p', 'wb') as f:
            #         torch.save(n.state_dict(), f)
            # plt.close()
            t.set_postfix(loss=loss, steps=steps, lr=o.lrs[0], accuracy=train_acc, best_val=best_val)
    return n, train_indices
