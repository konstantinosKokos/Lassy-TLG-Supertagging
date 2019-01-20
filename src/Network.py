import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..'))

from Transformer.src.Transformer import Transformer, Mask

import torch
from torch import nn, Tensor
from src.DataPrep import TLGDataset
from torch import optim
from torch.nn import NLLLoss
from torch.nn.utils.rnn import pad_sequence

import numpy as np

from src import DataPrep

from typing import List, Any, Tuple


def accuracy(predictions: Tensor, truth: Tensor, ignore_idx: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    predictions = predictions.argmax(dim=-1)
    correct_words = torch.ones(predictions.size())
    correct_words[predictions != truth] = 0
    correct_words[truth == ignore_idx] = 1

    correct_sentences = correct_words.prod(dim=1)
    num_correct_sentences = correct_sentences.sum().item()

    num_correct_words = correct_words.sum().item()
    num_masked_words = len(truth[truth == ignore_idx])

    return (predictions.shape[0], num_correct_sentences), \
           (predictions.shape[0] * predictions.shape[1] - num_masked_words, num_correct_words - num_masked_words)


class Supertagger(nn.Module):
    def __init__(self, num_classes: int, num_heads: int, encoder_layers: int, decoder_layers: int, d_intermediate: int,
                 device: str) -> None:
        super(Supertagger, self).__init__()
        self.num_classes = num_classes
        self.embedder = nn.Embedding(num_classes, 300, padding_idx=0, scale_grad_by_freq=True).to(device)
        self.transformer = Transformer(num_classes=num_classes, output_embedder=self.embedder, num_heads=num_heads,
                                       encoder_layers=encoder_layers, decoder_layers=decoder_layers,
                                       d_intermediate=d_intermediate, device=device)
        self.device = device

    def forward(self, encoder_input: Tensor, decoder_input: Tensor, encoder_mask: Tensor, decoder_mask: Tensor) \
            -> Tensor:
        return self.transformer.forward(encoder_input, decoder_input, encoder_mask, decoder_mask)

    def train_epoch(self, dataset: TLGDataset, batch_size: int, criterion: NLLLoss,
                    optimizer: optim.Optimizer, train_indices: List[int]) -> Any:
        self.train()

        permutation = np.random.permutation(train_indices)

        batch_start = 0
        loss = 0.
        BS, BTS, BW, BTW = 0, 0, 0, 0

        while batch_start < len(permutation):
            optimizer.zero_grad()
            batch_end = min([batch_start + batch_size, len(permutation)])

            batch_x = [dataset.X[permutation[i]] for i in range(batch_start, batch_end)]
            batch_y = [dataset.Y[permutation[i]] for i in range(batch_start, batch_end)]

            lens = list(map(len, batch_x))

            batch_x = pad_sequence(batch_x, batch_first=True).to(self.device)
            batch_y = pad_sequence(batch_y, batch_first=True).long().to(self.device)  # todo
            batch_e = self.embedder(batch_y).to(self.device)

            encoder_mask = torch.ones(batch_y.shape[0], batch_y.shape[1], batch_y.shape[1])
            # for i, l in enumerate(lens):
            #     encoder_mask[i, l::, :] = torch.zeros([1, batch_x.shape[1] - l, batch_x.shape[1]])
                # encoder_mask[i, :, l::] = torch.zeros([1, batch_x.shape[1], batch_x.shape[1] - l])
            encoder_mask = encoder_mask.to(self.device)
            # decoder_mask = torch.ones(batch_y.shape[0], batch_y.shape[1], batch_y.shape[1]).to(self.device)
            decoder_mask = Mask((batch_y.shape[0], batch_y.shape[1], batch_y.shape[1])).to(self.device)
            batch_p = self.forward(batch_x, batch_e, encoder_mask, decoder_mask)

            batch_loss = criterion(batch_p[:, :-1].permute(0, 2, 1), batch_y[:, 1:])
            loss += batch_loss
            batch_loss.backward()
            optimizer.step()
            (bs, bts), (bw, btw) = accuracy(batch_p[:, :-1], batch_y[:, 1:], 0)
            BS += bs
            BTS += bts
            BW += bw
            BTW += btw

            batch_start += batch_size

        return loss, BS, BTS, BW, BTW

    def eval_epoch(self, dataset: TLGDataset, batch_size: int, val_indices: List[int]) -> Any:
        self.eval()

        with torch.no_grad():

            permutation = val_indices

            batch_start = 0
            BS, BTS, BW, BTW = 0, 0, 0, 0

            while batch_start < len(permutation):
                batch_end = min([batch_start + batch_size, len(permutation)])

                batch_x = [dataset.X[permutation[i]] for i in range(batch_start, batch_end)]
                batch_y = [dataset.Y[permutation[i]] for i in range(batch_start, batch_end)]

                lens = list(map(len, batch_x))

                batch_x = pad_sequence(batch_x, batch_first=True).to(self.device)
                batch_y = pad_sequence(batch_y, batch_first=True).long().to(self.device)  # todo

                encoder_mask = torch.ones(batch_y.shape[0], batch_y.shape[1], batch_y.shape[1])
                # for i, l in enumerate(lens):
                #     encoder_mask[i, l::, :] = torch.zeros([1, batch_x.shape[1] - l, batch_x.shape[1]])
                # encoder_mask[i, :, l::] = torch.zeros([1, batch_x.shape[1], batch_x.shape[1] - l])
                encoder_mask = encoder_mask.to(self.device)
                # decoder_mask = torch.ones(batch_y.shape[0], batch_y.shape[1], batch_y.shape[1]).to(self.device)
                batch_p = self.transformer.infer(batch_x, encoder_mask, dataset.type_dict['SOS'])
                (bs, bts), (bw, btw) = accuracy(batch_p[:, :-1], batch_y[:, 1:], 0)
                BS += bs
                BTS += bts
                BW += bw
                BTW += btw

                batch_start += batch_size

        return BS, BTS, BW, BTW


def do_everything():
    tlg = DataPrep.do_everything()
    num_classes = len(tlg.type_dict) + 1  # todo
    n = Supertagger(num_classes, 4, 6, 6, 128, 'cuda')
    L = NLLLoss(ignore_index=0, reduction='sum')
    o = optim.Adam(n.parameters(), lr=2e-04)

    splitpoint = int(np.floor(0.25*len(tlg)))
    indices = list(range(len(tlg)))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[splitpoint:], indices[:splitpoint]  # todo

    for i in range(1000):
        loss, bs, bts, bw, btw = n.train_epoch(tlg, 128, L, o, train_indices)
        print('Epoch {}'.format(i))
        print(' Loss: {}, Sentence Accuracy: {}, Word Accuracy: {}'.format(loss, bts/bs, btw/bw))
        if i % 5 == 0:
            bs, bts, bw, btw = n.eval_epoch(tlg, 32, val_indices)
            print(' VALIDATION Sentence Accuracy: {}, Word Accuracy: {}'.format(bts / bs, btw / bw))
