import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.nn.utils.rnn import pad_sequence

from typing import Union, Callable, List, Tuple

from src import dataprep
from src.dataprep import TLGDataset

import numpy as np

try:
    from src.utils import *
except ImportError:
    from Transformer.src.utils import *

import pickle
import os

FloatTensor = Union[torch.cuda.FloatTensor, torch.FloatTensor]
LongTensor = Union[torch.cuda.LongTensor, torch.LongTensor]


def accuracy(predictions: LongTensor, truth: LongTensor, ignore_idx: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    correct_words = torch.ones(predictions.size())
    correct_words[predictions != truth] = 0
    correct_words[truth == ignore_idx] = 1

    correct_sentences = correct_words.prod(dim=1)
    num_correct_sentences = correct_sentences.sum().item()

    num_correct_words = correct_words.sum().item()
    num_masked_words = len(truth[truth == ignore_idx])

    return (predictions.shape[0], num_correct_sentences), \
           (predictions.shape[0] * predictions.shape[1] - num_masked_words, num_correct_words - num_masked_words)


class BaselineLSTM(nn.Module):
    def __init__(self, num_classes: int, d_src: int, d_h: int, d_trg: int, encoder_layers: int, decoder_layers: int,
                 device: str, dropout: float=0.1, activation: Callable[[FloatTensor], FloatTensor]=sigsoftmax):
        super(BaselineLSTM, self).__init__()
        self.num_classes = num_classes
        self.encoder = nn.LSTM(input_size=d_src, hidden_size=d_h, num_layers=encoder_layers, dropout=dropout,
                               bidirectional=True, batch_first=True).to(device)
        self.embedding_matrix = torch.nn.Parameter(torch.rand(num_classes, d_trg, device=device) * 0.02)
        self.output_embedder = lambda x: F.embedding(x, self.embedding_matrix, padding_idx=0, scale_grad_by_freq=True)
        self.decoder = nn.LSTM(input_size=d_trg, hidden_size=2 * d_h, num_layers=decoder_layers, dropout=dropout,
                               bidirectional=False, batch_first=True).to(device)
        self.predictor = lambda x: x @ (self.embedding_matrix.transpose(1, 0) + 1e-10)
        self.device = device
        self.activation = activation

    def forward(self, encoder_input: FloatTensor, decoder_input: FloatTensor):
        _, (sentence_encoding, _) = self.encoder(encoder_input)
        sentence_encoding = sentence_encoding.view(1, -1, 2 * self.encoder.hidden_size)
        type_decoding, _ = self.decoder(decoder_input, (sentence_encoding, torch.zeros_like(sentence_encoding,
                                                                                         device=self.device)))
        prediction = self.predictor(type_decoding)
        return torch.log(self.activation(prediction))

    def infer(self, encoder_input: FloatTensor, max_len: int, sos_symbol: int):
        _, (h, _) = self.encoder(encoder_input)
        h = h.view(1, -1, 2 * self.encoder.hidden_size)
        b = h.shape[1]
        td = torch.ones(b, device=self.device, dtype=torch.long) * sos_symbol
        td = self.output_embedder(td).unsqueeze(1)
        c = torch.zeros((1, b, self.decoder.hidden_size), device=self.device)

        probs = torch.Tensor().to(self.device)

        for i in range(max_len):
            next, (h, c) = self.decoder(td, (h, c))
            next = self.activation(self.predictor(next))
            probs = torch.cat((probs, next), dim=1)
            next = next.argmax(dim=-1)
            td = self.output_embedder(next)
        return probs

    def train_epoch(self, dataset: TLGDataset, batch_size: int,
                    criterion: Callable[[FloatTensor, LongTensor], FloatTensor],
                    optimizer: optim.Optimizer, train_indices: List[int]) -> Tuple[float, int, int, int, int]:
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
            batch_y = pad_sequence(batch_y, batch_first=True).long().to(self.device)
            batch_e = F.embedding(batch_y.to(self.device), self.embedding_matrix)

            batch_p = self.forward(batch_x, batch_e)

            batch_loss = criterion(batch_p[:, :-1].permute(0, 2, 1), batch_y[:, 1:])
            loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
            (bs, bts), (bw, btw) = accuracy(batch_p[:, :-1].argmax(dim=-1), batch_y[:, 1:], 0)
            BS += bs
            BTS += bts
            BW += bw
            BTW += btw

            batch_start += batch_size

        return loss, BS, BTS, BW, BTW

    def eval_epoch(self, dataset: TLGDataset, batch_size: int, val_indices: List[int],
                   criterion: Callable[[FloatTensor, LongTensor], FloatTensor]) -> Tuple[float, int, int, int, int]:
        self.eval()

        with torch.no_grad():

            permutation = val_indices

            batch_start = 0
            loss = 0.
            BS, BTS, BW, BTW = 0, 0, 0, 0

            while batch_start < len(permutation):
                batch_end = min([batch_start + batch_size, len(permutation)])

                batch_x = [dataset.X[permutation[i]] for i in range(batch_start, batch_end)]
                batch_y = [dataset.Y[permutation[i]] for i in range(batch_start, batch_end)]

                batch_x = pad_sequence(batch_x, batch_first=True).to(self.device)
                batch_y = pad_sequence(batch_y, batch_first=True).long().to(self.device)

                batch_p = self.infer(batch_x, batch_y.shape[1], dataset.type_dict['<SOS>'])
                batch_loss = criterion(torch.log(batch_p[:, :-1]).permute(0, 2, 1), batch_y[:, 1:])
                loss += batch_loss.item()
                (bs, bts), (bw, btw) = accuracy(batch_p[:, :-1].argmax(dim=-1), batch_y[:, 1:], 0)
                BS += bs
                BTS += bts
                BW += bw
                BTW += btw

                batch_start += batch_size

        return loss, BS, BTS, BW, BTW


def test_baseline(data_path='data/XY_0'):
    tlg = dataprep.bpe_elmo(data_path=data_path + '.p')
    split_path = 'data/XY_100_split.p'
    store_path = 'stored_models/baseline.p'

    d_src = 1024
    batch_size = 128
    num_epochs = 1000

    num_classes = len(tlg.type_dict) + 1
    n = BaselineLSTM(num_classes, d_src, d_h=512, d_trg=1024, encoder_layers=1, decoder_layers=1, device='cuda',
                     dropout=0.2)

    L = FuzzyLoss(torch.nn.KLDivLoss(reduction='batchmean'), num_classes, 0.2, ignore_index=0)
    a = torch.optim.Adam(n.parameters(), betas=(0.9, 0.98), eps=1e-09)
    o = CustomLRScheduler(a, [noam_scheme], d_model=1024, warmup_steps=4000)

    if os.path.isfile(split_path):
        print('Loading splits')
        with open(split_path, 'rb') as f:
            train_indices, val_indices, test_indices = pickle.load(f)
    else:
        print('Making splits')
        splitpoints = list(map(int, [np.floor(0.8 * len(tlg.X)), np.floor(0.9 * len(tlg.X))]))
        indices = list(range(len(tlg.X)))
        np.random.shuffle(indices)
        train_indices = indices[:splitpoints[0]]
        val_indices = sorted(indices[splitpoints[0]:splitpoints[1]], key=lambda x: len(tlg.Y[x]))
        test_indices = sorted(indices[splitpoints[1]:], key=lambda x: len(tlg.Y[x]))
        with open(split_path, 'wb') as f:
            pickle.dump([train_indices, val_indices, test_indices], f)

    best_v_loss = 1e20

    for i in range(num_epochs):
        try:
            loss, bs, bts, bw, btw = n.train_epoch(tlg, batch_size, L, o, train_indices)
            v_loss, v_bs, v_bts, v_bw, v_btw = n.eval_epoch(tlg, batch_size, val_indices, L)
            t_loss, t_bs, t_bts, t_bw, t_btw = n.eval_epoch(tlg, batch_size, test_indices, L)
            print('Epoch {}'.format(i))
            print('   Training')
            print('      Loss: {}, Sentence Accuracy: {}, Word Accuracy: {}'.format(loss, bts / bs, btw / bw))
            print('   Validation')
            print('      Loss: {}, Sentence Accuracy: {}, Word Accuracy: {}'.format(v_loss, v_bts / v_bs,
                                                                                    v_btw / v_bw))
            print('   Test')
            print('      Loss: {}, Sentence Accuracy: {}, Word Accuracy: {}'.format(t_loss, t_bts / t_bs,
                                                                                    t_btw / t_bw))
            print('-' * 64)
            if v_loss < best_v_loss:
                best_v_loss = v_loss
                with open(store_path, 'wb') as f:
                    torch.save(n.state_dict(), f)
        except KeyboardInterrupt:
            return n
    return n