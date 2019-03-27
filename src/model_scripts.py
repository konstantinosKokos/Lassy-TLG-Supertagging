from src import dataprep
from Transformer.src.utils import FuzzyLoss, CustomLRScheduler, noam_scheme, Mask
from src.supertagger import Supertagger

import torch
import numpy as np

from typing import List, Optional
from itertools import chain

import pickle
import os

from collections import Counter

snd = lambda x: x[1]


def bpe_elmo(data_path='data/XY_atomic_short'):
    tlg = dataprep.bpe_elmo(data_path=data_path + '.p')
    split_path = data_path + '_split.p'
    store_path = 'stored_models/bpe_elmo.p'

    d_model = 1024
    batch_size = 128
    num_epochs = 1000

    num_classes = len(tlg.type_dict) + 1
    n = Supertagger(num_classes, 3, 8, 1, 2, 1024, dropout=0.2, device='cuda', d_model=d_model)

    L = FuzzyLoss(torch.nn.KLDivLoss(reduction='batchmean'), num_classes, 0.2)
    a = torch.optim.Adam(n.parameters(), betas=(0.9, 0.98), eps=1e-09)
    o = CustomLRScheduler(a, [noam_scheme], d_model=d_model, warmup_steps=4000)

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


def bpe_evaluation():
    data_path = 'data/XY_atomic_short'
    split_path = data_path + '_split.p'
    store_path = 'stored_models/bpe_elmo.p'
    result_path = data_path + '_results.tsv'
    new_path = data_path + '_new.tsv'
    data_path = data_path + '.p'

    with open(split_path, 'rb') as f:
        train_indices, val_indices, test_indices = pickle.load(f)
    with open(data_path, 'rb') as f:
        X, Y, type_to_int = pickle.load(f)

    tlg = dataprep.bpe_elmo(data_path=data_path)

    num_classes = len(tlg.type_dict) + 1
    d_model = 1024
    batch_size = 64
    n = Supertagger(num_classes, 3, 8, 1, 2, 1024, dropout=0.2, device='cuda', d_model=d_model)
    with open(store_path, 'rb') as f:
        n.load_state_dict(torch.load(f))
    n.eval()
    max_len = 6

    P_val = list(chain.from_iterable(n.infer_epoch(tlg, batch_size, val_indices, max_len)))
    P_test = list(chain.from_iterable(n.infer_epoch(tlg, batch_size, test_indices, max_len)))
    del n, tlg

    int_to_type = {v: k for k, v in type_to_int.items()}
    X_train, Y_train = list(zip(*[(X[i], Y[i]) for i in train_indices]))
    X_val, Y_val = list(zip(*[(X[i], Y[i]) for i in val_indices]))
    X_test, Y_test = list(zip(*[(X[i], Y[i]) for i in test_indices]))

    def convert_ints_to_types(y: List[int]) -> List[str]:
        return list(map(lambda i: int_to_type[i], y))

    def remove_te(y: List[str]) -> List[str]:
        return (' '.join(y) + ' ').split(' <TE> ')[:-1]

    def tab_separate(y: List[str]) -> str:
        return '\t'.join(y)

    def process(y: List[int]) -> str:
        return tab_separate(remove_te(convert_ints_to_types(y)))

    Y_test_ = list(map(process, Y_test))
    P_test_ = list(map(process, P_test))
    Y_train_ = list(map(process, Y_train))

    unique_type_sequences = set(list(chain.from_iterable(map(lambda y: remove_te(convert_ints_to_types(y)), Y_train))))

    def get_unique(y: str, p: str) -> List[str]:
        return list(map(snd,
                        list(filter(lambda x: x[1] not in unique_type_sequences, zip(y.split('\t'), p.split('\t'))))))

    def get_unique_counts(y: str, p: str):
        return Counter(list(map(lambda x: x[0],
                                list(filter(lambda x: x, list(chain((map(get_unique, y, p)))))))))

    def unique_and_true(y: str, p: str):
        y = y.split('\t')
        p = p.split('\t')
        return list(filter(lambda x: x[1] not in unique_type_sequences and x[0] == x[1], zip(y, p)))

    def all_unique(y: List[str], p: List[str]):
        return Counter((chain.from_iterable(list(map(get_unique, y, p)))))

    def interleave(X: List[List[str]]) -> List[str]:
        return [X[j][i] for i in range(len(X[0])) for j in range(len(X))]

    X_test_ = list(map(lambda x: list(map(lambda w: w.encode('latin-1', 'replace').decode('latin-1'), x)), X_test))
    X_test_ = list(map(lambda p: '\t'.join(p), X_test_))

    new_indices = [i for i in range(len(Y_test_)) if get_unique(Y_test_[i], P_test_[i])]
    X_new_ = [X_test_[i] for i in new_indices]
    Y_new_ = [Y_test_[i] for i in new_indices]
    P_new_ = [P_test_[i] for i in new_indices]
    new_new_ = ['\t'.join(['1' if x not in unique_type_sequences else '0' for x in p.split('\t')])
                for p in P_new_]
    XYP_new_ = interleave([X_new_, Y_new_, P_new_, new_new_])

    XYP_test_ = interleave([X_test_, Y_test_, P_test_])

    def store(x: List[str], path: str) -> None:
        np.savetxt(path, np.array(x, dtype=object), fmt='%s')

    store(XYP_test_, result_path)
    store(XYP_new_, new_path)
