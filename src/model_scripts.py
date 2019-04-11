from src import dataprep
from Transformer.src.utils import FuzzyLoss, CustomLRScheduler, noam_scheme, Mask
from src.supertagger import Supertagger

import torch
import numpy as np

from typing import List, Tuple
from itertools import chain

import pickle
import os

from functools import reduce
from collections import Counter, defaultdict

snd = lambda x: x[1]


def bpe_elmo(data_path='data/XY_inf'):
    tlg = dataprep.bpe_elmo(data_path=data_path + '.p')
    split_path = 'data/XY_100_split.p'
    store_path = 'stored_models/inf_elmo.p'

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


def bpe_long_evaluation():
    data_path = 'data/XY_long_0.p'
    train_path = 'data/XY_0.p'
    split_path = 'data/XY_100_split.p'

    store_path = 'stored_models/bpe_elmo_0_1.p'

    with open(data_path, 'rb') as f:
        X, Y, type_to_int = pickle.load(f)
    with open(split_path, 'rb') as f:
        train_indices, _, _ = pickle.load(f)
    with open(train_path, 'rb') as f:
        X_train, Y_train, _ = pickle.load(f)

    tlg = dataprep.bpe_elmo(data_path=data_path)

    num_classes = len(tlg.type_dict) + 1
    d_model = 1024
    batch_size = 128
    n = Supertagger(num_classes, 3, 8, 1, 2, 1024, dropout=0.2, device='cuda', d_model=d_model)
    with open(store_path, 'rb') as f:
        n.load_state_dict(torch.load(f))
    n.eval()
    max_len = 5

    test_indices = [i for i in range(1024)]

    P_test = n.infer_epoch(tlg, batch_size, test_indices, max_len)
    P_test = list(chain.from_iterable(P_test))
    del n, tlg

    int_to_type = {v: k for k, v in type_to_int.items()}
    _, Y_train = list(zip(*[(X_train[i], Y_train[i]) for i in train_indices]))
    X_test, Y_test = list(zip(*[(X[i], Y[i]) for i in test_indices]))

    def convert_ints_to_atoms(y: List[int]) -> List[str]:
        return list(map(lambda i: int_to_type[i], y))

    def convert_atoms_to_types(y: List[str]) -> List[str]:
        return (' '.join(list(map(lambda a: a.replace('+', ' '), y))) + ' ').split(' <TE> ')[:-1]

    def convert_ints_to_types(y: List[int]) -> List[str]:
        return convert_atoms_to_types(convert_ints_to_atoms(y))

    def tab_join(y: List[str]) -> str:
        return '\t'.join(y)

    X_test_ = list(map(lambda x: list(map(lambda w: w.encode('latin-1', 'replace').decode('latin-1'), x)), X_test))

    Y_test_ = list(map(convert_ints_to_types, Y_test))
    P_test_ = list(map(convert_ints_to_types, P_test))
    Y_train_ = list(map(convert_ints_to_types, Y_train))

    type_counts = Counter(list(chain.from_iterable(Y_train_)))

    def occurrence_correct(y: List[str], p: List[str], occ: int) -> Tuple[int, int]:
        yp_o = list(filter(lambda yp: type_counts[yp[0]] == occ, zip(y, p)))
        return len(yp_o), len(list(filter(lambda yp: yp[0] == yp[1], yp_o)))

    def count_occurrence_correct(Y: List[List[str]], P: List[List[str]], occ: int) -> Tuple[int, int]:
        return reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]), list(map(lambda z, w: occurrence_correct(z, w, occ),
                                                                        Y, P)))

    def index_occurrence_correct(Y: List[List[str]], P: List[List[str]], occ: int) -> List[int]:
        yp = list(map(lambda y, p: occurrence_correct(y, p, occ), Y, P))
        return [i for i in range(len(Y)) if yp[i][1] > 0]

    def imagined(y: List[str], p: List[str], correct: bool=False) -> List[bool]:
        return [True if (type_counts[p[i]] == 0 and (y[i] == p[i] or not correct)) else False for i in range(
            min(len(y), len(p)))]

    def index_imagined(Y: List[List[str]], P: List[List[str]], correct: bool=False) -> List[int]:
        C = list(map(lambda y, p: imagined(y, p, correct), Y, P))
        return [idx for idx, im in enumerate(C) if any(im)]

    def count_imagined(Y: List[List[str]], P: List[List[str]]) -> int:
        C = list(map(lambda y, p: imagined(y, p), Y, P))
        return sum([len(list(filter(lambda x: x, im))) for im in C])

    def count_correct(y: List[str], p: List[str]) -> Tuple[int, int]:
        return len(y), len(list(filter(lambda x: x[0] == x[1], zip(y, p))))

    def accuracy(Y: List[List[str]], P: List[List[str]]) -> Tuple[int, int]:
        return reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]), list(map(count_correct, Y, P)))

    test_accuracy = accuracy(Y_test_, P_test_)
    print('Test accuracy: {}/{} ({})'.format(test_accuracy[1], test_accuracy[0], test_accuracy[1]/test_accuracy[0]))
    for occ in range(11):
        a = count_occurrence_correct(Y_test_, P_test_, occ)
        print(' {} Occurrence Accuracy: {}/{} ({})'.format(occ, a[1], a[0], a[1]/a[0]))
    print('Imagined new types {} times.'.format(count_imagined(Y_test_, P_test_)))


def bpe_evaluation():
    data_path = 'data/XY_0'
    split_path = 'data/XY_100_split.p'
    store_path = 'stored_models/bpe_elmo_0_1.p'
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

    P_test = list(chain.from_iterable(n.infer_epoch(tlg, batch_size, test_indices, max_len)))
    del n, tlg

    int_to_type = {v: k for k, v in type_to_int.items()}
    X_train, Y_train = list(zip(*[(X[i], Y[i]) for i in train_indices]))
    X_test, Y_test = list(zip(*[(X[i], Y[i]) for i in test_indices]))

    def convert_ints_to_atoms(y: List[int]) -> List[str]:
        return list(map(lambda i: int_to_type[i], y))

    def convert_atoms_to_types(y: List[str]) -> List[str]:
        return (' '.join(list(map(lambda a: a.replace('+', ' '), y))) + ' ').split(' <TE> ')[:-1]

    def convert_ints_to_types(y: List[int]) -> List[str]:
        return convert_atoms_to_types(convert_ints_to_atoms(y))

    def tab_join(y: List[str]) -> str:
        return '\t'.join(y)

    X_test_ = list(map(lambda x: list(map(lambda w: w.encode('latin-1', 'replace').decode('latin-1'), x)), X_test))

    Y_test_ = list(map(convert_ints_to_types, Y_test))
    P_test_ = list(map(convert_ints_to_types, P_test))
    Y_train_ = list(map(convert_ints_to_types, Y_train))

    type_counts = Counter(list(chain.from_iterable(Y_train_)))

    def range_correct(y: List[str], p: List[str], m: int, M: int) -> Tuple[int, int]:
        yp_o = list(filter(lambda yp: m < type_counts[yp[0]] <= M, zip(y, p)))
        return len(yp_o), len(list(filter(lambda yp: yp[0] == yp[1], yp_o)))

    def count_range_correct(Y: List[List[str]], P: List[List[str]], m: int, M: int) -> Tuple[int, int]:
        return reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]), list(map(lambda z, w: range_correct(z, w, m, M),
                                                                       Y, P)))

    def occurrence_correct(y: List[str], p: List[str], occ: int) -> Tuple[int, int]:
        yp_o = list(filter(lambda yp: type_counts[yp[0]] == occ, zip(y, p)))
        return len(yp_o), len(list(filter(lambda yp: yp[0] == yp[1], yp_o)))

    def count_occurrence_correct(Y: List[List[str]], P: List[List[str]], occ: int) -> Tuple[int, int]:
        return reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]), list(map(lambda z, w: occurrence_correct(z, w, occ),
                                                                        Y, P)))

    def index_occurrence_correct(Y: List[List[str]], P: List[List[str]], occ: int) -> List[int]:
        yp = list(map(lambda y, p: occurrence_correct(y, p, occ), Y, P))
        return [i for i in range(len(Y)) if yp[i][1] > 0]

    def imagined(y: List[str], p: List[str], correct: bool=False) -> List[bool]:
        return [True if (type_counts[p[i]] == 0 and (y[i] == p[i] or not correct)) else False for i in range(len(y))]

    def index_imagined(Y: List[List[str]], P: List[List[str]], correct: bool=False) -> List[int]:
        C = list(map(lambda y, p: imagined(y, p, correct), Y, P))
        return [idx for idx, im in enumerate(C) if any(im)]

    def count_imagined(Y: List[List[str]], P: List[List[str]]) -> int:
        C = list(map(lambda y, p: imagined(y, p), Y, P))
        return sum([len(list(filter(lambda x: x, im))) for im in C])

    def count_correct(y: List[str], p: List[str]) -> Tuple[int, int]:
        return len(y), len(list(filter(lambda x: x[0] == x[1], zip(y, p))))

    def accuracy(Y: List[List[str]], P: List[List[str]]) -> Tuple[int, int]:
        return reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]), list(map(count_correct, Y, P)))

    test_accuracy = accuracy(Y_test_, P_test_)
    print('Test accuracy: {}/{} ({})'.format(test_accuracy[1], test_accuracy[0], test_accuracy[1]/test_accuracy[0]))
    for occ in range(11):
        a = count_occurrence_correct(Y_test_, P_test_, occ)
        print(' {} Occurrence Accuracy: {}/{} ({})'.format(occ, a[1], a[0], a[1]/a[0]))
    for m, M in ([0, 10], [10, 100], [100, 1e10]):
        a = count_range_correct(Y_test, P_test_, m, M)
        print(' {}-{} Range Accuracy: {}/{} ({})'.format(m, M, a[1], a[0], a[1] / a[0]))
    print('Imagined new types {} times.'.format(count_imagined(Y_test_, P_test_)))

    def store(x: List[str], path: str) -> None:
        np.savetxt(path, np.array(x, dtype=object), fmt='%s')

    new_indices = index_imagined(Y_test_, P_test_, True)
    X_new_ = [tab_join(X_test_[i]) for i in new_indices]
    Y_new_ = [tab_join(Y_test_[i]) for i in new_indices]
    P_new_ = [tab_join(P_test_[i]) for i in new_indices]
    # todo
    new_new_ = [tab_join(list(map(lambda x: '1' if type_counts[x] == 0 else '0', P_test_[i]))) for i in new_indices]





    #
    # def get_unique(y: str, p: str) -> List[str]:
    #     return list(map(snd,
    #                     list(filter(lambda x: train_type_counts[x[1]] == 0, zip(y.split('\t'), p.split('\t'))))))
    #
    # def get_unique_counts(y: str, p: str):
    #     return Counter(list(map(lambda x: x[0],
    #                             list(filter(lambda x: x, list(chain((map(get_unique, y, p)))))))))
    #
    # def unique_and_true(y: str, p: str):
    #     y = y.split('\t')
    #     p = p.split('\t')
    #     return list(filter(lambda x: train_type_counts[x[1]] == 0 and x[0] == x[1], zip(y, p)))
    #
    # def all_unique(y: List[str], p: List[str]):
    #     return Counter((chain.from_iterable(list(map(get_unique, y, p)))))
    #
    # def interleave(X: List[List[str]]) -> List[str]:
    #     return [X[j][i] for i in range(len(X[0])) for j in range(len(X))]
    #
    #
    # new_indices = [i for i in range(len(Y_test_)) if get_unique(Y_test_[i], P_test_[i])]
    # X_new_ = [X_test_[i] for i in new_indices]
    # Y_new_ = [Y_test_[i] for i in new_indices]
    # P_new_ = [P_test_[i] for i in new_indices]
    # new_new_ = ['\t'.join(['1' if x not in unique_type_sequences else '0' for x in p.split('\t')])
    #             for p in P_new_]
    # XYP_new_ = interleave([X_new_, Y_new_, P_new_, new_new_])
    #
    # XYP_test_ = interleave([X_test_, Y_test_, P_test_])
    #
    #
    # store(XYP_test_, result_path)
    # store(XYP_new_, new_path)


def embedding_visualization():
    data_path = 'data/XY_0.p'
    store_path = 'stored_models/bpe_elmo_0_1.p'
    with open(data_path, 'rb') as f:
        X, Y, type_to_int = pickle.load(f)


    num_classes = len(type_to_int)+2
    d_model = 1024
    batch_size = 128
    n = Supertagger(num_classes, 3, 8, 1, 2, 1024, dropout=0.2, device='cuda', d_model=d_model)
    with open(store_path, 'rb') as f:
        n.load_state_dict(torch.load(f))
    n.eval()

    e = n.transformer.output_embedder
    idx = torch.arange(num_classes, device='cuda').view(-1, 1)
    embeddings = e(idx).detach().view(num_classes, -1).cpu()[:len(type_to_int)]
    int_to_type = {v:k for k, v in type_to_int.items()}
    labels = [int_to_type[i+1] for i in range(62)]
    from sklearn.manifold import TSNE
    tsne = TSNE(2, metric='cosine', early_exaggeration=50)
    em = tsne.fit_transform(embeddings)
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(em[:, 0], em[:, 1])

    for i, txt in enumerate(labels):
        ax.annotate(txt, (em[i, 0], em[i, 1]))

    return ax
