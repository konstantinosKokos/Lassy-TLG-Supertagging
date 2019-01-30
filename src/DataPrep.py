import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..'))
from LassyExtraction.src import WordType
from LassyExtraction.src.utils.PostProcess import indexize, get_unique
import pickle

sys.modules['src.WordType'] = WordType

from typing import Iterable, Dict, Set, TypeVar, Callable, Tuple
import subprocess
import io

import torch
from torch import Tensor
from functools import reduce
from torch.utils.data import Dataset

from tqdm import tqdm


T1 = TypeVar('T1')
T2 = TypeVar('T2')


def load(x: str='data/XYZ.p'):
    with open(x, 'rb') as f:
        X, Y, Z = pickle.load(f)
    return X, Y, Z


def map_type_sequence_to_index_sequence(ts: Iterable[WordType.WordType], d: Dict[WordType.WordType, int]) \
        -> Iterable[int]:
    return tuple(map(lambda x: Tensor([d[x]]), ts))


def map_sequences_to_sequences(T: Iterable[Iterable[T1]], d: Dict[T1, T2],
                               fun: Callable[[Iterable[T1], Dict[T1, T2]], Iterable[T2]]) \
        -> Iterable[Iterable[T2]]:
    return tuple(map(lambda x: fun(x, d), T))


def map_word_to_tensor(w: str, d: Dict[str, Tensor]) -> Tensor:
    subwords = w.split()
    subtensors = list(map(lambda x: d[x] / Tensor([len(subwords)]), subwords))
    return reduce(lambda x, y: x+y, subtensors).unsqueeze(0)


def map_word_sequence_to_tensor_sequence(ws: Iterable[str], d: Dict[str, Tensor]) -> Iterable[Tensor]:
    return tuple(map(lambda x: map_word_to_tensor(x, d), ws))


def write_vocab(voc: Set[str], voc_file: str) -> None:
    subwords = []
    with open(voc_file, 'w') as f:
        for word in voc:
            for subword in word.split():
                if subword not in subwords:
                    subwords.append(subword)
                    f.write(subword + '\n')


def vectorize_words(X: Iterable[Iterable[str]], fastText_path: str, model_path: str, storage_path: str,
                    vocab_path: str) -> None:
    vocabulary = get_unique(X)
    write_vocab(vocabulary, vocab_path)
    with open(storage_path, 'w') as f:
        ft = subprocess.Popen([fastText_path, 'print-word-vectors', model_path],
                              stdin=open(vocab_path), stdout=f)
        ft.wait()
        f.flush()


def load_vectors(fname: str) -> Dict[str, Tensor]:
    ret = dict()
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    for line in tqdm(fin.readlines()):
        if len(line.split()) == 2:
            continue
        tokens = line.rstrip().split(' ')
        ret[tokens[0]] = Tensor(list(map(float, tokens[1:])))
    return ret


def do_everything():
    fastText_path = '/home/kokos/Documents/Projects/FastText/fastText-0.1.0/fasttext'
    model_path = '/home/kokos/Documents/Projects/FastText/fastText-0.1.0/models/wiki.nl.bin'
    storage_path = 'data/vectors.vec'
    vocab_path = 'data/vocabulary.voc'
    X, Y, Z = load()

    # type_to_int = indexize(get_unique(Y).union(set(Z)))
    type_to_int = indexize(get_unique(Y))
    type_to_int[None] = 0
    type_to_int['SOS'] = len(type_to_int) + 1
    # int_to_type = {v: k for k, v in type_to_int.items()}
    type_sequences = map_sequences_to_sequences(Y, type_to_int, map_type_sequence_to_index_sequence)
    # results = map_type_sequence_to_index_sequence(Z, type_to_int)

    try:
        if not os.path.isfile(vocab_path):
            vectorize_words(X, fastText_path, model_path, storage_path, vocab_path)
        word_to_vec = load_vectors(storage_path)

        vector_sequences = map_sequences_to_sequences(X, word_to_vec, map_word_sequence_to_tensor_sequence)
    except KeyError:
        voc = get_unique(X)
        write_vocab(voc, vocab_path)
        vectorize_words(X, fastText_path, model_path, storage_path, vocab_path)
        word_to_vec = load_vectors(storage_path)
        vector_sequences = map_sequences_to_sequences(X, word_to_vec, map_word_sequence_to_tensor_sequence)

    return TLGDataset(vector_sequences, type_sequences, None, type_to_int)


class TLGDataset(Dataset):
    def __init__(self, X: Iterable[Iterable[Tensor]], Y: Iterable[Iterable[int]],
                 Z: Iterable[int], type_to_int: Dict[WordType.WordType, int]) -> None:
        super(TLGDataset, self).__init__()
        self.X = tuple(map(lambda x: torch.cat([torch.cat(x), torch.zeros([1, 300])]), X))
        self.Y = tuple(map(lambda x: torch.cat([Tensor([type_to_int['SOS']]), torch.cat(x)]), Y))
        self.Z = Z  # todo
        self.type_dict = type_to_int

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
        return self.X[item], self.Y[item]
