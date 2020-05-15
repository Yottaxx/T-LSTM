import torch
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
import numpy as np

text = data.Field(lower=True, include_lengths=True, batch_first=True)
label = data.Field(sequential=False)

# make splits for data
train, test = datasets.IMDB.splits(text, label)

# # build the vocabulary
# text.build_vocab(train, vectors=GloVe(name='6B', dim=300))
# label.build_vocab(train)
#
# # print vocab information
# print('len(TEXT.vocab)', len(text.vocab))
# print('TEXT.vocab.vectors.size()', text.vocab.vectors.size())
#
# test_iter = data.BucketIterator(
#     test, batch_size=3, device="cuda")
#
# for batch in test_iter:
#     print(batch)
