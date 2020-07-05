# -*- coding: utf-8 -*-
"""Python file offtargetCreateGuideSeqDataset.py for datasets generation."""

import sys
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import Bunch

# we import the off-targets from guideseq
fpath = 'guideseq.csv'
dfGuideSeq = pd.read_csv(fpath, sep=',')
dfGuideSeq = dfGuideSeq.drop(columns='nameLong')

dfGuideSeq = dfGuideSeq.drop_duplicates(
    subset=['otSeq'], keep=False, ignore_index=True)


# we encode the new validated off-targets
# as described in the references
def one_hot_encode_seq(data):
    """One-hot encoding of the sequences."""
    # define universe of possible input values
    alphabet = 'AGCT'
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    # integer encode input data
    integer_encoded = [char_to_int[char] for char in data]
    # print(integer_encoded)
    # one hot encode
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)
    # print(onehot_encoded)
    # invert encoding
    inverted = int_to_char[np.argmax(onehot_encoded[0])]
    # print(inverted)
    return onehot_encoded


def flatten_one_hot_encode_seq(seq):
    """Flatten one hot encoded sequences."""
    return np.asarray(seq).flatten(order='C')


enc_dna_list = []
enc_rna_list = []
encoded_list = []
for n in range(len(dfGuideSeq)):
    target_dna = dfGuideSeq.loc[n, 'otSeq']
    target_rna = dfGuideSeq.loc[n, 'guideSeq']
    arr1 = one_hot_encode_seq(target_dna)
    arr2 = one_hot_encode_seq(target_rna)
    arr = np.zeros((23, 4))
    for m in range(len(arr1)):
        if arr1[m] == arr2[m]:
            arr[m] = arr1[m]
        else:
            arr[m] = np.add(arr1[m], arr2[m])
    arr = flatten_one_hot_encode_seq(arr)
    enc_dna_list.append(np.asarray(arr1).flatten('C'))
    enc_rna_list.append(np.asarray(arr2).flatten('C'))
    encoded_list.append(np.asarray(arr).flatten('C'))

dfGuideSeq['enc_dna'] = pd.Series(enc_dna_list)
dfGuideSeq['enc_rna'] = pd.Series(enc_rna_list)
dfGuideSeq['encoded'] = pd.Series(encoded_list)

# we consider the encoded column for the 4x23 encoding
dfGuideSeq4x23 = pd.DataFrame(dfGuideSeq['encoded'].values.tolist())

# save the encoded results as 4x23 images
# we put the results in bunch
guideseq4x23 = Bunch(
    target_names=dfGuideSeq['name'].values,
    target=dfGuideSeq['label'].values,
    images=(dfGuideSeq4x23.values*254).reshape(-1, 4, 23, order='F'))
plt.imshow(guideseq4x23.images[0], cmap='Greys')

# we have to transform the RNA and DNA sequences to
# a 8x23 image
# we create a new column on the crispor data df
# we structure the images as the mnist dataset
# digits.target_names is the name
# digits.target is the binary classification
# digits.images is the 8x23 pixels of the image
# a. we do it for the put. off-target

# we store the image in im
im = np.zeros((len(dfGuideSeq), 8, 23))

cnt = 0
for n in range(len(dfGuideSeq)):
    arr1 = one_hot_encode_seq(dfGuideSeq.loc[n, 'guideSeq'])
    arr1 = np.asarray(arr1).T
    arr2 = one_hot_encode_seq(dfGuideSeq.loc[n, 'otSeq'])
    arr2 = np.asarray(arr2).T
    arr = np.concatenate((arr1, arr2)) * 254
    im[n] = arr
    cnt += 1

# we put the results in bunch
guideseq8x23 = Bunch(
    target_names=dfGuideSeq['name'].values,
    target=dfGuideSeq['label'].values,
    images=im)
plt.imshow(guideseq8x23.images[0], cmap='Greys')
plt.savefig('guideseq8x23.pdf')
