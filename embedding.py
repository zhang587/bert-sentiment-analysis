#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import torch
from torchtext import data

from torchtext.data import Field, Dataset, Example


class GloveEmbeddings():

    @classmethod
    def load_glove_embeddings(cls):
        SEED = 1234
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True

        TEXT = data.Field(tokenize='spacy', include_lengths=True)
        LABEL = data.LabelField(dtype=torch.float)
        MAX_VOCAB_SIZE = 25_000

        train_data = pd.read_csv('data/train.csv')[['Text', 'Score']].iloc[0:1000,:]
        print("train data", train_data)
        train_ds = DataFrameDataset(train_data, fields={'Text': TEXT, 'Score':  LABEL})

        TEXT.build_vocab(train_ds,
                         max_size=MAX_VOCAB_SIZE,
                         vectors="glove.6B.50d",
                         unk_init=torch.Tensor.normal_)

        vocabs = LABEL.build_vocab(train_ds)
        print("!", train_ds.Text)
        print("train ds", torch.tensor(train_ds.Text))
        print(vocabs)
        print("type", type(vocabs))

        train_dl = BatchWrapper(train_ds, "Text", [0,1])
        print(train_dl, train_dl)

GloveEmbeddings.load_glove_embeddings()

class BatchWrapper:
    def __init__(self, dl, x_var, y_vars):
        self.dl, self.x_var, self.y_vars = dl, x_var, y_vars  # we pass in the list of attributes for x

    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_var)  # we assume only one input in this wrapper

            if self.y_vars is None:  # we will concatenate y into a single tensor
                y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).float()
            else:
                y = torch.zeros((1))

            yield (x, y)

    def __len__(self):
        return len(self.dl)


class DataFrameDataset(Dataset):
    """Class for using pandas DataFrames as a datasource"""

    def __init__(self, examples, fields, filter_pred=None):
        """
        Create a dataset from a pandas dataframe of examples and Fields
        Arguments:
            examples pd.DataFrame: DataFrame of examples
            fields {str: Field}: The Fields to use in this tuple. The
                string is a field name, and the Field is the associated field.
            filter_pred (callable or None): use only exanples for which
                filter_pred(example) is true, or use all examples if None.
                Default is None
        """
        self.examples = examples.apply(SeriesExample.fromSeries, args=(fields,), axis=1).tolist()
        if filter_pred is not None:
            self.examples = filter(filter_pred, self.examples)
        self.fields = dict(fields)
        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]


class SeriesExample(Example):
    """Class to convert a pandas Series to an Example"""

    @classmethod
    def fromSeries(cls, data, fields):
        return cls.fromdict(data.to_dict(), fields)

    @classmethod
    def fromdict(cls, data, fields):
        ex = cls()

        for key, field in fields.items():
            if key not in data:
                raise ValueError("Specified key {} was not found in "
                                 "the input data".format(key))
            if field is not None:
                setattr(ex, key, field.preprocess(data[key]))
            else:
                setattr(ex, key, data[key])

        return ex
