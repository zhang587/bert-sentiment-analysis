from embedding import Embeddings
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from utils import pad_sents, batch_iter
import math
from cnn import CNN
import sys

class CNNClassifier(nn.Module):
    def __init__(self, embed_size, kernel_size, num_filter, p_dropout=0.1):
        super(CNNClassifier, self).__init__()
        self.embed_size = embed_size
        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.p_dropout = p_dropout
        
        self.cnn = CNN(embed_size, kernel_size, num_filter)
        self.linear = nn.Linear(in_features=self.num_filter, out_features=2)
        self.dropout = nn.Dropout(self.p_dropout)
        
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, input):
        x_conv_out = self.cnn(input)
        x_conv_out = x_conv_out.squeeze()
        output = self.dropout(self.linear(x_conv_out))
        return output
    

    @staticmethod
    def load(model_path: str, no_char_decoder=False):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = CNNClassifier(**args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.embed_size, 
            kernel_size=self.kernel_size,
            num_filter=self.num_filter,
            p_dropout=self.p_dropout),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
