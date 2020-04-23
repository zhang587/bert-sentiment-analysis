from embedding import Embeddings
from transformers import *
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from utils import pad_sents, batch_iter
import math
from cnn import CNN
import sys


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Bert_Embedding():
    embed_model = Embeddings()

    df_train = pd.read_csv('/Users/ziqunye/Documents/stanford/project/SentimentAnalysis/bert-sentiment-analysis/data/train.csv')
    df_test = pd.read_csv('/Users/ziqunye/Documents/stanford/project/SentimentAnalysis/bert-sentiment-analysis/data/test.csv')
    training_data_size = 200
    df_train = df_train.iloc[:training_data_size, :]
    df_test = df_test.iloc[:training_data_size, :]
    # you can experiment with more data to get a more realistic performance score. With a fewer datapoints the model tends to overfit

    text_train = df_train['Text'].iloc[0:training_data_size].values
    y_train = df_train['Score'].iloc[0:training_data_size].values

    text_test = df_train['Text'].iloc[0:training_data_size].values
    y_test = df_train['Score'].iloc[0:training_data_size].values

    x_train = []
    for sentence in tqdm(text_train):
        x_train.append(embed_model.sentence2vec(sentence, layers=2))
        
    x_test = []
    for sentence in tqdm(text_test):
        x_test.append(embed_model.sentence2vec(sentence, layers=2))
    return x_train, x_test, y_train, y_test


class CNNClassifier(nn.Module):
    def __init__(self, embed_size, kernel_size, num_filter, p_dropout=0.1):
        super(CNNClassifier, self).__init__()
        self.embed_size = embed_size
        self.kernel_size = kernel_size
        self.num_filter = num_filter
        
        self.cnn = CNN(embed_size, kernel_size, num_filter)
        self.linear = nn.Linear(in_features=self.num_filter, out_features=2)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, input):
        x_conv_out = self.cnn(input)
        x_conv_out = x_conv_out.squeeze()
        output = self.dropout(self.linear(x_conv_out))
        return output

def train(x_train, y_train, x_dev, y_dev, batch_size=30, epoch=100, valid_niter=2,
model_save_path ='/Users/ziqunye/Documents/stanford/project/SentimentAnalysis/bert-sentiment-analysis/model_cat/', 
patience_target = 20, max_num_trial=100, lr_decay=0.5, max_epoch=2000):
    criterion = torch.nn.CrossEntropyLoss()
    model = CNNClassifier(embed_size=768, kernel_size=5, num_filter=5)

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []


    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    ## Parameter Initialization
    # uniform_init = float(args['--uniform-init'])
    # if np.abs(uniform_init) > 0.:
    #     print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
    #     for p in model.parameters():
    #         p.data.uniform_(-uniform_init, uniform_init)

    model.train(mode=True)
    while epoch > 0:
        epoch_loss = 0
        epoch_acc = 0
        for sents, tgt in batch_iter(list(zip(x_train, y_train)), batch_size):
            this_batch_size = len(sents)
            optimizer.zero_grad()
            # tgt = torch.tensor(tgt)
            pred = model.forward(sents) # (batch_size,)
            loss = criterion(pred, tgt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss * this_batch_size
            y_pred = F.softmax(pred, dim=1)
            epoch_acc += binary_accuracy(y_pred, tgt) 
            # print('I am here')
            # # perform validation
            # if train_iter % valid_niter == 0:
            #     # print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
            #     #                                                                          cum_loss / cum_examples,
            #     #                                                                          np.exp(cum_loss / cum_tgt_words),
            #     #                                                                          cum_examples), file=sys.stderr)

            #     cum_loss = cum_examples = cum_tgt_words = 0.
            #     valid_num += 1

            #     # print('begin validation ...', file=sys.stderr)

            #     # compute dev. ppl and bleu
            #     dev_f1, dev_precision, dev_recall, dev_accuracy = (
            #         evaluate_F1(model, x_dev, y_dev, batch_size=batch_size)
            #         )  # dev batch size can be a bit larger
            #     valid_metric = dev_f1

            #     print('validation: iter %d, dev. f1 %f, dev. precision %f, dev. recall %f, dev. accuracy %f' \
            #         % (train_iter, dev_f1, dev_precision, dev_recall, dev_accuracy), file=sys.stderr)

            #     is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
            #     hist_valid_scores.append(valid_metric)

            #     if is_better:
            #         patience = 0
            #         print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
            #         model.save(model_save_path)

            #         # also save the optimizers' state
            #         torch.save(optimizer.state_dict(), model_save_path + '.optim')
            #     elif patience < patience_target:
            #         patience += 1
            #         print('hit patience %d' % patience, file=sys.stderr)

            #         if patience == patience_target:
            #             num_trial += 1
            #             print('hit #%d trial' % num_trial, file=sys.stderr)
            #             if num_trial == max_num_trial:
            #                 print('early stop!', file=sys.stderr)
            #                 exit(0)

            #             # decay lr, and restore from previously best checkpoint
            #             lr = optimizer.param_groups[0]['lr'] * float(lr_decay)
            #             print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

            #             # load model
            #             params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
            #             model.load_state_dict(params['state_dict'])
    

            #             print('restore parameters of the optimizers', file=sys.stderr)
            #             optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

            #             # set new lr
            #             for param_group in optimizer.param_groups:
            #                 param_group['lr'] = lr

            #             # reset patience
            #             patience = 0

            # if epoch == int(max_epoch):
            #     print('reached maximum number of epochs!', file=sys.stderr)
            #     exit(0)

        epoch -= 1

        if epoch % 30 == 0:
            print('loss', epoch_loss/len(y_train), 'accurary', epoch_acc/(200//batch_size+1))
    return model


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    # rounded_preds = torch.round(torch.sigmoid(preds))
    # print(torch.round(preds[:,1]))
    correct = (torch.round(preds[:,1]) == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc



def evaluate_F1(model, x_dev, y_dev, batch_size=32):
    """ Evaluate perplexity on dev sentences
    @param model (CNN): CNN Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (F1 on dev sentences)
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.
    criterion = torch.nn.CrossEntropyLoss()
    tp_cum = tn_cum = fp_cum = fn_cum = 0
    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for sents, y_true in batch_iter(list(zip(x_dev, y_dev)), batch_size):
            pred = model.forward(sents) # (batch_size,)
            y_pred += F.softmax(pred, dim=1)
            tp_cum += (y_true * y_pred).sum().to(torch.float32)
            tn_cum += ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
            fp_cum += ((1 - y_true) * y_pred).sum().to(torch.float32)
            fn_cum += (y_true * (1 - y_pred)).sum().to(torch.float32)
 
        epsilon = 1e-7
        precision = tp_cum / (tp_cum + fp_cum + epsilon)
        recall = tp_cum / (tp_cum + fn_cum + epsilon)
        f1 = 2* (precision*recall) / (precision + recall + epsilon)
        accuracy = (tp_cum + tn_cum)/len(y_dev)
    if was_training:
        model.train()

    return f1, precision, recall, accuracy




x_train, x_test, y_train, y_test = Bert_Embedding()
model = train(x_train, y_train, x_test, y_test)

def f1_loss(y_true:torch.Tensor, y_pred:torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
        
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1
# print(evaluate(x_test, y_test))

# print(model.forward(x_test))
    # clip gradient
    # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

    

    # batch_losses_val = batch_loss.item()
    # report_loss += batch_losses_val
    # cum_loss += batch_losses_val




# (12, 103, 768)
# class LogisticModel(nn.Module):
#     def __init__(self, embed_size):
#         super(LogisticModel, self).__init__()
#         self.embed_size = embed_size
#         self.linear = nn.Linear(in_features=self.embed_size, out_features=1)

#     def forward(self, input):
#         return F.sigmoid(self.linear(input))



# model = LogisticModel(768)
# y = model.forward(x_train)
# print(y.shape)
