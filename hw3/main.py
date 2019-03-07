import torch
import torchtext
from torchtext.vocab import Vectors, GloVe

from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField

import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import matplotlib.pyplot as plt

import numpy as np
import itertools as it
import spacy
import time


def save_checkpoint(mod_enc, mod_dec, filename='checkpoint.pth.tar'):
    state_dict = {'model_encoder' : mod_enc.state_dict(),
                  'model_decoder' : mod_dec.state_dict()}
    torch.save(state_dict, filename)
    
def load_checkpoint(filename='checkpoint.pth.tar'):
    state_dict = torch.load(filename)
    return state_dict['model_encoder'], state_dict['model_decoder']
  
def set_parameters(model, sv_model, cuda=True):
    for i,p in enumerate(model.parameters()):
        p.data = sv_model[list(sv_model)[i]]
    model.cuda()


spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

BOS_WORD = '<s>'
EOS_WORD = '</s>'
DE = data.Field(tokenize=tokenize_de)

# only target needs BOS/EOS:
EN = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, eos_token = EOS_WORD) 

train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE, EN), 
                                         filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
                                         len(vars(x)['trg']) <= MAX_LEN)


DE.build_vocab(train.src, min_freq=MIN_FREQ)
EN.build_vocab(train.trg, min_freq=MIN_FREQ)

pred_set = []
for i, line in enumerate(open("source_test.txt"), 1):
    words = line.split()# [:-1]
    pred_set.append([DE.vocab.stoi[s] for s in words])

device = torch.device('cuda')
train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test), batch_size=32, device=device,
                                                  repeat=False, sort_key=lambda x: len(x.src))

batch = next(iter(test_iter))
# sent_inspect(batch,4)
def sent_inspect(batch, idx=0):
    print("Source")
    print(' '.join([DE.vocab.itos[w] for w in batch.src.data[:,idx]]))
    print("Target")
    print(' '.join([EN.vocab.itos[w] for w in batch.trg.data[:,idx]]))


# # Only call these if starting from scratch. Better to load saved model attn.rpoch_10.ckpt.tar  below
# train_iter, val_iter = data.BucketIterator.splits((train, val), batch_size=32, device=device,
#                                                   repeat=False, sort_key=lambda x: len(x.src))
# bs_encoder = EncoderLSTM(DE, hidden_size=650, num_layers=4, embedding_size=650, 
#                          bidirectional=True, dropout=0.35)
# at_decoder = DecoderAttn(EN, hidden_size=650, num_layers=4, embedding_size=650, 
#                          dropout=0.35, tie_weights=True, linear_encoder=650, 
#                          bidirectional_encoder=True)
# trainer = ModelTrain([bs_encoder, at_decoder], DE, EN, lr=1.2, 
#                      lr_decay_type='adaptive', attention=True,
#                      clip_norm=5)
# evaluator = ModelEval([bs_encoder, at_decoder], DE, EN, attention=True,
#                         record_attention=False)
# trainer.train(train_iter, verbose=True, eval_=evaluator, val_iter=val_iter,
#               save_model='attn', num_iter=20)

# Generates predictions  
ld_enc, ld_dec = load_checkpoint('attn.epoch_10.ckpt.tar')
ld_encoder = EncoderLSTM(DE, hidden_size=650, num_layers=4, embedding_size=650, 
                         bidirectional=True, dropout=0.35)
ld_decoder = DecoderAttn(EN, hidden_size=650, num_layers=4, embedding_size=650, 
                         dropout=0.35, tie_weights=False, linear_encoder=650, 
                         bidirectional_encoder=True)
set_parameters(ld_encoder, ld_enc)
set_parameters(ld_decoder, ld_dec)
evaluator = ModelEval([ld_encoder, ld_decoder], DE, EN, attention=True,
                        record_attention=False)
evaluator.predict(pred_set, fname='predictions_100.txt', beam_size=100, ignore_eos=True)


