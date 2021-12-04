#modified to use phow2v in this file

import torch
from torch import nn
from torch.nn import functional as F


def position_embedding(input, d_model):
    input = input.view(-1, 1)
    dim = torch.arange(d_model // 2, dtype=torch.float32, device=input.device).view(1, -1)
    sin = torch.sin(input / 10000 ** (2 * dim / d_model))
    cos = torch.cos(input / 10000 ** (2 * dim / d_model))

    out = torch.zeros((input.shape[0], d_model), device=input.device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out


def sinusoid_encoding_table(max_len, d_model, padding_idx=None):
    pos = torch.arange(max_len, dtype=torch.float32)
    out = position_embedding(pos, d_model)

    if padding_idx is not None:
        out[padding_idx] = 0
    return out
def get_embedding_matrix():
    import numpy as np
    f =  open('/content/drive/MyDrive/Quan_task/embedded_M2forviic/word2vec_vi_words_300dims.txt')
    word_dict = []
    embeddings_index = {}
    embedding_dim = 300
    max_feature = len(embeddings_index) + 2
    for line in f:
        values = line.split(' ')
        word = values[0]
        word_dict.append(word)
        #try:
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        #except Exception as e:
            #pass
    import json
    with open('wtoi.json','r') as f:
        wtoi = json.load(f)
    num_words = len(wtoi)
    embedding_dim = 300
    # first create a matrix of zeros, this is our embedding matrix
    embedding_matrix = np.zeros((num_words, embedding_dim))

    # for each word in out tokenizer lets try to find that work in our w2v model
    for word,i in wtoi.items():
        # if i > max_feature:
        #     continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            # doesn't exist, assign a random vector
            embedding_matrix[i] = np.random.randn(embedding_dim)
    embedding_matrix = torch.tensor(embedding_matrix).type(torch.FloatTensor)
    expand_dim = nn.Linear(in_features=300,out_features=512)
    embedding_matrix = expand_dim(embedding_matrix)
    print('Embedding data loaded\n')
    print(embedding_matrix.shape)
    return embedding_matrix
    


def get_pretrained_encoding(path):
    import gensim
    model = gensim.models.KeyedVectors.load_word2vec_format(path)
    weights = torch.FloatTensor(model.vectors)
    expand_dim = nn.Linear(in_features=300,out_features=512)
    weights = expand_dim(weights)
    return weights

class PositionWiseFeedForward(nn.Module):
    '''
    Position-wise feed forward layer
    '''

    def __init__(self, d_model=512, d_ff=2048, dropout=.1, identity_map_reordering=False):
        super(PositionWiseFeedForward, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input):
        if self.identity_map_reordering:
            out = self.layer_norm(input)
            out = self.fc2(self.dropout_2(F.relu(self.fc1(out))))
            out = input + self.dropout(torch.relu(out))
        else:
            out = self.fc2(self.dropout_2(F.relu(self.fc1(input))))
            out = self.dropout(out)
            out = self.layer_norm(input + out)
        return out