import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

class word2vec(nn.Module):
    def __init__(self,vocab_len,dim=256):
        super(word2vec, self).__init__()

        self.dim=dim
        self.vocab_len=vocab_len
        self.embedding=nn.Embedding(self.vocab_len,self.dim)
        self.encoding_layer=nn.Linear(self.dim,128)
        self.decoding_layer=nn.Linear(128,self.vocab_len)


    def forward(self,x):

        embedds=self.embedding(x)
        output=F.relu(self.encoding_layer(embedds))
        output=self.decoding_layer(output)
        output=F.log_softmax(output,dim=0)


        return output
    
    
class negative_sampling(nn.Module):

    def __init__(self,model,vocab_len,n=10):
        super(negative_sampling, self).__init__()


        self.model=model['embedding.weight'].detach()
        self.vocab_len=vocab_len
        self.n=n


    def forward(self,y_true,y_pred):


        batch_size = y_true.size()[0]


    #    ivectors = self.embedding.forward_i(iword).unsqueeze(2)

        nwords = torch.FloatTensor(batch_size, self.n).uniform_(1, self.vocab_size).long()   

        n_vec=model[nwords].neg()
        i_vec=




    


