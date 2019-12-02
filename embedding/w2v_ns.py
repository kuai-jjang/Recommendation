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
        #self.decoding_layer=nn.Linear(128,self.vocab_len)


    def forward(self,x):
        
        print(self.embedding.weight.is_cuda)
        embedds=self.embedding(x).cuda() if self.embedding.weight.is_cuda else self.embedding(x)
        
        output=F.relu(self.encoding_layer(embedds))
        #output=self.decoding_layer(output)
        
        return output
    
    
class negative_sampling(nn.Module):

    def __init__(self,model,vocab_len,enc_dim=128,n=10):
        super(negative_sampling, self).__init__()


        self.model=model
        self.vocab_len=vocab_len
        self.enc_dim=enc_dim
        self.n=n


    def forward(self,x,y):


        batch_size = y.size()[0]

        nwords = torch.FloatTensor(batch_size, self.n).uniform_(1, self.vocab_len).long()   

        n_vec=self.model.forward(nwords).neg().view(batch_size,-1,self.n)
        i_vec=self.model.forward(x).view(batch_size,1,-1)
        o_vec=self.model.forward(y).view(batch_size,-1,1)

        # print(n_vec.shape)
        # print(i_vec.shape)
        # print(o_vec.shape)


        o_loss=torch.bmm(i_vec,o_vec).sigmoid().log().neg()
        n_loss=torch.bmm(i_vec,n_vec).sigmoid().log().mean()


        loss=o_loss+n_loss

        return loss.mean()  #batch니까 mean




    


