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
        #self.encoding_layer=nn.Linear(self.dim,128)
        #self.decoding_layer=nn.Linear(128,self.vocab_len)


    def forward(self,x):
        
        x=x.cuda() if self.embedding.weight.is_cuda else x
        embedds=self.embedding(x).cuda() if self.embedding.weight.is_cuda else self.embedding(x)
        
        #output=F.relu(self.encoding_layer(embedds))
        #output=self.decoding_layer(output)
        
        return embedds

'''
TODO : CBOW 모델 구현
'''
    
    
class negative_sampling(nn.Module):

    def __init__(self,model,vocab_len,freq_dic,enc_dim=256,n=5):
        super(negative_sampling, self).__init__()


        self.model=model
        self.vocab_len=vocab_len
        self.enc_dim=enc_dim
        self.freq_dic=torch.tensor(freq_dic)
        self.n=n


    def forward(self,x,y):


        batch_size = y.size()[0]

       # nwords = torch.FloatTensor(batch_size, self.n).uniform_(1, self.vocab_len).long()   

        nwords=torch.multinomial(self.freq_dic,batch_size*self.n).view(batch_size,self.n)  

        n_vec=self.model.forward(nwords).neg().view(batch_size,-1,self.n)
        i_vec=self.model.forward(x).view(batch_size,1,-1)
        o_vec=self.model.forward(y).view(batch_size,-1,1)

        o_loss=torch.bmm(i_vec,o_vec).sigmoid().log().neg()
        n_loss=torch.bmm(i_vec,n_vec).sigmoid().log().mean().neg()


        loss=o_loss+n_loss

        return loss.mean()  #batch니까 mean




    


