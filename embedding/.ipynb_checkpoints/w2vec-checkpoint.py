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
    
    
def negative_sampling(y_true,y_pred,vocab_len,n=10):
 

    for i in range(y_pred.size()[0]):
        if i==0:
            a=np.random.choice(np.arange(1,vocab_len+1),size=n,replace=False).reshape(1,-1)
        else:
            a=np.vstack((a,np.random.choice(np.arange(1,vocab_len+1),size=n,replace=False)))
   

    for idx,label in enumerate(y_true):
        if label.item() not in a[idx]:
            a[idx][0]=label
            
        y_is=torch.tensor([y_pred[idx][j-1] for j in a[idx]]).unsqueeze(0)
        y_is_true=np.where(a[idx]==label.item())[0]+1
        
        if idx==0:
            y_pred_batch=y_is
            y_true_batch=np.array(y_is_true)
        else:
            y_pred_batch=torch.cat([y_pred_batch,y_is])
            y_true_batch=np.hstack((y_true_batch,y_is_true)) 
        
    y_pred_batch=Variable(y_pred_batch,requires_grad=True)
    y_true_batch=torch.tensor(y_true_batch)
    
    criterion = nn.CrossEntropyLoss()
    
#     print(y_pred_batch)
#     print(y_true_batch)
    
    return criterion(y_pred_batch,y_true_batch)
    
    
    
    
    