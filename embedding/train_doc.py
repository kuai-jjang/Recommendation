from doc2vec_model import doc2vec

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset

import pickle
import argparse



def making_db(x):
    pass


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--docid_dir',default='./preprocessing/doc_id.pickle', help='doc_id?',type=str)
    parser.add_argument('--docskipgram_dir',default='./preprocessing/doc_skipgram_idx.pickle', help='vocab?',type=str)
  
    args = parser.parse_args()



    with open(args.docid_dir,'rb') as f:
        doc_id=pickle.load(f)
    with open(args.docskipgram_dir,'rb') as f:
        dataset=pickle.load(f)

    # X=torch.tensor([])
    # Y=torch.tensor([])

    #torch.cat 은 cpu에서 느림
    # for i in range(len(dataset)):
    #     try:
    #         for j in dataset[i]:
    #             X=torch.cat((X,torch.FloatTensor([[i,j[0],j[1]]])))
    #             Y=torch.cat((Y,torch.FloatTensor(j[2])))
    #     except:
    #         continue

    X=[]
    Y=[]


    for i in range(len(dataset)):
        try:
            for j in dataset[i]:
                X.append([[i,j[0],j[1]]])
                Y.append(j[2])
        except:
            continue

        if (i+1)%49==0:
            print(i)

    X=torch.FloatTensor(X).view(-1,3)
    Y=torch.FloatTensor(Y)

    my_dataset = TensorDataset(X,Y)#create your datset
    batchsize=16
    my_dataloader = DataLoader(my_dataset,batch_size=batchsize,shuffle = True,drop_last=True) #create your dataloader

    for i in my_dataloader:
        print(i)
        break