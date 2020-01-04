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
    parser.add_argument('--make_training',default=False, help='make training_set?',type=bool)
    parser.add_argument('--training_set',default='./doc2vec_training_set.pickle', help='training_set?',type=str)
    args = parser.parse_args()



    with open(args.docid_dir,'rb') as f:
        doc_id=pickle.load(f)
    with open(args.docskipgram_dir,'rb') as f:
        dataset=pickle.load(f)
    
    emb_model=torch.load('./w2v_withoud_ns_epoch_3_lr_0.0001',map_location='cpu')['state_dict']['embedding.weight'].detach()
    lecture_len=len(doc_id)
    print('document size : ',lecture_len)



    model=doc2vec(lecture_len=lecture_len,model=emb_model)

    if args.make_training:
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

        with open(args.training_set,'wb') as f:
            pickle.dump([X,Y],f)

    with open(args.training_set,'rb') as f:
        X,Y=pickle.load(f)
        print('로드완료')

    X=torch.LongTensor(X).view(-1,3)
    Y=torch.tensor(Y)

    my_dataset = TensorDataset(X,Y)#create your datset
    batchsize=16
    my_dataloader = DataLoader(my_dataset,batch_size=batchsize,shuffle = True,drop_last=True) #create your dataloader

    for x,y in my_dataloader:
        model.forward(x,y)
        break