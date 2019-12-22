from doc2vec_model import doc2vec

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import torch.optim as optim

import pickle
import argparse

from doc2vec_model import negative_sampling



def making_db(x):
    pass


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--docid_dir',default='./preprocessing/doc_id_2017_1.pickle', help='doc_id?',type=str)
    parser.add_argument('--docskipgram_dir',default='./preprocessing/doc_skipgram_idx_2017_1.pickle', help='vocab?',type=str)
    parser.add_argument('--make_training',default=False, help='make training_set?',type=bool)
    parser.add_argument('--training_set',default='./doc2vec_training_set_2017_1.pickle', help='training_set?',type=str)
    parser.add_argument('--embedding_model',default='./training_with_mecab_2018_2_epoch_14', help='model?',type=str)
    parser.add_argument('--load_dir',default='.', help='load directory?',type=str)
    parser.add_argument('--vocabfreq_dir',default="./preprocessing/vocab_mecab_multi_freq.pickle", help='vocabfreq dir?',type=str)
    parser.add_argument('--save_dir',default="./doc2vec_2017_1", help='vocabfreq dir?',type=str)



    parser.add_argument('--epoch',default=3, help='epoch?',type=int)
    parser.add_argument('--lr',default=0.0001, help='lr?',type=float)
    parser.add_argument('--early_stopping',default=10, help='early_stopping?',type=int)

    args = parser.parse_args()



    with open(args.docid_dir,'rb') as f:
        doc_id=pickle.load(f)
    with open(args.docskipgram_dir,'rb') as f:
        dataset=pickle.load(f)
    
    emb_model=torch.load(args.embedding_model,map_location='cpu')['state_dict']
    lecture_len=len(doc_id)
    print('document size : ',lecture_len)

    with open(args.vocabfreq_dir,'rb') as f: #defaultdict으로 바꿔야됨
        multi_freq=pickle.load(f)



    model=doc2vec(freq_dic=multi_freq,lecture_len=lecture_len,model=emb_model)

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
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    if len(args.load_dir)>1:
        checkpoint=torch.load(args.load_dir,map_location=device)
        print(checkpoint)
        start_epoch=checkpoint['epoch']+1
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        model.to(device)
        start_epoch=0


    epochs=start_epoch+args.epoch
    early_stopping=args.early_stopping
    es=0
    done=0

    running_loss=99
    previous_running_loss=100


    for epoch in range(start_epoch,epochs+1):
        

        for i,data in enumerate(my_dataloader,0):
            optimizer.zero_grad()
            outputs=model.forward(data[0],data[1])
            loss=negative_sampling(*outputs)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            if (i+1) % 500 == 0:    # print every 2000 mini-batches
                running_loss=running_loss/500
                print('[%d, %5d] loss: %.5f' %(epoch + 1, i + 1, running_loss))

                if abs(running_loss)<abs(previous_running_loss):
                    previous_running_loss=running_loss
                    es=0

                es+=1  
                running_loss=0

            if es==early_stopping:
                done=1
                epochs=epoch
                break

        running_loss=0
        if done:
            break

    state={'epoch':epochs,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict()}
    torch.save(state,args.save_dir+'_epoch_'+str(epochs))
        