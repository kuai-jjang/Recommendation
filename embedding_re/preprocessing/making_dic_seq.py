import pandas as pd
import ast
import re
import time
import numpy as np
import argparse

from konlpy.tag import Okt
import pickle
from compare_pos_tag import sampling_by_length,preprocessing
from collections import defaultdict
from eunjeon import Mecab




class making_dict:
    def __init__(self,tokenizer,sentences,freq,remove_pos,n=2):
        self.tokenizer=tokenizer
        self.sentences=sentences
        self.vocab=my_vocab
        self.len_sent=len(sentences)
        self.step=0  #진행상황
        self.freq={}
        self.remove_pos=remove_pos
        self.n=n

    def make_vocab(self):
        _=list(map(lambda x:self.tokenizing(x),self.sentences))

    def tokenizing(self,x):

        self.step+=1
        if self.step%1000==0:
            print(self.step,'/',self.len_sent)
        self.tokens=self.tokenizer.pos(x)
        
        for i in self.tokens:
            if i[1][0] not in self.remove_pos and i[1] not in self.remove_pos:
                if i[0] not in ['것','수']:
                    self.update_dict(i)
            
    def update_dict(self,token):
        self.check_freq(token)
            
    def check_freq(self,token):

        if token not in freq:
            freq[token]=1
        else:
            self.update_freq(token)

    def update_freq(self,token):
        
        freq[token]+=1
        if freq[token]==self.n:
            self.vocab[token]=len(self.vocab)
 
def preprocessing(data):
    lecture_sentences=list(map(lambda x:ast.literal_eval(x),data))
    lecture_sentences=list(filter(lambda x:(len(x)!=0 and len(x)!=1),lecture_sentences)) #강의평이 없가나 1개 밖에 없는 경우 날림
    lecture_sentences=[i for e in lecture_sentences for i in e]
    return lecture_sentences

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',default=r'C:\tensor_code\kluebot\data\raw\2018_2.csv', help='datafile',type=str)
    parser.add_argument('--make_vocab',default=True, help='make vocab?',type=bool)
    parser.add_argument('--make_seq',default=True, help='make seq?',type=bool)

    parser.add_argument('--online',default=True, help='update vocab?',type=bool)
    
    parser.add_argument('--vocab_name',default='./vocab_mecab.pickle', help='vocab_name?',type=str)
    parser.add_argument('--seq_name',default='./seq_mecab_2018_2', help='seq_name?',type=str)
    

    args = parser.parse_args()

    #강의평 데이터 불러오기
    my_data=pd.read_csv(args.data_dir)
    lecture_sentences=preprocessing(my_data.LectureEval.values)

    #ㄱ-ㅎ제거,'\n'
    k=re.compile('[ㄱ-ㅎ|ㅏ-ㅣ]')
    sample_sentence=list(map(lambda x:k.sub('',x),lecture_sentences))
    sample_sentence=list(map(lambda x:re.sub('\n','',x),sample_sentence))

    #tokenizer=Okt()
    tokenizer=Mecab()

    my_vocab={}
    freq={}
    remove_pos=['J','E','X','SF','SE','SSO','SSC','SC','SY']   #okt : ['Josa','Suffix','Foreign','Punctuation']




    if args.online:
        with open(args.vocab_name,'rb') as f:
            my_vocab=pickle.load(f)
        with open(args.vocab_name[:-7]+'_freq.pickle','rb') as f:
            freq=pickle.load(f)
                

    #사전 만들기
    if args.make_vocab:

        trial=making_dict(tokenizer,sample_sentence,my_vocab,remove_pos)
        trial.make_vocab()
        #사전 저장
        with open(args.vocab_name,'wb') as handle:  #빈도수랑 사전만드는 분리하고 싶다
            pickle.dump(my_vocab,handle)
        with open(args.vocab_name[:-7]+'_freq.pickle','wb') as handle:  #빈도수랑 사전만드는 분리하고 싶다
            total=sum(freq.values())
            pickle.dump(freq,handle)
            freq.update((x, y/total) for x, y in freq.items())
        multi_prob=np.array([])
        for i in my_vocab.keys():
            multi_prob=np.append(multi_prob,freq[i])
        with open(args.vocab_name[:-7]+'_multi_freq.pickle','wb') as f:
            pickle.dump(multi_prob,f)

    #사전 불러오기
    if args.make_seq:
        with open(args.vocab_name,'rb') as handle:
            w2i=pickle.load(handle)

        ####여기서부터 seq_idx 만들기
        vocab_size=len(w2i)
        print(vocab_size)
        w2i_default=defaultdict(lambda:vocab_size,w2i)  #'unk ' : vocab size
    
        #default dict 값이 unk
        #w2i['unk']=len(w2i)+1
        # make_seq=making_seq(Okt())
        
        step=0
        length_sent=len(sample_sentence)
        for i in sample_sentence:
            step+=1
            if step%1000==0:
                print(step,'/',length_sent)
            tokenized=tokenizer.pos(i)
            tokenized=list(filter(lambda x:x[1][0] not in remove_pos and x[1] not in remove_pos,tokenized))
            tokenized=list(filter(lambda x:x[0] not in ['것','수'],tokenized))
            seq=list(map(lambda x:w2i_default[x],tokenized))
            with open(args.seq_name+'.txt', 'a') as f:
                for item in seq:
                    f.write("%s " % item)
                f.write('\n')