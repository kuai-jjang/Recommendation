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

class making_dict:
    def __init__(self,tokenizer,sentences,freq,n=2):
        self.tokenizer=tokenizer
        self.sentences=sentences
        self.vocab=my_vocab
        self.len_sent=len(sentences)
        self.step=0  #진행상황
        self.freq={}
        self.n=n

    def make_vocab(self):
        _=list(map(lambda x:self.tokenizing(x),self.sentences))
        
    def tokenizing(self,x):
        self.step+=1
        if self.step%1000==0:
            print(self.step,'/',self.len_sent)
        self.tokens=self.tokenizer.pos(x)
        for i in self.tokens:
            self.update_dict(i)
            
    def update_dict(self,token):
        if token not in self.vocab:
            self.check_freq(token)
            
    def check_freq(self,token):
        if token not in freq:
            freq[token]=1
        else:
            self.istwo(token)
    def istwo(self,token):
        if freq[token]<=self.n-1:
            freq[token]+=1
        if freq[token]==self.n:
            self.vocab[token]=len(self.vocab)+1
            del  freq[token]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',default=r'C:\tensor_code\kluebot\data\raw\2017_1.csv', help='datafile',type=str)
    parser.add_argument('--making_vocab',default=False, help='make vocab?',type=bool)
    parser.add_argument('--make_seq',default=False, help='make seq?',type=bool)
    
    args = parser.parse_args()

    #강의평 데이터 불러오기
    my_data=pd.read_csv(args.data_dir)
    lecture_sentences=preprocessing(my_data.LectureEval.values)

    #ㄱ-ㅎ제거
    k=re.compile('[ㄱ-ㅎ]+|')
    sample_sentence=list(map(lambda x:k.sub('',x),lecture_sentences))
    tokenizer=Okt()

    #사전 만들기
    if args.making_vocab:
        my_vocab={}
        freq={}
        trial=making_dict(tokenizer,sample_sentence,my_vocab)
        trial.make_vocab()
        #사전 저장
        with open('./my_vocab_freq_3.pickle','wb') as handle:
            pickle.dump(my_vocab,handle)
    #사전 불러오기

    if args.make_seq:
        with open('./my_vocab_freq_3.pickle','rb') as handle:
            w2i=pickle.load(handle)

        ####여기서부터 seq_idx 만들기
        vocab_size=len(w2i)
        print(vocab_size)
        w2i_default=defaultdict(lambda:vocab_size+1,w2i)
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
            seq=list(map(lambda x:w2i_default[x],tokenized))
            with open('./sentence4idx.txt', 'a') as f:
                for item in seq:
                    f.write("%s " % item)
                f.write('\n')