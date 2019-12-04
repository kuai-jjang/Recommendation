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
            if i[1] not in ['Josa','Suffix','Foreign','Punctuation']:
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
 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',default=r'C:\tensor_code\kluebot\data\raw\2017_1.csv', help='datafile',type=str)
    parser.add_argument('--make_vocab',default=True, help='make vocab?',type=bool)
    parser.add_argument('--make_seq',default=True, help='make seq?',type=bool)
    
    parser.add_argument('--vocab_name',default='./vocab_without_josa_gut_su.pickle', help='vocab_name?',type=str)
    parser.add_argument('--seq_name',default='./seq_without_josa_gut_su', help='seq_name?',type=str)
    

    args = parser.parse_args()

    #강의평 데이터 불러오기
    my_data=pd.read_csv(args.data_dir)
    lecture_sentences=preprocessing(my_data.LectureEval.values)

    #ㄱ-ㅎ제거,'\n'
    k=re.compile('[ㄱ-ㅎ]+|')
    sample_sentence=list(map(lambda x:k.sub('',x),lecture_sentences))
    sample_sentence=list(map(lambda x:re.sub('\n','',x),sample_sentence))

    tokenizer=Okt()

    #사전 만들기
    if args.make_vocab:
        my_vocab={}
        freq={}
        trial=making_dict(tokenizer,sample_sentence,my_vocab)
        trial.make_vocab()
        #사전 저장
        with open(args.vocab_name,'wb') as handle:  #빈도수랑 사전만드는 분리하고 싶다
            pickle.dump(my_vocab,handle)
        with open(args.vocab_name[:-7]+'_freq.pickle','wb') as handle:  #빈도수랑 사전만드는 분리하고 싶다
            pickle.dump(freq,handle)

    #사전 불러오기
    if args.make_seq:
        with open(args.vocab_name,'rb') as handle:
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
            tokenized=list(filter(lambda x:x[1] not in ['Josa','Suffix','Foreign','Punctuation'],tokenized))
            tokenized=list(filter(lambda x:x[0] not in ['것','수'],tokenized))
            seq=list(map(lambda x:w2i_default[x],tokenized))
            with open(args.seq_name+'.txt', 'a') as f:
                for item in seq:
                    f.write("%s " % item)
                f.write('\n')