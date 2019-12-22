import pandas as pd
import numpy as np
import re
from compare_pos_tag import sampling_by_length,preprocessing
from konlpy.tag import Okt
import ast
import pickle
import argparse
from collections import defaultdict
from eunjeon import Mecab




class tokenizing:
    
    def __init__(self,tokenizer,dataframe,doc_id,w2i,window_size=3):
        
        self.tokenizer=tokenizer
        self.dataframe=dataframe
        self.doc_id=doc_id
        self.w2i=w2i
        self.doc_set={}
        self.step=0
        
        self.window_size=3

    def make_pos(self,x):
        self.step+=1
        if self.step%499==0:
            print(self.step)
        for i in x['LectureEval']:
            a=self.tokenizer.pos(i)
            a=list(filter(lambda x:x[1] not in ['Josa','Suffix','Foreign','Punctuation'],a))
            self.update_(x['doc_name'],a)
            
    def update_(self,x,a):
        
        token_length=len(a)
        if token_length>=self.window_size:
            skip_set=self.skipgram(a)        
            if doc_id[x] not in self.doc_set.keys():
                 self.doc_set[doc_id[x]]=[]    
                    
            for i in skip_set:
                self.doc_set[doc_id[x]].append(i)
        
    
    def skipgram(self,x):
        skipgram_set=[]
        for i in range(len(x)-self.window_size+1):
            idx_sets=self.w2i_(x[i:i+self.window_size])
            skipgram_set.append(idx_sets)
            
        return skipgram_set
    
    def w2i_(self,x):
        a=[]
        for i in x:
            a.append(self.w2i[i])
        return a
    
def pre_dataframe(dataframe):
    
    doc_frame=pd.Series(zip(dataframe.ProfessorName.values,dataframe.className.values))
    doc_merge=pd.concat([doc_frame.rename('doc_name'),dataframe.LectureEval],axis=1)

    return doc_merge
            
    
def make2string(x):

    lecutres=ast.literal_eval(x)

    return lecutres

def pre_re(y):
    k=re.compile('[ㄱ-ㅎ]+|')
    sample_sentence=list(map(lambda x:k.sub('',x),y))
    sample_sentence=list(map(lambda x:re.sub('\n','',x),sample_sentence))
    return sample_sentence



if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',default=r"C:\tensor_code\kluebot\data\raw\2017_1.csv", help='datadir?',type=str)

    args = parser.parse_args()


    my_data=pd.read_csv(args.data_dir)
    lecture_sentences=preprocessing(my_data.LectureEval.values)
    
    sem=args.data_dir.split('\\')[-1][:-4]
    
    tokenizer=Mecab()



    trial=pre_dataframe(my_data)
    trial['LectureEval']=trial['LectureEval'].map(lambda x:make2string(x))
    trial['LectureEval']=trial['LectureEval'].map(lambda x:pre_re(x))

    with open('./doc_id_'+sem+'.pickle','rb') as f:
        doc_id=pickle.load(f)

    with open('./vocab_mecab.pickle','rb') as f:
        w2i=pickle.load(f)
    
    vocab_size=len(w2i)
    w2i_default=defaultdict(lambda:vocab_size,w2i)

    hehe=tokenizing(tokenizer,trial,doc_id,w2i_default)
    trial.apply(lambda x:hehe.make_pos(x),axis=1)

    with open('./doc_skipgram_idx_'+sem+'.pickle','wb') as f:
        pickle.dump(hehe.doc_set,f)