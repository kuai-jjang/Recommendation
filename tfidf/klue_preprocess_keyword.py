#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import re
from konlpy.tag import Komoran 
import numpy as np
import argparse
from operator import itemgetter 
import ast
import re
import glob
import os

class prerpoces:
    
    def __init__(self,y):
        self.content=y
        
    def rem(self):   #1. string list로 변환, 2. ㄱ-ㅎ 제거
        mysent=re.sub('[ㄱ-ㅎ]','',self.content)
        mylist=ast.literal_eval(mysent)
        return list(filter(lambda x:len(x)!=0,mylist))
    
    def extract(self,keyword):  #키워드 문자리스트 추출
        self.keyword=keyword
        sent=[]
        for i,x in enumerate(self.content):
            if self.keyword in x:
                sent.append(x)
            else:
                continue
        return sent

if __name__=="__main__":
    
    keyword_dict={1:'',2:'과제',3:'출석',4:'선수'}

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str,required=True,
                help='./data')
    parser.add_argument('--keyword', type=int,required=True,
                help="What keyword?")

    args = parser.parse_args()
    base_dir = args.dir
    keyw = keyword_dict[args.keyword]
    data_dir=os.path.join(base_dir,'raw\*')
    save_dir=os.path.join(base_dir,'prep')
    files=glob.glob(data_dir)


    komoran = Komoran() 
    print(komoran.morphs(u'졸린데 수업 끝내주세요'))
    print(komoran.nouns(u'모기 물렸다'))
    print(data_dir)

    
    for i in files:
        df=pd.read_csv(i)
        df['trial']=df['LectureEval'].apply(lambda x:prerpoces(x).rem())
        df['keyword_sent']=df['trial'].apply(lambda x:prerpoces(x).extract(keyw))
        df_extract=df[df.keyword_sent.str.len()!=0] #keyword없는 것 제외
        df_extract.drop(['trial'],axis=1,inplace=True)

        print(df.keyword_sent.head(5))
        semester=i.split('\\')[-1][:-4]
        print(semester)
        save_name='/'.join(i.split('\\')[:-2])+'/'+'prep'+'/'+semester+'_'+str(args.keyword)
        print(save_name)
        if not os.path.isfile(save_name):
            df_extract.to_csv(save_name,encoding='utf-8',index=False)
