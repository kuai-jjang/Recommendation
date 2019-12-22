import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Komoran 
import ast
import collections
import copy
import random
import argparse
import re
import pickle
from eunjeon import Mecab


# class prepro:

#     def __init__(self):
#         None

#     def pick(self,x):                 #filter 역할, 입력: pos tagging 된 sentence
#         if x[1] in ['NNG','NNP','VV','VA','SN','NR']:
#             return x
#         else:
#             return None

#     def documentize(self,x):
#         try:
#             x=ast.literal_eval(x)
#             y=' '.join(x)
#             a=komoran.pos(y.replace('\n',' '))
#             a=' '.join(list(map(lambda x:x[0],list(filter(self.pick,a)))))
            
#             return a
#         except:
#             return float('nan') #강의평없는경우

class tf_idf:
    def __init__(self,x):  #x: lectures
        self.cont=x
    def cal_tf(self):
        tfidfv = TfidfVectorizer(max_df=0.9,min_df=0.2).fit(self.cont.values)   #transform: 행렬변환, max_df: 0.9할 이상의 doc에서 나오는 단어 삭제
        tfidf=tfidfv.transform(self.cont.values).toarray()
        vocab=tfidfv.vocabulary_
        return tfidf,vocab

class recommend:
    def __init__(self,x,num,metric):
        func_dict={'l2_dis':self.l2_dis,'cossim':self.cossim}
        self.x=x
        self.num=num
        self.metric=func_dict[metric]

    def getit(self):   #사이킷런 knn 활용하기
        my_vec=lec_vec[self.x]
        new_dict=copy.deepcopy(lec_vec)  #깊복, 얕복 차이
        
        del new_dict[self.x]
        
        q=collections.deque(maxlen=self.num)
        q.append((self.x,1000))
        
        for k,v in new_dict.items():
            dist=self.metric(my_vec,v)
        
            if (q[0][1] > dist): #첫번째를 max로 유지
                q.append((k,dist))
                q=collections.deque(sorted(q,reverse=True,key=lambda x:x[1]),maxlen=self.num)
            
        return q

    def l2_dis(self,x,y):
        return ((x-y)**2).sum()    

    def cossim(self,x,y):
        return 1/(((x*y).sum()/(x**2).sum()**0.5/(y**2).sum()**0.5))

class preprocessing:

    def __init__(self,data,tokenizer,remove_pos,start_idx):
        self.data=data
        self.tokenizer=tokenizer
        self.remove_pos=remove_pos
        self.start_index=start_idx
        self.k=re.compile('[ㄱ-ㅎ|ㅏ-ㅣ]')

    def removing(self,sent):
        
        sample_sentence=list(map(lambda x:self.k.sub('',x),sent))
        sample_sentence=list(map(lambda x:re.sub('\n','',x),sample_sentence))
        sample_sentence=list(map(lambda x:tokenizer.pos(x),sample_sentence))
        token=sample_sentence

        for idx,i in enumerate(sample_sentence):
            tokenized=list(filter(lambda x:x[1][0] not in remove_pos and x[1] not in remove_pos,i))
            tokenized=list(filter(lambda x:x[0] not in ['것','수'],tokenized))
            token[idx]=[i for e in tokenized for i in e]
        
        return token

    def prepro(self):
        
        self.idexes=[]
        def removing(x):
                       
            if len(x)!=0 and len(x)!=1:
                
                self.start_index+=1
                return x
            else:
                self.idexes.append(self.start_index)
                self.start_index+=1
                return None
        lecture_sentences=list(map(lambda x:ast.literal_eval(x),self.data))
        
        lecture_sentences=list(filter(lambda x:removing(x),lecture_sentences)) #강의평이 없가나 1개 밖에 없는 경우 날림
        
        lecture_sentences=list(map(lambda x:self.removing(x),lecture_sentences))
        #lecture_sentences=[i for e in lecture_sentences for i in e]

        return lecture_sentences,self.idexes



if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',default=r"C:\tensor_code\kluebot\data\raw\2018_2.csv", help='datadir?',type=str)

    args = parser.parse_args()

    sem=args.data_dir.split('\\')[-1][:-4]

    #komoran = Komoran() 
    tokenizer=Mecab()
    print(tokenizer.pos(u'졸린데 수업 끝내주세요'))
    print(tokenizer.nouns(u'모기 물렸다'))

    df=pd.read_csv(args.data_dir)
    
    df_art=df[df.Classification=='교양']
    df_start_idx=df_art.index.values[0]

    if 'raw' in args.data_dir:
        df_art['keyword_sent']=df_art['LectureEval']

    remove_pos=['J','E','X','SF','SE','SSO','SSC','SC','SY']


    M=preprocessing(df_art['keyword_sent'].values,tokenizer,remove_pos,df_start_idx)
    lecture_sentences,drop_idexes=M.prepro()
    

    drop_list=df_art.loc[drop_idexes]
    df_art.drop(drop_idexes,inplace=True,axis=0)

    df_art['doc_id']=list(zip(df_art['ProfessorName'],df_art['className']))

    df_extract=pd.concat([df_art['doc_id'],df_art['keyword_sent']],axis=1)
    
    print(df_extract.head())

    
    tfidf,vocab=tf_idf(df_extract['keyword_sent']).cal_tf()
  
    data=pd.DataFrame(tfidf).T
    print(data.head())

    df_new=pd.concat([df_extract.reset_index(drop=True),data.reset_index(drop=True)],axis=1, ignore_index=True)
    print(df_new.head())
    df_new.to_csv('./lecture_vector_'+sem+'.csv',index=False)
    lec_vec=dict(zip(df_extract['doc_id'],tfidf))


    # with open('./lec_vec_2017_1','wb') as f:
    #     pickle.dump(lec_vec,f)
    # x=df_extract.className.iloc[random.randint(1,df_extract.shape[0])]
    # print(x,'와 과제 관련 가장 가까운 강의들: ')
    # print("l2: ",recommend(x,5,'l2_dis').getit())
    # #print('\n')
    # #print("코사인유사도:" ,recommend(x,5,cossim))
