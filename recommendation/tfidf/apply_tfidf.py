import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Komoran 
import ast
import collections
import copy
import random
import argparse


class prepro:

    def __init__(self):
        None

    def pick(self,x):                 #filter 역할, 입력: pos tagging 된 sentence
        if x[1] in ['NNG','NNP','VV','VA','SN','NR']:
            return x
        else:
            return None

    def documentize(self,x):
        try:
            x=ast.literal_eval(x)
            y=' '.join(x)
            a=komoran.pos(y.replace('\n',' '))
            a=' '.join(list(map(lambda x:x[0],list(filter(self.pick,a)))))
            
            return a
        except:
            return float('nan') #강의평없는경우

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

if __name__=="__main__":

    komoran = Komoran() 
    print(komoran.morphs(u'졸린데 수업 끝내주세요'))
    print(komoran.nouns(u'모기 물렸다'))

    df=pd.read_csv(r"C:\tensor_code\kluebot\data\prep\2017_1_pre")
    df_art=df[df.Classification=='교양']

    #이 부분 바꿔야됨
    df_new=df_art['keyword_sent'].apply(lambda x:prepro().documentize(x))   
    df_extract=pd.concat([df_art['className'],df_new],axis=1).dropna().reset_index(drop=True)


    tfidf,vocab=tf_idf(df_extract['keyword_sent']).cal_tf()
    print(len(vocab))

    lec_vec=dict(zip(df_extract['className'],tfidf))



    x=df_extract.className.iloc[random.randint(1,df_extract.shape[0])]


    print(x,'와 과제 관련 가장 가까운 강의들: ')
    print("l2: ",recommend(x,5,'l2_dis').getit())
    #print('\n')
    #print("코사인유사도:" ,recommend(x,5,cossim))
