import pandas as pd
import numpy as np
from compare_pos_tag import sampling_by_length,preprocessing
from konlpy.tag import Okt
import ast
import pickle
import argparse


#doc id 만드는 함수 : ex_ ('서문경 교수','간호학 개론'):0 (0번 부터 시작)


class doc2vec_pre:

    def __init__(self,doc_id):
        
        self.doc_id=doc_id

    def make_doc_id(self,x):  #input : DataFrame[proff_name, class_name]
        if x not in doc_id.keys():
            doc_id[x]=len(self.doc_id)



if __name__=="__main__":



    my_data=pd.read_csv(r'C:\tensor_code\kluebot\data\raw\2017_1.csv')
    # lecture_sentences=preprocessing(my_data.LectureEval.values)
    # tokenizer=Okt()
    doc_id={}

    making_id=doc2vec_pre(doc_id)


    for i in zip(my_data[['ProfessorName','className']].values):
        x=tuple(i[0])
        making_id.make_doc_id(x)

    with open('./doc_id.pickle','wb') as f:
        pickle.dump(doc_id,f)