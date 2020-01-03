import pickle
import pandas as pd
from matplotlib import font_manager,rc
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import glob


def make_vector(vector,vocab):
    hw_dim=vocab['과제']
    ex_dim=vocab['시험']
    team_dim=vocab['팀플']
    hw_exam=vec[vec.columns[2:]].iloc[:,[ex_dim,hw_dim,team_dim]]
    hw_exam.columns=['exam','hw','team']
    
    
    return hw_exam


def make_idx(hw_exam,team=False):
    
    hw_exam[['exam_bool','hw_bool','team_1']]=hw_exam>hw_exam.quantile(.25)
    hw_exam['hw_dom']=hw_exam['hw']>hw_exam['exam']
    total_idx=hw_exam.index.values
    hw_idx=hw_exam.index[hw_exam['hw_dom']].values
    mid=hw_exam.index[(~hw_exam['hw_bool'])&(~hw_exam['exam_bool'])].values
    hw_idx=list(set(hw_idx)-set(mid))
    exam_idx=list(set(total_idx)-set(hw_idx)-set(mid))
    
    hw_exam['team_dom']=hw_exam['team']>hw_exam['team'].quantile(0.9)
    team_3_idx=hw_exam.index[hw_exam['team_dom']].values
    team_2_idx=hw_exam.index[hw_exam['team_1']].values
    team_2_idx=list(set(team_2_idx)-set(team_3_idx))
    team_1_idx=list(set(total_idx)-set(team_3_idx)-set(team_2_idx))
    
    return exam_idx,mid,hw_idx,[team_1_idx,team_2_idx,team_3_idx]


def marking(haha,s):
    for idx,j in enumerate(haha):
        for i in j:
            s[i]=idx+1
    return s

if __name__=="__main__":

    vec=pd.read_csv('./lecture_vector/lecture_vector_okt_2017_1.csv')
    with open('./vocab/voacb_okt_2017_1.pickle','rb') as f:
        vocab=pickle.load(f)

    hw_exam=make_vector(vec,vocab)
    classname_data=vec.iloc[:,0]

    exam_idx,mid,hw_idx,team_idx=make_idx(hw_exam)

    #make marks
    s1=np.array([0 for i in range(vec.shape[0])])
    exam_marks=marking([exam_idx,mid,hw_idx],s1)
    hmm=list(zip(vec['0'],exam_marks))

    s2=np.array([0 for i in range(vec.shape[0])])
    team_marks=marking(team_idx,s2)
    tmm=list(zip(vec['0'],team_marks))


