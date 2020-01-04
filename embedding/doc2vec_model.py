import torch
import torch.nn as nn


class doc2vec(nn.Module):

    def __init__(self,model, lecture_len, lecture_dim=256, num_words=2):
        super(doc2vec, self).__init__()
        
        self.lecture_len=lecture_len
        self.lecuter_dim=lecture_dim
        self.lecture = nn.Embedding(lecture_len,lecture_dim)
        self.word_emb=model.detach()


    def forward(self, inputs, target):



        doc_id=torch.LongTensor(inputs[:,0].unsqueeze(1))
        context=inputs[:,1:]

        lec_vec=self.lecture(doc_id)
        word_vec=self.word_emb[context]
        target_vec=self.word_emb[target].unsqueeze(1)

        d_vec=torch.cat((lec_vec,word_vec),1).mean(1).unsqueeze(1)
        i_loss=torch.bmm(d_vec,target_vec.permute(0,2,1))

        print(i_loss)
        print(i_loss.shape)
