import torch
import torch.nn as nn


class doc2vec(nn.Module):

    def __init__(self,model, lecture_len, lecture_dim=256, num_words=2):
        super(doc2vec, self).__init__()
        
        self.lecture_len=lecture_len
        self.lecuter_dim=lecture_dim
        # lecture matrix
        self.lecture = nn.Embedding(lecture_len,lecture_dim)
        # word matrix
        # output layer parameters
        self.word_emb=model
        self._O = nn.Parameter(torch.FloatTensor(lecture_dim, num_words).zero_(), requires_grad=True)

    def forward(self, context_ids, doc_ids, target_noise_ids):


        x = torch.add(
            self.lecture[doc_ids], torch.sum(self.word_emb[context_ids], dim=1))

        return torch.bmm(
            x.unsqueeze(1),
            self._O[:, target_noise_ids].permute(1, 0, 2)).squeeze()

    def get_paragraph_vector(self, index):
        return self._D[index, :].data.tolist()
