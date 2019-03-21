import torch
import torch.nn as nn
from torch.autograd import Variable
import itertools

class Fasttext(nn.Module):

    def __init__(self, embed_tensor, projection_dim, padding_idx, cuda_flag):

        """
        Args:
            embed_tensor: tensor with pre-trained embeds
            projection_dim:
            padding_idx:
            cuda_flag:
        """

        super(Fasttext, self).__init__()

        self.projection_dim = projection_dim
        self.vocab_size, self.embed_size = embed_tensor.size()
        self.padding_index = padding_idx
        self.cuda_flag = cuda_flag

        self.ft_encoder = nn.Embedding(self.vocab_size, self.embed_size, sparse=True, padding_idx= self.padding_index, _weight=embed_tensor)
        self.ft_encoder.weight.requires_grad = False

        self.projection = nn.Linear(self.embed_size,self.projection_dim,bias=True)
        self.projection.weight.data.fill_(0.0)
        self.projection.bias.data.fill_(0.0)


    def forward(self,ext_word_ids,seq_lengths):

        ext_ids = list(itertools.chain.from_iterable(ext_word_ids))
        ext_unique_ids = list(set(ext_ids))
        if self.cuda_flag:
            ext_ids_tensor = torch.LongTensor(ext_unique_ids).cuda()
        else:
            ext_ids_tensor = torch.LongTensor(ext_unique_ids)

        ext_embeds = self.ft_encoder(ext_ids_tensor)
        proj_ext_embeds = self.projection(ext_embeds)

        proj_embs = {}
        for idx in range(len(ext_unique_ids)):
            proj_embs[ext_unique_ids[idx]] = proj_ext_embeds[idx,:]

        max_seq = torch.max(seq_lengths).cpu().numpy().tolist()
        if self.cuda_flag:
            ext_emb_tensor = Variable(torch.zeros(len(ext_word_ids),max_seq,self.projection_dim)).cuda()
        else:
            ext_emb_tensor = Variable(torch.zeros(len(ext_word_ids),max_seq,self.projection_dim))

        for idx in range(len(ext_word_ids)):
            for jdx in range(len(ext_word_ids[idx])):
                ext_emb_tensor[idx,jdx,:] = proj_embs[ext_word_ids[idx][jdx]]

        return ext_emb_tensor
