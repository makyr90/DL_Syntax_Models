import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.autograd import Variable
from allennlp.modules.augmented_lstm import AugmentedLstm
from allennlp.modules.input_variational_dropout import InputVariationalDropout

class Char_RNN(nn.Module):

    def __init__(self, char_to_index, char_embed_size, hidden_size,output_size,dropout,cuda_flag, batch_first=True):

        """
        Args:
            char_to_index:
            char_embed_size: char embeddings dim
            hidden_size: lstm reccurent dim
            dropout: dropout probability
            batch_first: batch first option
        """


        super(Char_RNN, self).__init__()

        self.char_to_index = char_to_index
        self.char_embed_size = char_embed_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.output_size  = output_size
        self.batch_first = batch_first
        self.padding_index = self.char_to_index['__PADDING__']
        self.cuda_flag = cuda_flag


        self.char_encoder = nn.Embedding(len(self.char_to_index), self.char_embed_size, sparse=True, padding_idx= self.padding_index)
        torch.nn.init.xavier_uniform_(self.char_encoder.weight.data)

        self.char_rnn = AugmentedLstm(input_size= self.char_embed_size, hidden_size = self.hidden_size,go_forward = True, recurrent_dropout_probability = self.dropout,
                         use_highway = False, use_input_projection_bias = False)

        self.char_rnn.state_linearity.bias.data.fill_(0.0)
        self.var_drop = InputVariationalDropout(self.dropout)
        self.w_atten  = nn.Linear(self.hidden_size,1,bias=False)
        self.w_atten.weight.data.fill_(0.0)
        self.char_projection = nn.Linear(self.hidden_size*2,self.output_size,bias=True)
        self.char_projection.weight.data.fill_(0.0)
        self.char_projection.bias.data.fill_(0.0)
        self.drp = nn.Dropout(self.dropout)

    def forward(self,char_ids,seq_lengths):

        tokenIdChars = []
        for sent in char_ids:
            tokenIdChars.extend([idChars for idChars in sent])
        tokenIdChars_set = set(map(tuple,tokenIdChars))
        tokenIdChars = list(map(list,tokenIdChars_set))
        tokenIdChars.sort(key=lambda x: -len(x))
        max_len = len(max(tokenIdChars,key=len))
        batch_size = len(tokenIdChars)
        char_tensor = torch.zeros(batch_size,max_len).long()
        char_tensor.fill_(self.padding_index)

        for idx in range(len(tokenIdChars)):
            for jdx in range(len(tokenIdChars[idx])):
                char_tensor[idx,jdx] = tokenIdChars[idx][jdx]

        if self.cuda_flag:
            char_tensor = char_tensor.cuda()

        char_embed = self.char_encoder(char_tensor)
        char_embed  = self.var_drop(char_embed)
        char_seq_lengths = np.array([len(char) for char in tokenIdChars])
        packed_input = pack_padded_sequence(char_embed, char_seq_lengths,batch_first=True)
        packed_output, (ht,cell) = self.char_rnn(packed_input, None)
        out_rnn, lengths = pad_packed_sequence(packed_output, batch_first=True)
        out_rnn = self.var_drop(out_rnn)
        w_att = self.w_atten(out_rnn)

        if self.cuda_flag:
            mask = torch.ones(w_att.size()).cuda()
        else:
            mask = torch.ones(w_att.size())

        for i, l in enumerate(lengths):
            if l < out_rnn.size()[1]:
                mask[i, l:] = 0

        w_att = w_att.masked_fill(mask == 0, -1e9)

        #compute and apply attention
        attentions = F.softmax(w_att.squeeze(),dim=1)
        weighted = torch.mul(out_rnn, attentions.unsqueeze(-1).expand_as(out_rnn))
        char_att = weighted.sum(1).squeeze()
        char_embs = torch.cat((char_att,cell.squeeze(0)),1)
        char_embs = self.drp(char_embs)
        proj_char_embs = self.char_projection(char_embs)
        RNN_embs = {}
        for idx in range(len(tokenIdChars)):
            RNN_embs[str(tokenIdChars[idx])] = proj_char_embs[idx,:]

        max_seq = torch.max(seq_lengths).cpu().numpy().tolist()
        if self.cuda_flag:
            char_emb_tensor = Variable(torch.zeros(len(char_ids),max_seq,self.output_size)).cuda()
        else:
            char_emb_tensor = Variable(torch.zeros(len(char_ids),max_seq,self.output_size))

        for idx in range(len(char_ids)):
            for jdx in range(len(char_ids[idx])):
                char_emb_tensor[idx,jdx,:] = RNN_embs[str(char_ids[idx][jdx])]

        return char_emb_tensor
