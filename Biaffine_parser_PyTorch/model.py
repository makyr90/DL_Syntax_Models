import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from stacked_bidirectional_lstm import StackedBidirectionalLstm
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from char_lstm import Char_RNN
#from biaffine import Biaffine
from util import drop_word_pos_embeds
from pretrained_embs import Fasttext

class RNN(nn.Module):

    def __init__(self, word_to_index, ext_word_to_index, char_to_index,char_embed_size,pos_embed_size,char_lstm_hidden_size, word_embed_size,
                     ext_embed_tensor, num_layers, hidden_size,dropout,r_dropout, mlp_arc_size,mlp_label_size,pos_to_index, xpos_to_index,rel_to_index,cuda_flag, batch_first=True):

        """
        Args:
            word_to_index:
            char_to_index:
            char_embed_size: word embeddings dim
            pos_embed_size: pos/xpos embeddings dim
            char_lstm_hidden_size: char LSTM reccurent dim
            word_embed_size: word embeddings dim
            num_layers: Bi-LSTM  of layers
            hidden_size: Bi-lstm reccurent dim
            dropout: dropout probability
            r_dropout: dropout probability for reccurent units (Gal & Grahami)
            mlp_arc_size: arc mlp hidden dim
            mlp_label_size: label mlp hidden dim
            pos_to_index:
            xpos_to_index:
            rel_to_index:
            cuda_flag:
            batch_first: batch first option
        """


        super(RNN, self).__init__()

        self.word_to_index = word_to_index
        self.ext_word_to_index = ext_word_to_index
        self.char_to_index = char_to_index
        self.char_embed_size = char_embed_size
        self.pos_embed_size = pos_embed_size
        self.char_lstm_hidden_size = char_lstm_hidden_size
        self.word_embed_size = word_embed_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.r_dropout = r_dropout
        self.mlp_arc_size = mlp_arc_size
        self.mlp_label_size = mlp_label_size
        self.pos_to_index = pos_to_index
        self.xpos_to_index = xpos_to_index
        self.rel_to_index = rel_to_index
        self.rels = len(self.rel_to_index)
        self.cuda_flag = cuda_flag
        self.batch_first = batch_first


        self.word_encoder = nn.Embedding(len(self.word_to_index), self.word_embed_size, sparse=True, padding_idx = self.word_to_index['__PADDING__'])
        torch.nn.init.xavier_uniform_(self.word_encoder.weight.data)

        self.pos_encoder = nn.Embedding(len(self.pos_to_index), self.pos_embed_size, sparse=True, padding_idx = self.pos_to_index['__PADDING__'])
        torch.nn.init.xavier_uniform_(self.pos_encoder.weight.data)

        self.xpos_encoder = nn.Embedding(len(self.xpos_to_index), self.pos_embed_size, sparse=True, padding_idx = self.xpos_to_index['__PADDING__'])
        torch.nn.init.xavier_uniform_(self.xpos_encoder.weight.data)

        self.ROOT = nn.Parameter(torch.zeros(self.word_embed_size + self.pos_embed_size))
        torch.nn.init.normal_(self.ROOT,0,0.001)
        self.rnn = StackedBidirectionalLstm(input_size = self.word_embed_size + self.pos_embed_size, hidden_size = self.hidden_size, num_layers= self.num_layers,
                                             recurrent_dropout_probability =self.r_dropout, use_highway = False)

        self.char_embeds = Char_RNN(self.char_to_index,self.char_embed_size,self.char_lstm_hidden_size,self.word_embed_size, self.dropout,self.cuda_flag)
        self.fasttext_embs = Fasttext(ext_embed_tensor, self.word_embed_size, self.ext_word_to_index['__PADDING__'], self.cuda_flag)

        #Set forget bias to zero
        for layer_index in range(self.num_layers):
            eval("self.rnn.forward_layer_{}.state_linearity.bias.data.fill_(0.0)".format(layer_index))
            eval("self.rnn.backward_layer_{}.state_linearity.bias.data.fill_(0.0)".format(layer_index))

        self.var_drop = InputVariationalDropout(self.dropout)

        self.arc_head = nn.Linear(2*self.hidden_size,self.mlp_arc_size,bias=True)
        self.arc_dep = nn.Linear(2*self.hidden_size,self.mlp_arc_size,bias=True)
        torch.nn.init.orthogonal_(self.arc_head.weight.data,gain = np.sqrt(2/(1+0.1**2)))
        torch.nn.init.orthogonal_(self.arc_dep.weight.data,gain = np.sqrt(2/(1+0.1**2)))
        self.arc_head.bias.data.fill_(0.0)
        self.arc_head.bias.data.fill_(0.0)

        self.label_head = nn.Linear(2*self.hidden_size,self.mlp_label_size,bias=True)
        self.label_dep = nn.Linear(2*self.hidden_size,self.mlp_label_size,bias=True)
        torch.nn.init.orthogonal_(self.label_head.weight.data,gain = np.sqrt(2/(1+0.1**2)))
        torch.nn.init.orthogonal_(self.label_dep.weight.data,gain = np.sqrt(2/(1+0.1**2)))
        self.label_head.bias.data.fill_(0.0)
        self.label_head.bias.data.fill_(0.0)

        #Add biaffine layers
        #self.arc_biaffine = Biaffine(self.mlp_arc_size, self.mlp_arc_size, 1, bias=(True, False, False))
        #self.label_biaffine = Biaffine(self.mlp_label_size, self.mlp_label_size,self.rels,bias=(True, True, True))

        self.arc_biaf = nn.Parameter(torch.zeros(self.mlp_arc_size, self.mlp_arc_size))
        self.arc_head_aff = nn.Linear(self.mlp_arc_size,1, bias=False)
        self.arc_head_aff.weight.data.fill_(0.0)

        self.label_biaf = nn.Parameter(torch.zeros(self.mlp_label_size, self.mlp_label_size,self.rels))
        self.label_head_aff = nn.Linear(self.mlp_label_size,self.rels, bias=False)
        self.label_head_aff.weight.data.fill_(0.0)
        self.label_dep_aff = nn.Linear(self.mlp_label_size,self.rels, bias=False)
        self.label_dep_aff.weight.data.fill_(0.0)
        self.label_bias =  nn.Parameter(torch.zeros(self.rels))


        self.Relu = nn.LeakyReLU(0.1)

        self.param_group_sparse = []
        self.param_group_dense = []
        for name, param in self.named_parameters():
            if ((name=="word_encoder.weight") or (name=="char_embeds.char_encoder.weight") or (name=="pos_encoder.weight") or (name=="xpos_encoder.weight")):
                print("Sparse:",name)
                self.param_group_sparse.append(param)
            else:
                self.param_group_dense.append(param)

        self.param_group_sparse = iter(self.param_group_sparse)
        self.param_group_dense = iter(self.param_group_dense)


    def forward(self,word_tensor,ext_word_ids,char_ids,pos_tensor,xpos_tensor,seq_lengths):

        x_embed = self.word_encoder(word_tensor)
        char_embed = self.char_embeds(char_ids,seq_lengths)
        ext_embed = self.fasttext_embs(ext_word_ids,seq_lengths)
        x_embed = x_embed + ext_embed + char_embed

        pos_embed = self.pos_encoder(pos_tensor)
        xpos_embed = self.xpos_encoder(xpos_tensor)
        pos_xpos_embed = pos_embed + xpos_embed

        #idependently dropout and scaling for word & pos embs
        if self.training:
            w_mask, p_mask = drop_word_pos_embeds(xpos_embed.size()[0],seq_lengths,x_embed.size()[2],pos_xpos_embed.size()[2],self.dropout,self.cuda_flag)
            x_embed = x_embed * w_mask
            pos_xpos_embed = pos_xpos_embed * p_mask

        input_embed = torch.cat((x_embed,pos_xpos_embed),2)
        batch_size = input_embed.size()[0]

        #ADD root token
        root = self.ROOT.unsqueeze(1).permute(1,0).unsqueeze(1).repeat(batch_size,1,1)
        input_embed = torch.cat((root,input_embed),1)
        src_len = input_embed.size()[1]
        input_embed = self.var_drop(input_embed)
        seq_lengths_rnn =  seq_lengths + 1

        packed_input = pack_padded_sequence(input_embed, seq_lengths_rnn.cpu().numpy(),batch_first=True)
        packed_outputs, ht = self.rnn(packed_input, None)
        packed_output = packed_outputs[-1] #get the last layer's encodings
        out_rnn, lengths = pad_packed_sequence(packed_output, batch_first=True)
        out_rnn = self.var_drop(out_rnn)

        #Compute specialized vector representations for heads and dependents
        arc_head = self.var_drop(self.Relu(self.arc_head(out_rnn)))
        arc_dep = self.var_drop(self.Relu(self.arc_dep(out_rnn)))
        label_head = self.var_drop(self.Relu(self.label_head(out_rnn)))
        label_dep = self.var_drop(self.Relu(self.label_dep(out_rnn)))



        arc_affine = torch.mm(arc_head.view(batch_size*src_len,-1),self.arc_biaf) # bs*src_head,mlp_arc_size
        arc_biaffine = torch.bmm(arc_affine.view(batch_size,src_len,-1),arc_dep.permute(0,2,1)) # bs,src_head,src_dep
        arc_head_affine = self.arc_head_aff(arc_head).repeat(1,1,src_len)
        arc_scores  = arc_biaffine + arc_head_affine
        arc_scores = arc_scores.permute(0,2,1) # bs,src_dep,src_head

        label_biaf = self.label_biaf.view(self.mlp_label_size,self.rels*self.mlp_label_size) # mlp_label_size,rels*mlp_label_size
        label_affine  = torch.mm(label_head.view(batch_size*src_len,-1),label_biaf).view(batch_size,src_len*self.rels,self.mlp_label_size) # bs,src_head*rels,mlp_label_size
        label_biaffine = torch.bmm(label_affine,label_dep.permute(0,2,1)).view(batch_size,src_len,self.rels,src_len).permute(0,3,1,2) # bs,src_dep,src_head,rels

        label_head_affine = self.label_head_aff(label_head).unsqueeze(3).permute(0,3,1,2).repeat(1,src_len,1,1)
        label_dep_affine = self.label_dep_aff(label_dep).unsqueeze(3).permute(0,1,3,2).repeat(1,1,src_len,1)
        label_bias = self.label_bias.unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(3,2,1,0).repeat(batch_size,src_len,src_len,1)
        label_scores = label_biaffine + label_head_affine + label_dep_affine + label_bias

        #arc_scores = self.arc_biaffine(arc_dep, arc_head).squeeze(3) #[bs,src_len,src_len]
        #label_scores = self.label_biaffine(label_dep,label_head)    #[bs,src_len,src_len,labels]


        return arc_scores,label_scores
