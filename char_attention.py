import dynet as dy
import numpy as np
import Saxe

class HybridCharacterAttention(object):


    def __init__(self,model,layers=1,ldims=400,input_size=100,output_size=100,dropout=0):


        self.layers = layers
        self.input = input_size
        self.ldims = ldims
        self.output = output_size
        self.dropout = dropout
        self.charlstm  = dy.VanillaLSTMBuilder(self.layers, self.input, self.ldims, model, forget_bias = 0.0)

        Saxe_initializer = Saxe.Orthogonal()
        self.W_atten = model.add_parameters((self.ldims,1), init=dy.NumpyInitializer(Saxe_initializer(((self.ldims,1)))))
        self.W_linear = model.add_parameters((self.output,2*self.ldims),init = dy.ConstInitializer(0))
        self.b_linear = model.add_parameters((self.output),init = dy.ConstInitializer(0))

    def predict_sequence_batched(self,inputs,mask_array,wlen,src_len,batch_size,predictFlag = False):


        if (predictFlag):
            self.charlstm.disable_dropout()
        else:
            self.charlstm.set_dropouts(self.dropout,self.dropout)

        char_fwd = self.charlstm.initial_state()
        states = char_fwd.add_inputs(inputs)
        hidden_states = []
        for idx in range(src_len):
            mask = dy.inputVector(mask_array[idx])
            mask_expr = dy.reshape(mask,(1,),batch_size)
            hidden_states.append(states[idx].output() * mask_expr)

        H = dy.concatenate_cols(hidden_states)


        softmax_scores = []
        for idx in range(batch_size):
            if (predictFlag):
                a = dy.softmax(dy.transpose(self.W_atten.expr()) * (dy.select_cols(dy.pick_batch_elem(H,idx),range(wlen[idx]))))
            else:
                a = dy.dropout(dy.softmax(dy.transpose(self.W_atten.expr()) * (dy.select_cols(dy.pick_batch_elem(H,idx),range(wlen[idx])))),self.dropout)

            if (((src_len - wlen[idx]) > 0) and (wlen[idx] > 0)):
                softmax_scores.append(dy.concatenate([a,dy.zeros((1,(src_len - wlen[idx])))],d =1))
            elif (src_len == wlen[idx]):
                softmax_scores.append(a)
            else:
                softmax_scores.append(dy.zeros((1,src_len)))


        cell_states = []
        for idx in range(batch_size):
            if (wlen[idx] > 0):
                cell = dy.pick_batch_elem(states[wlen[idx]-1].s()[0],idx)
            else:
                cell = dy.zeros(self.ldims)
            cell_states.append(cell)

        C = dy.concatenate_to_batch(cell_states)
        a = dy.concatenate_to_batch(softmax_scores)


        H_atten = H*dy.transpose(a)
        char_emb = dy.concatenate([H_atten,C])
        proj_char_emb = dy.affine_transform([self.b_linear.expr(),self.W_linear.expr(),char_emb])
        proj_embs = []
        for idx in range(batch_size):
            if (wlen[idx] > 0):
                proj_embs.append(dy.pick_batch_elem(proj_char_emb,idx))
            else:
                proj_embs.append(dy.zeros(self.output))

        return dy.concatenate_to_batch(proj_embs)

    def predict_sequence(self,inputs):

        char_fwd = self.charlstm.initial_state()
        states = char_fwd.add_inputs(inputs)
        hidden_states = [s.output() for s in states]
        cell_state = states[-1].s()[0]
        H = dy.concatenate_cols(hidden_states)
        a = dy.softmax(dy.transpose(self.W_atten.expr()) * H)
        H_atten = H*dy.transpose(a)
        char_emb = dy.concatenate([H_atten,cell_state])
        proj_char_emb = dy.affine_transform([self.b_linear.expr(),self.W_linear.expr(),char_emb])

        return proj_char_emb
