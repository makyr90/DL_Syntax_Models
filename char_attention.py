import dynet as dy
from LSTMCell import LSTM

class HybridCharacterAttention(object):


    def __init__(self,model,ldims=400,input_size=100,output_size=100,dropout=0.33):


        self.input = input_size
        self.ldims = ldims
        self.output = output_size
        self.dropout = dropout
        self.charlstm  = LSTM(model, self.input, self.ldims, forget_bias = 0.0)
        self.W_atten = model.add_parameters((self.ldims,1),init = dy.ConstInitializer(0))
        self.W_linear = model.add_parameters((self.output,self.ldims*2),init = dy.ConstInitializer(0))
        self.b_linear = model.add_parameters((self.output),init = dy.ConstInitializer(0))

    def predict_sequence_batched(self,inputs,mask_array,wlen,predictFlag=False):

        batch_size = inputs[0].dim()[1]
        src_len = len(inputs)


        if not predictFlag:
            self.charlstm.set_dropouts(self.dropout,self.dropout)
            self.charlstm.set_dropout_masks(batch_size)

        char_fwd = self.charlstm.initial_state(batch_size)
        recur_states, cells = char_fwd.add_inputs(inputs,mask_array,predictFlag)

        hidden_states = []
        for idx in range(src_len):
            mask = dy.inputVector(mask_array[idx])
            mask_expr = dy.reshape(mask,(1,),batch_size)
            hidden_states.append(recur_states[idx] * mask_expr)

        H = dy.concatenate_cols(hidden_states)

        if (predictFlag):
            a = dy.softmax(dy.transpose(self.W_atten.expr()) * H)
        else:
            #dropout attention connections(keep the same dim across the sequence)
            a = dy.softmax(dy.transpose(self.W_atten.expr()) * dy.dropout_dim(H,1,self.dropout))

        cell_states = []
        for idx in range(batch_size):
            if (wlen[idx] > 0):
                cell = dy.pick_batch_elem(cells[wlen[idx]-1],idx)
            else:
                cell = dy.zeros(self.ldims)

            cell_states.append(cell)

        C = dy.concatenate_to_batch(cell_states)

        H_atten = H*dy.transpose(a)
        char_emb = dy.concatenate([H_atten,C])

        if predictFlag:
            proj_char_emb = dy.affine_transform([self.b_linear.expr(),self.W_linear.expr(),char_emb])
        else:
            proj_char_emb = dy.affine_transform([self.b_linear.expr(),self.W_linear.expr(),dy.dropout(char_emb,self.dropout)])

        return proj_char_emb
