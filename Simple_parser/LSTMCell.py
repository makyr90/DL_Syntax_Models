import dynet as dy
import Saxe
import numpy as np


class LSTM():

    def __init__(self,model,input_size,recur_size,forget_bias=0.0):

        self.input_size = input_size
        self.recur_size = recur_size
        self.input_drop_mask = dy.ones(self.input_size)
        self.recur_drop_mask = dy.ones(self.recur_size)
        self.forget_bias = forget_bias
        self.cell_previous = None
        self.hidden_previous = None
        self.init = False
        self.input_drop = 0
        self.recur_drop = 0

        Saxe_initializer = Saxe.Orthogonal()
        gates_init = Saxe_initializer(((self.recur_size,self.input_size+self.recur_size)))
        gates_init  = np.concatenate([gates_init]*4)
        self.WXH = model.add_parameters((self.recur_size*4,self.input_size+self.recur_size), init=dy.NumpyInitializer(gates_init))
        self.b = model.add_parameters((self.recur_size*4), init = dy.ConstInitializer(0))


    def initial_state(self,batch_size):

        self.cell_previous = dy.zeros((self.recur_size),batch_size)
        self.hidden_previous = dy.zeros((self.recur_size),batch_size)
        self.init = True
        return self

    def set_dropouts(self,input_drop=0,recur_drop=0):

        self.input_drop = input_drop
        self.recur_drop = recur_drop
        self.input_drop_mask = dy.dropout(dy.ones(self.input_size),self.input_drop)
        self.recur_drop_mask = dy.dropout(dy.ones(self.recur_size),self.recur_drop)

    def set_dropout_masks(self,batch_size):

        self.input_drop_mask = dy.dropout(dy.ones((self.input_size),batch_size),self.input_drop)
        self.recur_drop_mask = dy.dropout(dy.ones((self.recur_size),batch_size),self.recur_drop)


    def add_inputs(self,inputs,masks,predict=False):

        if not self.init:
            print("No Initial state provided")
            return

        recur_states = []
        cell_states = []

        for input_tensor in inputs:

            hidden = self.hidden_previous
            cell = self.cell_previous
            if not predict:
                input_tensor = dy.cmult(input_tensor,self.input_drop_mask)
                hidden = dy.cmult(hidden,self.recur_drop_mask)
            gates = dy.affine_transform([self.b.expr(),self.WXH.expr(),dy.concatenate([input_tensor,hidden])])
            iga = dy.pickrange(gates,0,self.recur_size)
            fga = dy.pickrange(gates,self.recur_size,2*self.recur_size)
            oga = dy.pickrange(gates,2*self.recur_size,3*self.recur_size)
            cga = dy.pickrange(gates,3*self.recur_size,4*self.recur_size)

            ig = dy.logistic(iga)
            fg = dy.logistic(fga) # +self.forget_bias
            og = dy.logistic(oga)
            c_tilda = dy.tanh(cga)
            new_cell = dy.cmult(cell,fg) + dy.cmult(c_tilda,ig)
            self.cell_previous = new_cell
            cell_states.append(new_cell)
            new_hidden = dy.cmult(dy.tanh(new_cell),og)
            self.hidden_previous = new_hidden
            recur_states.append(new_hidden)

        return recur_states,cell_states

    def transduce(self,inputs,masks,predict=False):

        if not self.init:
            print("No Initial state provided")
            return

        outputs = []
        batch_size = inputs[0].dim()[1]

        for idx,input_tensor in enumerate(inputs):
            recur_s = []
            cell_s = []
            out = []

            hidden = self.hidden_previous
            cell = self.cell_previous
            if not predict:
                input_tensor = dy.cmult(input_tensor,self.input_drop_mask)
                hidden = dy.cmult(hidden,self.recur_drop_mask)

            gates = dy.affine_transform([self.b.expr(),self.WXH.expr(),dy.concatenate([input_tensor,hidden])])
            iga = dy.pickrange(gates,0,self.recur_size)
            fga = dy.pickrange(gates,self.recur_size,2*self.recur_size)
            oga = dy.pickrange(gates,2*self.recur_size,3*self.recur_size)
            cga = dy.pickrange(gates,3*self.recur_size,4*self.recur_size)

            ig = dy.logistic(iga)
            fg = dy.logistic(fga)  # +self.forget_bias
            og = dy.logistic(oga)
            c_tilda = dy.tanh(cga)
            new_cell = dy.cmult(cell,fg) + dy.cmult(c_tilda,ig)
            new_hidden = dy.cmult(dy.tanh(new_cell),og)

            for jdx in range(batch_size):
                if masks[idx][jdx] == 1:
                    h_t = dy.pick_batch_elem(new_hidden,jdx)
                    recur_s.append(h_t)
                    cell_s.append(dy.pick_batch_elem(new_cell,jdx))
                    out.append(h_t)
                else:
                    recur_s.append(dy.pick_batch_elem(hidden,jdx))
                    cell_s.append(dy.pick_batch_elem(cell,jdx)) 
                    out.append(dy.zeros(self.recur_size))

            new_cell = dy.concatenate_to_batch(cell_s)
            new_hidden = dy.concatenate_to_batch(recur_s)
            self.cell_previous = new_cell
            self.hidden_previous = new_hidden
            outputs.append(dy.concatenate_to_batch(out))

        return outputs
