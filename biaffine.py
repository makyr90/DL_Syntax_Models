import copy
import dynet as dy
import numpy as np
from Edmonds_decoder import parse_proj
from collections import defaultdict
import Saxe

class DeepBiaffineAttentionDecoder(object):
    """ For MST dependency parsing (ICLR 2017)"""

    def __init__(self,
                 model,
                 n_labels,
                 src_ctx_dim=400,
                 n_arc_mlp_units=400,
                 n_label_mlp_units=100,
                 arc_mlp_dropout=0.33,
                 label_mlp_dropout=0.33):

        #Add lookup matrix for distance embedings in the range [-15,15]. Plus two
        #more embeddings for distances larger than +15 and -15
        # self.dist2id = defaultdict(lambda: len(self.dist2id))
        # for x in range(0,17):
        #     self.dist2id[x]
        # for x in range(-1,-17,-1):
        #     self.dist2id[x]
        # self.distembs = model.add_lookup_parameters((len(self.dist2id), 64),init = dy.NormalInitializer())


        #self.W_arc_dist = model.add_parameters((64), init = dy.ConstInitializer(0))
        Saxe_initializer = Saxe.Orthogonal(gain='leaky_relu',alpha = 0.1)
        self.src_ctx_dim = src_ctx_dim
        self.label_mlp_dropout = label_mlp_dropout
        self.arc_mlp_dropout = arc_mlp_dropout
        self.n_labels = n_labels
        self.W_arc_hidden_to_head = model.add_parameters((n_arc_mlp_units, src_ctx_dim), init=dy.NumpyInitializer(Saxe_initializer(((n_arc_mlp_units, src_ctx_dim)))))
        self.b_arc_hidden_to_head = model.add_parameters((n_arc_mlp_units,),init = dy.ConstInitializer(0))
        self.W_arc_hidden_to_dep = model.add_parameters((n_arc_mlp_units, src_ctx_dim), init=dy.NumpyInitializer(Saxe_initializer(((n_arc_mlp_units, src_ctx_dim)))))
        self.b_arc_hidden_to_dep = model.add_parameters((n_arc_mlp_units,),init = dy.ConstInitializer(0))

        self.W_label_hidden_to_head = model.add_parameters((n_label_mlp_units, src_ctx_dim), init=dy.NumpyInitializer(Saxe_initializer(((n_label_mlp_units, src_ctx_dim)))))
        self.b_label_hidden_to_head = model.add_parameters((n_label_mlp_units,),init = dy.ConstInitializer(0))
        self.W_label_hidden_to_dep = model.add_parameters((n_label_mlp_units, src_ctx_dim), init=dy.NumpyInitializer(Saxe_initializer(((n_label_mlp_units, src_ctx_dim)))))
        self.b_label_hidden_to_dep = model.add_parameters((n_label_mlp_units,),init = dy.ConstInitializer(0))

        self.U_arc_1 = model.add_parameters((n_arc_mlp_units, n_arc_mlp_units), init = dy.ConstInitializer(0))
        self.u_arc_2 = model.add_parameters((n_arc_mlp_units),init = dy.ConstInitializer(0))

        self.U_label_1 = [model.add_parameters((n_label_mlp_units, n_label_mlp_units), init = dy.ConstInitializer(0)) for _ in range(n_labels)]
        self.u_label_2_2 = [model.add_parameters((1, n_label_mlp_units), init = dy.ConstInitializer(0)) for _ in range(n_labels)]
        self.u_label_2_1 = [model.add_parameters((n_label_mlp_units, 1), init = dy.ConstInitializer(0)) for _ in range(n_labels)]
        #self.W_label_dist = [model.add_parameters((64),init = dy.ConstInitializer(0)) for _ in range(n_labels)]
        self.b_label = [model.add_parameters((1,),init = dy.ConstInitializer(0)) for _ in range(n_labels)]


    def leaky_ReLu(self,inputvec,alpha=0.1):

        return dy.rectify(inputvec) -alpha*dy.rectify(-inputvec)

    def cal_scores(self, src_encodings,predict=False):

        src_len = len(src_encodings)
        src_encodings = dy.concatenate_cols(src_encodings)  # src_ctx_dim, src_len, batch_size
        batch_size = src_encodings.dim()[1]

        W_arc_hidden_to_head = dy.parameter(self.W_arc_hidden_to_head)
        b_arc_hidden_to_head = dy.parameter(self.b_arc_hidden_to_head)
        W_arc_hidden_to_dep = dy.parameter(self.W_arc_hidden_to_dep)
        b_arc_hidden_to_dep = dy.parameter(self.b_arc_hidden_to_dep)

        W_label_hidden_to_head = dy.parameter(self.W_label_hidden_to_head)
        b_label_hidden_to_head = dy.parameter(self.b_label_hidden_to_head)
        W_label_hidden_to_dep = dy.parameter(self.W_label_hidden_to_dep)
        b_label_hidden_to_dep = dy.parameter(self.b_label_hidden_to_dep)

        U_arc_1 = dy.parameter(self.U_arc_1)
        u_arc_2 = dy.parameter(self.u_arc_2)
        #W_arc_dist = dy.parameter(self.W_arc_dist)

        U_label_1 = [dy.parameter(x) for x in self.U_label_1]
        u_label_2_1 = [dy.parameter(x) for x in self.u_label_2_1]
        u_label_2_2 = [dy.parameter(x) for x in self.u_label_2_2]
        #W_label_dist = [dy.parameter(x) for x in self.W_label_dist]
        b_label = [dy.parameter(x) for x in self.b_label]

        if predict:
            h_arc_head = self.leaky_ReLu(dy.affine_transform([b_arc_hidden_to_head, W_arc_hidden_to_head, src_encodings]))  # n_arc_ml_units, src_len, bs
            h_arc_dep  = self.leaky_ReLu(dy.affine_transform([b_arc_hidden_to_dep, W_arc_hidden_to_dep, src_encodings]))
            h_label_head = self.leaky_ReLu(dy.affine_transform([b_label_hidden_to_head, W_label_hidden_to_head, src_encodings]))
            h_label_dep = self.leaky_ReLu(dy.affine_transform([b_label_hidden_to_dep, W_label_hidden_to_dep, src_encodings]))
        else:

            src_encodings = dy.dropout_dim(src_encodings,1,self.arc_mlp_dropout)

            h_arc_head = self.leaky_ReLu(dy.affine_transform([b_arc_hidden_to_head, W_arc_hidden_to_head, src_encodings]))  # n_arc_ml_units, src_len, bs
            h_arc_head  = dy.dropout_dim(h_arc_head,1,self.arc_mlp_dropout)
            h_arc_dep = self.leaky_ReLu(dy.affine_transform([b_arc_hidden_to_dep, W_arc_hidden_to_dep, src_encodings]))
            h_arc_dep = dy.dropout_dim(h_arc_dep,1,self.arc_mlp_dropout)

            h_label_head = self.leaky_ReLu(dy.affine_transform([b_label_hidden_to_head, W_label_hidden_to_head, src_encodings]))
            h_label_head  = dy.dropout_dim(h_label_head,1,self.label_mlp_dropout)
            h_label_dep = self.leaky_ReLu(dy.affine_transform([b_label_hidden_to_dep, W_label_hidden_to_dep, src_encodings]))
            h_label_dep  = dy.dropout_dim(h_label_dep,1,self.label_mlp_dropout)


        #Add W^T * dist_emb[head_i,dep_j]
        # arc_Dist = []
        # for row in range(src_len):
        #     for col in range(src_len):
        #         if ((row - col) < 16 and (row - col) > -16):
        #             dist_emb = dy.lookup_batch(self.distembs,[self.dist2id[int(row-col)]] * batch_size)
        #         elif ((row - col)) >= 16:
        #             dist_emb = dy.lookup_batch(self.distembs,[self.dist2id[16]] * batch_size)
        #         else:
        #             dist_emb = dy.lookup_batch(self.distembs,[self.dist2id[-16]] * batch_size)
        #
        #         arc_Dist.append(dist_emb)
        #
        #
        # arc_Dist = dy.transpose(dy.concatenate_cols(arc_Dist))

        h_arc_head_transpose = dy.transpose(h_arc_head)
        h_label_head_transpose = dy.transpose(h_label_head)

        s_arc = h_arc_head_transpose * dy.colwise_add(U_arc_1 * h_arc_dep, u_arc_2) #+ dy.reshape((arc_Dist * W_arc_dist),(src_len,src_len),batch_size = batch_size)

        s_label = []
        for U_1, u_2_1, u_2_2, b in zip(U_label_1, u_label_2_1, u_label_2_2, b_label):
            e1 = h_label_head_transpose * U_1 * h_label_dep
            e2 = h_label_head_transpose * u_2_1 * dy.ones((1, src_len))
            e3 = dy.ones((src_len, 1)) * u_2_2 * h_label_dep
            #e4 = dy.reshape((arc_Dist * w_l),(src_len,src_len),batch_size = batch_size)
            s_label.append(e1 + e2 + e3 +  b)
        return s_arc, s_label

    def decode_loss(self, src_encodings, masks, tgt_seqs):
        """
        :param tgt_seqs: (tgt_heads, tgt_labels): list (length=batch_size) of (src_len)
        """

        tgt_heads, tgt_labels = tgt_seqs

        src_len = len(tgt_heads[0])
        batch_size = len(tgt_heads)
        np_tgt_heads = np.array(tgt_heads).flatten()  # (src_len * batch_size)
        np_tgt_labels = np.array(tgt_labels).flatten()
        np_masks = np.array(masks).transpose().flatten()
        masks_expr = dy.inputVector(np_masks)
        masks_expr = dy.reshape(masks_expr, (1,), batch_size = src_len * batch_size)
        s_arc, s_label = self.cal_scores(src_encodings)  # (src_len, src_len, bs), ([(src_len, src_len, bs)])
        s_arc_value = s_arc.npvalue()
        s_arc_choice = np.argmax(s_arc_value, axis=0).transpose().flatten()  # (src_len * batch_size)
        s_pick_labels = [dy.pick_batch(dy.reshape(score, (src_len,), batch_size=src_len * batch_size), s_arc_choice)
                     for score in s_label]
        s_argmax_labels = dy.concatenate(s_pick_labels, d=0)  # n_labels, src_len * batch_size

        reshape_s_arc = dy.reshape(s_arc, (src_len,), batch_size=src_len * batch_size)
        arc_loss = dy.pickneglogsoftmax_batch(reshape_s_arc, np_tgt_heads)
        arc_loss = arc_loss * masks_expr
        label_loss = dy.pickneglogsoftmax_batch(s_argmax_labels, np_tgt_labels)
        label_loss = label_loss * masks_expr
        loss = dy.sum_batches(arc_loss + label_loss)
        return loss

    def decoding(self, src_encodings,sentences_length):

        pred_heads = []
        pred_labels = []
        s_arc, s_label = self.cal_scores(src_encodings,True)
        for idx in range(len(sentences_length)):


            s_arc_values = dy.pick_batch_elem(s_arc,idx).npvalue()[:sentences_length[idx],:sentences_length[idx]]  # src_len, src_len
            s_label_values = np.asarray([dy.pick_batch_elem(label,idx).npvalue() for label in s_label]).transpose((2, 1, 0))[:sentences_length[idx],:sentences_length[idx],:] # src_len, src_len, n_labels
            weights = s_arc_values
            spred_heads = parse_proj(weights)
            spred_labels = [np.argmax(labels[head]) for head, labels in zip(spred_heads, s_label_values)]
            pred_heads.append(spred_heads)
            pred_labels.append(spred_labels)

        return pred_heads, pred_labels
