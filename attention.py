import copy
import dynet as dy
import numpy as np
from Edmonds_decoder import parse_proj
from collections import defaultdict
import Saxe

class AttentionDecoder(object):


    def __init__(self,
                 model,
                 n_labels,
                 src_ctx_dim=400,
                 hidden = 400,
                 dropout =0.33):

        self.src_ctx_dim = src_ctx_dim
        self.dropout = dropout
        self.n_labels = n_labels
        self.hidden = hidden
        self.dist_max = 10
        self.dist_dims = 32
        self.dlookup = model.add_lookup_parameters((self.dist_max * 2, self.dist_dims),init = dy.ConstInitializer(0))
        Saxe_initializer = Saxe.Orthogonal(gain='leaky_relu',alpha = 0.1)
        self.W_head = model.add_parameters((self.src_ctx_dim,self.src_ctx_dim), init=dy.NumpyInitializer(Saxe_initializer(((self.src_ctx_dim,self.src_ctx_dim)))))
        self.b_head = model.add_parameters((self.src_ctx_dim),init = dy.ConstInitializer(0))
        self.W_mod = model.add_parameters((self.src_ctx_dim,self.src_ctx_dim), init=dy.NumpyInitializer(Saxe_initializer(((self.src_ctx_dim,self.src_ctx_dim)))))
        self.b_mod = model.add_parameters((self.src_ctx_dim),init = dy.ConstInitializer(0))
        self.W_arc1 = model.add_parameters((self.hidden, self.src_ctx_dim + self.dist_dims), init=dy.NumpyInitializer(Saxe_initializer(((self.hidden,  self.src_ctx_dim + self.dist_dims)))))
        self.b_arc1 = model.add_parameters((self.hidden),init = dy.ConstInitializer(0))
        self.W_arc2 = model.add_parameters((self.n_labels, self.hidden), init=dy.NumpyInitializer(Saxe_initializer(((self.n_labels, self.hidden)))))
        self.b_arc2 = model.add_parameters((self.n_labels),init = dy.ConstInitializer(0))


    def leaky_ReLu(self,inputvec,alpha=0.1):

        return dy.bmax(alpha*inputvec,inputvec)

    def scoreHeadModLabel(self, input_vec, train):

        if train:
            output = self.leaky_ReLu(self.b_arc2.expr() + self.W_arc2.expr() * dy.dropout(self.leaky_ReLu(self.b_arc1.expr() + self.W_arc1.expr() * dy.dropout(input_vec,self.dropout)), self.dropout))
        else:
            output = self.leaky_ReLu(self.b_arc2.expr() + self.W_arc2.expr() * self.leaky_ReLu(self.b_arc1.expr() + self.W_arc1.expr() * input_vec))

        return output

    def cal_scores(self, src_encodings,masks,train):

        src_len = len(src_encodings)
        batch_size = src_encodings[0].dim()[1]
        heads_LRlayer = []
        mods_LRlayer = []
        for encoding in src_encodings:
            heads_LRlayer.append(self.leaky_ReLu(self.b_head.expr() + self.W_head.expr() * encoding))
            mods_LRlayer.append(self.leaky_ReLu(self.b_mod.expr() + self.W_mod.expr() * encoding))

        heads_labels =[]
        heads = []
        labels = []
        neg_inf = dy.constant(1,-float("inf"))
        for row in range(1,src_len): #exclude root @ index=0 since roots do not have heads

            scores = []
            scores_idx = []
            for col in range(src_len):

                dist = col-row
                mdist = self.dist_max
                dist_i = (min(dist, mdist-1) + mdist if dist >= 0 else int(min(-1.0 * dist, mdist-1)))
                dist_vec = dy.lookup_batch(self.dlookup,[dist_i] * batch_size)
                if train:
                    input_vec = dy.concatenate([dy.esum([dy.dropout(heads_LRlayer[col],self.dropout),dy.dropout(mods_LRlayer[row],self.dropout)]),dist_vec])
                else:
                    input_vec = dy.concatenate([dy.esum([heads_LRlayer[col],mods_LRlayer[row]]),dist_vec])
                score = self.scoreHeadModLabel(input_vec,train)
                mask = masks[row] and masks[col]
                mask_expr = dy.inputVector(mask)
                mask_expr = dy.reshape(mask_expr,(1,),batch_size)
                scores.append(score * mask_expr)
                join_scores = []
                for bdx in range(batch_size):
                    if (mask[bdx] == 1):
                        join_scores.append(dy.pick_batch_elem(score,bdx))
                    else:
                        join_scores.append(dy.concatenate([neg_inf]*self.n_labels))
                scores_idx.append(dy.concatenate_to_batch(join_scores))
            heads_labels.append(dy.concatenate(scores_idx))
            heads.append(dy.concatenate([dy.sum_elems(s) for s in scores_idx]))
            labels.append(dy.esum(scores))

        return heads_labels,heads,labels

    def decode_loss(self, src_encodings, masks, tgt_seqs):


        batch_heads,batch_labels,batch_heads_labels_idx = tgt_seqs
        src_len = len(batch_heads[0])
        batch_size = len(batch_heads)
        heads_labels_idx,heads,labels  = self.cal_scores(src_encodings,masks,True)
        #mask, gold head,labels,idx  +1
        loss =[]
        for idx in range(len(heads)):

            heads_labels_loss = dy.pickneglogsoftmax_batch(heads_labels_idx[idx],[h[idx] for h in batch_heads_labels_idx])
            heads_loss = dy.pickneglogsoftmax_batch(heads[idx],[h[idx] for h in batch_heads])
            labels_loss = dy.pickneglogsoftmax_batch(labels[idx],[h[idx] for h in batch_labels])
            mask = masks[idx+1]
            mask_expr = dy.inputVector(mask)
            mask_expr = dy.reshape(mask_expr,(1,),batch_size)
            loss.append((heads_labels_loss + heads_loss + labels_loss)  * mask_expr)


        loss = dy.sum_batches(dy.esum(loss))

        return loss

    def decoding(self, src_encodings,masks,sentences_length):

        src_len = len(src_encodings)
        batch_size = src_encodings[0].dim()[1]

        pred_heads = [[] for _ in range(batch_size)]
        pred_labels = [[] for _ in range(batch_size)]

        heads_labels_idx,heads,labels  = self.cal_scores(src_encodings,masks,False)

        scores = dy.concatenate_cols(heads_labels_idx)

        for idx in range(batch_size):


             scores_np = dy.pick_batch_elem(scores,idx).npvalue()[:(sentences_length[idx]+1)*self.n_labels,:sentences_length[idx]]
             pred_idx = np.argmax(scores_np,axis=0)
             pred_label = pred_idx % self.n_labels
             pred_head = (pred_idx - pred_label) /  self.n_labels
             pred_head = pred_head.astype(int)
             pred_label = pred_label.astype(int)

             pred_heads[idx].extend(pred_head.tolist())
             pred_labels[idx].extend(pred_label.tolist())


        return pred_heads, pred_labels
