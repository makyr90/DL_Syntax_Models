import dynet as dy
import numpy as np
import Saxe



class affineAttentionDecoder(object):


    def __init__(self,
                 model,
                 pos_labels, xpos_labels,
                 src_ctx_dim=400,
                 n_pos_tagger_mlp_units=200,
                 n_xpos_tagger_mlp_units=200,
                 mlps_dropout=0.33):


        self.src_ctx_dim = src_ctx_dim
        self.dropout = mlps_dropout
        self.pos_labels = pos_labels
        self.xpos_labels = xpos_labels

        Saxe_initializer = Saxe.Orthogonal(gain='leaky_relu',alpha = 0.1)

        self.W_pos = model.add_parameters((n_pos_tagger_mlp_units, src_ctx_dim), init=dy.NumpyInitializer(Saxe_initializer(((n_pos_tagger_mlp_units, src_ctx_dim)))))
        self.b_pos = model.add_parameters((n_pos_tagger_mlp_units,),init = dy.ConstInitializer(0))
        self.W_xpos = model.add_parameters((n_xpos_tagger_mlp_units, src_ctx_dim), init=dy.NumpyInitializer(Saxe_initializer(((n_xpos_tagger_mlp_units, src_ctx_dim)))))
        self.b_xpos = model.add_parameters((n_xpos_tagger_mlp_units,),init = dy.ConstInitializer(0))

        self.W_affine_pos = model.add_parameters((n_pos_tagger_mlp_units,pos_labels), init = dy.ConstInitializer(0))
        self.b_affine_pos = model.add_parameters((pos_labels),init = dy.ConstInitializer(0))
        self.W_affine_xpos = model.add_parameters((n_xpos_tagger_mlp_units,xpos_labels), init = dy.ConstInitializer(0))
        self.b_affine_xpos = model.add_parameters((xpos_labels),init = dy.ConstInitializer(0))



    def leaky_ReLu(self,inputvec,alpha=0.1):

        return dy.bmax(alpha*inputvec, inputvec)

    def cal_scores(self, src_encodings,predict=False):

        src_len = len(src_encodings)
        src_encodings = dy.concatenate_cols(src_encodings)  # src_ctx_dim, src_len, batch_size
        batch_size = src_encodings.dim()[1]

        W_pos = dy.parameter(self.W_pos)
        b_pos = dy.parameter(self.b_pos)
        W_xpos = dy.parameter(self.W_xpos)
        b_xpos = dy.parameter(self.b_xpos)


        W_affine_pos = dy.parameter(self.W_affine_pos)
        b_affine_pos = dy.parameter(self.b_affine_pos)
        W_affine_xpos = dy.parameter(self.W_affine_xpos)
        b_affine_xpos = dy.parameter(self.b_affine_xpos)

        if predict:
            pos = self.leaky_ReLu(dy.affine_transform([b_pos, W_pos, src_encodings]))  # n_pos_mlp_units, src_len, bs
            xpos = self.leaky_ReLu(dy.affine_transform([b_xpos, W_xpos, src_encodings]))

        else:
            src_encodings = dy.dropout_dim(src_encodings,1,self.dropout)
            pos = dy.dropout_dim(self.leaky_ReLu(dy.affine_transform([b_pos, W_pos, src_encodings])),1,self.dropout)  # n_pos_mlp_units, src_len, bs
            xpos = dy.dropout_dim(self.leaky_ReLu(dy.affine_transform([b_xpos, W_xpos, src_encodings])),1,self.dropout)


        pos_label = dy.affine_transform([b_affine_pos, dy.transpose(W_affine_pos), pos])
        xpos_label = dy.affine_transform([b_affine_xpos, dy.transpose(W_affine_xpos), xpos])

        return pos_label, xpos_label

    def decode_loss(self, src_encodings, masks, src_len, batch_size, pos_ids,xpos_ids):

        np_masks = np.array(masks).transpose().flatten()
        masks_expr = dy.inputVector(np_masks)
        masks_expr = dy.reshape(masks_expr, (1,), batch_size = src_len * batch_size)
        np_pos = np.array(pos_ids).flatten()
        np_xpos = np.array(xpos_ids).flatten()

        pos_labels, xpos_labels = self.cal_scores(src_encodings)
        pos_labels = dy.reshape(pos_labels, (self.pos_labels,), batch_size=src_len * batch_size)
        xpos_labels = dy.reshape(xpos_labels, (self.xpos_labels,), batch_size=src_len * batch_size)  #n_xpos_labels, batch_size * src_len

        pos_loss = dy.pickneglogsoftmax_batch(pos_labels, np_pos)
        pos_loss = pos_loss * masks_expr
        xpos_loss = dy.pickneglogsoftmax_batch(xpos_labels, np_xpos)
        xpos_loss = xpos_loss * masks_expr

        loss = dy.sum_batches(pos_loss + xpos_loss)

        return loss

    def decoding(self, src_encodings,sentences_length):

        pred_pos = []
        pred_xpos = []
        pos_label, xpos_label = self.cal_scores(src_encodings,True)
        for idx in range(len(sentences_length)):

            s_pos_values = dy.pick_batch_elem(pos_label,idx).npvalue()[:,:sentences_length[idx]] #  n_pos_labels*sent_length
            spred_pos = list(np.argmax(s_pos_values,axis =0).astype(int))
            pred_pos.append(spred_pos)
            s_xpos_values = dy.pick_batch_elem(xpos_label,idx).npvalue()[:,:sentences_length[idx]] #  n_xpos_labels*sent_length
            spred_xpos = list(np.argmax(s_xpos_values,axis =0).astype(int))
            pred_xpos.append(spred_xpos)



        return pred_pos, pred_xpos
