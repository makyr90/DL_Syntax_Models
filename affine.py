import dynet as dy
import numpy as np
import Saxe



class affineAttentionDecoder(object):


    def __init__(self,
                 model,
                 pos_labels, xpos_labels,
                 src_ctx_dim=400,
                 n_pos_tagger_mlp_units=500,
                 n_xpos_tagger_mlp_units=500,
                 mlps_dropout=0.5):



        self.src_ctx_dim = src_ctx_dim
        self.dropout = mlps_dropout
        self.pos_labels = pos_labels
        self.xpos_labels = xpos_labels

        Saxe_initializer = Saxe.Orthogonal(gain='leaky_relu',alpha = 0.1)

        self.W_pos = model.add_parameters((n_pos_tagger_mlp_units, src_ctx_dim), init=dy.NumpyInitializer(Saxe_initializer(((n_pos_tagger_mlp_units, src_ctx_dim)))))
        self.b_pos = model.add_parameters((n_pos_tagger_mlp_units,),init = dy.ConstInitializer(0))
        self.W_xpos = model.add_parameters((n_xpos_tagger_mlp_units, src_ctx_dim), init=dy.NumpyInitializer(Saxe_initializer(((n_xpos_tagger_mlp_units, src_ctx_dim)))))
        self.b_xpos = model.add_parameters((n_xpos_tagger_mlp_units,),init = dy.ConstInitializer(0))

        self.W_affine_pos = [model.add_parameters((n_pos_tagger_mlp_units), init = dy.ConstInitializer(0)) for _ in range(pos_labels)]
        self.b_affine_pos = [model.add_parameters((1,),init = dy.ConstInitializer(0)) for _ in range(pos_labels)]
        self.W_affine_xpos = [model.add_parameters((n_xpos_tagger_mlp_units), init = dy.ConstInitializer(0)) for _ in range(xpos_labels)]
        self.b_affine_xpos = [model.add_parameters((1,),init = dy.ConstInitializer(0)) for _ in range(xpos_labels)]




    def leaky_ReLu(self,inputvec,alpha=0.1):

        return dy.rectify(inputvec) -alpha*dy.rectify(-inputvec)

    def cal_scores(self, src_encodings,predict=False):

        src_len = len(src_encodings)
        src_encodings = dy.concatenate_cols(src_encodings)  # src_ctx_dim, src_len, batch_size
        batch_size = src_encodings.dim()[1]

        W_pos = dy.parameter(self.W_pos)
        b_pos = dy.parameter(self.b_pos)
        W_xpos = dy.parameter(self.W_xpos)
        b_xpos = dy.parameter(self.b_xpos)


        W_affine_pos = [dy.parameter(x) for x in self.W_affine_pos]
        b_affine_pos = [dy.parameter(x) for x in self.b_affine_pos]
        W_affine_xpos = [dy.parameter(x) for x in self.W_affine_xpos]
        b_affine_xpos = [dy.parameter(x) for x in self.b_affine_xpos]


        if predict:
            pos = self.leaky_ReLu(dy.affine_transform([b_pos, W_pos, src_encodings]))  # n_pos_mlp_units, src_len, bs
            xpos = self.leaky_ReLu(dy.affine_transform([b_xpos, W_xpos, src_encodings]))

        else:

            pos = self.leaky_ReLu(dy.affine_transform([b_pos, W_pos, src_encodings]))  # n_pos_mlp_units, src_len, bs
            pos = dy.dropout(pos, self.dropout)
            xpos = self.leaky_ReLu(dy.affine_transform([b_xpos, W_xpos, src_encodings]))
            xpos = dy.dropout(xpos, self.dropout)


        pos_label = []
        for w,b in zip(W_affine_pos,b_affine_pos):
            pos_score = dy.affine_transform([b, dy.transpose(w), pos])
            pos_label.append(pos_score)

        xpos_label = []
        for w,b in zip(W_affine_xpos,b_affine_xpos):
            xpos_score = dy.affine_transform([b, dy.transpose(w), xpos])
            xpos_label.append(xpos_score)

        return pos_label, xpos_label

    def decode_loss(self, src_encodings, masks, src_len, batch_size, pos_ids,xpos_ids):


        np_masks = np.array(masks).transpose().flatten()
        masks_expr = dy.inputVector(np_masks)
        masks_expr = dy.reshape(masks_expr, (1,), batch_size = src_len * batch_size)
        np_pos = np.array(pos_ids).flatten()
        np_xpos = np.array(xpos_ids).flatten()

        pos_label, xpos_label = self.cal_scores(src_encodings)

        for idx in range(len(pos_label)):
            pos_label[idx] = dy.reshape(pos_label[idx], (1,), batch_size=src_len * batch_size)
        pos_labels = dy.concatenate(pos_label) #n_pos_labels, batch_size * src_len

        for idx in range(len(xpos_label)):
            xpos_label[idx] = dy.reshape(xpos_label[idx], (1,), batch_size=src_len * batch_size)
        xpos_labels = dy.concatenate(xpos_label) #n_xpos_labels, batch_size * src_len

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

            s_pos_values = np.asarray([dy.pick_batch_elem(label,idx).npvalue().flatten() for label in pos_label])[:,:sentences_length[idx]] #  n_pos_labels*sent_length
            spred_pos = list(np.argmax(s_pos_values,axis =0).astype(int))
            pred_pos.append(spred_pos)
            s_xpos_values = np.asarray([dy.pick_batch_elem(label,idx).npvalue().flatten() for label in xpos_label])[:,:sentences_length[idx]] #  n_xpos_labels*sent_length
            spred_xpos = list(np.argmax(s_xpos_values,axis =0).astype(int))
            pred_xpos.append(spred_xpos)



        return pred_pos, pred_xpos
