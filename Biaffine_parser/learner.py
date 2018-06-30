import dynet as dy
from utils import read_conll, write_conll, batch_data
from biaffine import DeepBiaffineAttentionDecoder
from char_attention import HybridCharacterAttention
from NN import Lin_Projection
import  time, random, utils, pickle
from LSTMCell import LSTM
import numpy as np


class parser:
    def __init__(self, vocab, pos, xpos, rels, w2i, c2i, ext_words_train, ext_words_devtest, options):

        self.model = dy.ParameterCollection()
        self.pretrained_embs = dy.ParameterCollection()
        self.learning_rate = options.learning_rate
        self.trainer = dy.AdamTrainer(self.model,alpha = self.learning_rate,beta_1=0.9,beta_2=0.9,eps=1e-12)

        self.dropout = float(options.dropout)
        self.ldims = options.lstm_dims
        self.wdims = options.wembedding_dims
        self.cdims = options.cembedding_dims
        self.posdims = options.posembedding_dims
        self.pred_batch_size = options.pred_batch_size
        self.ext_words_train = {word: ind+2 for word, ind in ext_words_train.items()}
        self.ext_words_devtest = {word: ind+2 for word, ind in ext_words_devtest.items()}
        self.wordsCount = vocab
        self.vocab = {word: ind+2 for word, ind in w2i.items()}
        self.pos = {word: ind+2 for ind, word in enumerate(pos)}
        self.id2pos = {ind: word for word,ind in self.pos.items()}
        self.xpos = {word: ind+2 for ind, word in enumerate(xpos)}
        self.id2xpos = {ind: word for word,ind in self.xpos.items()}
        self.c2i = c2i
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.irels = {ind: word for word, ind in self.rels.items()}
        self.vocab['PAD'] = 1
        self.pos['PAD'] = 1
        self.xpos['PAD'] = 1


        self.external_embedding, self.edim, self.edim_out = None, 0, 0
        if options.external_embedding is not None:

            self.external_embedding = np.load(options.external_embedding)
            self.ext_voc=  pickle.load( open( options.external_embedding_voc, "rb" ) )
            self.edim = self.external_embedding.shape[1]
            self.projected_embs = Lin_Projection(self.model, self.edim, self.wdims)
            self.elookup_train = self.pretrained_embs.add_lookup_parameters((len(self.ext_words_train)+2 , self.edim))
            for word, i in self.ext_words_train.items():
                self.elookup_train.init_row(i, self.external_embedding[self.ext_voc[word],:])
            self.elookup_train.init_row(0, np.zeros(self.edim))
            self.elookup_train.init_row(1, np.zeros(self.edim))

            self.elookup_devtest = self.pretrained_embs.add_lookup_parameters((len(self.ext_words_devtest)+2 , self.edim))
            for word, i in self.ext_words_devtest.items():
                self.elookup_devtest.init_row(i, self.external_embedding[self.ext_voc[word],:])
            self.elookup_devtest.init_row(0, np.zeros(self.edim))
            self.elookup_devtest.init_row(1, np.zeros(self.edim))

            self.ext_words_train['PAD'] = 1
            self.ext_words_devtest['PAD'] = 1

            print('Load external embeddings. External embeddings vectors dimension', self.edim)


        #LSTMs
        self.fwdLSTM1 = LSTM(self.model, self.wdims+self.posdims, self.ldims, forget_bias = 0.0)
        self.bwdLSTM1 = LSTM(self.model, self.wdims+self.posdims, self.ldims, forget_bias = 0.0)
        self.fwdLSTM2 = LSTM(self.model, self.ldims, self.ldims, forget_bias = 0.0)
        self.bwdLSTM2 = LSTM(self.model, self.ldims, self.ldims, forget_bias = 0.0)
        self.fwdLSTM3 = LSTM(self.model, self.ldims, self.ldims, forget_bias = 0.0)
        self.bwdLSTM3 = LSTM(self.model, self.ldims, self.ldims, forget_bias = 0.0)



        self.biaffineParser =  DeepBiaffineAttentionDecoder(self.model, len(self.rels), src_ctx_dim=self.ldims * 2,
                    n_arc_mlp_units=400, n_label_mlp_units=100, arc_mlp_dropout=self.dropout, label_mlp_dropout=self.dropout)

        self.HybridCharembs = HybridCharacterAttention(self.model,ldims=400,input_size=self.cdims,output_size=self.wdims,dropout=self.dropout)

        self.wlookup = self.model.add_lookup_parameters((len(vocab) + 2, self.wdims),init = dy.ConstInitializer(0))
        #0 for unknown 1 for [PAD]
        self.poslookup = self.model.add_lookup_parameters((len(self.pos) + 2, self.posdims),init = dy.ConstInitializer(0))
        #0 for unknown 1 for  [PAD]
        self.xposlookup = self.model.add_lookup_parameters((len(self.xpos) + 2, self.posdims),init = dy.ConstInitializer(0))
        #0 for unknown 1 for  [PAD]

        self.clookup = self.model.add_lookup_parameters((len(c2i), self.cdims), init = dy.NormalInitializer())
        self.ROOT= self.model.add_parameters((self.wdims*2),init = dy.ConstInitializer(0))


    def Save(self, filename):
        self.model.save(filename)


    def Load(self, filename):
        self.model.populate(filename)

    def leaky_ReLu(self, inputvec, alpha=0.1):
        return dy.bmax(alpha*inputvec, inputvec)

    def RNN_embeds(self,sentences,predictFlag=False):

        tokenIdChars = []
        for sent in sentences:
            tokenIdChars.extend([entry.idChars for entry in sent])
        tokenIdChars_set = set(map(tuple,tokenIdChars))
        tokenIdChars = list(map(list,tokenIdChars_set))
        tokenIdChars.sort(key=lambda x: -len(x))

        char_src_len =  len(max(tokenIdChars,key=len))
        chars_mask = []
        char_ids = []
        for i in range(char_src_len):
            char_ids.append([(chars[i] if len(chars) > i else 4) for chars in tokenIdChars])
            char_mask = [(1 if len(chars) > i else 0) for chars in tokenIdChars]
            chars_mask.append(char_mask)
        char_embs = []
        for cid in char_ids:
            char_embs.append(dy.lookup_batch(self.clookup, cid))
        wordslen = list(map(lambda x:len(x),tokenIdChars))

        chr_embs = self.HybridCharembs.predict_sequence_batched(char_embs,chars_mask,wordslen,predictFlag)

        RNN_embs = {}
        for idx in range(len(tokenIdChars)):
            RNN_embs[str(tokenIdChars[idx])] = dy.pick_batch_elem(chr_embs,idx)

        return RNN_embs

    def Ext_embeds(self,sentences,predictFlag=False):

        if predictFlag:
            wordtoidx = self.ext_words_devtest
            lookup_matrix = self.elookup_devtest
        else:
            wordtoidx = self.ext_words_train
            lookup_matrix = self.elookup_train

        idxtoword = {ind: word for word, ind in wordtoidx.items()}

        ext_embs = []
        for sent in sentences:
            ext_embs.extend([entry.norm for entry in sent])
        ext_embs_set = list(set(ext_embs))
        ext_embs_idx = []
        for emb in ext_embs_set:
            try:
                w_ind = wordtoidx[emb]
                ext_embs_idx.append(w_ind)
            except KeyError:
                continue
        ext_lookup_batch = dy.lookup_batch(lookup_matrix,ext_embs_idx)
        projected_embs = self.projected_embs(ext_lookup_batch)

        proj_embs = {}
        for idx in range(len(ext_embs_idx)):
            proj_embs[idxtoword[ext_embs_idx[idx]]] = dy.pick_batch_elem(projected_embs,idx)

        return proj_embs


    def Predict(self,conll_sentences,test=False):

        # Batched predictions
        print("Predictions batch size = ",self.pred_batch_size)
        if not test:
            conll_sentences.sort(key=lambda x: -len(x))
            sents_len_r = reversed(list(map(lambda x:len(x),conll_sentences)))
            ones = 0
            for senlen in sents_len_r:
                if senlen == 1:
                    ones+=1
                else:
                    break
            ones+=2
            test_batches = [x*self.pred_batch_size for x in range(int((len(conll_sentences)-1-ones)/self.pred_batch_size + 1))]
        else:
            test_batches = [x*self.pred_batch_size for x in range(int((len(conll_sentences)-1)/self.pred_batch_size + 1))]

        for bdx in range(len(test_batches)):

            dy.renew_cg()
            if not test:
                if (bdx+1 < len(test_batches)):
                    sentences = conll_sentences[test_batches[bdx]:test_batches[bdx+1]]
                else:
                    sentences = conll_sentences[test_batches[bdx]:]
            else:
                batch = test_batches[bdx]
                sentences = conll_sentences[batch:min(batch+self.pred_batch_size,len(conll_sentences))]
            sents_len = list(map(lambda x:len(x),sentences))
            sents_len = list(map(lambda x:len(x),sentences))
            dy.renew_cg()
            batch_size = len(sentences)
            wids = []
            posids = []
            xposids = []
            ext_embs = []
            char_embs =[]
            RNN_embs = self.RNN_embeds(sentences,True)
            fasttext_embs = self.Ext_embeds(sentences,True)
            masks = []
            for i in range(max([len(x) for x in sentences])):
                wids.append([(int(self.vocab.get(sent[i].norm, 0)) if len(sent) > i else 1) for sent in sentences]) #1 is the word id for pad symbol
                posids.append([(int(self.pos.get(sent[i].pos,0)) if len(sent) > i else 1) for sent in sentences])
                xposids.append([(int(self.xpos.get(sent[i].xpos,0)) if len(sent) > i else 1) for sent in sentences])
                char_embs.append([RNN_embs[str(sent[i].idChars)] if len(sent) > i else dy.zeros(self.cdims) for sent in sentences])
                ext_emb =[]
                for sent in sentences:
                    if len(sent) > i:
                        try:
                            ext_emb.append(fasttext_embs[sent[i].norm])
                        except KeyError:
                            ext_emb.append(dy.zeros(self.wdims))
                    else:
                        ext_emb.append(dy.zeros(self.wdims))
                ext_embs.append(ext_emb)

                mask = [(1 if len(sent) > i else 0) for sent in sentences]
                masks.append(mask)


            input_vecs = []
            input_vecs.append(dy.concatenate_to_batch([self.ROOT.expr()]*batch_size))
            assert len(wids)==len(char_embs)==len(ext_embs),"Error in batches input construction"
            for idx,(wid,char_emb,ext_emb,posid,xposid,mask) in enumerate(zip(wids,char_embs,ext_embs,posids,xposids,masks)):

                wembs = dy.lookup_batch(self.wlookup, wid)
                chr_embs = dy.concatenate_to_batch(char_emb)
                eemb =dy.concatenate_to_batch(ext_emb)

                finalwembs = dy.esum([wembs,eemb,chr_embs])

                posembs = dy.lookup_batch(self.poslookup, posid)
                xposembs = dy.lookup_batch(self.xposlookup, xposid)
                finalposembs = dy.esum([posembs,xposembs])

                #Concatenate word and pos tag embeddings
                input_vecs.append(dy.concatenate([finalwembs,finalposembs]))

            masks = [[1]* batch_size] + masks
            rmasks = list(reversed(masks))
            fwd1 = self.fwdLSTM1.initial_state(batch_size)
            bwd1 = self.bwdLSTM1.initial_state(batch_size)
            fwd_embs1 = fwd1.transduce(input_vecs,masks,True)
            bwd_embs1 = bwd1.transduce(list(reversed(input_vecs)),rmasks,True)
            fwd2 = self.fwdLSTM2.initial_state(batch_size)
            bwd2 = self.bwdLSTM2.initial_state(batch_size)
            fwd_embs2 = fwd2.transduce(fwd_embs1,masks,True)
            bwd_embs2 = bwd2.transduce(bwd_embs1,rmasks,True)
            fwd3 = self.fwdLSTM3.initial_state(batch_size)
            bwd3 = self.bwdLSTM3.initial_state(batch_size)
            fwd_embs3 = fwd3.transduce(fwd_embs2,masks,True)
            bwd_embs3 = bwd3.transduce(bwd_embs2,rmasks,True)

            src_encodings = [dy.concatenate([f, b]) for f, b in zip(fwd_embs3, list(reversed(bwd_embs3)))]
            pred_heads, pred_labels = self.biaffineParser.decoding(src_encodings,sents_len,test)

            for idx,sent in enumerate(sentences):
                for entry,head, relation in zip(sent,pred_heads[idx], pred_labels[idx]):
                        entry.pred_parent_id = head
                        entry.pred_relation = self.irels[relation]



                yield sent

    def drop_input_embs(self,wids):

        #Independently dropout word & pos embeddings. If both are dropped replace with zeros.
        #If only one is dropped scale the other to compensate. Otherwise keep both
        w_dropout = []
        p_dropout = []
        for wid in wids:
            if (wid != 1):
                wemb_Dropflag = random.random() < self.dropout
                posemb_Dropflag = random.random() < self.dropout
                if (wemb_Dropflag and posemb_Dropflag):
                    w_dropout.append(0)
                    p_dropout.append(0)
                elif wemb_Dropflag:
                    w_dropout.append(0)
                    p_dropout.append(1/ (1 - (float(self.wdims) / (self.wdims+self.posdims))))
                elif posemb_Dropflag:
                    w_dropout.append(1/ (1 - (float(self.posdims) / (self.wdims+self.posdims))))
                    p_dropout.append(0)
                else:
                    w_dropout.append(1)
                    p_dropout.append(1)
            else:
                w_dropout.append(0)
                p_dropout.append(0)

        return w_dropout,p_dropout

    def calculate_loss(self,sentences):

        dy.renew_cg()
        batch_size = len(sentences)
        batch_heads = []
        batch_labels = []

        for sentence in sentences:

            heads = [entry.parent_id for entry in sentence]
            labels = [self.rels[sentence[modifier].relation] for modifier, head in enumerate(heads)]
            heads.extend([0]*((len(sentences[0]) - len(heads))))
            labels.extend([0]*((len(sentences[0]) - len(labels))))
            batch_heads.append(heads)
            batch_labels.append(labels)

        sents_len = list(map(lambda x: len(x),sentences))
        RNN_embs = self.RNN_embeds(sentences)
        fasttext_embs = self.Ext_embeds(sentences)
        total_words = 0
        wids = []
        posids = []
        xposids = []
        ext_embs = []
        char_embs =[]
        masks = []
        wemb_Dropout = []
        posemb_Dropout = []
        for i in range(len(sentences[0])):
            wids.append([(int(self.vocab.get(sent[i].norm, 0)) if len(sent) > i else 1) for sent in sentences]) #1 is the word id for pad symbol
            posids.append([(int(self.pos.get(sent[i].pos,0)) if len(sent) > i else 1) for sent in sentences])
            xposids.append([(int(self.xpos.get(sent[i].xpos,0)) if len(sent) > i else 1) for sent in sentences])
            char_embs.append([RNN_embs[str(sent[i].idChars)] if len(sent) > i else dy.zeros(self.cdims) for sent in sentences])
            ext_emb =[]
            for sent in sentences:
                if len(sent) > i:
                    try:
                        ext_emb.append(fasttext_embs[sent[i].norm])
                    except KeyError:
                        ext_emb.append(dy.zeros(self.wdims))
                else:
                    ext_emb.append(dy.zeros(self.wdims))
            ext_embs.append(ext_emb)

            mask = [(1 if len(sent) > i else 0) for sent in sentences]
            masks.append(mask)
            total_words+=sum(mask)

            w_dropout,p_dropout = self.drop_input_embs(wids[-1])
            wemb_Dropout.append(w_dropout)
            posemb_Dropout.append(p_dropout)

        input_vecs = []
        input_vecs.append(dy.concatenate_to_batch([self.ROOT.expr()]*batch_size))
        assert len(wids)==len(char_embs)==len(ext_embs),"Error in batches input construction"
        for idx,(wid,char_emb,ext_emb,posid,xposid,mask) in enumerate(zip(wids,char_embs,ext_embs,posids,xposids,masks)):

            wembs = dy.lookup_batch(self.wlookup, wid)
            chr_embs = dy.concatenate_to_batch(char_emb)
            eemb =dy.concatenate_to_batch(ext_emb)
            finalwembs = dy.esum([wembs,eemb,chr_embs])

            posembs = dy.lookup_batch(self.poslookup, posid)
            xposembs = dy.lookup_batch(self.xposlookup, xposid)
            finalposembs = dy.esum([posembs,xposembs])
            #Apply word embeddings dropout mask
            word_dropout_mask = dy.inputVector(wemb_Dropout[idx])
            word_dropout_mask = dy.reshape(word_dropout_mask, (1,), batch_size)
            finalwembs = finalwembs * word_dropout_mask
            #Apply pos tag embeddings dropout mask
            pos_dropout_mask = dy.inputVector(posemb_Dropout[idx])
            pos_dropout_mask = dy.reshape(pos_dropout_mask, (1,), batch_size)
            finalposembs = finalposembs * pos_dropout_mask
            #Concatenate word and pos tag embeddings
            input_vecs.append(dy.concatenate([finalwembs,finalposembs]))

        masks = [[1]* batch_size] + masks
        rmasks = list(reversed(masks))
        self.fwdLSTM1.set_dropouts(self.dropout,self.dropout)
        self.bwdLSTM1.set_dropouts(self.dropout,self.dropout)
        self.fwdLSTM2.set_dropouts(self.dropout,self.dropout)
        self.bwdLSTM2.set_dropouts(self.dropout,self.dropout)
        self.fwdLSTM3.set_dropouts(self.dropout,self.dropout)
        self.bwdLSTM3.set_dropouts(self.dropout,self.dropout)



        self.fwdLSTM1.set_dropout_masks(batch_size)
        self.bwdLSTM1.set_dropout_masks(batch_size)
        self.fwdLSTM2.set_dropout_masks(batch_size)
        self.bwdLSTM2.set_dropout_masks(batch_size)
        self.fwdLSTM3.set_dropout_masks(batch_size)
        self.bwdLSTM3.set_dropout_masks(batch_size)

        fwd1 = self.fwdLSTM1.initial_state(batch_size)
        bwd1 = self.bwdLSTM1.initial_state(batch_size)
        fwd_embs1 = fwd1.transduce(input_vecs,masks)
        bwd_embs1 = bwd1.transduce(list(reversed(input_vecs)),rmasks)
        fwd2 = self.fwdLSTM2.initial_state(batch_size)
        bwd2 = self.bwdLSTM2.initial_state(batch_size)
        fwd_embs2 = fwd2.transduce(fwd_embs1,masks)
        bwd_embs2 = bwd2.transduce(bwd_embs1,rmasks)
        fwd3 = self.fwdLSTM3.initial_state(batch_size)
        bwd3 = self.bwdLSTM3.initial_state(batch_size)
        fwd_embs3 = fwd3.transduce(fwd_embs2,masks)
        bwd_embs3 = bwd3.transduce(bwd_embs2,rmasks)


        src_encodings = [dy.concatenate([f, b]) for f, b in zip(fwd_embs3, list(reversed(bwd_embs3)))]

        return  self.biaffineParser.decode_loss(src_encodings,masks[1:],(batch_heads, batch_labels),sents_len),total_words


    def Train(self,conll_sentences,mini_batch,t_step,lr=False):

        if (lr):
            self.learning_rate = self.learning_rate * 0.75
            self.trainer.learning_rate = self.learning_rate
            print("Trainer learning rate is updated")
            print(self.trainer.status())


        start = time.time()
        train_loss = 0
        total_words = 0


        loss,words = self.calculate_loss(conll_sentences[mini_batch[0]:mini_batch[1]])
        train_loss += loss.value()
        loss.backward()
        total_words += words
        self.trainer.update()



        print("Finish training step: %i, Train loss/token=%.4f, time=%.2fs" % (t_step, train_loss / total_words, time.time() - start))
