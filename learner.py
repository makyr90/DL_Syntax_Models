import dynet as dy
from utils import read_conll, write_conll
from biaffine import DeepBiaffineAttentionDecoder
from char_attention import HybridCharacterAttention
from NN import Lin_Projection
import  time, random, utils, pickle
import numpy as np


class biAffine_parser:
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
        self.layers = options.lstm_layers
        self.ext_words_train = {word: ind+3 for word, ind in ext_words_train.items()}
        self.ext_words_devtest = {word: ind+3 for word, ind in ext_words_devtest.items()}
        self.wordsCount = vocab
        self.vocab = {word: ind+3 for word, ind in w2i.items()}
        self.pos = {word: ind+3 for ind, word in enumerate(pos)}
        self.id2pos = {ind: word for word,ind in self.pos.items()}
        self.xpos = {word: ind+3 for ind, word in enumerate(xpos)}
        self.id2xpos = {ind: word for word,ind in self.xpos.items()}
        self.c2i = c2i
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.irels = {ind: word for word, ind in self.rels.items()}
        self.pred_batch_size = options.pred_batch_size
        self.vocab['INITIAL'] = 1
        self.vocab['PAD'] = 2
        self.pos['INITIAL'] = 1
        self.pos['PAD'] = 2
        self.xpos['INITIAL'] = 1
        self.xpos['PAD'] = 2


        self.external_embedding, self.edim, self.edim_out = None, 0, 0
        if options.external_embedding is not None:

            self.external_embedding = np.load(options.external_embedding)
            self.ext_voc=  pickle.load( open( options.external_embedding_voc, "rb" ) )
            self.edim = self.external_embedding.shape[1]
            self.projective_embs = Lin_Projection(self.model, self.edim, self.wdims)
            self.elookup_train = self.pretrained_embs.add_lookup_parameters((len(self.ext_words_train)+3 , self.edim))
            for word, i in self.ext_words_train.items():
                self.elookup_train.init_row(i, self.external_embedding[self.ext_voc[word],:])
            self.elookup_train.init_row(0, np.zeros(self.edim))
            self.elookup_train.init_row(1, np.zeros(self.edim))
            self.elookup_train.init_row(2, np.zeros(self.edim))

            self.elookup_devtest = self.pretrained_embs.add_lookup_parameters((len(self.ext_words_devtest)+3 , self.edim))
            for word, i in self.ext_words_devtest.items():
                self.elookup_devtest.init_row(i, self.external_embedding[self.ext_voc[word],:])
            self.elookup_devtest.init_row(0, np.zeros(self.edim))
            self.elookup_devtest.init_row(1, np.zeros(self.edim))
            self.elookup_devtest.init_row(2, np.zeros(self.edim))

            self.ext_words_train['INITIAL'] = 1
            self.ext_words_train['PAD'] = 2
            self.ext_words_devtest['INITIAL'] = 1
            self.ext_words_devtest['PAD'] = 2

            print('Load external embeddings. External embeddings vectors dimension', self.edim)

        print("Dropout probability for MLP's hidden layers & LSTM's hidden/reccurent units:", self.dropout)

        self.fwdLSTM = dy.VanillaLSTMBuilder(self.layers, self.wdims + self.posdims, self.ldims, self.model,forget_bias = 0.0)
        self.bwdLSTM = dy.VanillaLSTMBuilder(self.layers, self.wdims + self.posdims, self.ldims, self.model, forget_bias = 0.0)


        self.biaffineParser = DeepBiaffineAttentionDecoder(self.model, len(self.rels), src_ctx_dim=self.ldims * 2,
                    n_arc_mlp_units=500, n_label_mlp_units=100, arc_mlp_dropout=self.dropout, label_mlp_dropout=self.dropout)

        self.HybridCharembs = HybridCharacterAttention(self.model,layers=1,ldims=400,input_size=self.cdims,output_size=self.wdims,dropout=self.dropout)

        self.wlookup = self.model.add_lookup_parameters((len(vocab) + 3, self.wdims),init = dy.ConstInitializer(0))
        #0 for unknown 1 for [initial] and 2 for [PAD]
        self.poslookup = self.model.add_lookup_parameters((len(self.pos) + 3, self.posdims),init = dy.ConstInitializer(0))
        #0 for unknown 1 for [initial] and 2 for [PAD]
        self.xposlookup = self.model.add_lookup_parameters((len(self.xpos) + 3, self.posdims),init = dy.ConstInitializer(0))
        #0 for unknown 1 for [initial] and 2 for [PAD]

        self.clookup = self.model.add_lookup_parameters((len(c2i), self.cdims), init = dy.NormalInitializer())


    def Save(self, filename):
        self.model.save(filename)


    def Load(self, filename):
        self.model.populate(filename)


    def Predict(self, conll_path, test=False):

        # Batched predictions
        self.fwdLSTM.disable_dropout()
        self.bwdLSTM.disable_dropout()
        with open(conll_path, 'r') as conllFP:
            testData = list(read_conll(conllFP, self.c2i))

        conll_sentences = []
        for sentence in testData:
            conll_sentence = [entry for entry in sentence  if isinstance(entry, utils.ConllEntry)]
            conll_sentences.append(conll_sentence)

        if not test:
            conll_sentences.sort(key=lambda x: -len(x))
        test_batches = [x*self.pred_batch_size for x in range(int((len(conll_sentences)-1)/self.pred_batch_size + 1))]

        for batch in test_batches:

            dy.renew_cg()
            sentences = conll_sentences[batch:min(batch+self.pred_batch_size,len(conll_sentences))]
            sents_len = list(map(lambda x:len(x),sentences))
            dy.renew_cg()
            batch_size = len(sentences)
            wids = []
            posids = []
            xposids = []
            extwids = []
            wordChars =[]
            masks = []
            for i in range(max([len(x) for x in sentences])):
                wids.append([(int(self.vocab.get(sent[i].norm, 0)) if len(sent) > i else 2) for sent in sentences]) #2 is the word id for pad symbol
                posids.append([(int(self.pos.get(sent[i].pos,0)) if len(sent) > i else 2) for sent in sentences])
                xposids.append([(int(self.xpos.get(sent[i].xpos,0)) if len(sent) > i else 2) for sent in sentences])

                wordChars.append([sent[i].idChars if len(sent) > i else [] for sent in sentences]) #5 is the char id for pad symbol
                extwids.append([(int(self.ext_words_devtest.get(sent[i].norm, 0)) if len(sent) > i else 2) for sent in sentences])
                mask = [(1 if len(sent) > i else 0) for sent in sentences]
                masks.append(mask)


            input_vecs = []

            for idx,(wid,wch,extwid,posid,xposid,mask) in enumerate(zip(wids,wordChars,extwids,posids,xposids,masks)):

                wembs = dy.lookup_batch(self.wlookup, wid)

                char_src_len =  len(max(wch,key=len))
                chars_mask = []
                char_ids = []
                for i in range(char_src_len):
                    char_ids.append([(char[i] if len(char) > i else 5) for char in wch])
                    char_mask = [(1 if len(chars) > i else 0) for chars in wch]
                    chars_mask.append(char_mask)

                char_embs = []
                for cid in char_ids:
                    char_embs.append(dy.lookup_batch(self.clookup, cid))
                wordslen = list(map(lambda x:len(x),wch))

                chr_embs = self.HybridCharembs.predict_sequence_batched(char_embs,chars_mask,wordslen,char_src_len,batch_size,True)


                extwembs = dy.lookup_batch(self.elookup_devtest,extwid)
                proj_ext_word_embs = self.projective_embs(extwembs)
                finalwembs = dy.esum([wembs,proj_ext_word_embs,chr_embs])

                posembs = dy.lookup_batch(self.poslookup, posid)
                xposembs = dy.lookup_batch(self.xposlookup, xposid)
                finalposembs = dy.esum([posembs,xposembs])

                if mask[-1] != 1:
                    mask_expr = dy.inputVector(mask)
                    mask_expr = dy.reshape(mask_expr, (1,), batch_size)
                    finalwembs = finalwembs * mask_expr
                    finalposembs = finalposembs * mask_expr

                #Concatenate word and pos tag embeddings
                input_vecs.append(dy.concatenate([finalwembs,posembs]))

            fwd = self.fwdLSTM.initial_state()
            bwd = self.bwdLSTM.initial_state()
            fwd_embs = fwd.transduce(input_vecs)
            bwd_embs = bwd.transduce(reversed(input_vecs))

            src_encodings = [dy.reshape(dy.concatenate([f, b]), (self.ldims * 2, 1)) for f, b in zip(fwd_embs, reversed(bwd_embs))]
            pred_heads, pred_labels = self.biaffineParser.decoding(src_encodings,sents_len)

            for idx,sent in enumerate(sentences):
                for entry,head, relation in zip(sent,pred_heads[idx], pred_labels[idx]):
                        entry.pred_parent_id = head
                        entry.pred_relation = self.irels[relation]



                yield sent

    def drop_input_embs(self,wids):

        #Independently dropout word & pos embeddings. If both are dropped replace with zeros.
        #If only one is dropped scale the other to compensate. Otherwise keep both
        #Use the same dropout mask for both forward and backword LSTMs
        w_dropout = []
        p_dropout = []
        for wid in wids:
            if (wid != 2):
                wemb_Dropflag = random.random() < self.dropout
                posemb_Dropflag = random.random() < self.dropout
                if (wemb_Dropflag and posemb_Dropflag):
                    w_dropout.append(0)
                    p_dropout.append(0)
                elif wemb_Dropflag:
                    w_dropout.append(0)
                    p_dropout.append(1/ (1 - (float(self.wdims) / (self.wdims+self.posdims))))
                elif posemb_Dropflag:
                    w_dropout.append(1 / (1 - (float(self.posdims) / (self.wdims+self.posdims))))
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

        total_words = 0
        wids = []
        posids = []
        xposids = []
        extwids = []
        wordChars =[]
        masks = []
        wemb_Dropout = []
        posemb_Dropout = []
        for i in range(len(sentences[0])):
            wids.append([(int(self.vocab.get(sent[i].norm, 0)) if len(sent) > i else 2) for sent in sentences]) #2 is the word id for pad symbol
            posids.append([(int(self.pos.get(sent[i].pos,0)) if len(sent) > i else 2) for sent in sentences])
            xposids.append([(int(self.xpos.get(sent[i].xpos,0)) if len(sent) > i else 2) for sent in sentences])

            wordChars.append([sent[i].idChars if len(sent) > i else [] for sent in sentences]) #5 is the char id for pad symbol
            extwids.append([(int(self.ext_words_train.get(sent[i].norm, 0)) if len(sent) > i else 2) for sent in sentences])
            mask = [(1 if len(sent) > i else 0) for sent in sentences]
            masks.append(mask)
            total_words+=sum(mask)

            w_dropout,p_dropout = self.drop_input_embs(wids[-1])
            wemb_Dropout.append(w_dropout)
            posemb_Dropout.append(p_dropout)


        input_vecs = []

        for idx,(wid,wch,extwid,posid,xposid) in enumerate(zip(wids,wordChars,extwids,posids,xposids)):

            wembs = dy.lookup_batch(self.wlookup, wid)

            #Batched character embeddings

            char_src_len =  len(max(wch,key=len))
            chars_mask = []
            char_ids = []
            for i in range(char_src_len):
                char_ids.append([(char[i] if len(char) > i else 5) for char in wch])
                char_mask = [(1 if len(chars) > i else 0) for chars in wch]
                chars_mask.append(char_mask)

            char_embs = []
            for cid in char_ids:
                char_embs.append(dy.lookup_batch(self.clookup, cid))
            wordslen = list(map(lambda x:len(x),wch))

            chr_embs = self.HybridCharembs.predict_sequence_batched(char_embs,chars_mask,wordslen,char_src_len,batch_size)

            #Unbatched character embeddings
            # chr_embs = []
            # for char_list in wch:
            #     if len(char_list) > 0:
            #         chr_embs.append(self.HybridCharembs.predict_sequence([self.clookup[char] for char in char_list]))
            #     else:
            #         chr_embs.append(dy.zeros(self.wdims))
            #
            # chr_embs = dy.concatenate_to_batch(chr_embs)

            extwembs = dy.lookup_batch(self.elookup_train,extwid)
            proj_ext_word_embs = self.projective_embs(extwembs)
            finalwembs = dy.esum([wembs,proj_ext_word_embs,chr_embs])

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
            posembs = finalposembs * pos_dropout_mask
            #Concatenate word and pos tag embeddings
            input_vecs.append(dy.concatenate([finalwembs,posembs]))


        fwd = self.fwdLSTM.initial_state()
        bwd = self.bwdLSTM.initial_state()
        fwd_embs = fwd.transduce(input_vecs)
        bwd_embs = bwd.transduce(reversed(input_vecs))

        src_encodings = [dy.reshape(dy.concatenate([f, b]), (self.ldims * 2, 1)) for f, b in zip(fwd_embs, reversed(bwd_embs))]
        return self.biaffineParser.decode_loss(src_encodings,masks, (batch_heads, batch_labels)),total_words


    def Train(self,conll_sentences,mini_batch,t_step,lr=False):

        if (lr):
            self.learning_rate = self.learning_rate * 0.75
            self.trainer.learning_rate = self.learning_rate
            print("Trainer learning rate is updated")
            print(self.trainer.status())

        self.fwdLSTM.set_dropouts(self.dropout,self.dropout)
        self.bwdLSTM.set_dropouts(self.dropout,self.dropout)

        start = time.time()
        train_loss = 0
        total_words = 0


        loss,words = self.calculate_loss(conll_sentences[mini_batch[0]:mini_batch[1]])
        train_loss += loss.value()
        loss.backward()
        total_words += words
        self.trainer.update()



        print("Finish training step: %i, Train loss/token=%.4f, time=%.2fs" % (t_step, train_loss / total_words, time.time() - start))
