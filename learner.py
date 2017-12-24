import dynet as dy
from utils import read_conll, write_conll
from affine import affineAttentionDecoder
from char_attention import HybridCharacterAttention
from NN import Lin_Projection
import  time, random, utils, pickle
import numpy as np


class Affine_tagger:
    def __init__(self, vocab, pos, xpos,  w2i, c2i, ext_words_train, ext_words_devtest, options):

        self.model = dy.ParameterCollection()
        self.pretrained_embs = dy.ParameterCollection()
        self.learning_rate = options.learning_rate
        self.trainer = dy.AdamTrainer(self.model,alpha = self.learning_rate,beta_1=0.9,beta_2=0.9,eps=1e-12)

        self.dropout = float(options.dropout)
        self.ldims = options.lstm_dims
        self.wdims = options.wembedding_dims
        self.cdims = options.cembedding_dims
        self.layers = options.lstm_layers
        self.pos = {word: ind for ind, word in enumerate(pos)}
        self.ipos = {ind: word for word, ind in self.pos.items()}
        self.xpos = {word: ind for ind, word in enumerate(xpos)}
        self.ixpos = {ind: word for word, ind in self.xpos.items()}
        self.ext_words_train = {word: ind+3 for word, ind in ext_words_train.items()}
        self.ext_words_devtest = {word: ind+3 for word, ind in ext_words_devtest.items()}
        self.wordsCount = vocab
        self.vocab = {word: ind+3 for word, ind in w2i.items()}
        self.c2i = c2i
        self.pred_batch_size = options.pred_batch_size
        self.vocab['INITIAL'] = 1
        self.vocab['PAD'] = 2



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


        self.fwdLSTM = dy.VanillaLSTMBuilder(self.layers, self.wdims, self.ldims, self.model,forget_bias = 0.0)
        self.bwdLSTM = dy.VanillaLSTMBuilder(self.layers, self.wdims, self.ldims, self.model, forget_bias = 0.0)


        self.affineTagger = affineAttentionDecoder(self.model, len(self.ipos), len(self.ixpos), src_ctx_dim=self.ldims * 2, n_pos_tagger_mlp_units=500,
                            n_xpos_tagger_mlp_units=500, mlps_dropout = 0.5)

        self.HybridCharembs = HybridCharacterAttention(self.model,layers=1,ldims=400,input_size=self.cdims,output_size=self.wdims,dropout=self.dropout)

        self.wlookup = self.model.add_lookup_parameters((len(vocab) + 3, self.wdims), init = dy.ConstInitializer(0))
        #0 for unknown 1 for [initial] and 2 for [PAD]

        self.clookup = self.model.add_lookup_parameters((len(c2i), self.cdims), init = dy.NormalInitializer())


    def Save(self, filename):
        self.model.save(filename)


    def Load(self, filename):
        self.model.populate(filename)


    def Predict(self, conll_path,test=False):

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
            batch_size = len(sentences)

            wids = []
            extwids = []
            wordChars =[]
            masks = []
            for i in range(max([len(x) for x in sentences])):
                wids.append([(int(self.vocab.get(sent[i].norm, 0)) if len(sent) > i else 2) for sent in sentences]) #2 is the word id for pad symbol
                wordChars.append([sent[i].idChars if len(sent) > i else [] for sent in sentences]) #5 is the char id for pad symbol
                extwids.append([(int(self.ext_words_devtest.get(sent[i].norm, 0)) if len(sent) > i else 2) for sent in sentences])
                mask = [(1 if len(sent) > i else 0) for sent in sentences]
                masks.append(mask)


            input_vecs = []

            for idx,(wid,wch,extwid,mask) in enumerate(zip(wids,wordChars,extwids,masks)):

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


                if mask[-1] != 1:
                    mask_expr = dy.inputVector(mask)
                    mask_expr = dy.reshape(mask_expr, (1,), batch_size)
                    finalwembs = finalwembs * mask_expr

                input_vecs.append(finalwembs)

            fwd = self.fwdLSTM.initial_state()
            bwd = self.bwdLSTM.initial_state()
            fwd_embs = fwd.transduce(input_vecs)
            bwd_embs = bwd.transduce(reversed(input_vecs))

            src_encodings = [dy.reshape(dy.concatenate([f, b]), (self.ldims * 2, 1)) for f, b in zip(fwd_embs, reversed(bwd_embs))]
            pred_pos, pred_xpos = self.affineTagger.decoding(src_encodings,sents_len)

            for idx,sent in enumerate(sentences):
                for entry, pos, xpos in zip(sent,pred_pos[idx], pred_xpos[idx]):
                    entry.pred_pos = self.ipos[pos]
                    entry.pred_xpos = self.ixpos[xpos]

                yield sent


    def calculate_loss(self,sentences):

        dy.renew_cg()
        batch_size = len(sentences)
        src_len = len(sentences[0])
        pos_ids = []
        xpos_ids = []
        for sentence in sentences:
            pos = [self.pos[entry.pos] for entry in sentence]
            xpos = [self.xpos[entry.xpos] for entry in sentence]
            pos.extend([0]*((len(sentences[0]) - len(pos))))
            xpos.extend([0]*((len(sentences[0]) - len(xpos))))
            pos_ids.append(pos)
            xpos_ids.append(xpos)

        total_words = 0
        wids = []
        extwids = []
        wordChars =[]
        masks = []
        for i in range(len(sentences[0])):
            wids.append([(int(self.vocab.get(sent[i].norm, 0)) if len(sent) > i else 2) for sent in sentences]) #2 is the word id for pad symbol

            wordChars.append([sent[i].idChars if len(sent) > i else [] for sent in sentences]) #5 is the char id for pad symbol
            extwids.append([(int(self.ext_words_train.get(sent[i].norm, 0)) if len(sent) > i else 2) for sent in sentences])
            mask = [(1 if len(sent) > i else 0) for sent in sentences]
            masks.append(mask)
            total_words+=sum(mask)


        input_vecs = []

        for idx,(wid,wch,extwid) in enumerate(zip(wids,wordChars,extwids)):

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



            extwembs = dy.lookup_batch(self.elookup_train,extwid)
            proj_ext_word_embs = self.projective_embs(extwembs)
            finalwembs = dy.esum([wembs,proj_ext_word_embs,chr_embs])

            input_vecs.append(finalwembs)


        fwd = self.fwdLSTM.initial_state()
        bwd = self.bwdLSTM.initial_state()
        fwd_embs = fwd.transduce(input_vecs)
        bwd_embs = bwd.transduce(reversed(input_vecs))

        src_encodings = [dy.reshape(dy.concatenate([f, b]), (self.ldims * 2, 1)) for f, b in zip(fwd_embs, reversed(bwd_embs))]
        return self.affineTagger.decode_loss(src_encodings,masks,src_len,batch_size, pos_ids,xpos_ids),total_words


    def Train(self,conll_sentences,mini_batch,t_step,lr=False):

        if (lr):
            self.learning_rate = self.learning_rate * 0.75
            self.trainer.learning_rate = self.learning_rate
            print("Trainer learning rate is updated")
            print(self.trainer.status())

        self.fwdLSTM.set_dropouts(self.dropout,0.5)
        self.bwdLSTM.set_dropouts(self.dropout,0.5)

        start = time.time()
        train_loss = 0
        total_words = 0


        loss,words = self.calculate_loss(conll_sentences[mini_batch[0]:mini_batch[1]])
        train_loss += loss.value()
        loss.backward()
        total_words += words
        self.trainer.update()



        print("Finish training step: %i, Train loss/token=%.4f, time=%.2fs" % (t_step, train_loss / total_words, time.time() - start))
