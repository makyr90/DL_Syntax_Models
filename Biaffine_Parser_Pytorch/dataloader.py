import torch
import pandas as pd
import numpy as np
from ast import literal_eval
import math
import random

class ClassDataLoader(object):

    def __init__(self, path_file, word_to_index,fasttext_to_index,char_to_index,pos_to_index,xpos_to_index,rel_to_index,batch_size=32,predict_flag=0,train=0):
        """

        Args:
            path_file:
            word_to_index:
            fast_to_index:
            char_to_index:
            pos_to_index:
            xpos_to_index:
            predict_flag:
            train:
            batch_size:
        """

        self.batch_size = batch_size
        self.word_to_index = word_to_index
        self.fasttext_to_index = fasttext_to_index
        self.pos_to_index = pos_to_index
        self.xpos_to_index = xpos_to_index
        self.rel_to_index = rel_to_index
        self.char_to_index = char_to_index
        self.predict_flag = predict_flag
        self.train = train
        print("Train flag: ",self.train)
        print("Predict flag: ",self.predict_flag)

        # read file
        df = pd.read_csv(path_file)
        df['Text_id'] = df['Text'].apply(self.generate_indexifyer(self.word_to_index))
        df['Text_ext_id'] = df['Text'].apply(self.generate_indexifyer(self.fasttext_to_index))
        df['Text_char_id'] = df['Text'].apply(self.generate_indexifyer_char())
        df['POS_id'] = df['POS'].apply(self.generate_indexifyer_tag(self.pos_to_index))
        df['XPOS_id'] = df['XPOS'].apply(self.generate_indexifyer_tag(self.xpos_to_index))
        if self.predict_flag:
            data = df[['Text_id','Text_ext_id','Text_char_id','POS_id','XPOS_id']]
        else:
            df['Drel_id'] = df['Drel'].apply(self.generate_indexifyer_tag(self.rel_to_index))
            data = df[['Text_id','Text_ext_id','Text_char_id','POS_id','XPOS_id','Head_id','Drel_id']]
        self.samples = data.values.tolist()

        # for batch
        self.n_samples = len(self.samples)
        self.n_batches = math.ceil(self.n_samples / self.batch_size)
        self.max_length = self._get_max_length()
        self.index = 0
        self.batch_index = 0
        self.indices = np.arange(self.n_samples)
        if self.train:
            self._shuffle_indices()

        self.report()

    def _shuffle_indices(self):
        self.indices = np.random.permutation(self.n_samples)
        self.index = 0
        self.batch_index = 0

    def _get_max_length(self):
        length = 0
        for sample in self.samples:
            length = max(length, len(sample[0]))
        return length

    def generate_indexifyer(self,lookup_dict):

        def indexify(sentence):

            sent_indices = []
            for word in literal_eval(sentence):
                if word.lower() in lookup_dict:
                    sent_indices.append(lookup_dict[word.lower()])
                else:
                    sent_indices.append(lookup_dict['__UNK__'])
            return sent_indices

        return indexify

    def generate_indexifyer_char(self):

        def indexify(sentence):

            char_indices = []
            for word in literal_eval(sentence):
                chars = []
                chars.append(self.char_to_index['__START__'])
                for char in list(word):
                    if char in self.char_to_index:
                        chars.append(self.char_to_index[char])
                    else:
                        chars.append(self.char_to_index['__UNK__'])
                chars.append(self.char_to_index['__END__'])
                char_indices.append(chars)

            return char_indices

        return indexify

    def generate_indexifyer_tag(self,lookup_dict):

        def indexify(sentence):

            tag_indices = []
            for tag in literal_eval(sentence):
                if tag in lookup_dict:
                    tag_indices.append(lookup_dict[tag])
                else:
                    print(tag)
                    tag_indices.append(lookup_dict['__UNK__'])

            return tag_indices

        return indexify


    def _create_batch(self):

        batch = []
        n = 0
        while ((n < self.batch_size) and (self.index < len(self.samples))):
            _index = self.indices[self.index]
            batch.append(self.samples[_index])
            self.index += 1
            n += 1
        self.batch_index += 1

        #Fix for the extreme case that last batch has size == 1. Append the last sample to the
        #previous batch
        if (self.index+1 == len(self.samples)):
            _index = self.indices[self.index]
            batch.append(self.samples[_index])
            self.index += 1
            self.batch_index += 1

        word_ids = []
        ext_word_ids = []
        char_ids = []
        pos_ids = []
        xpos_ids = []
        head_ids = []
        drel_ids = []

        for bat in batch:

            word_ids.append(bat[0])
            ext_word_ids.append(bat[1])
            char_ids.append(bat[2])
            pos_ids.append(bat[3])
            xpos_ids.append(bat[4])

            if not self.predict_flag:
                head_ids.append(literal_eval(bat[5]))
                drel_ids.append(bat[6])


        # get the length of each seq in your batch
        seq_lengths = torch.LongTensor(list(map(len, word_ids)))
        #padding
        word_tensor = torch.zeros((len(word_ids), seq_lengths.max())).long()
        pos_tensor = torch.zeros((len(word_ids), seq_lengths.max())).long()
        # fill with pos Padding id
        pos_tensor.fill_(self.pos_to_index["__PADDING__"])
        xpos_tensor = torch.zeros((len(word_ids), seq_lengths.max())).long()
        # fill with xpos Padding id
        xpos_tensor.fill_(self.xpos_to_index["__PADDING__"])

        for idx, (wid,pos_id,xpos_id,seqlen) in enumerate(zip(word_ids,pos_ids,xpos_ids,seq_lengths)):
            word_tensor[idx, :seqlen] = torch.LongTensor(wid)
            pos_tensor[idx, :seqlen] = torch.LongTensor(pos_id)
            xpos_tensor[idx, :seqlen] = torch.LongTensor(xpos_id)

        if not self.predict_flag:

            head_targets = torch.zeros((len(word_ids), seq_lengths.max())).long()
            # fill with  Padding id -1
            head_targets.fill_(-1)
            rel_targets = torch.zeros((len(word_ids), seq_lengths.max())).long()
            # fill with rel Padding id
            rel_targets.fill_(self.rel_to_index["__PADDING__"])
            for idx, (head_id,drel_id,seqlen) in enumerate(zip(head_ids,drel_ids,seq_lengths)):
                head_targets[idx, :seqlen] = torch.LongTensor(head_id)
                rel_targets[idx, :seqlen] = torch.LongTensor(drel_id)

        else:
            head_targets = None
            rel_targets = None

        #sort in decreasing order
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        word_tensor = word_tensor[perm_idx]
        pos_tensor = pos_tensor[perm_idx]
        xpos_tensor = xpos_tensor[perm_idx]

        if not self.predict_flag:
            head_targets = head_targets[perm_idx]
            rel_targets = rel_targets[perm_idx]

        sort_idx = perm_idx.data.numpy().tolist()
        char_ids_sorted = []
        ext_word_ids_sorted = []
        for idx in sort_idx:
            char_ids_sorted.append(char_ids[idx])
            ext_word_ids_sorted.append(ext_word_ids[idx])

        return word_tensor,ext_word_ids_sorted,char_ids_sorted,pos_tensor,xpos_tensor,head_targets,rel_targets,seq_lengths,perm_idx

    def __len__(self):
        return self.n_batches

    def __iter__(self):

        self.index = 0
        self.batch_index = 0
        if  self.train:
            self._shuffle_indices()
        for i in range(self.n_batches):
            if self.batch_index == self.n_batches:
                raise StopIteration()
            yield self._create_batch()

    def show_samples(self, n=10):
        for sample in self.samples[:n]:
            print(sample)

    def report(self):
        print('# samples: {}'.format(len(self.samples)))
        print('max len: {}'.format(self.max_length))
        print('# vocab: {}'.format(len(self.word_to_index)))
        print('# batches: {} (batch_size = {})'.format(self.n_batches, self.batch_size))
