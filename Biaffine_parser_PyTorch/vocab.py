from collections import Counter
import pandas as pd
import torch
import numpy as np
import pickle
from ast import literal_eval


class VocabBuilder(object):
    '''
    Read file and create word_to_index dictionary.
    This can truncate low-frequency words with min_sample option.
    '''
    def __init__(self, path_file):
        # word count
        self.word_count = VocabBuilder.count_from_file(path_file)
        self.word_to_index = {}

    @staticmethod
    def count_from_file(path_file):
        """
        count word frequencies in a file.
        Args:
            path_file:

        Returns:
            dict: {word_n :count_n, ...}

        """
        df = pd.read_csv(path_file)

        # count
        word_count = Counter()
        for text in df['Text'].values.tolist():
            word_count.update([tok.lower() for tok in literal_eval(text)])

        return word_count

    def get_word_index(self, min_sample=2, padding_marker='__PADDING__', unknown_marker='__UNK__'):
        """
        create word-to-index mapping. Padding and unknown are added to first 2 indices.

        Args:
            min_sample: for Truncation
            padding_marker: padding mark
            unknown_marker: unknown-word mark

        Returns:
            dict: {word_n: index_n, ... }

        """
        # truncate low fq word
        _word_count = filter(lambda x:  min_sample<=x[1], self.word_count.items())
        tokens = list(zip(*_word_count))[0]

        # inset padding and unknown
        self.word_to_index = { tkn: i for i, tkn in enumerate([padding_marker, unknown_marker] + sorted(tokens))}
        print('Turncated vocab size:{} (removed:{})'.format(len(self.word_to_index),
                                                            len(self.word_count) - len(self.word_to_index)))
        return self.word_to_index, tokens

class CharBuilder(object):
    '''
    Read file and create char_to_index dictionary.
    '''
    def __init__(self, path_file):
        # word count
        self.char_count = CharBuilder.count_from_file(path_file)
        self.char_to_index = {}

    @staticmethod
    def count_from_file(path_file):
        """
        count char frequencies in a file.
        Args:
            path_file:

        Returns:
            dict: {char_n :count_n, ...}

        """
        df = pd.read_csv(path_file)

        # count
        char_count = Counter()
        for text in df['Text'].values.tolist():
            for tok in literal_eval(text):
                char_count.update([char for char in list(tok)])

        return char_count

    def get_char_index(self, start_marker='__START__', end_marker='__END__',padding_marker='__PADDING__', unknown_marker='__UNK__'):
        """
        create char-to-index mapping. Padding and unknown are added to first 2 indices.

        Args:
            padding_marker: padding mark
            unknown_marker: unknown-word mark

        Returns:
            dict: {char_n: index_n, ... }

        """

        chars = self.char_count.keys()

        # inset padding and unknown
        self.char_to_index = {chr: i for i, chr in enumerate([start_marker,end_marker,padding_marker,unknown_marker] + sorted(chars))}

        return self.char_to_index, chars

class TagBuilder(object):
    '''
    Read file and create tag_to_index dictionary.
    '''
    def __init__(self, path_file,field):
        # word count
        self.tag_count = TagBuilder.count_from_file(path_file,field)
        self.tag_to_index = {}

    @staticmethod
    def count_from_file(path_file,field):
        """
        count tag frequencies in a file.
        Args:
            path_file:
            field:

        Returns:
            dict: {tag_n :count_n, ...}

        """
        df = pd.read_csv(path_file)

        # count
        tag_count = Counter()
        for tags in df[field].values.tolist():
            tag_count.update([tag for tag in literal_eval(tags)])

        return tag_count

    def get_tag_index(self, padding_marker='__PADDING__', unknown_marker='__UNK__'):
        """
        create tag-to-index mapping. Padding and unknown are added to last 2 indices.

        Args:
            padding_marker: padding mark
            unknown_marker: unknown-word mark

        Returns:
            dict: {tag_n: index_n, ... }

        """

        tags = self.tag_count.keys()

        # inset padding 
        self.tag_to_index = { tag: i for i, tag in enumerate(sorted(tags))} #+ [padding_marker])} #, unknown_marker])}
        self.tag_to_index[padding_marker] = -1
        return self.tag_to_index, tags

    def get_tag_index_padded(self, padding_marker='__PADDING__', unknown_marker='__UNK__'):
        """
        create tag-to-index mapping. Padding and unknown are added to last 2 indices.

        Args:
            padding_marker: padding mark
            unknown_marker: unknown-word mark

        Returns:
            dict: {tag_n: index_n, ... }

        """

        tags = self.tag_count.keys()

        # inset padding and unknown
        self.tag_to_index = { tag: i for i, tag in enumerate(sorted(tags) + [padding_marker,unknown_marker])}

        return self.tag_to_index, tags


class FastextVocabBuilder(object) :

    def __init__(self, path_fasttext):
        self.vec = None
        self.vocab = {}
        self.path_fasttext = path_fasttext

    def get_word_index(self,tensor_path, voc_path, padding_marker='__PADDING__', unknown_marker='__UNK__'):


        idx = 0
        with open(self.path_fasttext, 'r', encoding="utf-8", newline='\n',errors='ignore') as f:
            for l in f:
                line = l.rstrip().split(' ')
                if idx == 0:
                    vocab_size = int(line[0]) + 2
                    dim = int(line[1])
                    self.vec = torch.zeros((vocab_size,dim))
                    self.vocab["__PADDING__"] = 0
                    self.vocab["__UNK__"] = 1
                    idx = 2
                else:
                    self.vocab[line[0]] = idx
                    emb = np.array(line[1:]).astype(np.float)
                    if (emb.shape[0] == dim):
                        self.vec[idx,:] = torch.tensor(np.array(line[1:]).astype(np.float))
                        idx+=1
                    else:
                        continue

            pickle.dump(self.vocab,open(voc_path,'wb'))
            torch.save(self.vec,tensor_path)


        return self.vocab, self.vec
