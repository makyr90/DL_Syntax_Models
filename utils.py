# coding=utf-8
from collections import Counter
import re, pickle
import numpy as np


class ConllEntry:
    def __init__(self, id, form, lemma, pos, xpos, feats=None, parent_id=None, relation=None, deps=None, misc=None):
        self.id = id
        self.form = form
        self.norm = normalize(form)
        self.xpos = xpos.upper()
        self.pos = pos.upper()
        self.parent_id = parent_id
        self.relation = relation

        self.lemma = lemma
        self.feats = feats
        self.deps = deps
        self.misc = misc

        self.pred_parent_id = None
        self.pred_relation = None

        self.idChars = []

    def __str__(self):
        values = [str(self.id), self.form, self.lemma, self.pos, self.xpos, self.feats, str(self.pred_parent_id) if self.pred_parent_id is not None else None, self.pred_relation, self.deps, self.misc]
        return '\t'.join(['_' if v is None else v for v in values])


def vocab(conll_path):
    wordsCount = Counter()
    posCount = Counter()
    xposCount = Counter()
    relCount = Counter()

    #Character vocabulary
    c2i = {}
    c2i["UNK"] = 0  # unk char
    c2i["NUM"] = 1
    c2i["START"] = 2
    c2i["STOP"] = 3
    c2i["ROOT"] = 4
    c2i["PAD"] = 5

    root = ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-CPOS', '_', 0, 'rroot', '_', '_')
    root.idChars =[2,4,3]
    tokens = [root]

    for line in open(conll_path, 'r'):
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            if len(tokens)>1:
                wordsCount.update([node.norm for node in tokens if isinstance(node, ConllEntry)])
                posCount.update([node.pos for node in tokens if isinstance(node, ConllEntry)])
                xposCount.update([node.xpos for node in tokens if isinstance(node, ConllEntry)])
                relCount.update([node.relation for node in tokens if isinstance(node, ConllEntry)])
            tokens = [root]
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                tokens.append(line.strip())
            else:
                entry = ConllEntry(int(tok[0]), tok[1], tok[2], tok[3], tok[4], tok[5], int(tok[6]) if tok[6] != '_' else 0, tok[7], tok[8], tok[9])

                if entry.norm == 'NUM':
                    entry.idChars = [2,1,3]
                else:
                    chars_of_word = [2]

                    for char in tok[1].lower():
                        if char not in c2i:
                            c2i[char] = len(c2i)
                        chars_of_word.append(c2i[char])
                    chars_of_word.append(3)
                    entry.idChars = chars_of_word

                tokens.append(entry)


    if len(tokens) > 1:
        wordsCount.update([node.norm for node in tokens if isinstance(node, ConllEntry)])
        posCount.update([node.pos for node in tokens if isinstance(node, ConllEntry)])
        xposCount.update([node.xpos for node in tokens if isinstance(node, ConllEntry)])
        relCount.update([node.relation for node in tokens if isinstance(node, ConllEntry)])

    #Keep words that appears at least twice in the trainning corpus
    wordsCount = {k: v for k, v in wordsCount.items() if v > 1}

    return (wordsCount, {w: i for i, w in enumerate(list(wordsCount.keys()))}, c2i, list(posCount.keys()),list(xposCount.keys()), list(relCount.keys()))

def ext_vocab(conll_path,ext_emb_file):

    ext_voc=  pickle.load( open( ext_emb_file, "rb" ) )
    wordsCount = Counter()
    root = ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-CPOS', '_', 0, 'rroot', '_', '_')
    tokens = [root]

    for line in open(conll_path, 'r'):
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            if len(tokens)>1:
                wordsCount.update([node.norm for node in tokens if isinstance(node, ConllEntry)])
            tokens = [root]
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                tokens.append(line.strip())
            else:
                entry = ConllEntry(int(tok[0]), tok[1], tok[2], tok[3], tok[4], tok[5], int(tok[6]) if tok[6] != '_' else 0, tok[7], tok[8], tok[9])
                tokens.append(entry)


    if len(tokens) > 1:
        wordsCount.update([node.norm for node in tokens if isinstance(node, ConllEntry)])

    #Keep only words that actually have external embedding
    wordsCount = {w: c for w,c in wordsCount.items() if (ext_voc.get(w,0) !=0) }

    return ({w: i for i, w in enumerate(wordsCount.keys())})


def read_conll(fh,c2i):
    #Character vocabulary
    root = ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-CPOS', '_', 0, 'rroot', '_', '_')
    root.idChars =[2,4,3]
    tokens = [root]


    for line in fh:
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            if len(tokens)>1: yield tokens
            tokens = [root]
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                tokens.append(line.strip())
            else:
                entry = ConllEntry(int(tok[0]), tok[1], tok[2], tok[3], tok[4], tok[5], int(tok[6]) if tok[6] != '_' else 0, tok[7], tok[8], tok[9])

                if entry.norm == 'NUM':
                    entry.idChars = [2,1,3]
                else:

                    chars_of_word = [2]
                    for char in tok[1].lower():
                        if char in c2i:
                            chars_of_word.append(c2i[char])
                        else:
                            chars_of_word.append(0)
                    chars_of_word.append(3)
                    entry.idChars = chars_of_word

                tokens.append(entry)


    if len(tokens) > 1:
        yield tokens


def write_conll(fn, conll_gen):
    with open(fn, 'w') as fh:
        for sentence in conll_gen:
            for entry in sentence[1:]:
                fh.write(str(entry) + '\n')
            fh.write('\n')


numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");
def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()



def batch_train_data(conll_path,c2i,tokens_size=5000):

    #Batch train data. Each mini-batch has approximately 5000 tokens(default)
    #The size of minibatches is adjustable by the tokens_size parameter
    with open(conll_path, 'r') as conllFP:
        trainData = list(read_conll(conllFP, c2i))

    conll_sentences = []
    for sentence in trainData:
        conll_sentence = [entry for entry in sentence  if isinstance(entry, ConllEntry)]
        conll_sentences.append(conll_sentence)

    conll_sentences.sort(key=lambda x: -len(x))
    train_batches = []
    tokens = 0
    start_idx = 0
    for idx,sent in enumerate(conll_sentences):
        tokens+=len(sent)
        if ((tokens -tokens_size) > 0):
            train_batches.append((start_idx,idx+1))
            start_idx = idx+1
            tokens = 0

    return conll_sentences,train_batches
