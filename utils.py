from collections import Counter
import re, pickle
import random


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
    c2i["<w>"] = 1   # word start
    c2i["</w>"] = 2  # word end index
    c2i["NUM"] = 3
    c2i["PAD"] = 4

    # root = ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-CPOS', '_', -1,'rroot', '_', '_')
    # root.idChars = [1,2]
    tokens = []

    for line in open(conll_path, 'r'):
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            if len(tokens)>0:
                wordsCount.update([node.norm for node in tokens if isinstance(node, ConllEntry)])
                posCount.update([node.pos for node in tokens if isinstance(node, ConllEntry)])
                xposCount.update([node.xpos for node in tokens if isinstance(node, ConllEntry)])
                relCount.update([node.relation for node in tokens if isinstance(node, ConllEntry)])
            tokens = []
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                tokens.append(line.strip())
            else:
                entry = ConllEntry(int(tok[0]), tok[1], tok[2], tok[3], tok[4], tok[5], int(tok[6]) if tok[6] != '_' else 0, tok[7], tok[8], tok[9])

                chars_of_word = [1]

                for char in tok[1]:
                    if char not in c2i:
                        c2i[char] = len(c2i)
                    chars_of_word.append(c2i[char])
                chars_of_word.append(2)
                entry.idChars = chars_of_word

                tokens.append(entry)


    if len(tokens) > 0:
        wordsCount.update([node.norm for node in tokens if isinstance(node, ConllEntry)])
        posCount.update([node.pos for node in tokens if isinstance(node, ConllEntry)])
        xposCount.update([node.xpos for node in tokens if isinstance(node, ConllEntry)])
        relCount.update([node.relation for node in tokens if isinstance(node, ConllEntry)])

    #Keep words that appears at least 3 times in the trainning corpus
    wordsCount = {k: v for k, v in wordsCount.items() if v > 1}

    return (wordsCount, {w: i for i, w in enumerate(list(wordsCount.keys()))}, c2i, list(posCount.keys()),list(xposCount.keys()), list(relCount.keys()))

def ext_vocab(conll_path,ext_emb_file):

    ext_voc=  pickle.load( open( ext_emb_file, "rb" ) )
    wordsCount = Counter()
    #root = ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-CPOS', '_', -1, 'rroot', '_', '_')
    tokens = []

    for line in open(conll_path, 'r'):
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            if len(tokens)>0:
                wordsCount.update([node.norm for node in tokens if isinstance(node, ConllEntry)])
            tokens = []
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                tokens.append(line.strip())
            else:
                entry = ConllEntry(int(tok[0]), tok[1], tok[2], tok[3], tok[4], tok[5], int(tok[6]) if tok[6] != '_' else 0, tok[7], tok[8], tok[9])
                tokens.append(entry)


    if len(tokens) > 0:
        wordsCount.update([node.norm for node in tokens if isinstance(node, ConllEntry)])

    #Keep only words that actually have external embedding
    wordsCount = {w: c for w,c in wordsCount.items() if (ext_voc.get(w,-1) !=-1) }

    return ({w: i for i, w in enumerate(wordsCount.keys())})


def read_conll(fh,c2i):
    #Character vocabulary
    # root = ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-CPOS', '_', -1, 'rroot', '_', '_')
    # root.idChars =[1,2]
    tokens = []


    for line in fh:
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            if len(tokens)>0: yield tokens
            tokens = []
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                tokens.append(line.strip())
            else:
                entry = ConllEntry(int(tok[0]), tok[1], tok[2], tok[3], tok[4], tok[5], int(tok[6]) if tok[6] != '_' else 0, tok[7], tok[8], tok[9])

                chars_of_word = [1]
                for char in tok[1]:
                    if char in c2i:
                        chars_of_word.append(c2i[char])
                    else:
                        chars_of_word.append(0)
                chars_of_word.append(2)
                entry.idChars = chars_of_word

                tokens.append(entry)


    if len(tokens) > 0:
        yield tokens


def write_conll(fn, conll_gen):
    with open(fn, 'w') as fh:
        for sentence in conll_gen:
            for entry in sentence:
                fh.write(str(entry) + '\n')
            fh.write('\n')


#numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");
def normalize(word):
    return  word.lower()



def batch_data(conll_path,c2i,tokens_size,train):

    #Batch train/dev/test data. Each mini-batch has approximately 5000 tokens(default)
    #The size of minibatches is adjustable by the tokens_size parameter
    with open(conll_path, 'r') as conllFP:
        data = list(read_conll(conllFP, c2i))

    conll_sentences = []
    for sentence in data:
        conll_sentence = [entry for entry in sentence  if isinstance(entry, ConllEntry)]
        conll_sentences.append(conll_sentence)

    if train:
        conll_sentences.sort(key=lambda x: -len(x))
        conll_sentences = list(filter(lambda x: len(x)<=80,conll_sentences))
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
