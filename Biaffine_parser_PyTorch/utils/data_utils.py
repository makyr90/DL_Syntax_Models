import pandas as pd
from ast import literal_eval

def conll_to_pd(conll_path):

    with open(conll_path, 'r') as conllFP:

        sentences = []
        tokens = []
        pos = []
        xpos = []
        head_id = []
        rel = []

        for line in conllFP:
            tok = line.strip().split('\t')
            if not tok or line.strip() == '':

                if len(tokens)>0:
                    assert len(tokens) == len(pos) == len(xpos) == len(head_id) == len(rel), "Error tokens,pos,xpos,head_id,rel lists should have the same size"
                    sentences.append([tokens,pos,xpos,head_id,rel])
                tokens = []
                pos = []
                xpos = []
                head_id = []
                rel = []
            else:
                if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                    continue
                else:
                    tokens.append(tok[1])
                    pos.append(tok[3].upper())
                    xpos.append(tok[4].upper())
                    head_id.append(int(tok[6]) if tok[6] != '_' else 0)
                    rel.append(tok[7])


        if len(tokens) > 0:
            sentences.append([tokens,pos,xpos,head_id,rel])

        df = pd.DataFrame(sentences, columns=["Text","POS","XPOS","Head_id","Drel"])

        return df

class ConllEntry:

    def __init__(self, id, form, lemma, pos, xpos, feats=None, parent_id=None, relation=None, deps=None, misc=None):
      self.id = id
      self.form = form
      self.norm = form
      self.xpos = xpos.upper()
      self.pos = pos.upper()
      self.parent_id = parent_id
      self.relation = relation

      self.lemma = lemma
      self.feats = feats
      self.deps = deps
      self.misc = misc


    def __str__(self):
      values = [str(self.id), self.form, self.lemma, self.pos, self.xpos, self.feats, str(self.parent_id) if self.parent_id is not None else None, self.relation, self.deps, self.misc]
      return '\t'.join(['_' if v is None else v for v in values])

def pd_to_conll(csv_path):

    df = pd.read_csv(csv_path)
    sentences = []
    for index, row in df.iterrows():
        sentence = []
        text = literal_eval(row["Text"])
        pos = literal_eval(row["POS"])
        xpos = literal_eval(row["XPOS"])
        head_id = literal_eval(row["Head_id"])
        rel = literal_eval(row["Drel"])
        for idx,(t,p,x,h,r) in enumerate(zip(text,pos,xpos,head_id,rel)):
            entry = ConllEntry(idx+1, t, None, p, x, feats=None, parent_id=h, relation=r, deps=None, misc=None)
            sentence.append(entry)
        sentences.append(sentence)

    return sentences


def write_conll(fn, conll_gen):
    with open(fn, 'w') as fh:
        for sentence in conll_gen:
            for entry in sentence:
                fh.write(str(entry) + '\n')
            fh.write('\n')

def write_csv(fn, df):
    df.to_csv(fn,index=False)
