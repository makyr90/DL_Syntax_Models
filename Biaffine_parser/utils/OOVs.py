import sys
import utils
#Find % of OOVs on dev or test dataset
#Usage python OOVs.py path_to_train path_to_dev/test
train =  sys.argv[1]
dev_test = sys.argv[2]
words, w2i, c2i, pos, rels = utils.vocab(train)
words_dev, w2i_dev, c2i_dev, pos_dev, rels_dev = utils.vocab(dev_test)

OOVs = 0
for k,v in words_dev.items():
	if not(k in words.keys()):
		OOVs += 1

print str(format(float(OOVs)/(len(words_dev)) *100, '.2f'))+"% OOVs on test/dev dataset"