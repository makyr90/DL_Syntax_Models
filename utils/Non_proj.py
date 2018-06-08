import utils
from utils import read_conll
import sys

'''Percentage of non-projective sentences of the given dataset.
Usage: python Non_proj.py path_to_dataset
'''

def non_proj_sent(sentence):
	#Return true if sentence is non-projective
	id_head = {}
	for entry in sentence:
 		id_head[entry.id] = entry.parent_id
 	for k,v in id_head.items():
 		if (k < v):
 	 		spann = range(k,v+1)
 	 		nodes = range(k+1,v)
 	 	else:
 	 		spann = range(v,k+1)
 	 		nodes = range(v+1,k)
 	 	for node in nodes:
 	 		if (not(id_head[node] in spann)):
 	 			return True 
 	return False
 	 	



count = 0
count_nproj = 0
with open(sys.argv[1], 'r') as conllFP:
	Data = list(read_conll(conllFP, []))
 	for sent in Data:
 		count+=1
 		conll_sent = [entry for entry in sent if isinstance(entry, utils.ConllEntry)]
 		if (non_proj_sent(conll_sent)):
 			count_nproj +=1


print str(format(float(count_nproj)/count*100, '.2f'))+"%  of total sentences are non-projective"
 	 	
 	 			



