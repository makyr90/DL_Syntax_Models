import numpy as np
import sys
import pickle

def file_len(fname):
    with open(fname) as f:
        for idx, line in enumerate(f):
            pass
    return idx

rows =  file_len(sys.argv[1])

fh=open(sys.argv[1],'r')
foutname=sys.argv[2]
dim = int(sys.argv[3])

vocab={}
wvecs = np.zeros(rows*dim).reshape(rows,dim)
for i,line in enumerate(fh):
    line = line.strip().split(' ')
    values = line[1:]
       #if (len(values)==300):
    vocab[line[0].lower()] = i-1
    wvecs[i-1,:] = values
    if (i % 5000 == 0):
        print(i,line[0])

np.save(foutname+".npy",wvecs)
pickle.dump( vocab, open( foutname+".vocab", "wb" ) )
