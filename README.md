# Dozat et al. model for dependency parsing implemented in DyNet Framework using Python3

## Vanilla model greek dev data(Accuracy):

- **UAS:** 88.21  

- **LAS:** 85.75

## Vanilla model + Dist/dir embeddings(64D,[-15,15]) greek dev data(Accuracy):
###Just Add to arc and arc_label biaffine formulas the term W^T * Dist_emb[head_i,dep_j]

- **UAS:** 88.68  

- **LAS:** 86.48
