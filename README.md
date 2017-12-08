# Dozat et al. model for dependency parsing implemented in DyNet Framework using Python3

## Best Results so far(greek dev data):

- **UAS: 88.21**  

- **LAS: 85.75**

## Results using Dist/dir embeddings(64-D,[-15,15]).
###Just Add to arc and arc_label biaffine formulas the term W^T * Dist_emb[head_i,dep_j]

- **UAS: 88.68**  

- **LAS: 86.48**
