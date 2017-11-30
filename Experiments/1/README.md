## Model without character embeddings

-Batched manually and run on CPU for 584 epochs
-Best model @ epoch 242
-UAS: 86.71
-LAS: 84.01

### Possible problems:

-I have applied independently word/pos embedding dropout more frequent since
i calculated the respective drop flags based on the maximum sentence length for all
 batched sentences. The correct implementation is to calculate the drop flags based
on the length of each individual sentence. For padded words(wid = 2) just replace both embeddings
with zeros. 

-Also  masks must be applied to the loss function in order to turn off traiining
for padding!! Very Important!!
