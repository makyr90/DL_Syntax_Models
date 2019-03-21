import time
import os
import argparse
import pickle
import torch
import numpy as np
from model import RNN
import pandas as pd
from dataloader import ClassDataLoader
from data_utils import pd_to_conll,write_conll
from Edmonds_decoder import parse_proj

np.random.seed(1990)
torch.manual_seed(1990)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--wembedding-size', default=100, type=int, metavar='N', help='trainable word embedding size')
parser.add_argument('--posembedding-size', default=100, type=int, metavar='N', help='trainable pos/xpos embedding size')
parser.add_argument('--cembedding-size', default=100, type=int, metavar='N', help='trainable character embedding size')
parser.add_argument('--hidden-size', default=200, type=int, metavar='N', help='rnn hidden size')
parser.add_argument('--char-hidden-size', default=400, type=int, metavar='N', help='rnn hidden size')
parser.add_argument('--layers', default=3, type=int, metavar='N', help='number of rnn layers')
parser.add_argument('--mlp-arc-size', default=400, type=int, metavar='N', help='size of pos mlp hidden layer')
parser.add_argument('--mlp-label-size', default=100, type=int, metavar='N', help='size of xpos mlp hidden layer')
parser.add_argument('--cuda', default=False, action='store_true', help='use cuda')
parser.add_argument('--model-params', default='gen/parser_model.params', help='path to pickle file with model params')
parser.add_argument('--model', default='gen/parser_model.pt', help='path to file with model weights')
parser.add_argument('--fasttext-tensor', default='data/fasttext.pt', help='path to fasttext embeddings tensor')
parser.add_argument('--print-freq', default=10, help='print frequency' )
parser.add_argument('--fasttext-voc', default='data/fasttext_voc.pkl', help='path to fasttext embeddings tensor')
parser.add_argument('--test-path', default="data/en-ud-test.csv", help='path to dev data csv')
args = parser.parse_args()

print()
print("===> loading word, tag, char, dep_rel vocabs and pre-trained embeddings ...")

start = time.time()
fasttext_embed = torch.load(args.fasttext_tensor)
fasttext_word_to_index = pickle.load(open(args.fasttext_voc, 'rb'))

with open(args.model_params, 'rb') as paramsfp:
    word_to_index,char_to_index,pos_to_index,xpos_to_index,rel_to_index = pickle.load(paramsfp)

index_to_rel = {ind: rel for rel, ind in rel_to_index.items()}

print("===> creating dataloader ...")
end = time.time()
test_loader = ClassDataLoader(args.test_path, word_to_index,fasttext_word_to_index,char_to_index,pos_to_index,xpos_to_index,rel_to_index,2*args.batch_size,predict_flag=1,train=0)
print('===> dataloader creatinng in: {t:.3f}s'.format(t=time.time()-end))


#create model
print("===> creating rnn model ...")
model = RNN(word_to_index,fasttext_word_to_index,char_to_index,args.cembedding_size,args.posembedding_size,args.char_hidden_size, args.wembedding_size, fasttext_embed, args.layers, args.hidden_size,
            0,0, args.mlp_arc_size, args.mlp_label_size,pos_to_index, xpos_to_index,rel_to_index,args.cuda, batch_first=True)
print(model)

#load model
print("===> Loading pretrained model ...")
if args.cuda:
    checkpoint = torch.load(args.model)
else:
    checkpoint = torch.load(args.model,map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])


if args.cuda:
    model.cuda()

# switch to evaluate mode
model.eval()

arcs_preds = []
labels_preds = []
start = time.time()

for i, (word_tensor, ext_word_ids,char_ids,pos_tensor,xpos_tensor,head_targets,rel_targets,seq_lengths,perm_idx) in enumerate(test_loader):


    if args.cuda:
        word_tensor = word_tensor.cuda()
        pos_tensor = pos_tensor.cuda()
        xpos_tensor = xpos_tensor.cuda()

    # compute output
    arc_scores,label_scores = model(word_tensor,ext_word_ids,char_ids,pos_tensor,xpos_tensor,seq_lengths)
    batch_size, src_len = word_tensor.size()
    src_len += 1
    label_scores = label_scores.cpu().detach() # [bs,src_len+1,src_len+1,labels]
    arc_scores =  arc_scores.cpu().detach()  #  [bs,src_len+1,src_len+1]

    perm_idx = perm_idx.data.numpy().tolist()
    ordered_arc_scores = np.empty_like(arc_scores)
    ordered_label_scores = np.empty_like(label_scores)
    s_lengths = seq_lengths.cpu().numpy().tolist()
    ordered_s_lengths = s_lengths.copy()
    for idx,perm in enumerate(perm_idx):
        ordered_arc_scores[perm,:,:] = arc_scores[idx,:,:]
        ordered_label_scores[perm,:,:,:] = label_scores[idx,:,:,:]
        ordered_s_lengths[perm] = s_lengths[idx]

    for idx in range(len(s_lengths)):
        arc_scores_p = ordered_arc_scores[idx,:ordered_s_lengths[idx]+1,:ordered_s_lengths[idx]+1].transpose(1,0) # [src_len+1,src_len+1]
        label_scores_p = ordered_label_scores[idx,:ordered_s_lengths[idx]+1,:ordered_s_lengths[idx]+1,:] # [src_len+1,src_len+1,labels]
        pred_heads = parse_proj(arc_scores_p)
        pred_labels = [np.argmax(labels[head]) for head, labels in zip(pred_heads,label_scores_p)]
        pred_labels = [index_to_rel[label] for label in pred_labels] # Map label indexes to relation tags
        arcs_preds.append(pred_heads[1:])
        labels_preds.append(pred_labels[1:])

    if (((i+1) % args.print_freq) == 0):
        print('Test: [{0}/{1}] elapsed time: {t:.3f}s'.format(i+1, len(test_loader),t=(time.time()-start)))

#Read csv input file to pd overwrite pos and xpos
test_data = pd.read_csv(args.test_path)

for index, row in test_data.iterrows():

    test_data.at[index,'Head_id'] = arcs_preds[index]
    test_data.at[index,'Drel'] = labels_preds[index]

filename, file_extension = os.path.splitext(args.test_path)
outout_csv = filename+"_pred.csv"
output_conllu = filename+"_pred.conllu"
test_data.to_csv(outout_csv,index=False)
conll_sentences = pd_to_conll(outout_csv)
write_conll(output_conllu,conll_sentences)
