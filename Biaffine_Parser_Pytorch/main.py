import time
import os
import sys
import argparse
import pickle
import torch
import torch.nn as nn
import numpy as np
from vocab import  VocabBuilder,TagBuilder,CharBuilder
from model import RNN
from dataloader import ClassDataLoader
from util import adjust_learning_rate


np.random.seed(1990)
torch.manual_seed(1990)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--steps', default=30000, type=int, metavar='N', help='number of total trainninf steps (updates) to run')
parser.add_argument('--epochs', default=3000, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--wembedding-size', default=100, type=int, metavar='N', help='trainable word embedding size')
parser.add_argument('--posembedding-size', default=100, type=int, metavar='N', help='trainable pos/xpos embedding size')
parser.add_argument('--cembedding-size', default=100, type=int, metavar='N', help='trainable character embedding size')
parser.add_argument('--hidden-size', default=200, type=int, metavar='N', help='rnn hidden size')
parser.add_argument('--char-hidden-size', default=400, type=int, metavar='N', help='rnn hidden size')
parser.add_argument('--layers', default=3, type=int, metavar='N', help='number of rnn layers')
parser.add_argument('--mlp-arc-size', default=400, type=int, metavar='N', help='size of pos mlp hidden layer')
parser.add_argument('--mlp-label-size', default=100, type=int, metavar='N', help='size of xpos mlp hidden layer')
parser.add_argument('--dropout', default=0.33, type=float, metavar='drp', help='dropout probability')
parser.add_argument('--var-dropout', default=0.33,type=float, metavar='vdrp', help='reccurent dropout probability')
parser.add_argument('--min-samples', default=2, type=int, metavar='N', help='min number of tokens')
parser.add_argument('--cuda', default=False, action='store_true', help='use cuda')
parser.add_argument('--fasttext-tensor', default='data/fasttext.pt', help='path to fasttext embeddings tensor')
parser.add_argument('--fasttext-voc', default='data/fasttext_voc.pkl', help='path to fasttext embeddings tensor')
parser.add_argument('--train-path', default="data/en-ud-train.csv", help='path to train data csv')
parser.add_argument('--dev-path', default="data/en-ud-dev.csv", help='path to dev data csv')
parser.add_argument('--clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

print()
# create vocab
print("===> creating word, tag, char, dep_rel vocabs and loading pre-trained embeddings ...")

start = time.time()
fasttext_embed = torch.load(args.fasttext_tensor)
fasttext_word_to_index = pickle.load(open(args.fasttext_voc, 'rb'))
w_builder = VocabBuilder(path_file=args.train_path)
word_to_index, words = w_builder.get_word_index(min_sample=args.min_samples)
char_builder  = CharBuilder(path_file=args.train_path)
char_to_index, chars = char_builder.get_char_index()
pos_builder = TagBuilder(args.train_path,"POS")
pos_to_index, pos_tags = pos_builder.get_tag_index_padded()
xpos_builder = TagBuilder(args.train_path,"XPOS")
xpos_to_index, xpos_tags = xpos_builder.get_tag_index_padded()
rel_builder = TagBuilder(args.train_path,"Drel")
rel_to_index, rel_tags = rel_builder.get_tag_index()

if not os.path.exists('gen'):
    os.mkdir('gen')
with open("gen/parser_model.params", 'wb') as paramsfp:
    pickle.dump((word_to_index,char_to_index,pos_to_index,xpos_to_index,rel_to_index), paramsfp)
print('===> vocab creating in: {t:.3f}s'.format(t=time.time()-start))


print("===> creating dataloaders ...")
end = time.time()
train_loader = ClassDataLoader(args.train_path,word_to_index,fasttext_word_to_index,char_to_index,pos_to_index,xpos_to_index,rel_to_index,args.batch_size,predict_flag=0,train=1)
val_loader = ClassDataLoader(args.dev_path, word_to_index,fasttext_word_to_index,char_to_index,pos_to_index,xpos_to_index,rel_to_index,args.batch_size,predict_flag=0,train=0)
print('===> dataloaders creatinng in: {t:.3f}s'.format(t=time.time()-end))


#create model
print("===> creating rnn model ...")
model = RNN(word_to_index,fasttext_word_to_index,char_to_index,args.cembedding_size,args.posembedding_size,args.char_hidden_size, args.wembedding_size, fasttext_embed, args.layers, args.hidden_size,
            args.dropout,args.var_dropout, args.mlp_arc_size, args.mlp_label_size,pos_to_index, xpos_to_index,rel_to_index,args.cuda, batch_first=True)
print(model)

if args.cuda:
    model.cuda()


#optimizer and losses
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.param_group_dense), lr=args.lr, betas=(0.9, 0.9), eps=1e-12)
optimizer_sparse = torch.optim.SparseAdam(filter(lambda p: p.requires_grad, model.param_group_sparse), lr=args.lr,  betas=(0.9, 0.9), eps=1e-12)
criterion_arc = nn.CrossEntropyLoss(ignore_index=-1) # ignore PADDED targets
criterion_label = nn.CrossEntropyLoss(ignore_index=model.rel_to_index['__PADDING__']) # ignore PADDED targets


def test(val_loader, model):

    # switch to evaluate mode
    model.eval()
    gold_arcs = np.array([])
    pred_arcs = np.array([])
    gold_labels = np.array([])
    pred_labels = np.array([])

    for i, (word_tensor, ext_word_ids,char_ids,pos_tensor,xpos_tensor,head_targets,rel_targets,seq_lengths,perm_idx) in enumerate(val_loader):

        start = time.time()
        # switch to evaluation mode
        model.eval()
        if args.cuda:
            word_tensor = word_tensor.cuda()
            pos_tensor = pos_tensor.cuda()
            xpos_tensor = xpos_tensor.cuda()
            head_targets = head_targets.cuda()
            rel_targets = rel_targets.cuda()

        # compute output
        arc_logits,label_logits = model(word_tensor,ext_word_ids,char_ids,pos_tensor,xpos_tensor,seq_lengths)
        arc_logits = arc_logits[:,1:,:]
        label_logits = label_logits[:,1:,:,:]
        batch_size, src_len = word_tensor.size()
        head_targets = head_targets.cpu().detach().numpy()
        rel_targets = rel_targets.cpu().detach().numpy()

        s_arc_scores, s_arc_indices = torch.max(arc_logits, 2)
        label_logits = label_logits.contiguous().view(batch_size*src_len,src_len+1,len(model.rel_to_index)) #[bs*src_len,src_len+1,labels]
        label_logits = label_logits.gather(1, s_arc_indices.view(-1, 1, 1).expand(label_logits.size(0), 1, label_logits.size(2))).squeeze(1).view(batch_size,src_len,-1)
        label_logits = label_logits.cpu().detach().numpy()
        label_logits = np.argmax(label_logits,axis=2)
        s_arc_scores =  s_arc_indices.cpu().detach().numpy()

        for idx in range(len(seq_lengths)):
            gold_arcs = np.concatenate((gold_arcs,head_targets[idx,:seq_lengths[idx]]),axis=0)
            pred_arcs = np.concatenate((pred_arcs,s_arc_scores[idx,:seq_lengths[idx]]),axis=0)
            gold_labels = np.concatenate((gold_labels,rel_targets[idx,:seq_lengths[idx]]),axis=0)
            pred_labels = np.concatenate((pred_labels,label_logits[idx,:seq_lengths[idx]]),axis=0)

    arcs_acc = np.mean(gold_arcs==pred_arcs)
    labls_acc  = np.mean(np.logical_and(gold_arcs==pred_arcs, gold_labels==pred_labels))

    return arcs_acc,labls_acc


highestScore = 0
tsid = 0
name_model = 'parser_model2.pt'
path_save_model = os.path.join('gen', name_model)
for epoch in range(1, args.epochs+1):

    for i, (word_tensor, ext_word_ids,char_ids,pos_tensor,xpos_tensor,head_targets,rel_targets,seq_lengths,perm_idx) in enumerate(train_loader):

        start = time.time()
        # switch to train mode
        model.train()
        ts = (((epoch -1) * train_loader.n_batches) + (i+1))
        if (ts%5000 == 0):
            adjust_learning_rate(args.lr, optimizer,optimizer_sparse)

        if args.cuda:
            word_tensor = word_tensor.cuda()
            pos_tensor = pos_tensor.cuda()
            xpos_tensor = xpos_tensor.cuda()
            head_targets = head_targets.cuda()
            rel_targets = rel_targets.cuda()

        # compute output
        arc_logits,label_logits = model(word_tensor,ext_word_ids,char_ids,pos_tensor,xpos_tensor,seq_lengths)
        arc_logits = arc_logits[:,1:,:]
        label_logits = label_logits[:,1:,:,:]
        head_targets = head_targets.view(-1)
        rel_targets = rel_targets.view(-1)
        s_arc_scores, s_arc_indices = torch.max(arc_logits, 2)
        arc_logits = arc_logits.contiguous().view(-1,word_tensor.size()[1]+1)
        arc_loss = criterion_arc(arc_logits,head_targets)
        label_logits = label_logits.contiguous().view(word_tensor.size()[0]*word_tensor.size()[1],word_tensor.size()[1]+1,len(model.rel_to_index)) #[bs*src_len,src_len+1,labels]
        label_logits = label_logits.gather(1, s_arc_indices.view(-1, 1, 1).expand(label_logits.size(0), 1, label_logits.size(2))).squeeze(1)
        label_loss = criterion_label(label_logits,rel_targets)

        loss = arc_loss + label_loss
        optimizer_sparse.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.param_group_dense, args.clip)
        nn.utils.clip_grad_value_(model.param_group_sparse, args.clip)
        optimizer.step()
        optimizer_sparse.step()
        print("Finish training step: %i, Avg batch loss= %.4f, time= %.2fs" % (ts,loss.data.cpu().numpy().tolist() , time.time() - start))

        if (ts<=1000):
            if (ts%100== 0):
                print("Save Model...")
                torch.save({'t_step': ts,'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),'optimizer_sparse_state_dict': optimizer_sparse.state_dict()}, path_save_model)

        else:
            print("Performance on Dev data")
            start = time.time()
            arcs_acc,labls_acc = test(val_loader, model)
            print("Finish predictions on dev data in %.2fs" %  (time.time() - start))
            print("---\nUAS accuracy:\t%.2f" % (float(arcs_acc) * 100 ))
            print("---\nLAS accuracy:\t%.2f" % (float(labls_acc) * 100 ))
            print("------------------------------------------------------------------")

            score = (float(labls_acc) * 100)
            if score > highestScore:
                print("Save Model...")
                torch.save({'t_step': ts,'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),'optimizer_sparse_state_dict': optimizer_sparse.state_dict()}, path_save_model)
                highestScore = score
                tsid = ts

            print("Highest LAS(@Dev): %.2f at trainning step %d" % (highestScore,tsid))
            print("-------------------------------------------------------------------")
            if ((ts - tsid) > 5000):
                print("Model trainning finish..")
                print("Model  didn't improve during the last 5000 trainning steps")
                sys.exit()
            elif (ts > args.steps):
                print("Model trainning finish..")
                sys.exit()
