import torch
import random

''' from https://github.com/pytorch/examples/blob/master/imagenet/main.py'''
class AverageMeter(object):

    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def adjust_learning_rate(lr, optimizer,optimizer_sparse):

    """Decay LR """
    lr = lr * (0.75)
    print("New larning rate = ",lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer_sparse.param_groups:
        param_group['lr'] = lr

def drop_word_pos_embeds(batch_size,seq_lengths,w_size,p_size,dropout,cuda_flag):

    """
    Independently dropout word & pos embeddings. If both are dropped replace with zeros.
    If only one is dropped scale the other to compensate. Otherwise keep both
    """

    if cuda_flag:
        word_drp_tensor = torch.zeros((batch_size, seq_lengths.max())).cuda()
        pos_drp_tensor = torch.zeros((batch_size, seq_lengths.max())).cuda()
    else:
        word_drp_tensor = torch.zeros((batch_size, seq_lengths.max()))
        pos_drp_tensor = torch.zeros((batch_size, seq_lengths.max()))

    seq_lengths = seq_lengths.cpu().numpy()
    for idx in range(batch_size):
        for jdx in range(seq_lengths[idx]):
            wemb_Dropflag = random.random() < dropout
            posemb_Dropflag = random.random() < dropout
            if (wemb_Dropflag and posemb_Dropflag):
                word_drp_tensor[idx,jdx] = 0
                pos_drp_tensor[idx,jdx] = 0
            elif wemb_Dropflag:
                word_drp_tensor[idx,jdx] = 0
                pos_drp_tensor[idx,jdx] = (1/ (1 - (float(w_size) / (w_size+p_size))))
            elif posemb_Dropflag:
                word_drp_tensor[idx,jdx] = (1/ (1 - (float(p_size) / (w_size+p_size))))
                pos_drp_tensor[idx,jdx] = 0
            else:
                word_drp_tensor[idx,jdx] = 1
                pos_drp_tensor[idx,jdx] = 1

    word_drp_tensor = word_drp_tensor.unsqueeze(2).repeat(1,1,w_size)
    pos_drp_tensor = pos_drp_tensor.unsqueeze(2).repeat(1,1,p_size)

    return word_drp_tensor,pos_drp_tensor
