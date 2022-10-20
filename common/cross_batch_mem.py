import torch

def cdist2(A, B):
    return  (A.pow(2).sum(1, keepdim = True)
             - 2 * torch.mm(A, B.t())
             + B.pow(2).sum(1, keepdim = True).t())

def top_dist_idx(A, B, k):
    return cdist2(A, B).topk(k, dim=1, largest=False, sorted=True)[1]  #  0 value,  1 index

def top_dist_value(A, B, k):
    return cdist2(A, B).topk(k, dim=1, largest=False, sorted=True)[0] 

class XBM:
    def __init__(self, size=4096 ,dim = 256, device = torch.device('cpu')):
        self.K = size
        self.feat_dim = dim
        self.feats = torch.zeros(self.K, dim).to(device)
        self.targets = (torch.zeros(self.K)-1).long().to(device)
        self.ptr = 0
        print('XBM size=',size, ' dim=',dim, ' cuda:',device)
    @property
    def is_full(self):
        return self.targets[-1].item() != -1

    def get(self):
        if self.is_full:
            return self.feats, self.targets
        else:
            return self.feats[:self.ptr], self.targets[:self.ptr]

    def enqueue_dequeue(self, feats, targets):
        q_size = len(targets)
        if self.ptr + q_size > self.K:
            self.feats[-q_size:] = feats
            self.targets[-q_size:] = targets
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.ptr += q_size


class XBM_isclean:
    def __init__(self, size=4096 ,dim = 256, device = torch.device('cpu')):
        self.K = size
        self.feat_dim = dim
        self.feats = torch.zeros(self.K, dim).to(device)
        self.targets = (torch.zeros(self.K)-1).long().to(device)
        self.isclean = (torch.zeros(self.K)).bool().to(device)
        self.groundtruth = (torch.zeros(self.K)-1).long().to(device)
        self.ptr = 0
        print('XBM size=',size, ' dim=',dim, ' cuda:',device)
    @property
    def is_full(self):
        return self.targets[-1].item() != -1

    def get(self):
        if self.is_full:
            return self.feats, self.targets, self.isclean, self.groundtruth
        else:
            return self.feats[:self.ptr], self.targets[:self.ptr], self.isclean[:self.ptr], self.groundtruth[:self.ptr]

    def enqueue_dequeue(self, feats, targets, isclean, groundtruth):
        q_size = len(targets)
        if self.ptr + q_size > self.K:
            self.feats[-q_size:] = feats
            self.targets[-q_size:] = targets
            self.isclean[-q_size:] = isclean
            self.groundtruth[-q_size:] = groundtruth
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.isclean[self.ptr: self.ptr + q_size] = isclean
            self.groundtruth[self.ptr: self.ptr + q_size] = groundtruth
            self.ptr += q_size


class XBM_truth:
    def __init__(self, size=4096 ,dim = 256, device = torch.device('cpu')):
        self.K = size
        self.feat_dim = dim
        self.feats = torch.zeros(self.K, dim).to(device)
        self.targets = (torch.zeros(self.K)-1).long().to(device)
        self.truth = (torch.zeros(self.K)-1).long().to(device)
        self.ptr = 0
        print('XBM size=',size, ' dim=',dim, ' cuda:',device)
    @property
    def is_full(self):
        return self.targets[-1].item() != -1

    def get(self):
        if self.is_full:
            return self.feats, self.targets, self.truth
        else:
            return self.feats[:self.ptr], self.targets[:self.ptr], self.truth[:self.ptr]

    def enqueue_dequeue(self, feats, targets, truth):
        q_size = len(targets)
        if self.ptr + q_size > self.K:
            self.feats[-q_size:] = feats
            self.targets[-q_size:] = targets
            self.truth[-q_size:] = truth
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.truth[self.ptr: self.ptr + q_size] = truth
            self.ptr += q_size
            
# total_loss = raw_loss +  concat_loss + partcls_loss  #  rank_loss   tripletloss +
# xbm.enqueue_dequeue(concat_out.detach(), label.detach())
# if epoch > 1:
#     xbm_feats, xbm_targets = xbm.get()
#     tripletloss = 10*Tripletloss(concat_out, label, xbm_feats, xbm_targets)
#     t_triplet_loss += tripletloss.item() *batch_size                  # ======================
#     total_loss = total_loss +  tripletloss


