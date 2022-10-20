"""
    Understanding Image Retrieval Re-Ranking: A Graph Neural Network Perspective

    Xuanmeng Zhang, Minyue Jiang, Zhedong Zheng, Xiao Tan, Errui Ding, Yi Yang

    Project Page : https://github.com/Xuanmeng-Zhang/gnn-re-ranking

    Paper: https://arxiv.org/abs/2012.07620v2

    ======================================================================
   
    On the Market-1501 dataset, we accelerate the re-ranking processing from 89.2s to 9.4ms
    with one K40m GPU, facilitating the real-time post-processing. Similarly, we observe 
    that our method achieves comparable or even better retrieval results on the other four 
    image retrieval benchmarks, i.e., VeRi-776, Oxford-5k, Paris-6k and University-1652, 
    with limited time cost.
"""

from operator import index
from re import X
import torch
import torch.nn.functional as F

def pairwise_distance(x, y):
    m, n = x.size(0), y.size(0)
    
    x = x.view(m, -1)
    y = y.view(n, -1)

    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n,m).t()
    dist.addmm_(mat1=x, mat2=y.t(), beta=1, alpha=-2)

    return dist

def cosine_similarity(x, y):
    m, n = x.size(0), y.size(0)

    x = x.view(m, -1)
    y = y.view(n, -1)

    y = y.t()
    score = torch.mm(x, y)

    return score




def kk_reranking(X_q,X_g,k1,k2,label_g,device):
    query_num, gallery_num = X_q.shape[0], X_g.shape[0]
    ind = pairwise_distance(X_q, X_g).topk(k=k1, dim=-1, largest=False, sorted=True)[1]    # query_num × k1   index  hard negative example
    
    ind_ind = pairwise_distance(X_g[ind.view(-1),:], torch.cat((X_q, X_g), axis = 0)).topk(k=k1, dim=-1, largest=False, sorted=True)[1]    #  (query_num * k1) × (query_num +gallery_num) ===>  (query_num * k1) × k1
    ind_q = torch.arange(query_num).unsqueeze(-1).repeat(1,k1*k1).view(-1,k1).to(device)                #   (query_num * k1) × k1  

    mask = (ind_ind==ind_q).sum(-1).view(query_num, k1)                                 #    query_num × k1
    # print(torch.nn.functional.one_hot(label_g)[ind].size())
    confident_kk_softsum = (torch.nn.functional.one_hot(label_g)[ind].to(device)*mask.unsqueeze(-1)).sum(1).float()
    confident_indgt = confident_kk_softsum.max(1)[0].gt(6)
    

    # mask[mask.sum(-1)==torch.zeros(query_num).to(device),:k2]+=1
        
    # pseudo_label = torch.true_divide((torch.nn.functional.one_hot(label_g)[ind].to(device)*mask.unsqueeze(-1)).sum(1), mask.sum(-1,keepdim=True))

    return confident_kk_softsum ,confident_indgt



def kk_smoothing_acc(X_q, X_g, k1,sigma,label_g,device, num_classes, truth_q):
    query_num, gallery_num = X_q.shape[0], X_g.shape[0]
    ind = pairwise_distance(X_q, X_g).topk(k=k1, dim=-1, largest=False, sorted=True)[1]
    ind_ind = pairwise_distance(X_g[ind.view(-1),:], torch.cat((X_q, X_g), axis = 0)).topk(k=k1, dim=-1, largest=False, sorted=True)[1]
    ind_q = torch.arange(query_num).unsqueeze(-1).repeat(1,k1*k1).view(-1,k1).to(device)
    mask = (ind_ind==ind_q).sum(-1).view(query_num, k1)

    mask[mask.sum(-1)==torch.zeros(query_num).to(device),:k1]+=1
    soft_pseudo_label = torch.true_divide((torch.nn.functional.one_hot(label_g, num_classes)[ind].to(device)*mask.unsqueeze(-1)).sum(1),   mask.sum(-1,keepdim=True))
    max_ind = soft_pseudo_label.topk(k=1, dim=-1, largest=True, sorted=True)[1].view(-1)
    id_ind = soft_pseudo_label.topk(k=1, dim=-1, largest=True, sorted=True)[0].ge(sigma).view(-1)
    if id_ind.sum().item() == 0:
        return torch.zeros(query_num, num_classes).to(device), None, torch.zeros(1), torch.zeros(1)
    softcount = (soft_pseudo_label[torch.arange(query_num), truth_q][id_ind]>0).sum() / (id_ind.sum()) *100
    hardcount =  (max_ind[id_ind]==truth_q[id_ind]).sum() / (id_ind.sum()) *100
    # print( (soft_pseudo_label[torch.arange(query_num), truth_q]>0).size(), (soft_pseudo_label.topk(k=1, dim=-1, largest=True, sorted=True)[1]==truth_q).size() )
    return soft_pseudo_label, id_ind, softcount, hardcount




def kk_smoothing(X_q,X_g,k1,k2,label_g,device, num_classes):
    query_num, gallery_num = X_q.shape[0], X_g.shape[0]
    ind = pairwise_distance(X_q, X_g).topk(k=k1, dim=-1, largest=False, sorted=True)[1]
    ind_ind = pairwise_distance(X_g[ind.view(-1),:], torch.cat((X_q, X_g), axis = 0)).topk(k=k1, dim=-1, largest=False, sorted=True)[1]
    ind_q = torch.arange(query_num).unsqueeze(-1).repeat(1,k1*k1).view(-1,k1).to(device)
    mask = (ind_ind==ind_q).sum(-1).view(query_num, k1)
    mask[mask.sum(-1)==torch.zeros(query_num).to(device),:k2]+=1
    soft_pseudo_label = torch.true_divide((torch.nn.functional.one_hot(label_g, num_classes)[ind].to(device)*mask.unsqueeze(-1)).sum(1),   mask.sum(-1,keepdim=True))
    return soft_pseudo_label



def ktop_pseudo_label(X_q,X_g,k1,k2,label_g,device):
    # query_num, gallery_num = X_q.shape[0], X_g.shape[0]
    ind = pairwise_distance(X_q, X_g).topk(k=k1, dim=-1, largest=False, sorted=True)[1]
    pseudo_label = torch.nn.functional.one_hot(label_g).to(device)[ind].float().mean(1)
    return pseudo_label
     


def kkvsk(X_q, X_g, k1,sigma,label_g,device, num_classes, truth_q):

    query_num, gallery_num = X_q.shape[0], X_g.shape[0]
    ind = pairwise_distance(X_q, X_g).topk(k=k1, dim=-1, largest=False, sorted=True)[1]
    ind_ind = pairwise_distance(X_g[ind.view(-1),:], torch.cat((X_q, X_g), axis = 0)).topk(k=k1, dim=-1, largest=False, sorted=True)[1]
    ind_q = torch.arange(query_num).unsqueeze(-1).repeat(1,k1*k1).view(-1,k1).to(device)
    mask = (ind_ind==ind_q).sum(-1).view(query_num, k1)

    mask[mask.sum(-1)==torch.zeros(query_num).to(device),:k1]+=1
    soft_pseudo_label = torch.true_divide((torch.nn.functional.one_hot(label_g, num_classes)[ind].to(device)*mask.unsqueeze(-1)).sum(1),   mask.sum(-1,keepdim=True))
    max_ind = soft_pseudo_label.topk(k=1, dim=-1, largest=True, sorted=True)[1].view(-1)
    
    id_ind = soft_pseudo_label.topk(k=1, dim=-1, largest=True, sorted=True)[0].ge(sigma).view(-1)
    
    # k_ind = torch.nn.functional.one_hot(label_g, num_classes)[ind].to(device).sum(1).topk(k=1, dim=-1, largest=True, sorted=True)[0].ge(sigma).view(-1)
    
    if id_ind.sum().item() == 0:
        return torch.zeros(query_num, num_classes).to(device), id_ind, max_ind, torch.zeros(1), torch.zeros(1)

    # softcount = (soft_pseudo_label[torch.arange(query_num), truth_q][id_ind]>0).sum() / (id_ind.sum()) *100
    hardcount =  (max_ind[id_ind]==truth_q[id_ind]).sum() / (id_ind.sum()) *100
    # print( (soft_pseudo_label[torch.arange(query_num), truth_q]>0).size(), (soft_pseudo_label.topk(k=1, dim=-1, largest=True, sorted=True)[1]==truth_q).size() )
    k_count = (torch.nn.functional.one_hot(label_g, num_classes)[ind].to(device).sum(1).topk(k=1, dim=-1, largest=True, sorted=True)[1].view(-1) == truth_q).sum() / query_num *100
    
    return soft_pseudo_label, id_ind, max_ind, hardcount, k_count



def kreciprocal(X_q, X_g, k1,sigma,label_g,device, num_classes):
    
    query_num, gallery_num = X_q.shape[0], X_g.shape[0]
    ind = pairwise_distance(X_q, X_g).topk(k=k1, dim=-1, largest=False, sorted=True)[1]
    ind_ind = pairwise_distance(X_g[ind.view(-1),:], torch.cat((X_q, X_g), axis = 0)).topk(k=k1, dim=-1, largest=False, sorted=True)[1]
    ind_q = torch.arange(query_num).unsqueeze(-1).repeat(1,k1*k1).view(-1,k1).to(device)
    mask = (ind_ind==ind_q).sum(-1).view(query_num, k1)

    mask[mask.sum(-1)==torch.zeros(query_num).to(device),:k1]+=1
    soft_pseudo_label = torch.true_divide((torch.nn.functional.one_hot(label_g, num_classes)[ind].to(device)*mask.unsqueeze(-1)).sum(1),   mask.sum(-1,keepdim=True))
    max_ind = soft_pseudo_label.topk(k=1, dim=-1, largest=True, sorted=True)[1].view(-1)
    
    id_ind = soft_pseudo_label.topk(k=1, dim=-1, largest=True, sorted=True)[0].ge(sigma).view(-1)
    
    # k_ind = torch.nn.functional.one_hot(label_g, num_classes)[ind].to(device).sum(1).topk(k=1, dim=-1, largest=True, sorted=True)[0].ge(sigma).view(-1)
    
    # if id_ind.sum().item() == 0:
    #     return torch.zeros(query_num, num_classes).to(device), id_ind, max_ind, torch.zeros(1), torch.zeros(1)

    return soft_pseudo_label, id_ind, max_ind




if __name__ == '__main__':
    torch.manual_seed(2)
    X_q = torch.randn((2,4))
    X_g = torch.randn((5,4))
    k1 = 3
    k2 = 2
    a=kk_reranking(X_q,X_g,k1,k2,torch.randint(0,3,(5,)))
