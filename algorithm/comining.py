# -*- coding:utf-8 -*-
import imp
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from model.cnn import CNNfeat
from model.resnet import ResNetfeat
import numpy as np
from common.utils import *
from common.builder import *
from common.loss import cross_entropy
from common.triplet import TopksimpleLoss
from common.reranking import kk_smoothing, kreciprocal
from common.cross_batch_mem import XBM, XBM_isclean
from common.meter import MaxMeter, AverageMeter

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def norml2(x):
    return torch.nn.functional.normalize(x, p=2, dim=1)

def get_smoothed_label_distribution(labels, num_class, epsilon):
    # one-hot label ----> smoothing label
    smoothed_label = torch.full(size=(labels.size(0), num_class), fill_value=epsilon / (num_class - 1))
    smoothed_label.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1).cpu(), value=1 - epsilon)
    return smoothed_label.to(labels.device)



class comining:
    def __init__(self, args, input_channel, num_classes):

        # Hyper Parameters
        learning_rate = args.lr

        if args.forget_rate is None:
            forget_rate = (args.closeset_ratio + args.openset_ratio/(1-args.openset_ratio)) / ( 1+ args.openset_ratio/(1-args.openset_ratio))
        else:
            forget_rate = args.forget_rate
        
        # define drop rate schedule
        self.rate_schedule = np.ones(args.epochs) * forget_rate
        # self.rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate ** args.exponent, args.num_gradual)
        self.device = device
        self.epochs = args.epochs
        self.dataset = args.dataset
        self.adjust_lr = args.adjust_lr
        self.num_classes = num_classes
        self.epsilon = args.epsilon
        self.w_pc = args.w_pc
        self.w_tri = args.w_tri
        self.w_re = args.w_re
        self.re_k1 =args.re_k1
        self.re_sigma = args.re_sigma
        self.warmup_epochs = args.warmup_epochs

        if args.dataset.startswith('cifar'):
            self.model1 = CNNfeat(input_channel=input_channel, n_outputs=num_classes,)
            self.model1.to(device)
            # Adjust learning rate and betas for Adam Optimizer
            mom1 = 0.9
            mom2 = 0.1
            self.alpha_plan = [learning_rate] * args.epochs
            self.beta1_plan = [mom1] * args.epochs

            for i in range(args.epoch_decay_start, args.epochs):
                self.alpha_plan[i] = float(args.epochs - i) / (args.epochs - args.epoch_decay_start) * learning_rate
                self.beta1_plan[i] = mom2
            self.optimizer = torch.optim.Adam(self.model1.parameters(), lr=learning_rate)
            
        elif args.dataset.startswith('web'):
            self.model1 = ResNetfeat(args.arch,num_classes)
            self.model1.to(device)
            self.optimizer = build_sgd_optimizer(self.model1.parameters(), args.lr, args.weight_decay, nesterov=True)
            self.lr_plan = [learning_rate] * args.epochs
            for i in range(args.warmup_epochs, args.epochs):
                self.lr_plan[i] = 0.5 * args.lr * (1 + math.cos((i - args.warmup_epochs + 1) * math.pi / (args.epochs - args.warmup_epochs + 1)))  # cosine decay
            
        self.TripletLoss = TopksimpleLoss(args.m, args.metric, args.plk, args.puk, args.nlk, args.nuk,  reduce=False )
        self.cleanerbank1 = XBM(size=args.bank,  dim= self.model1.fc.in_features , device = device) 
        self.cleanerbank2 = XBM(size=args.bank,  dim= self.model1.fc.in_features , device = device)  
        self.max_classnum = MaxMeter()
        

    # Evaluate the Model
    def evaluate(self, test_loader):
        self.model1.eval()  # Change model to 'eval' mode.

        correct1 = 0
        total1 = 0
        for images, labels, _ in test_loader:
            images = Variable(images).to(self.device)
            logits1,_ = self.model1(images)
            outputs1 = F.softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)
            total1 += labels.size(0)
            correct1 += (pred1.cpu() == labels).sum()

        acc1 = 100 * float(correct1) / float(total1)
        return acc1  

    # Train the Model
    def train(self, train_loader, epoch):
        self.model1.train()  # Change model to 'train' mode.

        if self.adjust_lr == 1:
            if self.dataset.startswith('web'):
                adjust_lr(self.optimizer, self.lr_plan[epoch])
            else:
                self.adjust_learning_rate(self.optimizer, epoch)


        for i, ((images1,images2), labels, truth) in enumerate(train_loader):

            images1 = Variable(images1).to(self.device)
            images2 = Variable(images2).to(self.device)
            labels = Variable(labels).to(self.device)
            N = labels.size(0)
            self.max_classnum.update(labels.max().item()+1)
            logits1, feat1 = self.model1(images1)
            feat1 = norml2(feat1)
            logits2,feat2 = self.model1(images2)
            feat2 = norml2(feat2)

            smooth_label = get_smoothed_label_distribution(labels, self.num_classes, self.epsilon)
            
            if epoch< self.warmup_epochs:
                loss = 0.5 * cross_entropy(logits1, smooth_label, reduction='mean') + 0.5 * cross_entropy(logits2, smooth_label, reduction='mean')
                self.cleanerbank1.enqueue_dequeue(feat1.detach(), labels.detach(),)
                self.cleanerbank2.enqueue_dequeue(feat2.detach(), labels.detach(),)
            else: 
                probs1 = logits1.softmax(dim=1)
                probs2 = logits2.softmax(dim=1)
                
                loss_cls = 0.5 * cross_entropy(logits1, smooth_label, reduction='none') + 0.5 * cross_entropy(logits2, smooth_label, reduction='none')
                loss_pc = 0.5 * symmetric_kl_div(probs1, probs2) + 0.5 * symmetric_kl_div(probs2, probs1) 
                
                sort_ind = torch.argsort(loss_cls + self.w_pc *loss_pc)
                clean_allid = sort_ind[ :int(N*(1-self.rate_schedule[epoch]) )]
                cleanerid = sort_ind[ :int(N*(1-self.rate_schedule[epoch])* 0.5)]
                ind_drop = sort_ind[ int(N*self.rate_schedule[epoch]):]                
                triploss1, hard_p1,  hard_n1   = self.TripletLoss(feat1[cleanerid],  labels[cleanerid], self.cleanerbank1.get()[0], self.cleanerbank1.get()[1], )
                triploss2, hard_p2,  hard_n2   = self.TripletLoss(feat2[cleanerid],  labels[cleanerid], self.cleanerbank2.get()[0], self.cleanerbank2.get()[1], )
                loss_trip = ( triploss1+ triploss2)/2.
                
                soft_label1, id_ind1,  maxid1 = kreciprocal(feat1[ind_drop], self.cleanerbank1.get()[0], self.re_k1, \
                    self.re_sigma, self.cleanerbank1.get()[1], self.device, self.max_classnum.val)
                soft_label2, id_ind2,  maxid2 = kreciprocal(feat2[ind_drop], self.cleanerbank2.get()[0], self.re_k1, \
                    self.re_sigma, self.cleanerbank2.get()[1], self.device, self.max_classnum.val)
                
                if (id_ind1 & id_ind2 & (maxid1==maxid2)).sum()==0:    
                    loss_re = torch.zeros(len(ind_drop)).to(self.device)  
                    re_idx = torch.zeros(0)
                else:          
                    re_idx = id_ind1 & id_ind2 & (maxid1==maxid2)
                    soft_label = (soft_label1.detach() + soft_label2.detach())/2
                    loss_re = - 0.5 *(F.log_softmax(logits1[ind_drop,:][re_idx], dim=1)*soft_label[re_idx]).sum(-1)  - 0.5 *(F.log_softmax(logits2[ind_drop][re_idx], dim=1)*soft_label[re_idx]).sum(-1)
                
                loss = loss_cls[clean_allid].mean()  + self.w_pc * loss_pc[clean_allid].mean()  + self.w_tri * loss_trip.mean() + self.w_re * loss_re.mean()

                self.cleanerbank1.enqueue_dequeue(feat1[cleanerid].detach(), labels[cleanerid].detach())
                self.cleanerbank2.enqueue_dequeue(feat2[cleanerid].detach(), labels[cleanerid].detach())
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
                                       


    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1
