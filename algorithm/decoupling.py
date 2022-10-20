import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from model.cnn import CNN
from common.utils import accuracy


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class decoupling:
    def __init__(
            self, 
            args, 
            input_channel: int = 3, 
            num_classes: int = 10,
        ):

        self.lr = args.lr
        self. epochs = args.epochs
        self.epoch_decay_start = args.epoch_decay_start
        self.adjust_lr = args.adjust_lr
        self.device = device
        
        # Adjust learning rate and betas for Adam Optimizer
        mom1 = 0.9
        mom2 = 0.1
        self.alpha_plan = [self.lr] * args.epochs
        self.beta1_plan = [mom1] * args.epochs

        for i in range(args.epoch_decay_start, args.epochs):
            self.alpha_plan[i] = float(args.epochs - i) / (args.epochs - args.epoch_decay_start) * self.lr
            self.beta1_plan[i] = mom2


        # model
        self.model1 = CNN(input_channel=input_channel, n_outputs=num_classes)
        self.model2 = CNN(input_channel=input_channel, n_outputs=num_classes)
        self.model1.to(device)
        self.model2.to(device)

        self.optimizer1 = torch.optim.Adam(self.model1.parameters(), lr=self.lr)
        self.optimizer2 = torch.optim.Adam(self.model2.parameters(), lr=self.lr)
        
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.adjust_lr = args.adjust_lr

    def evaluate(self, test_loader):
        # print('Evaluating ...')
        self.model1.eval()  # Change model to 'eval' mode.
        self.model2.eval()  # Change model to 'eval' mode

        correct1 = 0
        total1 = 0
        for images, labels, _ in test_loader:
            images = Variable(images).to(self.device)
            logits1 = self.model1(images)
            outputs1 = F.softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)
            total1 += labels.size(0)
            correct1 += (pred1.cpu() == labels).sum()

        correct2 = 0
        total2 = 0
        for images, labels, _ in test_loader:
            images = Variable(images).to(self.device)
            logits2 = self.model2(images)
            outputs2 = F.softmax(logits2, dim=1)
            _, pred2 = torch.max(outputs2.data, 1)
            total2 += labels.size(0)
            correct2 += (pred2.cpu() == labels).sum()

        acc1 = 100 * float(correct1) / float(total1)
        acc2 = 100 * float(correct2) / float(total2)
        return acc1, acc2

    def train(self, train_loader, epoch):
        # print('Training ...')
        self.model1.train()  
        if self.adjust_lr == 1:
            self.adjust_learning_rate(self.optimizer1, epoch)
            
        self.model2.train()

        if self.adjust_lr == 1:
            self.adjust_learning_rate(self.optimizer2, epoch)

        for (images, labels, _) in train_loader:
            images = Variable(images).to(self.device)
            labels = Variable(labels).to(self.device)

            logits1 = self.model1(images)
            _, pred1 = torch.max(logits1, dim=1)
            logits2 = self.model2(images)
            _, pred2 = torch.max(logits2, dim=1)

            inds = torch.where(pred1 != pred2)    
            loss_1 = self.loss_fn(logits1[inds], labels[inds])
            loss_2 = self.loss_fn(logits2[inds], labels[inds])

            self.optimizer1.zero_grad()
            loss_1.backward()
            self.optimizer1.step()
            self.optimizer2.zero_grad()
            loss_2.backward()
            self.optimizer2.step()


    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1