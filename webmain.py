from distutils.command.config import config
import os
import sys
import pathlib
import time
import datetime
import argparse
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torch.autograd import Variable
import warnings
warnings.filterwarnings('ignore')
from common.core import accuracy
from common.builder import *
from common.utils import *
from common.meter import AverageMeter, MaxMeter
from common.logger import Logger
from common.loss import *
from common.cross_batch_mem import XBM, XBM_isclean

from common.cross_batch_mem import XBM
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import math
from common.triplet import TopksimpleLoss
from common.reranking import kk_smoothing, kreciprocal
from losses.loss_utils import kl_loss_compute
from model.resnet import *




class TwoDataTransform(object):
    def __init__(self, transform_weak, transform_strong):
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong

    def __call__(self, sample):
        x_w1 = self.transform_weak(sample)
        # x_w2 = transform_weak(sample)
        x_s = self.transform_strong(sample)
        return x_w1,  x_s

class CLDataTransform(object):
    def __init__(self, transform_weak, transform_strong):
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong

    def __call__(self, sample):
        x_w1 = self.transform_weak(sample)
        x_w2 = self.transform_weak(sample)
        x_s = self.transform_strong(sample)
        return x_w1, x_w2, x_s
    

def save_current_script(log_dir):
    current_script_path = __file__
    shutil.copy(current_script_path, log_dir)


def record_network_arch(result_dir, net):
    with open(f'{result_dir}/network.txt', 'w') as f:
        f.writelines(net.__repr__())


def init_weights(module, init_method='He'):
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if init_method == 'He':
                nn.init.kaiming_normal_(m.weight.data)
            elif init_method == 'Xavier':
                nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, val=0)
                

def build_logger(params):
    logger_root = f'Results/{params.dataset}'
    if not os.path.isdir(logger_root):
        os.makedirs(logger_root, exist_ok=True)

    logtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join(logger_root,  params.net +'_', \
        f'{params.log}-{logtime}'+'_'+params.aug +'_'+ params.schedule +'_'+ str(params.epsilon))
    
    logger = Logger(logging_dir=result_dir, DEBUG=True)
    logger.set_logfile(logfile_name='log.txt')
    # save_config(params, f'{result_dir}/params.cfg') 
    save_params(params, f'{result_dir}/params.json', json_format=True)
    save_current_script(result_dir)
    logger.msg(f'Result Path: {result_dir}')
    return logger, result_dir


def build_model_optim_scheduler(params,net,n_classes, device, build_scheduler=True):
    if params.opt == 'sgd':
        optimizer = build_sgd_optimizer(net.parameters(), params.lr, params.weight_decay, nesterov=True)
    elif params.opt == 'adam':
        optimizer = build_adam_optimizer(net.parameters(), params.lr)
    else:
        raise AssertionError(f'{params.opt} optimizer is not supported yet.')
    if build_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True, threshold=1e-4)
    else:
        scheduler = None
    return net.to(device), optimizer, scheduler, n_classes


def build_lr_plan(params, factor=10, decay='linear'):
    epoch_decay_start = 80
    lr_plan = [params.lr] * params.epochs
    for i in range(0, params.warmup_epochs):
        lr_plan[i] *= factor
    for i in range(epoch_decay_start, params.epochs):
        if decay == 'linear':
            lr_plan[i] = float(params.epochs - i) / (params.epochs - params.warmup_epochs) * params.lr  # linearly decay
        elif decay == 'cosine':
            lr_plan[i] = 0.5 * params.lr * (1 + math.cos((i - params.warmup_epochs + 1) * math.pi / (params.epochs - params.warmup_epochs + 1)))  # cosine decay
        else:
            raise AssertionError(f'lr decay method: {decay} is not implemented yet.')
    return lr_plan
    



def get_baseline_stats(result_file):
    with open(result_file, 'r') as f:
        lines = f.readlines()
    test_acc_list = []
    test_acc_list2 = []
    valid_epoch = []
    # valid_epoch = [191, 192, 193, 194, 195, 196, 197, 198, 199, 200]
    for idx in range(1, 11):
        line = lines[-idx].strip()
        epoch,  test_acc = line.split(' | ')[0],line.split(' | ')[-3]
        ep = int(epoch.split(': ')[1])
        valid_epoch.append(ep)
        # assert ep in valid_epoch, ep
        if '/' not in test_acc:
            test_acc_list.append(float(test_acc.split(': ')[1]))
        else:
            test_acc1, test_acc2 = map(lambda x: float(x), test_acc.split(': ')[1].lstrip('(').rstrip(')').split('/'))
            test_acc_list.append(test_acc1)
            test_acc_list2.append(test_acc2)
    if len(test_acc_list2) == 0:
        test_acc_list = np.array(test_acc_list)
        print(valid_epoch)
        print(f'mean: {test_acc_list.mean():.2f}, std: {test_acc_list.std():.2f}')
        print(f' {test_acc_list.mean():.2f}±{test_acc_list.std():.2f}')
        return {'mean': test_acc_list.mean(), 'std': test_acc_list.std(), 'valid_epoch': valid_epoch}
    else:
        test_acc_list = np.array(test_acc_list)
        test_acc_list2 = np.array(test_acc_list2)
        print(valid_epoch)
        print(f'mean: {test_acc_list.mean():.2f} , std: {test_acc_list.std():.2f}')
        print(f'mean: {test_acc_list2.mean():.2f} , std: {test_acc_list2.std():.2f}')
        print(f' {test_acc_list.mean():.2f}±{test_acc_list.std():.2f}  ,  {test_acc_list2.mean():.2f}±{test_acc_list2.std():.2f} ')
        return {'mean1': test_acc_list.mean(), 'std1': test_acc_list.std(),
                'mean2': test_acc_list2.mean(), 'std2': test_acc_list2.std(),
                'valid_epoch': valid_epoch}
        
def wrapup_training(result_dir, best_accuracy):
    stats = get_baseline_stats(f'{result_dir}/log.txt')
    with open(f'{result_dir}/result_stats.txt', 'w') as f:
        f.write(f"valid epochs: {stats['valid_epoch']}\n")
        if 'mean' in stats.keys():
            f.write(f"mean: {stats['mean']:.4f}, std: {stats['std']:.4f}\n")
        else:
            f.write(f"mean1: {stats['mean1']:.4f}, std2: {stats['std1']:.4f}\n")
            f.write(f"mean2: {stats['mean2']:.4f}, std2: {stats['std2']:.4f}\n")
    os.rename(result_dir, f'{result_dir}-bestAcc_{best_accuracy:.4f}')

def norml2(x):
    return torch.nn.functional.normalize(x, p=2, dim=1)



def get_test_acc(acc):
        return (acc[0] + acc[1]) / 2. if isinstance(acc, tuple) else acc
       
def main(cfg, device):
    init_seeds(1)
    assert cfg.dataset.startswith('web')
    logger, result_dir = build_logger(cfg)

    # model = ResNetfeat(cfg.net, cfg.n_classes,pretrained=True, ).to(device)  
    model = ResNet18(cfg.n_classes).to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9, nesterov=True)
    if cfg.schedule == 'max':
        lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, threshold=1e-6, verbose=True)
    elif cfg.schedule == 'cosine':
        lr_plan = [cfg.lr] * cfg.epochs
        for i in range(cfg.warmup_epochs, cfg.epochs):
            lr_plan[i] = 0.5 * cfg.lr * (1 + math.cos((i - cfg.warmup_epochs + 1) * math.pi / (cfg.epochs - cfg.warmup_epochs + 1)))  # cosine decay
    else:
        raise AssertionError(f'lr decay method: {cfg.schedule} must be max or cosine.')
    
    transform = build_transform(rescale_size=512, crop_size=448)
    if cfg.aug == 'ws':
        dataset = build_webfg_dataset(os.path.join(cfg.data_root, cfg.dataset), TwoDataTransform(transform['train'], transform['train_strong_aug']), transform['test'])
    elif cfg.aug == 'ww':
        dataset = build_webfg_dataset(os.path.join(cfg.data_root, cfg.dataset), TwoDataTransform(transform['train'], transform['train']), transform['test'])
    elif cfg.aug == 'ss':
        dataset = build_webfg_dataset(os.path.join(cfg.data_root, cfg.dataset), TwoDataTransform(transform['train_strong_aug'], transform['train_strong_aug']), transform['test'])
    train_loader = DataLoader(dataset['train'], batch_size=cfg.bs, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(dataset['test'], batch_size=16, shuffle=False, num_workers=8, pin_memory=False)

    logger.msg(f"Categories: {cfg.n_classes}, Training Samples: {dataset['n_train_samples']}, Testing Samples: {dataset['n_test_samples']}")

    # record_network_arch(result_dir, net)
    # drop rate 
    rate_schedule = np.ones(cfg.epochs) * cfg.forget_rate
    rate_schedule[:cfg.warmup_epochs] = np.linspace(0, cfg.forget_rate, cfg.warmup_epochs)


    TripletLoss = TopksimpleLoss(cfg.m, cfg.metric, cfg.plk, cfg.puk, cfg.nlk, cfg.nuk,  reduce=False )
    cleanerbank1 = XBM(size=cfg.bank,  dim=model.fc.in_features , device = device)  
    cleanerbank2 = XBM(size=cfg.bank,  dim=model.fc.in_features , device = device)  
    max_classnum = MaxMeter()
    # meters -----------------------------------------------------------------------------------------------------------------------------------------
    train3_loss = AverageMeter()
    n_p = AverageMeter()
    
    best_accuracy, best_epoch = 0.0, None

    # training ---------------------------------------------------------------------------------------------------------------------------------------
    for epoch in range(0, cfg.epochs):
        start_time = time.time()

        test_total = 0
        test_correct = 0        
        model.train()
        if cfg.schedule == 'cosine':
            adjust_lr(optimizer, lr_plan[epoch])
            
        for i, ((images1,images2), labels) in enumerate(train_loader):
            images1 = Variable(images1).to(device)
            images2 = Variable(images2).to(device)
            max_classnum.update(labels.max().item()+1)
            labels = Variable(labels).to(device)
            N = labels.size(0)
            
            # Forward + Backward + Optimize
            logits1, feat1 = model(images1)
            feat1 = norml2(feat1)
                        
            logits2,feat2 = model(images2)
            feat2 = norml2(feat2)

            smooth_label = get_smoothed_label_distribution(labels, cfg.n_classes, cfg.epsilon)
            
            if epoch< cfg.warmup_epochs:
                loss = 0.5 * cross_entropy(logits1, smooth_label, reduction='mean') + 0.5 * cross_entropy(logits2, smooth_label, reduction='mean')
                cleanerbank1.enqueue_dequeue(feat1.detach(), labels.detach(),)
                cleanerbank2.enqueue_dequeue(feat2.detach(), labels.detach(),)
            else: 
                probs1 = logits1.softmax(dim=1)
                probs2 = logits2.softmax(dim=1)
                
                loss_cls = 0.5 * cross_entropy(logits1, smooth_label, reduction='none') + 0.5 * cross_entropy(logits2, smooth_label, reduction='none')
                loss_pc = 0.5 * symmetric_kl_div(probs1, probs2) + 0.5 * symmetric_kl_div(probs2, probs1) 
                
                sort_ind = torch.argsort(loss_cls + cfg.w_pc *loss_pc)
                clean_allid = sort_ind[ :int(N*(1-rate_schedule[epoch]) )]
                cleanerid = sort_ind[ :int(N*(1- rate_schedule[epoch])* cfg.er)]
                ind_drop = sort_ind[ int(N* rate_schedule[epoch]):]   
                
                #  semi-hard mining --------------------------------------------------------------------------------------------------------------             
                triploss1, hard_p1,  hard_n1    = TripletLoss(feat1[cleanerid],  labels[cleanerid], cleanerbank1.get()[0], cleanerbank1.get()[1], )
                triploss2, hard_p2,  hard_n2    = TripletLoss(feat2[cleanerid],  labels[cleanerid], cleanerbank2.get()[0], cleanerbank2.get()[1], )
                loss_trip = ( triploss1+ triploss2)/2.
                
                #  relabel --------------------------------------------------------------------------------------------------------------
                soft_label1, id_ind1,  maxid1 = kreciprocal(feat1[ind_drop], cleanerbank1.get()[0], cfg.re_k1, cfg.re_sigma, cleanerbank1.get()[1], device, max_classnum.val)
                soft_label2, id_ind2,  maxid2 = kreciprocal(feat2[ind_drop], cleanerbank2.get()[0], cfg.re_k1, cfg.re_sigma, cleanerbank2.get()[1], device, max_classnum.val)
                
                if (id_ind1 & id_ind2 & (maxid1==maxid2)).sum()==0:    
                    loss_re = torch.zeros(len(ind_drop)).to(device)  
                    re_idx = torch.zeros(0)
                else:          
                    re_idx = id_ind1 & id_ind2 & (maxid1==maxid2)
                    soft_label = (soft_label1.detach() + soft_label2.detach())/2
                    loss_re = - 0.5 *(F.log_softmax(logits1[ind_drop,:][re_idx], dim=1)*soft_label[re_idx]).sum(-1)  - 0.5 *(F.log_softmax(logits2[ind_drop][re_idx], dim=1)*soft_label[re_idx]).sum(-1)
                
                loss = loss_cls[clean_allid].mean()  + cfg.w_pc * loss_pc[clean_allid].mean()  + cfg.w_tri * loss_trip.mean() + cfg.w_re * loss_re.mean()

                cleanerbank1.enqueue_dequeue(feat1[cleanerid].detach(), labels[cleanerid].detach())
                cleanerbank2.enqueue_dequeue(feat2[cleanerid].detach(), labels[cleanerid].detach())
                
                #  statistics --------------------------------------------------------------------------------------------------------------
                train3_loss.update(cfg.w_tri * loss_trip.mean().detach() )
                n_p.update( (hard_n1-hard_p1).mean().detach().cpu() )
                n_p.update( (hard_n2-hard_p2).mean().detach().cpu() )
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
        # print(train_accuracy)
        model.eval()
        for images, labels in test_loader:
            images = Variable(images).to(device)
            logits,_ = model(images)
            outputs = F.softmax(logits, dim=1)
            _, pred = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (pred.cpu() == labels).sum()       
        test_accuracy = 100 * float(test_correct) / float(test_total)   
        
        if cfg.schedule =='max': 
            lr_schedule.step(test_accuracy)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch + 1

        # logging this epoch
        runtime = time.time() - start_time

        logger.info(f'epoch: {epoch + 1:>3d} | '
                f'train3 loss: {train3_loss.avg:>6.4f} | '
                f'n-p: {n_p.avg:>6.3f} | '
                # f'test loss: {test_loss:>6.4f} | '
                f'test accuracy: {test_accuracy:>6.3f} | '
                f'epoch runtime: {runtime:6.2f} sec | '
                f'best accuracy: {best_accuracy:6.3f} @ epoch: {best_epoch:03d}')

    wrapup_training(result_dir, best_accuracy)


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, required=True)

    parser.add_argument('--dataset', type=str, default='web-bird') 

    parser.add_argument('--noise-rate', type=float, default='0.25')   #   closeset-ratio
    parser.add_argument('--gpu', type=str,  default='2')
    parser.add_argument('--net', type=str, default='resnet50')
    parser.add_argument('--bs', type=int, default=25)

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr-decay', type=str, default='linear')
    parser.add_argument('--schedule', type=str, default='cosine', help=' max or cosine')
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--forget_rate', type=float,  default=None)
    parser.add_argument('--epochs', type=int, default=60)
    
    parser.add_argument('--aug', type=str, default='ws')     
    parser.add_argument('--data_root', type=str, default='~/data') 
    
    parser.add_argument('--log', type=str, default='')

    parser.add_argument('--epsilon', type=float, default='0.5',help=' label smoothing distribution')
    parser.add_argument('--anno', type=str, default='', help='annotation')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--ablation', action='store_true')

    parser.add_argument('--w_pc', type=float, default=1.0, help='weight for the prediction consistency loss')
    parser.add_argument('--w_tri', type=float, default=3.0, help='weight for the triplet loss')
    parser.add_argument('--w_re', type=float, default=0.01, help='weight for the relabel loss')
    
    parser.add_argument('--bank', type=int, default=4096, help='the size of feature memory bank')
    parser.add_argument('--m', default='0', type=str, help='margin')
    parser.add_argument('--metric', default='cosine', type=str, help='metric: cosine or euclidean')
    parser.add_argument('--plk', default=0, type=int, help='topk hard example mining')
    parser.add_argument('--nlk', default=0, type=int, help='topk hard example mining')
    parser.add_argument('--puk', default=5, type=int, help='topk hard example mining')
    parser.add_argument('--nuk', default=5, type=int, help='topk hard example mining')
    parser.add_argument('--re_k1', default=15, type=int, help='reranking qurey top k1')
    parser.add_argument('--re_sigma', default=0.5, type=float, help='the threshold for id or ood ')
    parser.add_argument('--er', type=float, default=0.5)
    args = parser.parse_args()

    # config = load_from_cfg(args.config)
    # override_config_items = [k for k, v in args.__dict__.items() if k != 'config' and v is not None]
    # for item in override_config_items:
    #     config.set_item(item, args.__dict__[item])
    config = args

    if config.dataset.endswith('bird'):
        config.n_classes=200
    elif config.dataset.endswith('aircraft'):
        config.n_classes=100
    elif config.dataset.endswith('car'):    
        config.n_classes=196

    if config.forget_rate is None:
        config.forget_rate = config.noise_rate
    print(config)
    return config


if __name__ == '__main__':

    params = parse_args()
    dev = set_device(params.gpu)
    print(dev)
    script_start_time = time.time()
    main(params, dev)
    script_runtime = time.time() - script_start_time
    print(f'Runtime of this script {str(pathlib.Path(__file__))} : {script_runtime:.1f} seconds ({script_runtime/3600:.3f} hours)')
