import os
import sys
import pathlib
import time
import datetime
import argparse
import shutil
import torch
import torchvision
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import DataLoader
from common.core import accuracy, evaluate
from common.builder import *
from common.utils import *
from common.meter import AverageMeter
from common.logger import Logger
from common.loss import *

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import math
import algorithm


class TwoDataTransform(object):
    def __init__(self, transform_weak, transform_strong):
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong

    def __call__(self, sample):
        x_w = self.transform_weak(sample)
        x_s = self.transform_strong(sample)
        return x_w,  x_s


def save_current_script(log_dir):
    current_script_path = __file__
    shutil.copy(current_script_path, log_dir)


def build_logger(params):
    logger_root = f'Results/{params.synthetic_data}'
    if not os.path.isdir(logger_root):
        os.makedirs(logger_root, exist_ok=True)
    percentile = int(params.closeset_ratio * 100)
    noise_condition = f'symm_{percentile:2d}' if params.noise_type == 'symmetric' else f'asym_{percentile:2d}'
    logtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join(logger_root, noise_condition, params.method, f'{params.log}-{logtime}')
    logger = Logger(logging_dir=result_dir, DEBUG=True)
    logger.set_logfile(logfile_name='log.txt')
    save_params(params, f'{result_dir}/params.json', json_format=True)
    save_current_script(result_dir)
    logger.msg(f'Result Path: {result_dir}')
    return logger, result_dir
    
    
def build_dataset_loader(params):
    assert params.dataset.startswith('cifar')
    if params.dataset == 'cifar100':
        dataset = build_cifar100n_dataset(os.path.join(params.data_root, params.dataset),torchvision.transforms.ToTensor(),torchvision.transforms.ToTensor(), \
            noise_type=params.noise_type, openset_ratio= params.openset_ratio, closeset_ratio=params.closeset_ratio)
    elif params.dataset == 'cifar10':
        dataset = build_cifar10n_dataset(os.path.join(params.data_root, params.dataset), torchvision.transforms.ToTensor(),torchvision.transforms.ToTensor(), \
            noise_type=params.noise_type, openset_ratio= params.openset_ratio, closeset_ratio=params.closeset_ratio)
    else:
        raise AssertionError(f'{params.dataset} dataset is not supported yet.')
    train_loader = DataLoader(dataset['train'], batch_size=params.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(dataset['test'], batch_size=16, shuffle=False, num_workers=8, pin_memory=False)
    return dataset, train_loader, test_loader

def build_2dataset_loader(params):
    assert params.dataset.startswith('cifar')
    transform = build_transform(rescale_size=params.rescale_size, crop_size=params.crop_size)
    if params.dataset == 'cifar100':
        if params.aug == 'ws':
            dataset = build_cifar100n_dataset(os.path.join(params.data_root, params.dataset), TwoDataTransform(transform['cifar_train'], transform['cifar_train_strong_aug']), \
                transform['cifar_test'], noise_type=params.noise_type, openset_ratio= params.openset_ratio, closeset_ratio=params.closeset_ratio)
        elif params.aug == 'ww':
            dataset = build_cifar100n_dataset(os.path.join(params.data_root, params.dataset), TwoDataTransform(transform['cifar_train'], transform['cifar_train']), \
                transform['cifar_test'], noise_type=params.noise_type, openset_ratio= params.openset_ratio, closeset_ratio=params.closeset_ratio)
        elif params.aug == 'ss':
            dataset = build_cifar100n_dataset(os.path.join(params.data_root, params.dataset), TwoDataTransform(transform['cifar_train_strong_aug'], transform['cifar_train_strong_aug']), \
                transform['cifar_test'], noise_type=params.noise_type, openset_ratio= params.openset_ratio, closeset_ratio=params.closeset_ratio)

    elif params.dataset == 'cifar10':
        if params.aug == 'ws':
            dataset = build_cifar10n_dataset(os.path.join(params.data_root, params.dataset), TwoDataTransform(transform['cifar_train'], transform['cifar_train_strong_aug']), \
                transform['cifar_test'], noise_type=params.noise_type, openset_ratio= params.openset_ratio, closeset_ratio=params.closeset_ratio)
        elif params.aug == 'ww':
            dataset = build_cifar10n_dataset(os.path.join(params.data_root, params.dataset), TwoDataTransform(transform['cifar_train'], transform['cifar_train']), \
                transform['cifar_test'], noise_type=params.noise_type, openset_ratio= params.openset_ratio, closeset_ratio=params.closeset_ratio)
        elif params.aug == 'ss':
            dataset = build_cifar10n_dataset(os.path.join(params.data_root, params.dataset), TwoDataTransform(transform['cifar_train_strong_aug'], transform['cifar_train_strong_aug']), \
                transform['cifar_test'], noise_type=params.noise_type, openset_ratio= params.openset_ratio, closeset_ratio=params.closeset_ratio)
    else:
        raise AssertionError(f'{params.dataset} dataset is not supported yet.')
    train_loader = DataLoader(dataset['train'], batch_size=params.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(dataset['test'], batch_size=16, shuffle=False, num_workers=8, pin_memory=False)
    return dataset, train_loader, test_loader


def get_baseline_stats(result_file):
    with open(result_file, 'r') as f:
        lines = f.readlines()
    test_acc_list = []
    test_acc_list2 = []
    valid_epoch = []
    # valid_epoch = [191, 192, 193, 194, 195, 196, 197, 198, 199, 200]
    for idx in range(1, 11):
        line = lines[-idx].strip()
        epoch,  test_acc = line.split(' | ')[0], line.split(' | ')[-3]
        ep = int(epoch.split(': ')[1])
        valid_epoch.append(ep)
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
    os.rename(result_dir, f"{result_dir}-bestAcc_{best_accuracy:.4f}-lastAcc_{stats['mean']:.4f}")


def get_test_acc(acc):
        return (acc[0] + acc[1]) / 2. if isinstance(acc, tuple) else acc
       
def main(cfg, device):
    init_seeds(1)
    assert cfg.dataset.startswith('cifar')
    logger, result_dir = build_logger(cfg)

    n_classes = int(cfg.n_classes * (1 - cfg.openset_ratio))
    model = algorithm.__dict__[cfg.method](cfg, input_channel=3, num_classes= n_classes)   
    
    if cfg.method == 'comining' :
        dataset, train_loader, test_loader = build_2dataset_loader(cfg)
    else:
        dataset, train_loader, test_loader = build_dataset_loader(cfg)

    logger.msg(f"Categories: {n_classes}, Training Samples: {dataset['n_train_samples']}, Testing Samples: {dataset['n_test_samples']}")
    logger.msg(f"Noise Type: {dataset['train'].noise_type}, Openset Noise Ratio: {dataset['train'].openset_noise_ratio}, Closedset Noise Ratio: {dataset['train'].closeset_noise_rate}")
    logger.msg(f'Optimizer: {cfg.opt}')

    # meters -----------------------------------------------------------------------------------------------------------------------------------------
    best_accuracy, best_epoch = 0.0, None
    # training ---------------------------------------------------------------------------------------------------------------------------------------
    for epoch in range(0, cfg.epochs):
        start_time = time.time()

        model.train(train_loader, epoch )
        test_accuracy = get_test_acc(model.evaluate(test_loader))
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch + 1

        runtime = time.time() - start_time

        logger.info(f'epoch: {epoch + 1:>3d} | '
                # f'train loss: {train_loss.avg:>6.4f} | '
                # f'train accuracy: {train_accuracy.avg:>6.3f} | '
                # f'test loss: {test_loss:>6.4f} | '
                f'test accuracy: {test_accuracy:>6.3f} | '
                f'epoch runtime: {runtime:6.2f} sec | '
                f'best accuracy: {best_accuracy:6.3f} @ epoch: {best_epoch:03d}')

    wrapup_training(result_dir, best_accuracy)


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--synthetic-data', type=str, default='cifar100nc')  # 'cifar100nc'   cifar80no  
    parser.add_argument('--dataset', type=str, default='cifar100') 
    parser.add_argument('--rescale_size', type=int, default=32) 
    parser.add_argument('--crop_size', type=int, default=32) 
    parser.add_argument('--input_channel', type=int, default=3) 
    parser.add_argument('--aug', type=str, default='ws')     
    
    parser.add_argument('--noise-type', type=str, default='symmetric')   #  symmetric  asymmetric
    parser.add_argument('--closeset_ratio', type=float, default='0.5')   #   closeset-ratio
    parser.add_argument('--gpu', type=str, default='0',)
    parser.add_argument('--net', type=str, default='cnn')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr-decay', type=str, default='linear')
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200)
    
    parser.add_argument('--method', type=str, default='comining', help='standard, decoupling, coteaching, coteachingplus, jocor, peerlearning, comining')  
    parser.add_argument('--data_root', type=str, default='~/data') 

    parser.add_argument('--log', type=str, default='')

    parser.add_argument('--epsilon', type=float, default='0.5',help=' label smoothing distribution')
    parser.add_argument('--forget_rate', type=float,  default=None)
    parser.add_argument('--adjust_lr', type=int,  default=1)
    parser.add_argument('--epoch_decay_start', type=int,  default=80)
    
    parser.add_argument('--num_gradual', type=int,  default=10)
    parser.add_argument('--exponent', type=float,  default=1)

    parser.add_argument('--w_pc', type=float, default=1.0, help='weight for the prediction consistency loss')
    parser.add_argument('--w_tri', type=float, default=3.0, help='weight for the triplet loss')
    parser.add_argument('--w_re', type=float, default=0.01, help='weight for the relabel loss')
    
    parser.add_argument('--bank', type=int, default=4096, help='the size of feature memory bank')
    parser.add_argument('--m', default='0', type=str, help="margin: 'soft' or float value, such as 0")
    parser.add_argument('--metric', default='cosine', type=str, help='metric: cosine or euclidean')
    parser.add_argument('--plk', default=0, type=int, help='topk hard example mining')
    parser.add_argument('--nlk', default=0, type=int, help='topk hard example mining')
    parser.add_argument('--puk', default=5, type=int, help='topk hard example mining')
    parser.add_argument('--nuk', default=5, type=int, help='topk hard example mining')
    parser.add_argument('--re_k1', default=15, type=int, help='reranking qurey/gallery top k1')
    parser.add_argument('--re_sigma', default=0.15, type=float, help='sigma')
    args = parser.parse_args()

    # config = load_from_cfg(args.config)
    # override_config_items = [k for k, v in args.__dict__.items() if k != 'config' and v is not None]
    # for item in override_config_items:
    #     config.set_item(item, args.__dict__[item])
    config = args
    assert config.synthetic_data in ['cifar100nc', 'cifar80no', 'cifar10nc']
    assert config.noise_type in ['symmetric', 'asymmetric']
    config.openset_ratio = 0.2 if config.synthetic_data == 'cifar80no' else 0
    config.n_classes=100 if config.dataset == 'cifar100' else 10

    print(config)
    return config


if __name__ == '__main__':

    params = parse_args()
    dev = set_device(params.gpu)
    script_start_time = time.time()
    main(params, dev)
    script_runtime = time.time() - script_start_time
    print(f'Runtime of this script {str(pathlib.Path(__file__))} : {script_runtime:.1f} seconds ({script_runtime/3600:.3f} hours)')
