""" train arcface model 
    with pytorch DistributedDataParallel
    The learning time has been reduced by 1/5.
   """
import argparse
from email.policy import default
import os
import random
import time
import cv2
import numpy as np
import logging
import warnings

# albumentation
import albumentations as A

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.backends import cudnn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
import torch.utils.data

# local
from modules.data_augmetation import DukeDataset
from modules.model import ArcMarginProduct
from modules.backbone import ResNet50
from modules.loss import ArcFaceLoss

class Logger:
    """set logging format"""

    def __init__(self, log_file_name, log_level, logger_name):
        # firstly, create a logger
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(log_level)

        # secondly, create a handler
        file_handler = logging.FileHandler(log_file_name)
        console_handler = logging.StreamHandler()

        # thirdly, define the output form of handler
        formatter = logging.Formatter(
            "[%(asctime)s]-[%(filename)s line:%(lineno)d]:%(message)s "
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # finally, add the Hander to logger
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger


checkpoint_dir = time.strftime("%y%m%d_%H%M")
SAVE_PATH = "./checkpoints/"+checkpoint_dir+"_"

logger = Logger(log_file_name=os.path.join('./logs', "{}_log.txt".format(checkpoint_dir)),
                log_level=logging.DEBUG,
                logger_name="Arcface with Distributed Data Parallel"
                ).get_log()


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--bs', dest='batch_size', type=int, default=36)
    p.add_argument('--n_epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=10e-5)
    p.add_argument('--verbose', type=int, default=2)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--num_workers', type=int, default=4, help='number of worker in dataloader')
    p.add_argument('--train_dir', type=str, default="./dataset/AI_Hub_reID/train")
    p.add_argument('--valid_dir', type=str, default="./dataset/AI_Hub_reID/valid")
    p.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    p.add_argument('--dist-url', default='tcp://14.49.44.186:8890', type=str,
                        help='url used to set up distributed training')
    p.add_argument('--local_rank', default=-1, type=int, help="node rank for distributed training")

    config = p.parse_args()
    return config

def set_albumentation():
    transform = A.Compose([
    A.OneOf([
        A.Sharpen(),
        A.HorizontalFlip(),
        A.ImageCompression(quality_lower=50, quality_upper=100, always_apply=False, p=0.3),
        A.ElasticTransform(alpha=10, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.Downscale(scale_min=0.25, scale_max=0.25, interpolation=0, always_apply=False, p=0.2),
        A.ShiftScaleRotate(rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        
    ])
],)
    return transform


def multiclass_accuracy(prediction, target):
    """Accuracy for multi-class classification

    Args:
        prediction (torch.FloatTensor): network output
        target (torch.LongTensor): ground truth

    Returns:
        torch.LongTensor: sum over examples
    """
    batch_size = target.shape[0]
    prediction = torch.argmax(prediction, dim=-1)

    result = (target == prediction).long()
    return sum(result / batch_size) * 100


def collate_fn(batch):
    transposed_data = list(zip(*batch))
    img = torch.stack(transposed_data[0], dim=0)
    label = torch.stack(transposed_data[1], dim=0)
    return img, label


def main():
    config = define_argparser()
    config.nprocs = torch.cuda.device_count()
    print('==== number of processes : ', config.nprocs)
    
    if config.seed is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    
    mp.spawn(main_worker, nprocs=config.nprocs, args=(config.nprocs, logger, config))


def main_worker(local_rank, nprocs, logger, config):
    dist.init_process_group(backend=config.dist_backend,
                            init_method=config.dist_url,
                            world_size=nprocs,
                            rank=local_rank)
    
    print('rank : {} is loaded'.format(local_rank))
        
    # use albumentation
    train_data = DukeDataset(file_dir=config.train_dir, transform=set_albumentation())
    train_sampler = DistributedSampler(train_data)
    train_loader = data.DataLoader(train_data,
                                   batch_size=config.batch_size,
                                   shuffle=(train_sampler is None),
                                   num_workers=config.num_workers,
                                   prefetch_factor=2,
                                   pin_memory=True,
                                   sampler=train_sampler
                                   )

    valid_data = DukeDataset(file_dir=config.valid_dir, transform=set_albumentation())
    valid_sampler = DistributedSampler(valid_data)
    valid_loader = data.DataLoader(valid_data,
                                   batch_size=config.batch_size,
                                   shuffle=(train_sampler is None),
                                   num_workers=config.num_workers,
                                   prefetch_factor=2,
                                   pin_memory=True,
                                   sampler=valid_sampler
                                   )
    
    num_classes = len(os.listdir(config.train_dir))
    torch.cuda.set_device(local_rank)
    
    model = ResNet50(512).cuda(local_rank)
    model = DistributedDataParallel(model, device_ids=[local_rank])
    metric = ArcMarginProduct(in_features=512, num_classes=num_classes, local_rank=local_rank).cuda(local_rank)
    criterion = ArcFaceLoss().cuda(local_rank)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold=0.00001)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, )

    logger.info("Training start, batchsize is: {:d}, learning rate: {}, train_data: {}, work number is: {}"\
        .format(config.batch_size, config.lr, config.train_dir, config.num_workers)
        )
    logger.info("Train datasets number is : {}".format(len(train_data)))
    logger.info("       =======   start  training   ======     ")

    for epoch in range(config.n_epochs):
        train_sampler.set_epoch(epoch)
        logger.info('===Epoch:[{}/{}]==='.format(
                    epoch + 1,
                    config.n_epochs
                    ))

        loss_avg = train(train_loader, model, metric, criterion, optimizer, local_rank)
        val_acc = validate(valid_loader, model, metric, criterion, local_rank)
        scheduler.step(val_acc)
        
        for param_group in optimizer.param_groups:
            print("current learning rate : ", str(param_group['lr']))
            
        logger.info("avg loss : {}".format(loss_avg))
        
# TODO : best model 만 저장하도록 변경
        if val_acc > 80:
            checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
            }

            torch.save(checkpoint, SAVE_PATH+str(epoch)+".pth")

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }

    torch.save(checkpoint, SAVE_PATH+".pth")


def train(train_loader, model, metric, criterion, optimizer, local_rank):
    # switch to train mode
    model.train()
    
    loss_list = []
    for idx, batch in enumerate(train_loader):
        input_data, label = batch
        input_data = input_data.cuda(local_rank)
        label = label.flatten().long().cuda(local_rank)
        feature = model(input_data).cuda(local_rank)
        pred = metric(feature, label, local_rank).cuda(local_rank)
        loss = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        if idx % 1000 == 0:
            logger.info("  === step:[{:3}/{}],|loss:{:.4f}|".format(
                idx + 1,
                len(train_loader),
                loss.item()
                )
            )
    return np.mean(loss_list)


def validate(valid_loader, model, metric, criterion, local_rank):
    model.eval()
    loss_list = []
    acc_list = []
    for batch in valid_loader:
        with torch.no_grad():
            input_data, label = batch
            input_data = input_data.cuda(local_rank)
            label = label.flatten().long().cuda(local_rank)
            feature = model(input_data).cuda(local_rank)
            pred = metric(feature, label, local_rank).cuda(local_rank)
            loss = criterion(pred, label)
            
            loss_list.append(loss.item())
            acc = multiclass_accuracy(pred, label)
            acc_list.append(acc.item())

    logger.info(" ============== validation ==============")
    logger.info(" valid loss : {:.4f}, |  accuracy : {:.2f}%".format(
        np.mean(loss_list), np.mean(acc_list)))

    return np.mean(acc_list)

if __name__ == '__main__':
    main()
