"""train arcface"""
import argparse
import os
import glob
from random import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
from modules.data_augmetation import DukeDataset
from modules.model import ArcMarginProduct
from modules.backbone import ResNet50
from modules.loss import ArcFaceLoss
from torch.utils import data
import time
import cv2
import numpy as np
import logging
import albumentations as A



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
                logger_name="Arcface using DukeMTMC data"
                ).get_log()


def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--bs', dest='batch_size', type=int, default=41)
    p.add_argument('--n_epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=10e-5)
    p.add_argument('--verbose', type=int, default=2)
    p.add_argument('--train_dir', type=str, default="./dataset/AI_Hub_reID/train")
    p.add_argument('--valid_dir', type=str, default="./dataset/AI_Hub_reID/valid")

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


def main(config):
    device = torch.device('cuda:' + str(config.gpu_id))
    print('device : ', device)

    # use albumentation
    train_data = DukeDataset(file_dir=config.train_dir, transform=set_albumentation())
    train_loader = data.DataLoader(train_data,
                                   batch_size=config.batch_size,
                                   shuffle=True,
                                   collate_fn=collate_fn)

    valid_data = DukeDataset(file_dir=config.valid_dir, transform=set_albumentation())
    valid_loader = data.DataLoader(valid_data,
                                   batch_size=config.batch_size,
                                   shuffle=True,
                                   collate_fn=collate_fn)
    
    num_classes = len(os.listdir(config.train_dir))
    model = ResNet50(512, device).to(device)
    metric = ArcMarginProduct(device, in_features=512, num_classes=num_classes).to(device)
    
    criterion = ArcFaceLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold=0.00001)


    logger.info("Training start, batchsize is: {:d}, learning rate: {}, train_data: {}".format(
        config.batch_size, config.lr, config.train_dir
            )
        )
    logger.info("Train datasets number is : {}".format(len(train_data)))
    logger.info("       =======   start  training   ======     ")

    for epoch in range(config.n_epochs):
        logger.info('===Epoch:[{}/{}]==='.format(
                    epoch + 1,
                    config.n_epochs
                    ))

        loss_avg = train(train_loader, device, model, metric, criterion, optimizer)
        val_acc = validate(valid_loader, device, model, metric, criterion)
        scheduler.step(val_acc)
        
        for param_group in optimizer.param_groups:
            print("current learning rate : ", str(param_group['lr']))
            
        logger.info("avg loss : {}".format(loss_avg))
        
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


def train(train_loader, device, model, metric, criterion, optimizer):
    model.train()
    loss_list = []
    for idx, batch in enumerate(train_loader):
        input_data, label = batch
        input_data = input_data.to(device)
        label = label.flatten().long().to(device)
        feature = model(input_data)
        pred = metric(device, feature, label)
        
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


def validate(valid_loader, device, model, metric, criterion):
    model.eval()
    loss_list = []
    acc_list = []
    for batch in valid_loader:
        with torch.no_grad():
            input_data, label = batch
            input_data = input_data.to(device)
            label = label.flatten().long().to(device)
            feature = model(input_data)
            pred = metric(device, feature, label)
            loss = criterion(pred, label)
            
            loss_list.append(loss.item())
            acc = multiclass_accuracy(pred, label)
            acc_list.append(acc.item())

    logger.info(" ============== validation ==============")
    logger.info(" valid loss : {:.4f}, |  accuracy : {:.2f}%".format(
        np.mean(loss_list), np.mean(acc_list)))

    return np.mean(acc_list)


if __name__ == '__main__':
    config = define_argparser()
    main(config)
