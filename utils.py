# -*- coding: utf-8 -*-
"""
@Time ： 2023/10/10 18:47
@Auth ： dongb
@File ：utils.py
@IDE ：PyCharm
"""
import argparse
import datetime
import logging
import os

import numpy as np
import torch
import torchvision.datasets as datasets
from torchvision import transforms


# =================== Argument Parser ===================
def args_parser():
    """
    Parses command-line arguments for the model configuration.
    Returns the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='the sample example')

    # Add various argument options for model configuration
    parser.add_argument('--c', type=int, default=32, help="Out-channel of the Encoder's last convolution")
    parser.add_argument('--P', type=int, default=1, help="Power normalization limit before input channel")
    parser.add_argument('--snr', type=int, default=10, help="Channel signal-to-noise ratio")
    parser.add_argument('--epochs', type=int, default=200, help="Total number of epochs")
    parser.add_argument('--device', type=str, default='cuda:1', help='GPU device')
    parser.add_argument('--print_frequency', type=int, default=150)
    parser.add_argument('--save_log', type=bool, default=True, help="Whether to save logs")
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument("--phase", default="test", type=str, help="Train or test phase")
    parser.add_argument('--dataset', type=str, default='cifar', help='Dataset type: cifar10 or imagenet')
    parser.add_argument('--model', type=str, default='bdjscc', help='Model type: cnn, swin, etc.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate')

    # Add more options for network and training configuration...
    # (Add the other arguments for network parameters as required)

    args = parser.parse_args()
    return args


args = args_parser()


# =================== Logger Setup ===================
def makedirs(directory):
    """
    Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def beijing(sec, what):
    """
    Returns the time in Beijing timezone.
    """
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=12)
    return beijing_time.timetuple()


def logger_configuration(filename, phase, save_log=True):
    """
    Configures logging for training/testing with time adjustments for Beijing timezone.
    Logs are saved to file if `save_log` is True.
    """
    logger = logging.getLogger("DeepJSCC")
    workdir = "./history/{}".format(filename)
    if phase == "test":
        filename = filename + "_test"
    log = workdir + "/{}.log".format(filename)
    models = workdir + "/models"

    # Make necessary directories
    makedirs(workdir)
    makedirs(models)

    # Set the custom time format for logging
    logging.Formatter.converter = beijing
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set up file handler for logging to a file
    if save_log:
        filehandler = logging.FileHandler(log)
        filehandler.setLevel(logging.INFO)
        logger.addHandler(filehandler)

    return workdir, logger


# =================== Average Meter Class ===================
class AverageMeter:
    """
    Computes and stores the average value for a given metric.
    Useful for tracking loss, accuracy, or other metrics during training.
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the average with a new value.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def clear(self):
        """
        Resets the running average and total values.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


# =================== Data Loader Functions ===================
def train_data_loader(batch_size=args.batchsize, imgz=128, workers=0, pin_memory=True):
    """
    Loads training data for a given dataset.
    You need to specify your dataset path before using.
    """
    # Specify dataset path
    data_dir = os.path.join('/home/caobin/dongb/dataset/ImageNet/train_data')

    # Create dataset with transformations (center crop, random flip, etc.)
    dataset = datasets.ImageFolder(
        data_dir,
        transforms.Compose([
            transforms.CenterCrop(imgz),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
        ])
    )

    # Return DataLoader for training
    train_data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return train_data_loader


def test_data_loader(batch_size=args.batchsize, imgz=128, workers=0, pin_memory=True):
    """
    Loads test data for a given dataset.
    You need to specify your dataset path before using.
    """
    # Specify dataset path
    data_dir = os.path.join('/home/caobin/dongb/dataset/ImageNet/test_data')

    # Create dataset with transformations (center crop, no augmentation)
    dataset = datasets.ImageFolder(
        data_dir,
        transforms.Compose([
            transforms.CenterCrop(imgz),
            transforms.ToTensor(),
        ])
    )

    # Return DataLoader for testing
    test_data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return test_data_loader
