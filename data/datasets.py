from torch.utils.data import Dataset
from PIL import Image
# import cv2
import os
import numpy as np
from glob import glob
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset
import torch
import math
import torch.utils.data as data
NUM_DATASET_WORKERS = 8
SCALE_MIN = 0.75
SCALE_MAX = 0.95


class HR_image(Dataset):
    files = {"train": "train", "test": "test", "val": "validation"}

    def __init__(self, config, data_dir):
        self.imgs = []
        for dir in data_dir:
            self.imgs += glob(os.path.join(dir, '*.jpg'))
            self.imgs += glob(os.path.join(dir, '*.png'))
        _, self.im_height, self.im_width = config.image_dims
        self.crop_size = self.im_height
        self.image_dims = (3, self.im_height, self.im_width)
        self.transform = self._transforms()

    def _transforms(self,):
        """
        Up(down)scale and randomly crop to `crop_size` x `crop_size`
        """
        transforms_list = [
            transforms.RandomCrop((self.im_height, self.im_width)),
            transforms.ToTensor()]

        return transforms.Compose(transforms_list)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        img = img.convert('RGB')
        transformed = self.transform(img)
        return transformed

    def __len__(self):
        return len(self.imgs)


class Datasets(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.imgs = []
        for dir in self.data_dir:
            self.imgs += glob(os.path.join(dir, '*.jpg'))
            self.imgs += glob(os.path.join(dir, '*.png'))
        self.imgs.sort()


    def __getitem__(self, item):
        image_ori = self.imgs[item]
        image = Image.open(image_ori).convert('RGB')
        self.im_height, self.im_width = image.size
        if self.im_height % 32 != 0 or self.im_width % 32 != 0:
            self.im_height = self.im_height - self.im_height % 32
            self.im_width = self.im_width - self.im_width % 32
        self.transform = transforms.Compose([
            transforms.CenterCrop((self.im_width, self.im_height)),
            transforms.ToTensor()])
        img = self.transform(image)
        return img
    def __len__(self):
        return len(self.imgs)

class CIFAR10(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.len = dataset.__len__()

    def __getitem__(self, item):
        return self.dataset.__getitem__(item % self.len)

    def __len__(self):
        return self.len * 10


def get_loader(args, config):
    if args.trainset == 'DIV2K':
        train_dataset = HR_image(config, config.train_data_dir)
        test_dataset = Datasets(config.test_data_dir)
    elif args.trainset == 'CIFAR10':
        dataset_ = datasets.CIFAR10
        if config.norm is True:
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])

            transform_test = transforms.Compose([
                transforms.ToTensor()])
        train_dataset = dataset_(root=config.train_data_dir,
                                 train=True,
                                 transform=transform_train,
                                 download=False)

        test_dataset = dataset_(root=config.test_data_dir,
                                train=False,
                                transform=transform_test,
                                download=False)

        train_dataset = CIFAR10(train_dataset)

    else:
        train_dataset = Datasets(config.train_data_dir)
        test_dataset = Datasets(config.test_data_dir)

    def worker_init_fn_seed(worker_id):
        seed = 10
        seed += worker_id
        np.random.seed(seed)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               num_workers=NUM_DATASET_WORKERS,
                                               pin_memory=True,
                                               batch_size=config.batch_size,
                                               worker_init_fn=worker_init_fn_seed,
                                               shuffle=True,
                                               drop_last=True)
    if args.trainset == 'CIFAR10':
        test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=1024,
                                  shuffle=False)

    else:
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    test_data_dir = ['/home/caobin/dongb/dataset/test_cifar/test_cifar10/airplane']
    test_dataset = Datasets(test_data_dir)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)
    import time

    import torch
    import torchvision
    from torch.nn import MSELoss
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from torchvision import transforms
    from utils import *
    from models.Mingyu2021.DynaAWGN import DynaAWGNModel

    # 定义变量
    args = args_parser()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")  # 指定GPU
    # train_losses, test_losses, train_psnrs, test_psnrs, elapsed = [AverageMeter() for _ in range(5)]
    loss_fn = MSELoss()  # 损失函数

    args.phase = "test"
    # test_snrs = [-3, 1, 3, 5, 7, 9, 10, 11, 13, 15, 17, 19, 21, 23]
    test_snrs = [10]

    log_filename = "{}_{}_awgn_{}epochs_{}db".format(args.model, args.dataset, args.epochs,
                                                     args.snr)  # 修改args里面的参数，则训练产生的日志在不同的文件夹
    workdir, logger = logger_configuration(log_filename, args.phase, save_log=args.save_log)
    logger.info("======Begin to test: " + log_filename + "======")

    # # 加载测试数据集
    # # transform_train = transforms.Compose([transforms.ToTensor()])
    # transform_test = transforms.Compose(
    #     [transforms.ToTensor(), ])
    # test_data = torchvision.datasets.CIFAR10("../../dataset/cifar", train=False,
    #                                          transform=transform_test)
    # test_loader = DataLoader(test_data, batch_size=args.batchsize, shuffle=True, num_workers=2, drop_last=True)

    checkpoints = torch.load(workdir + "/models/" + "DynaAWGN_cifar_awgn_10epochs_10db.model")
    print(workdir + "/models/" + "DynaAWGN_cifar_awgn_4epochs_10db.model")
    jscc = DynaAWGNModel().to(device)
    jscc.load_state_dict(checkpoints)
    for test_snr in test_snrs:
        jscc.snr = test_snr
        test_losses, test_psnrs, cbrs, elapsed, snrs = [AverageMeter() for _ in range(5)]
        metrics = [test_losses, test_psnrs, cbrs, elapsed, snrs]
        # jscc.load_state_dict(checkpoints)
        jscc.eval()
        with torch.no_grad():
            start_time = time.time()
            for data in test_loader:
                input_imgs, _ = data
                input_imgs = input_imgs.to(device)

                output_imgs, count, cbr, snr = jscc(input_imgs)
                loss_mse = loss_fn(input_imgs * 255., output_imgs * 255.)
                loss = args.lambda_L2 * loss_mse + args.lambda_reward * torch.mean(count)
                psnr = 10 * (torch.log(255. * 255. / loss_mse) / np.log(10))

                test_losses.update(loss)
                test_psnrs.update(psnr)
                cbrs.update(cbr)
                snrs.update(snr)

                elapsed.update(time.time() - start_time)

            log = (" | ".join([
                f'Test Loss {test_losses.val:.4f} ({test_losses.avg:.4f})',
                f'CBR {cbrs.val:.8f} ({cbrs.avg:.4f})',
                f'Time {elapsed.avg:.4f}',
                f'PSNR {test_psnrs.val:.4f}({test_psnrs.avg:.4f})',
                f'SNR {int(snrs.val)}',
            ]))
            logger.info(log)
            for i in metrics:
                i.clear()
