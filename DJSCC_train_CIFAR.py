# -*- coding: utf-8 -*-
"""
@Time ： 2023/10/9 10:50
@Auth ： dongb
@IDE ：PyCharm
"""
import time
import torch
import torchvision
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils import *
from models.DJSCC import Jscc  # Deep JSCC model

# =================== Argument and Environment Setup ===================
args = args_parser()
learning_rate = 1e-4
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
loss_fn = MSELoss()  # Mean Squared Error Loss
best_loss = float("inf")  # For tracking the best test loss
log_filename = "{}_{}_awgn_{}epochs_{}db_cbr{}".format(args.model, args.dataset, args.epochs, args.snr, args.c)

# Logger and TensorBoard Writer
workdir, logger = logger_configuration(log_filename, args.phase, save_log=args.save_log)
writter = SummaryWriter("history/" + log_filename + "/Tensorboard_logs")
logger.info("======Begin to train: " + log_filename + "======")
logger.info(args)

# =================== Dataset Preparation ===================
# Data augmentation for training
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=5, pad_if_needed=True, fill=0, padding_mode='reflect'),
    transforms.ToTensor()
])

# Load CIFAR-10 training set
train_data = torchvision.datasets.CIFAR10("../../dataset/cifar", train=True, transform=transform_train)
train_loader = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=0, drop_last=True)

# Test data (no augmentation)
transform_test = transforms.Compose([transforms.ToTensor()])
test_data = torchvision.datasets.CIFAR10("../../dataset/cifar", train=False, transform=transform_test)
test_loader = DataLoader(test_data, batch_size=args.batchsize, shuffle=True, num_workers=0, drop_last=True)

# =================== Model Initialization ===================
jscc = Jscc().to(device)  # Create model and move to device
optimizer = torch.optim.Adam(jscc.parameters(), lr=args.learning_rate)

# =================== Training Loop ===================
for epoch in range(args.epochs):
    jscc.train()
    # Metrics
    train_losses, test_losses, train_psnrs, test_psnrs, elapsed = [AverageMeter() for _ in range(5)]
    metrics = [train_losses, test_losses, train_psnrs, test_psnrs, elapsed]
    start_time = time.time()

    # ========== Train Phase ==========
    for data in train_loader:
        input_imgs, _ = data
        input_imgs = input_imgs.to(device)

        # Random SNR values
        SNR_TRAIN = torch.randint(0, 20, (args.batchsize, 1)).to(args.device)
        output_imgs = jscc(input_imgs, SNR_TRAIN)
        loss = loss_fn(input_imgs * 255., output_imgs * 255.)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update training time and metrics
        elapsed.update(time.time() - start_time)
        psnr = 10 * (torch.log(255. * 255. / loss) / np.log(10))
        train_losses.update(loss)
        train_psnrs.update(psnr)

    # Log training results
    log = ' | '.join([
        f'Epoch {epoch}',
        f'Train Loss {train_losses.val:.4f} ({train_losses.avg:.4f})',
        f'Time {elapsed.avg:.4f}',
        f'PSNR {train_psnrs.val:.4f} ({train_psnrs.avg:.4f})'
    ])
    logger.info(log)
    writter.add_scalar("train_loss", train_losses.val, global_step=epoch)

    # ========== Test Phase ==========
    jscc.eval()
    with torch.no_grad():
        start_time = time.time()
        for data in test_loader:
            input_imgs, _ = data
            input_imgs = input_imgs.to(device)

            # Random SNR values
            SNR_TEST = torch.randint(0, 20, (args.batchsize, 1)).to(args.device)
            output_imgs = jscc(input_imgs, SNR_TEST)
            loss = loss_fn(input_imgs * 255., output_imgs * 255.)

            # Update test time and metrics
            elapsed.update(time.time() - start_time)
            psnr = 10 * (torch.log(255. * 255. / loss) / np.log(10))
            test_losses.update(loss)
            test_psnrs.update(psnr)

        # Log testing results
        log = " | ".join([
            f'Epoch {epoch}',
            f'Test Loss {test_losses.val:.4f} ({test_losses.avg:.4f})',
            f'Time {elapsed.avg:.4f}',
            f'PSNR {test_psnrs.val:.4f} ({test_psnrs.avg:.4f})'
        ])
        logger.info(log)
        writter.add_scalar("test_loss", test_losses.avg, global_step=epoch)

        # ========== Save Model ==========
        checkpoint = {
            "models": jscc.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }

        if test_losses.avg < best_loss and epoch != (args.epochs - 1):
            # Save best-performing model during training
            save_model_name = "{}_{}_awgn_{}db_cbr{}.model".format(args.model, args.dataset, args.snr, args.c)
            torch.save(jscc.state_dict(), workdir + "/models/" + save_model_name)
            best_loss = test_losses.avg
        elif epoch == (args.epochs - 1):
            # Save final model at the end of training
            save_model_name = "{}_{}_awgn_{}epochs_{}db_cbr{}.model".format(args.model, args.dataset, epoch, args.snr, args.c)
            torch.save(checkpoint, workdir + "/models/" + save_model_name)
            best_loss = loss

        # Reset metrics for next epoch
        for i in metrics:
            i.clear()

# Close TensorBoard writer
writter.close()
logger.info("Model training completed")
