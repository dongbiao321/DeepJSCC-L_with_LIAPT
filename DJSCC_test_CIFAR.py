# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/14 19:59
@Auth ： dongb
@IDE ：PyCharm
"""
import time
import torchvision
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils import *
from models.DJSCC import *
# ===================Definitions
args = args_parser()
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
# train_losses, test_losses, train_psnrs, test_psnrs, elapsed = [AverageMeter() for _ in range(5)]
loss_fn = MSELoss()
args.phase = "test"
# test_snrs = [ -3, 1, 3, 5, 7, 9,10, 11, 13, 15, 17, 19, 21, 23]
test_snrs = [10]

log_filename = "{}_{}_awgn_{}epochs_{}db_cbr{}".format(args.model,args.dataset, args.epochs, args.snr, args.c) #修改args里面的参数，则训练产生的日志在不同的文件夹

workdir, logger = logger_configuration(log_filename, args.phase, save_log=args.save_log)
logger.info("======Begin to test: "+log_filename + "======")

# ===================Datasets
if args.dataset == "cifar":# The test set of cifar10
    transform_train = transforms.Compose([transforms.ToTensor()])
    transform_test = transforms.Compose(
        [transforms.ToTensor(),])
    test_data = torchvision.datasets.CIFAR100("../../dataset/cifar100", train=False,
                                              transform=transform_train)
    test_loader = DataLoader(test_data, batch_size=args.batchsize, shuffle=True, num_workers=2,drop_last=True)
else:                      # The average rate for different classes -> for data,_ in test_loader: delete _
    from data.datasets import *
    test_data_dir = ['/home/caobin/dongb/dataset/test_cifar/test_cifar10/horse']
    test_dataset = Datasets(test_data_dir)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batchsize, shuffle=True, num_workers=2,drop_last=True)

# ===================Load model
checkpoints = torch.load(workdir+"/models/"+"bdjscc_cifar_awgn_199epochs_10db_cbr32.model")
jscc = Jscc(args.snr).to(device)
jscc.load_state_dict(checkpoints["models"])
# ===================Test
# results_cbr = np.zeros(len(test_snrs))
from pytorch_wavelets import DWTInverse, DWTForward
DWT = DWTForward(wave='db1', mode='zero').to(args.device)
IDWT = DWTInverse(wave='db1', mode='zero').to(args.device)
results_psnr = np.zeros(len(test_snrs))
for i, test_snr in enumerate(test_snrs):
    jscc.snr = test_snr*torch.ones((args.batchsize, 1), dtype=torch.float32).to(args.device)
    test_losses, test_psnrs, cbrs, elapsed,snrs = [AverageMeter() for _ in range(5)]
    metrics = [test_losses, test_psnrs, cbrs, elapsed,snrs]
    # jscc.load_state_dict(checkpoints)
    jscc.eval()
    with torch.no_grad():
        start_time = time.time()
        for data,_ in test_loader:
            input_imgs = data
            input_imgs = input_imgs.to(device)
            input_shape = input_imgs.shape
            test_snrs = [20, 10, 5, 0]
            snr = 7 * torch.ones((args.batchsize, 1), dtype=torch.float32).to(args.device)
            z = jscc.encoder(input_imgs,snr)
            yl, yh = DWT(z)
            yll = yl.unsqueeze(2)
            out = torch.cat((yll, yh[0]), dim=2)
            c_a, c_h, c_v, c_d = torch.split(out, split_size_or_sections=1, dim=2)
            coeffs_shape = c_a.shape
            z_shape = z.shape
            # ===================Power Normalize
            c_a, c_h, c_v, c_d = [jscc.powerNorm(c) for c in [c_a, c_h, c_v, c_d]]
            # ===================Pass channel
            for i in range(4):
                coeffs[i] = jscc.add_noise(coeffs[i], snrs[i])
            # ===================Wavelet synthesis
            c_a, c_h, c_v, c_d = [
                torch.view_as_real(c).view(*coeffs_shape) for c in [c_a, c_h, c_v, c_d]
            ]
            coeffs = (c_a.squeeze(2), [torch.cat((c_h, c_v, c_d), dim=2)])
            z = IDWT(coeffs=coeffs)
            # ===================Decoding
            output_imgs = jscc.decoder(z, snr)
            loss= loss_fn(input_imgs * 255., output_imgs * 255.)
            psnr = 10 * (torch.log(255. * 255. / loss) / np.log(10))
            test_losses.update(loss)
            test_psnrs.update(psnr)
            elapsed.update(time.time() - start_time)

    log = (" | ".join([
        f'Test Loss {test_losses.val:.4f} ({test_losses.avg:.4f})',
        f'Time {elapsed.avg:.4f}',
        f'PSNR {test_psnrs.val:.4f}({test_psnrs.avg:.4f})',
    ]))
    logger.info(log)

    # results_cbr[i] = round(cbrs.avg, 4)
    results_psnr[i] = round(float(test_psnrs.avg), 4)

# print("CBR: {}".format(results_cbr.tolist()))
print("PSNR: {}".format(results_psnr.tolist()))
print(results_psnr[6])



