# -*- coding: utf-8 -*-
"""
@Time ： 2023/11/3 10:55
@Auth ： dongb
@File ：DJSCC.py
@IDE ：PyCharm
"""
import torch.nn as nn
import torch.fft, copy
from models.Dynamic_Convolution import Involution2d
from models.GDN import GDN
from models.ESAM import *
from utils import *
from pytorch_wavelets import DWTInverse, DWTForward
args = args_parser()

class AF_block(nn.Module):
    def __init__(self, Nin, Nh, No, args):
        super(AF_block, self).__init__()
        self.fc1 = nn.Linear(Nin + 1, Nh)
        self.fc2 = nn.Linear(Nh, No)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.args = args

    def forward(self, x, snr):
        mu = torch.mean(x, (2, 3)).to(self.args.device)
        snr = snr.to(self.args.device)
        out = torch.cat((mu, snr), 1).to(self.args.device)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.unsqueeze(2)
        out = out.unsqueeze(3)
        out = out * x
        return out


class Encoder_stage_ACmix(nn.Module):
    def __init__(self, in_channels, out_channels, use_conv1x1=False, kernel_size=3, stride=1, padding=1):
        super(Encoder_stage_ACmix, self).__init__()
        self.conv1 = ACmix(in_channels, out_channels, stride=stride)
        self.conv2 = conv(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.gdn1 = GDN(out_channels)
        self.gdn2 = GDN(out_channels)
        self.prelu = nn.PReLU()
        self.use_conv1x1 = use_conv1x1
        if use_conv1x1 == True:
            self.conv3 = conv(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.gdn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.gdn2(out)

        if self.use_conv1x1 == True:
            x = self.conv3(x)

        out = out + x
        out = self.prelu(out)
        return out

class Encoder_stage_Invo(nn.Module):
    def __init__(self, in_channels, out_channels, use_conv1x1=False, kernel_size=3, stride=1, padding=1,groups=1,reduce_ratio=4):
        super(Encoder_stage_Invo, self).__init__()
        # self.conv1 = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv1 = Involution2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups, reduce_ratio=reduce_ratio, dilation=1, padding=padding, bias=False)
        self.conv2 = conv(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        # self.conv2 = ACmix(out_channels, out_channels, stride=stride)
        # self.conv1 = conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.gdn1 = GDN(out_channels)
        self.gdn2 = GDN(out_channels)
        self.prelu = nn.PReLU()
        self.use_conv1x1 = use_conv1x1
        if use_conv1x1 == True:
            self.conv3 = conv(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.gdn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.gdn2(out)

        if self.use_conv1x1 == True:
            x = self.conv3(x)

        out = out + x
        out = self.prelu(out)
        return out


class Decoder_stage(nn.Module):
    def __init__(self, in_channels, out_channels, use_deconv1x1=False, kernel_size=3, stride=1, padding=1,
                 output_padding=0):
        super(Decoder_stage, self).__init__()
        self.deconv1 = deconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              output_padding=output_padding)
        self.deconv2 = deconv(out_channels, out_channels, kernel_size=1, stride=1, padding=0, output_padding=0)
        self.gdn1 = GDN(out_channels)
        self.gdn2 = GDN(out_channels)
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.use_deconv1x1 = use_deconv1x1
        if use_deconv1x1 == True:
            self.deconv3 = deconv(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                  output_padding=output_padding)

    def forward(self, x, activate_func='prelu'):
        out = self.deconv1(x)
        out = self.gdn1(out)
        out = self.prelu(out)
        out = self.deconv2(out)
        out = self.gdn2(out)
        if self.use_deconv1x1 == True:
            x = self.deconv3(x)
        out = out + x
        if activate_func == 'prelu':
            out = self.prelu(out)
        elif activate_func == 'sigmoid':
            out = self.sigmoid(out)
        return out

class Decoder_stage_ACmix(nn.Module):
    def __init__(self, in_channels, out_channels, use_deconv1x1=False, kernel_size=3, stride=1, padding=1,
                 output_padding=0):
        super(Decoder_stage_ACmix, self).__init__()
        self.deconv1 = ACmix(in_channels, out_channels, stride=stride)
        self.deconv2 = deconv(out_channels, out_channels, kernel_size=1, stride=1, padding=0, output_padding=0)
        self.gdn1 = GDN(out_channels)
        self.gdn2 = GDN(out_channels)
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.use_deconv1x1 = use_deconv1x1
        if use_deconv1x1 == True:
            self.deconv3 = deconv(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                  output_padding=output_padding)

    def forward(self, x, activate_func='prelu'):
        out = self.deconv1(x)
        out = self.gdn1(out)
        out = self.prelu(out)
        out = self.deconv2(out)
        out = self.gdn2(out)
        if self.use_deconv1x1 == True:
            x = self.deconv3(x)
        out = out + x
        if activate_func == 'prelu':
            out = self.prelu(out)
        elif activate_func == 'sigmoid':
            out = self.sigmoid(out)
        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.kernel_sz = 5
        self.enc_shape = [32, 8, 8]
        self.enc_N = self.enc_shape[0]
        channel_N = 256
        self.padding_L = (self.kernel_sz - 1) // 2
        self.encoder_stage1 = Encoder_stage_ACmix(3, channel_N, use_conv1x1=True, kernel_size=self.kernel_sz,
                                                  stride=2, padding=self.padding_L)
        self.encoder_stage2 = Encoder_stage_Invo(channel_N, channel_N, use_conv1x1=True, kernel_size=3, stride=2,
                                                 padding=1, groups=8, reduce_ratio=4)
        self.encoder_stage3 = Encoder_stage_Invo(channel_N, self.enc_N, use_conv1x1=True, kernel_size=self.kernel_sz, stride=1,
                                                 padding=self.padding_L, groups=1, reduce_ratio=1)
        self.AF1 = AF_block(channel_N, channel_N, channel_N,args)
        self.AF2 = AF_block(channel_N, channel_N, channel_N,args)
        self.AF3 = AF_block(self.enc_N, self.enc_N // 2, self.enc_N, args)

    def forward(self, z, SNR):
        z = self.encoder_stage1(z)
        z = self.AF1(z, SNR)
        z = self.encoder_stage2(z)
        z = self.AF2(z, SNR)
        z = self.encoder_stage3(z)
        z = self.AF3(z, SNR)
        return z

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.kernel_sz = 5
        self.enc_shape = [32, 8, 8]
        self.enc_N = self.enc_shape[0]
        channel_N = 256
        self.padding_L = (self.kernel_sz - 1) // 2
        self.decoder_stage1 = Decoder_stage(self.enc_N, channel_N, use_deconv1x1=True, kernel_size=self.kernel_sz, stride=2, padding=self.padding_L, output_padding=1)
        self.decoder_stage2 = Decoder_stage(channel_N, channel_N, use_deconv1x1=True, kernel_size=self.kernel_sz, stride=2, padding=self.padding_L, output_padding=1)
        self.decoder_stage3 = Decoder_stage(channel_N, channel_N, use_deconv1x1=False, kernel_size=self.kernel_sz, stride=1, padding=self.padding_L)
        self.decoder_stage4 = Decoder_stage(channel_N, channel_N, use_deconv1x1=False, kernel_size=self.kernel_sz, stride=1, padding=self.padding_L)
        self.decoder_stage5 = Decoder_stage(channel_N, 3, use_deconv1x1=True, kernel_size=self.kernel_sz, stride=1, padding=self.padding_L)
        self.AF1 = AF_block(self.enc_N, self.enc_N // 2, self.enc_N, args)
        self.AF2 = AF_block(channel_N, channel_N, channel_N, args)
        self.AF3 = AF_block(channel_N, channel_N, channel_N, args)
        self.AF4 = AF_block(channel_N, channel_N, channel_N, args)
        self.AF5 = AF_block(channel_N, channel_N, channel_N, args)

    def forward(self, z, SNR):
        z = self.AF1(z,SNR)
        z = self.decoder_stage1(z)
        z = self.AF2(z, SNR)
        z = self.decoder_stage2(z)
        z = self.AF3(z, SNR)
        z = self.decoder_stage3(z)
        z = self.AF4(z, SNR)
        z = self.decoder_stage4(z)
        z = self.AF5(z, SNR)
        z = self.decoder_stage5(z,"sigmod")
        return z

class Jscc(nn.Module):
    def __init__(self):
        super(Jscc, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.DWT = DWTForward(wave='db1', mode='zero').to(args.device)
        self.IDWT = DWTInverse(wave='db1', mode='zero').to(args.device)

    def powerNorm(self, z):
        input_shape = z.shape
        z_com = z.contiguous().view(input_shape[0],-1,2)
        z_com = torch.view_as_complex(z_com)
        pwr_z_com = torch.sum(torch.abs(z_com)**2,dim=-1)/torch.numel(z_com[0])
        z_com = z_com / torch.sqrt(pwr_z_com.unsqueeze(-1))
        return z_com

    def add_noise(self, z, SNR):
        input_shape = z.shape
        noise_shape = [input_shape[0], input_shape[1]]
        sig_real = torch.randn(*noise_shape)
        sig_imag = torch.randn(*noise_shape)
        noise_sd = (torch.complex(sig_real, sig_imag) / np.sqrt(2)).to(args.device)
        noise_sd_pwr = 10 ** (-SNR[0] / 10)
        z = z + torch.sqrt(torch.tensor(noise_sd_pwr)) * noise_sd

        # h_I = torch.randn(*noise_shape).to(args.device)
        # h_R = torch.randn(*noise_shape).to(args.device)
        # h_com = torch.complex(h_I, h_R)
        # z = h_com * z + torch.sqrt(torch.tensor(noise_sd_pwr)) * noise_sd  # [64, 1024]
        # z = z / h_com
        return z

    def forward(self, x, SNR):
        # ===================Encoding
        z = self.encoder(x,SNR)
        # ===================Wavelet decomposition
        yl, yh = self.DWT(z)
        yll = yl.unsqueeze(2)
        out = torch.cat((yll, yh[0]), dim=2)
        c_a, c_h, c_v, c_d = torch.split(out, split_size_or_sections=1, dim=2)
        coeffs_shape = c_a.shape
        # ===================Power Normalize & Pass channel
        c_a, c_h, c_v, c_d = [self.add_noise(self.powerNorm(c), SNR) for c in [c_a, c_h, c_v, c_d]]
        # ===================Wavelet synthesis
        c_a, c_h, c_v, c_d = [
            torch.view_as_real(c).view(*coeffs_shape) for c in [c_a, c_h, c_v, c_d]
        ]
        coeffs = (c_a.squeeze(2), [torch.cat((c_h, c_v, c_d), dim=2)])
        z = self.IDWT(coeffs=coeffs)
        x_hat = self.decoder(z,SNR)
        return x_hat

if __name__ == "__main__":
    SNR = 2 * torch.ones((args.batchsize, 1), dtype=torch.float32).to(args.device)
    jscc = Jscc().to(args.device)
    z = torch.ones(32, 3, 32, 32).to(args.device)
    # summary(jscc, z)
    z_hat = jscc(z,SNR)
    print(z_hat.shape)
