import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_wavelets import DWTForward, DWTInverse
import matplotlib.pyplot as plt
from odconv import ODConv2d



class od_attention(nn.Module):
    def __init__(self, channels):
        super(od_attention, self).__init__()

        self.od_conv = ODConv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        od_out = self.od_conv(x)
        
        out = self.conv(x)
        attention = F.gelu(od_out)

        return out*attention


class SSL(nn.Module): 
    def __init__(self, channels):
        super(SSL, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)#conv_block_my(channels, channels, kernel_size = 3, stride=1, padding = 1, dilation=1)
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)#conv_block_my(channels, channels, kernel_size = 3, stride=1, padding = 5, dilation=5)
        self.conv7 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)#conv_block_my(channels, channels, kernel_size = 3, stride=1, padding = 7, dilation=7)
        self.conv9 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)#conv_block_my(channels, channels, kernel_size = 3, stride=1, padding = 9, dilation=9)

        self.conv_cat = nn.Conv2d(channels*4, channels, kernel_size=3, padding=1, groups=channels, bias=False)#conv_block_my(channels*4, channels, kernel_size = 3, stride = 1, padding = 1, dilation=1)

    def forward(self, x):

        aa =  DWTForward(J=1, mode='zero', wave='db3').cuda()
        yl, yh = aa(x)

        yh_out = yh[0]
        ylh = yh_out[:,:,0,:,:]
        yhl = yh_out[:,:,1,:,:]
        yhh = yh_out[:,:,2,:,:]

        conv_rec1 = self.conv5(yl)
        conv_rec5 = self.conv5(ylh)
        conv_rec7 = self.conv7(yhl)
        conv_rec9 = self.conv9(yhh)

        cat_all = torch.stack((conv_rec5, conv_rec7, conv_rec9),dim=2)
        rec_yh = []
        rec_yh.append(cat_all)


        ifm = DWTInverse(wave='db3', mode='zero').cuda()
        Y = ifm((conv_rec1, rec_yh))

        return Y

class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
        self.query = SSL(channels)
        self.qkv_conv = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, groups=channels * 2, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        k, v = self.qkv_conv(self.qkv(x)).chunk(2, dim=1)
        q = self.query(x)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))
        return x

class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Inpainting(nn.Module):
    def __init__(self, num_blocks=[4, 6, 6, 8], num_heads=[1, 2, 4, 8], channels=[48//3, 96//3, 192//3, 384//3], num_refinement=4,
                 expansion_factor=2.66):
        super(Inpainting, self).__init__()

        self.embed_conv = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)

        self.encoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(
            num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) for num_tb, num_ah, num_ch in
                                       zip(num_blocks, num_heads, channels)])

        # self.res_layers = nn.ModuleList([nn.Sequential(*[TransformerBlock_Res(
        #     num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) for num_tb, num_ah, num_ch in
        #                                zip(num_blocks, num_heads, [384//3, 384//3,384//3,384//3])])
        # the number of down sample or up sample == the number of encoder - 1
        self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in channels[:-1]])
        
        self.skips = nn.ModuleList([od_attention(num_ch) for num_ch in list(reversed(channels))[1:]])
        self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])
        # the number of reduce block == the number of decoder - 1
        self.reduces = nn.ModuleList([nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
                                      for i in reversed(range(2, len(channels)))])
        # the number of decoder == the number of encoder - 1
        self.decoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(channels[2], num_heads[2], expansion_factor)
                                                       for _ in range(num_blocks[2])])])
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[1], expansion_factor)
                                             for _ in range(num_blocks[1])]))
        # the channel of last one is not change
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                             for _ in range(num_blocks[0])]))

        self.refinement = nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                          for _ in range(num_refinement)])
        self.output = nn.Conv2d(channels[1], 3, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        fo = self.embed_conv(x)
        out_enc1 = self.encoders[0](fo)
        out_enc2 = self.encoders[1](self.downs[0](out_enc1))
        out_enc3 = self.encoders[2](self.downs[1](out_enc2))
        out_enc4 = self.encoders[3](self.downs[2](out_enc3))

        out_dec3 = self.decoders[0](self.reduces[0](torch.cat([self.ups[0](out_enc4), self.skips[0](out_enc3)], dim=1)))
        out_dec2 = self.decoders[1](self.reduces[1](torch.cat([self.ups[1](out_dec3), self.skips[1](out_enc2)], dim=1)))
        fd = self.decoders[2](torch.cat([self.ups[2](out_dec2), self.skips[2](out_enc1)], dim=1))
        fr = self.refinement(fd)
        out = self.output(fr)
        return out