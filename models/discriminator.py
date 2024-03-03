from torch import nn
import functools
import torch

class LatentCodesDiscriminator(nn.Module):
    def __init__(self, style_dim, n_mlp):
        super().__init__()

        self.style_dim = style_dim

        layers = []
        for i in range(n_mlp-1):
            layers.append(
                nn.Linear(style_dim, style_dim)
            )
            layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(512, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, w):
        return self.mlp(w)

'''
class SNTemporalPatchGANDiscriminator(BaseModule):
    def __init__(
        self, nc_in, nf=64, norm='SN', use_sigmoid=True, use_bias=True, conv_type='vanilla',
        conv_by='3d'
    ):
        super().__init__(conv_type)
        use_bias = use_bias
        self.use_sigmoid = use_sigmoid

        ######################
        # Convolution layers #
        ######################
        self.conv1 = self.ConvBlock(
            nc_in, nf * 1, kernel_size=(3, 5, 5), stride=(1, 2, 2),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv2 = self.ConvBlock(
            nf * 1, nf * 2, kernel_size=(3, 5, 5), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv3 = self.ConvBlock(
            nf * 2, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv4 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv5 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv6 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=None, activation=None,
            conv_by=conv_by
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, xs):
        # B, L, C, H, W = xs.shape
        xs_t = torch.transpose(xs, 1, 2)
        c1 = self.conv1(xs_t)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)
        if self.use_sigmoid:
            c6 = torch.sigmoid(c6)
        out = torch.transpose(c6, 1, 2)
        return out
'''

# Based on https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class TemporalPatchGANDiscriminator(nn.Module):
    def __init__(self, input_nc, output_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False):
        super().__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = (3, 4, 4)
        padw = (1, 1, 1)
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=(1, 2, 2), padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=(1, 2, 2), padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, output_nc, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        input = torch.transpose(input, 1, 2)
        return self.model(input)
        