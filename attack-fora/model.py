# import tensorflow as tf
# import numpy as np
import functools
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, bn=False, stride=1):
        super(ResBlock, self).__init__()
        self.bn = bn
        if bn:
            self.bn0 = nn.BatchNorm2d(in_planes)

        
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1)
        if bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1)
        self.shortcut = nn.Sequential()

        if stride > 1:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn0(x))
        else:
            out = F.relu(x)

        if self.bn:
            out = F.relu(self.bn1(self.conv1(out)))
        else:
            out = F.relu(self.conv1(out))

        out = self.conv2(out)
        out += self.shortcut(x)
        return out

class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.num_classes = num_classes
        self.in_channels = 3
        self.conv1 = self._make_layers([64, 64, "M"])
        self.conv2 = self._make_layers([128, 128, "M"])
        self.conv3 = self._make_layers([256, 256, 256, "M"])
        self.conv4 = self._make_layers([512, 512, 512, "M"])
        self.conv5 = self._make_layers([512, 512, 512, "M"])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 1 * 1, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg, batch_norm=True):
        layers = []
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(self.in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                self.in_channels = v
        return nn.Sequential(*layers)

def vgg16_make_layers(cfg, batch_norm=True, in_channels=3):
    layers = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg16(level, batch_norm, num_class = 10):

    client_net = []
    server_net = []
    # print(level)
    if level == 1 :
        client_net += vgg16_make_layers([32, 32, "M"], batch_norm, in_channels=3)
        server_net += vgg16_make_layers([128, 128, "M"], batch_norm, in_channels=64)
        server_net += vgg16_make_layers([256, 256, 256, "M"], batch_norm, in_channels=128)
        server_net += vgg16_make_layers([512, 512, 512, "M"], batch_norm, in_channels=256)
        server_net += vgg16_make_layers([512, 512, 512, "M"], batch_norm, in_channels=512)
        server_net += [nn.AdaptiveAvgPool2d((1, 1))]
        server_net += [nn.Flatten(),nn.Linear(512 * 1 * 1, num_class)]
        return nn.Sequential(*client_net),nn.Sequential(*server_net)


    if level == 2 :
        client_net += vgg16_make_layers([64,64,"M"], batch_norm, in_channels=3)
        server_net += vgg16_make_layers([128, 128, "M"], batch_norm, in_channels=64)
        server_net += vgg16_make_layers([256, 256, 256, "M"], batch_norm, in_channels=128)
        server_net += vgg16_make_layers([512, 512, 512, "M"], batch_norm, in_channels=256)
        server_net += vgg16_make_layers([512, 512, 512, "M"], batch_norm, in_channels=512)
        server_net += [nn.AdaptiveAvgPool2d((1, 1))]
        server_net += [nn.Flatten(),nn.Linear(512 * 1 * 1, num_class)]
        return nn.Sequential(*client_net),nn.Sequential(*server_net)

    if level == 3:
        client_net += vgg16_make_layers([64, 64, "M"], batch_norm, in_channels=3)
        client_net += vgg16_make_layers([128, 128, "M"], batch_norm, in_channels=64)
        server_net += vgg16_make_layers([256, 256, 256, "M"], in_channels=128)
        server_net += vgg16_make_layers([512, 512, 512, "M"], in_channels=256)
        server_net += vgg16_make_layers([512, 512, 512, "M"], in_channels=512)
        server_net += [nn.AdaptiveAvgPool2d((1, 1))]
        server_net += [nn.Flatten(),nn.Linear(512 * 1 * 1, num_class)]
        return nn.Sequential(*client_net),nn.Sequential(*server_net)

    if level == 4:
        client_net += vgg16_make_layers([64, 64, "M"], batch_norm, in_channels=3)
        client_net += vgg16_make_layers([128, 128, "M"], batch_norm, in_channels=64)
        client_net += vgg16_make_layers([256, 256, 256, "M"], batch_norm, in_channels=128)
        server_net += vgg16_make_layers([512, 512, 512, "M"], in_channels=256)
        server_net += vgg16_make_layers([512, 512, 512, "M"], in_channels=512)
        server_net += [nn.AdaptiveAvgPool2d((1, 1))]
        server_net += [nn.Flatten(),nn.Linear(512 * 1 * 1, num_class)]
        return nn.Sequential(*client_net),nn.Sequential(*server_net)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


def cifar_mobilenet(level):
    client = []
    server = []

    if level == 1:
        client += conv_bn(  3,  32, 2)
    #     client += nn.Sequential(
    #     nn.Conv2d(3,32, 3, 2, 1, bias=False),
    #     nn.ReLU(inplace=True)
    # )
        server += conv_dw( 32,  64, 1)
        server += conv_dw( 64, 128, 2)
        server += conv_dw(128, 128, 1)
        server += conv_dw(128, 256, 2)
        server += conv_dw(256, 256, 1)
        server += conv_dw(256, 512, 2)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 1024, 2)
        server += conv_dw(1024, 1024, 1)
        server += nn.Sequential(nn.AvgPool2d(1),nn.Flatten(),nn.Linear(1024, 10))
        return nn.Sequential(*client),nn.Sequential(*server)
    if level == 2:
        client += conv_bn(  3,  32, 2)
        client += conv_dw( 32,  64, 1)
        server += conv_dw( 64, 128, 2)
        server += conv_dw(128, 128, 1)
        server += conv_dw(128, 256, 2)
        server += conv_dw(256, 256, 1)
        server += conv_dw(256, 512, 2)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 1024, 2)
        server += conv_dw(1024, 1024, 1)
        server += nn.Sequential(nn.AvgPool2d(1),nn.Flatten(),nn.Linear(1024, 10))
        return nn.Sequential(*client),nn.Sequential(*server)
    if level == 3:
        client += conv_bn(  3,  32, 2)
        client += conv_dw( 32,  64, 1)
        client += conv_dw( 64, 128, 2)
        client += conv_dw(128, 128, 1)
        server += conv_dw(128, 256, 2)
        server += conv_dw(256, 256, 1)
        server += conv_dw(256, 512, 2)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 1024, 2)
        server += conv_dw(1024, 1024, 1)
        server += nn.Sequential(nn.AvgPool2d(1),nn.Flatten(),nn.Linear(1024, 10))
        return nn.Sequential(*client),nn.Sequential(*server)
    if level == 4:
        client += conv_bn(  3,  32, 2)
        client += conv_dw( 32,  64, 1)
        client += conv_dw( 64, 128, 2)
        client += conv_dw(128, 128, 1)
        client += conv_dw(128, 256, 2)
        client += conv_dw(256, 256, 1)
        server += conv_dw(256, 512, 2)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 512, 1)
        server += conv_dw(512, 1024, 2)
        server += conv_dw(1024, 1024, 1)
        server += nn.Sequential(nn.AvgPool2d(1),nn.Flatten(),nn.Linear(1024, 10))
        return nn.Sequential(*client),nn.Sequential(*server)




def cifar_decoder(input_shape, level, channels=3):
    
    net = []
    #act = "relu"
    act = None
    print("[DECODER] activation: ", act)

    net += [nn.ConvTranspose2d(input_shape[0], 256, 3, 2, 1, output_padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True)]

    if level <= 2:
        net += [nn.Conv2d(256, channels, 3, 1, 1), nn.BatchNorm2d(channels)]
        net += [nn.Tanh()]
        return nn.Sequential(*net)
    
    net += [nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True)]

    if level == 3:
        net += [nn.Conv2d(128, channels, 3, 1, 1), nn.BatchNorm2d(channels)]
        net += [nn.Tanh()]
        return nn.Sequential(*net)
    
    net += [nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True)]

    if level == 4:
        net += [nn.Conv2d(64, channels, 3, 1, 1), nn.BatchNorm2d(channels)]
        net += [nn.Tanh()]
        return nn.Sequential(*net)


class SparseLiftDecoder(nn.Module):
    """Inversion decoder with a sparse-aware 'lift' front-end (stronger attacker for
    top-k sparse smashed data).

    topk_sparse sends the EXACT activation values at the top-k_s magnitude positions
    and zeros the rest; crucially every UNSENT coord j is bounded by tau = the
    smallest sent magnitude (it lost the top-k race), i.e. |h_j| <= tau. This module
    exploits both facts before the usual cifar_decoder:
      - support mask (which coords are exact) + the per-sample bound tau are fed in,
      - a learned lift (the attacker's R^T-analogue) infers the missing coords,
      - the infill is CLAMPED to [0, tau] (post-ReLU bound) and applied only at the
        unsent positions, while the sent values pass through EXACTLY ("raw" at support).
    On dense inputs (fixed_random / block_random) the mask is all-ones so the infill
    term vanishes and this reduces to the baseline cifar_decoder -- a strictly
    stronger, general attacker. Opt-in via main.py --inv-lift.

    Optional class-conditional prior (num_classes>0): the attacker knows the victim's
    label (server-side CE loss) and has labelled shadow data, so it can estimate the
    per-class mean activation mu_c. The unsent coords are then filled with mu_c[label]
    (+ a learned residual, clamped to [0, tau]) instead of 0 -- a far better prior than
    zero. prior_only=True is the label-only CONTROL: reconstruct from mu_c alone with the
    activation ignored, to isolate label-prior leakage from true activation leakage.
    """

    def __init__(self, input_shape, level, channels=3, num_classes=0,
                 prior_only=False, ema=0.05):
        super().__init__()
        C, H, W = input_shape
        self.C, self.H, self.W = C, H, W
        self.class_prior = num_classes > 0      # class-conditional fill enabled
        self.prior_only = bool(prior_only)      # label-only control (ignore activation)
        self.ema = ema
        self.lift = nn.Sequential(
            nn.Conv2d(2 * C + 1, C, 1), nn.BatchNorm2d(C), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(C, C, 3, 1, 1), nn.BatchNorm2d(C), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(C, C, 3, 1, 1),
        )
        # Zero-init the final conv so the residual == 0 at start: the attacker begins
        # identical to the plain sparse decoder (dense == x, or == the class prior when
        # enabled) and LEARNS the residual gradually, instead of flooding the 16k empty
        # coords with random noise (which made early reconstructions washed-out/bright).
        nn.init.zeros_(self.lift[-1].weight)
        nn.init.zeros_(self.lift[-1].bias)
        self.decoder = cifar_decoder(input_shape, level, channels)
        if self.class_prior:
            # per-class running mean activation ("what class c looks like in activation
            # space"); EMA-updated from observed activations + labels (training calls only).
            self.register_buffer("mu_c", torch.zeros(num_classes, C * H * W))

    @torch.no_grad()
    def _update_prior(self, x, labels):
        xf = x.flatten(1)                                        # (B, D), activations >= 0
        mf = (xf != 0).float()
        for c in labels.unique():
            sel = labels == c
            denom = mf[sel].sum(0).clamp(min=1.0)
            batch_mu = (xf[sel] * mf[sel]).sum(0) / denom        # per-coord class mean (where seen)
            seen = mf[sel].sum(0) > 0
            self.mu_c[c] = torch.where(seen, (1 - self.ema) * self.mu_c[c] + self.ema * batch_mu,
                                       self.mu_c[c])

    def forward(self, x, labels=None):
        use_prior = self.class_prior and labels is not None
        # Update mu_c only on genuine training calls (grad enabled); the attack-application
        # calls run under no_grad, so they only USE the prior, never pollute it.
        if use_prior and self.training and torch.is_grad_enabled():
            self._update_prior(x.detach(), labels)
        prior = (self.mu_c[labels].view(-1, self.C, self.H, self.W).clamp(min=0.0)
                 if use_prior else torch.zeros_like(x))
        if self.prior_only and use_prior:                        # label-only control
            return self.decoder(prior)
        mask = (x != 0).float()                                  # 1 at sent (exact) coords
        flat = x.abs().flatten(1)
        masked = flat.masked_fill(flat == 0, float("inf"))
        tau = masked.min(dim=1).values                           # per-sample top-k threshold
        tau = torch.where(torch.isinf(tau), torch.zeros_like(tau), tau).view(-1, 1, 1, 1)
        tau_map = tau.expand(-1, 1, x.size(2), x.size(3))
        feat = torch.cat([x, mask, tau_map], dim=1)              # values + support + bound
        # Cut is post-ReLU: unsent coords lie in [0, tau]. Fill = class prior + learned
        # residual, constrained to that known range, applied only at the unsent coords.
        infill = (self.lift(feat) + prior).clamp(min=0.0)
        infill = torch.minimum(infill, tau.expand_as(x))
        dense = x + (1.0 - mask) * infill                        # exact at support, bounded infill else
        return self.decoder(dense)


def cifar_discriminator_model(input_shape, level):

    net = []
    if level <=2:
        net += [nn.Conv2d(input_shape[0], 128, 3, 2, 1)]
        net += [nn.ReLU()]
        net += [nn.Conv2d(128, 256, 3, 2, 1)]
    elif level == 3:
        net += [nn.Conv2d(input_shape[0], 256, 3, 2, 1)]
    elif level == 4:
        net += [nn.Conv2d(input_shape[0], 256, 3, 1, 1)]


    bn = False
        
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]

    net += [nn.Conv2d(256, 256, 3, 2, 1)]
    net += [nn.Flatten()]
    net += [nn.Linear(1024,1)]
    return nn.Sequential(*net)

def cifar_pseudo(level):
    client = []
    if level == 1:
        client += conv_bn(  3,  32, 2)
        client += conv_dw(  32,  32, 1)
 
        return nn.Sequential(*client)
    if level == 2:
        client += conv_bn(  3,  32, 2)
        client += conv_dw( 32,  64, 1)
        client += conv_dw( 64,  64, 1)
        return nn.Sequential(*client)
    if level == 3:
        client += conv_bn(  3,  32, 2)
        client += conv_dw( 32,  64, 1)
        client += conv_dw( 64, 128, 2)
        client += conv_dw(128, 128, 1)
        client += conv_dw(128, 128, 1)
        return nn.Sequential(*client)
    if level == 4:
        client += conv_bn(  3,  32, 2)
        client += conv_dw( 32,  64, 1)
        client += conv_dw( 64, 128, 2)
        client += conv_dw(128, 128, 1)
        client += conv_dw(128, 256, 2)
        client += conv_dw(256, 256, 1)
        client += conv_dw(256, 256, 1)
        return nn.Sequential(*client)
    