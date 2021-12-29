#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import re
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import utils.block as B

STATE_T = OrderedDict[str, Tensor]


class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0], [0, 0, 0], [0, 1, 0]]
        kernel_h = [[0, 0, 0], [-1, 0, 1], [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)

        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim=1)

        return x


####################
# Generator
####################

# Borrowed from https://github.com/rlaphoenix/VSGAN/blob/master/vsgan/archs/ESRGAN.py
# Which enhanced stuff that was already here
class ESRGAN(nn.Module):
    def __init__(
        self,
        state_dict: str,
        norm=None,
        act: str = "leakyrelu",
        upsampler: str = "upconv",
        mode: str = "CNA",
    ) -> None:
        """
        ESRGAN - Enhanced Super-Resolution Generative Adversarial Networks.
        By Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Yu Qiao,
        and Chen Change Loy.
        This is old-arch Residual in Residual Dense Block Network and is not
        the newest revision that's available at github.com/xinntao/ESRGAN.
        This is on purpose, the newest Network has severely limited the
        potential use of the Network with no benefits.
        This network supports model files from both new and old-arch.
        Args:
            norm: Normalization layer
            act: Activation layer
            upsampler: Upsample layer. upconv, pixel_shuffle
            mode: Convolution mode
        """
        super(ESRGAN, self).__init__()

        self.state = state_dict
        self.norm = norm
        self.act = act
        self.upsampler = upsampler
        self.mode = mode

        self.state_map = {
            # currently supports old, new, and newer RRDBNet arch models
            # ESRGAN, BSRGAN/RealSR, Real-ESRGAN
            "model.0.weight": ("conv_first.weight",),
            "model.0.bias": ("conv_first.bias",),
            "model.1.sub./NB/.weight": ("trunk_conv.weight", "conv_body.weight"),
            "model.1.sub./NB/.bias": ("trunk_conv.bias", "conv_body.bias"),
            "model.3.weight": ("upconv1.weight", "conv_up1.weight"),
            "model.3.bias": ("upconv1.bias", "conv_up1.bias"),
            "model.6.weight": ("upconv2.weight", "conv_up2.weight"),
            "model.6.bias": ("upconv2.bias", "conv_up2.bias"),
            "model.8.weight": ("HRconv.weight", "conv_hr.weight"),
            "model.8.bias": ("HRconv.bias", "conv_hr.bias"),
            "model.10.weight": ("conv_last.weight",),
            "model.10.bias": ("conv_last.bias",),
            r"model.1.sub.\1.RDB\2.conv\3.0.\4": (
                r"RRDB_trunk\.(\d+)\.RDB(\d)\.conv(\d+)\.(weight|bias)",
                r"body\.(\d+)\.rdb(\d)\.conv(\d+)\.(weight|bias)",
            ),
        }
        if "params_ema" in self.state:
            self.state = self.state["params_ema"]
        self.num_blocks = self.get_num_blocks()
        self.plus = any("conv1x1" in k for k in self.state.keys())

        self.state: STATE_T = self.new_to_old_arch(self.state)
        self.in_nc = self.state["model.0.weight"].shape[1]

        self.out_nc = (
            self.get_out_nc() or self.in_nc
        )  # assume same as in nc if not found

        self.scale = self.get_scale()
        self.num_filters = self.state["model.0.weight"].shape[0]

        # Detect if pixelunshuffle was used (Real-ESRGAN)
        if self.in_nc in (self.out_nc * 4, self.out_nc * 16) and self.out_nc in (
            self.in_nc / 4,
            self.in_nc / 16,
        ):
            self.shuffle_factor = int(math.sqrt(self.in_nc / self.out_nc))
        else:
            self.shuffle_factor = None

        upsample_block = {
            "upconv": B.upconv_block,
            "pixel_shuffle": B.pixelshuffle_block,
        }.get(self.upsampler)
        if upsample_block is None:
            raise NotImplementedError(
                "Upsample mode [%s] is not found" % self.upsampler
            )

        if self.scale == 3:
            upsample_blocks = upsample_block(
                in_nc=self.num_filters,
                out_nc=self.num_filters,
                upscale_factor=3,
                act_type=self.act,
            )
        else:
            upsample_blocks = [
                upsample_block(
                    in_nc=self.num_filters, out_nc=self.num_filters, act_type=self.act
                )
                for _ in range(int(math.log(self.scale, 2)))
            ]

        self.model = B.sequential(
            # fea conv
            B.conv_block(
                in_nc=self.in_nc,
                out_nc=self.num_filters,
                kernel_size=3,
                norm_type=None,
                act_type=None,
            ),
            B.ShortcutBlock(
                B.sequential(
                    # rrdb blocks
                    *[
                        B.RRDB(
                            nf=self.num_filters,
                            kernel_size=3,
                            gc=32,
                            stride=1,
                            bias=True,
                            pad_type="zero",
                            norm_type=self.norm,
                            act_type=self.act,
                            mode="CNA",
                            plus=self.plus,
                        )
                        for _ in range(self.num_blocks)
                    ],
                    # lr conv
                    B.conv_block(
                        in_nc=self.num_filters,
                        out_nc=self.num_filters,
                        kernel_size=3,
                        norm_type=self.norm,
                        act_type=None,
                        mode=self.mode,
                    ),
                )
            ),
            *upsample_blocks,
            # hr_conv0
            B.conv_block(
                in_nc=self.num_filters,
                out_nc=self.num_filters,
                kernel_size=3,
                norm_type=None,
                act_type=self.act,
            ),
            # hr_conv1
            B.conv_block(
                in_nc=self.num_filters,
                out_nc=self.out_nc,
                kernel_size=3,
                norm_type=None,
                act_type=None,
            ),
        )

        self.load_state_dict(self.state, strict=False)

    def new_to_old_arch(self, state: STATE_T) -> STATE_T:
        """Convert a new-arch model state dictionary to an old-arch dictionary."""
        if "params_ema" in state:
            state = state["params_ema"]

        if "conv_first.weight" not in state:
            # model is already old arch, this is a loose check, but should be sufficient
            return state

        # add nb to state keys
        for kind in ("weight", "bias"):
            self.state_map[f"model.1.sub.{self.num_blocks}.{kind}"] = self.state_map[
                f"model.1.sub./NB/.{kind}"
            ]
            del self.state_map[f"model.1.sub./NB/.{kind}"]

        old_state = OrderedDict()
        for old_key, new_keys in self.state_map.items():
            for new_key in new_keys:
                if r"\1" in old_key:
                    for k, v in state.items():
                        sub = re.sub(new_key, old_key, k)
                        if sub != k:
                            old_state[sub] = v
                else:
                    if new_key in state:
                        old_state[old_key] = state[new_key]

        return old_state

    def get_out_nc(self) -> Optional[int]:
        max_part = 0
        out_nc = None
        for part in list(self.state):
            parts = part.split(".")[1:]
            if len(parts) == 2:
                part_num = int(parts[0])
                if part_num > max_part:
                    max_part = part_num
                    out_nc = self.state[part].shape[0]
        return out_nc

    def get_scale(self, min_part: int = 6) -> int:
        n = 0
        for part in list(self.state):
            parts = part.split(".")[1:]
            if len(parts) == 2:
                part_num = int(parts[0])
                if part_num > min_part and parts[1] == "weight":
                    n += 1
        return 2 ** n

    def get_num_blocks(self) -> int:
        nbs = []
        state_keys = self.state_map[r"model.1.sub.\1.RDB\2.conv\3.0.\4"] + (
            r"model\.\d+\.sub\.(\d+)\.RDB(\d+)\.conv(\d+)\.0\.(weight|bias)",
        )
        for state_key in state_keys:
            for k in self.state:
                m = re.search(state_key, k)
                if m:
                    nbs.append(int(m.group(1)))
            if nbs:
                break
        return max(*nbs) + 1

    def forward(self, x):
        if self.shuffle_factor:
            x = torch.pixel_unshuffle(x, downscale_factor=self.shuffle_factor)
        return self.model(x)


class SRVGGNetCompact(nn.Module):
    """A compact VGG-style network structure for super-resolution.
    It is a compact network structure, which performs upsampling in the last layer and no convolution is
    conducted on the HR feature space.
    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_conv (int): Number of convolution layers in the body network. Default: 16.
        upscale (int): Upsampling factor. Default: 4.
        act_type (str): Activation type, options: 'relu', 'prelu', 'leakyrelu'. Default: prelu.
    """

    def __init__(
        self,
        state_dict: str,
        act_type: str = "prelu",
    ):
        super(SRVGGNetCompact, self).__init__()
        self.act_type = act_type

        self.state = state_dict

        if "params" in self.state:
            self.state = self.state["params"]

        self.key_arr = list(self.state.keys())

        self.num_in_ch = self.get_in_nc()
        self.num_feat = self.get_num_feats()
        self.num_conv = self.get_num_conv()
        self.num_out_ch = self.num_in_ch  # :(
        self.scale = self.get_scale()

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(self.num_in_ch, self.num_feat, 3, 1, 1))
        # the first activation
        if act_type == "relu":
            activation = nn.ReLU(inplace=True)
        elif act_type == "prelu":
            activation = nn.PReLU(num_parameters=self.num_feat)
        elif act_type == "leakyrelu":
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # the body structure
        for _ in range(self.num_conv):
            self.body.append(nn.Conv2d(self.num_feat, self.num_feat, 3, 1, 1))
            # activation
            if act_type == "relu":
                activation = nn.ReLU(inplace=True)
            elif act_type == "prelu":
                activation = nn.PReLU(num_parameters=self.num_feat)
            elif act_type == "leakyrelu":
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        # the last conv
        self.body.append(nn.Conv2d(self.num_feat, self.pixelshuffle_shape, 3, 1, 1))
        # upsample
        self.upsampler = nn.PixelShuffle(self.scale)

        self.load_state_dict(self.state, strict=False)

    def get_num_conv(self) -> int:
        return (int(self.key_arr[-1].split(".")[1]) - 2) // 2

    def get_num_feats(self) -> int:
        return self.state[self.key_arr[0]].shape[0]

    def get_in_nc(self) -> int:
        return self.state[self.key_arr[0]].shape[1]

    def get_scale(self) -> int:
        self.pixelshuffle_shape = self.state[self.key_arr[-1]].shape[0]
        # Assume out_nc is the same as in_nc
        # I cant think of a better way to do that
        self.num_out_ch = self.num_in_ch
        scale = math.sqrt(self.pixelshuffle_shape / self.num_out_ch)
        if scale - int(scale) > 0:
            print(
                "out_nc is probably different than in_nc, scale calculation might be wrong"
            )
        scale = int(scale)
        return scale

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        out = self.upsampler(out)
        # add the nearest upsampled image, so that the network learns the residual
        base = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        out += base
        return out


# older esrgan net
class RRDBNet(nn.Module):
    def __init__(
        self,
        in_nc,
        out_nc,
        nf,
        nb,
        gc=32,
        upscale=4,
        norm_type=None,
        act_type="leakyrelu",
        mode="CNA",
        upsample_mode="upconv",
        convtype="Conv2D",
        finalact=None,
        plus=False,
    ):
        super(RRDBNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        rb_blocks = [
            B.RRDB(
                nf,
                kernel_size=3,
                gc=32,
                stride=1,
                bias=1,
                pad_type="zero",
                norm_type=norm_type,
                act_type=act_type,
                mode="CNA",
                convtype=convtype,
                plus=plus,
            )
            for _ in range(nb)
        ]
        LR_conv = B.conv_block(
            nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode
        )

        if upsample_mode == "upconv":
            upsample_block = B.upconv_block
        elif upsample_mode == "pixelshuffle":
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError(
                "upsample mode [{:s}] is not found".format(upsample_mode)
            )
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [
                upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)
            ]
        HR_conv0 = B.conv_block(
            nf, nf, kernel_size=3, norm_type=None, act_type=act_type
        )
        HR_conv1 = B.conv_block(
            nf, out_nc, kernel_size=3, norm_type=None, act_type=None
        )

        # Note: this option adds new parameters to the architecture, another option is to use "outm" in the forward
        outact = B.act(finalact) if finalact else None

        self.model = B.sequential(
            fea_conv,
            B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),
            *upsampler,
            HR_conv0,
            HR_conv1,
            outact,
        )

    def forward(self, x, outm=None):
        x = self.model(x)

        if (
            outm == "scaltanh"
        ):  # limit output range to [-1,1] range with tanh and rescale to [0,1] Idea from: https://github.com/goldhuang/SRGAN-PyTorch/blob/master/model.py
            return (torch.tanh(x) + 1.0) / 2.0
        elif outm == "tanh":  # limit output to [-1,1] range
            return torch.tanh(x)
        elif outm == "sigmoid":  # limit output to [0,1] range
            return torch.sigmoid(x)
        elif outm == "clamp":
            return torch.clamp(x, min=0.0, max=1.0)
        else:  # Default, no cap for the output
            return x


class SPSRNet(nn.Module):
    def __init__(
        self,
        in_nc,
        out_nc,
        nf,
        nb,
        gc=32,
        upscale=4,
        norm_type=None,
        act_type="leakyrelu",
        mode="CNA",
        upsample_mode="upconv",
    ):
        super(SPSRNet, self).__init__()

        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        rb_blocks = [
            B.RRDB(
                nf,
                kernel_size=3,
                gc=32,
                stride=1,
                bias=True,
                pad_type="zero",
                norm_type=norm_type,
                act_type=act_type,
                mode="CNA",
            )
            for _ in range(nb)
        ]
        LR_conv = B.conv_block(
            nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode
        )

        if upsample_mode == "upconv":
            upsample_block = B.upconv_blcok
        elif upsample_mode == "pixelshuffle":
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError(
                "upsample mode [{:s}] is not found".format(upsample_mode)
            )
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [
                upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)
            ]
        self.HR_conv0_new = B.conv_block(
            nf, nf, kernel_size=3, norm_type=None, act_type=act_type
        )
        self.HR_conv1_new = B.conv_block(
            nf, nf, kernel_size=3, norm_type=None, act_type=None
        )

        self.model = B.sequential(
            fea_conv,
            B.ShortcutBlockSPSR(B.sequential(*rb_blocks, LR_conv)),
            *upsampler,
            self.HR_conv0_new,
        )

        self.get_g_nopadding = Get_gradient_nopadding()

        self.b_fea_conv = B.conv_block(
            in_nc, nf, kernel_size=3, norm_type=None, act_type=None
        )

        self.b_concat_1 = B.conv_block(
            2 * nf, nf, kernel_size=3, norm_type=None, act_type=None
        )
        self.b_block_1 = B.RRDB(
            nf * 2,
            kernel_size=3,
            gc=32,
            stride=1,
            bias=True,
            pad_type="zero",
            norm_type=norm_type,
            act_type=act_type,
            mode="CNA",
        )

        self.b_concat_2 = B.conv_block(
            2 * nf, nf, kernel_size=3, norm_type=None, act_type=None
        )
        self.b_block_2 = B.RRDB(
            nf * 2,
            kernel_size=3,
            gc=32,
            stride=1,
            bias=True,
            pad_type="zero",
            norm_type=norm_type,
            act_type=act_type,
            mode="CNA",
        )

        self.b_concat_3 = B.conv_block(
            2 * nf, nf, kernel_size=3, norm_type=None, act_type=None
        )
        self.b_block_3 = B.RRDB(
            nf * 2,
            kernel_size=3,
            gc=32,
            stride=1,
            bias=True,
            pad_type="zero",
            norm_type=norm_type,
            act_type=act_type,
            mode="CNA",
        )

        self.b_concat_4 = B.conv_block(
            2 * nf, nf, kernel_size=3, norm_type=None, act_type=None
        )
        self.b_block_4 = B.RRDB(
            nf * 2,
            kernel_size=3,
            gc=32,
            stride=1,
            bias=True,
            pad_type="zero",
            norm_type=norm_type,
            act_type=act_type,
            mode="CNA",
        )

        self.b_LR_conv = B.conv_block(
            nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode
        )

        if upsample_mode == "upconv":
            upsample_block = B.upconv_blcok
        elif upsample_mode == "pixelshuffle":
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError(
                "upsample mode [{:s}] is not found".format(upsample_mode)
            )
        if upscale == 3:
            b_upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            b_upsampler = [
                upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)
            ]

        b_HR_conv0 = B.conv_block(
            nf, nf, kernel_size=3, norm_type=None, act_type=act_type
        )
        b_HR_conv1 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None)

        self.b_module = B.sequential(*b_upsampler, b_HR_conv0, b_HR_conv1)

        self.conv_w = B.conv_block(
            nf, out_nc, kernel_size=1, norm_type=None, act_type=None
        )

        self.f_concat = B.conv_block(
            nf * 2, nf, kernel_size=3, norm_type=None, act_type=None
        )

        self.f_block = B.RRDB(
            nf * 2,
            kernel_size=3,
            gc=32,
            stride=1,
            bias=True,
            pad_type="zero",
            norm_type=norm_type,
            act_type=act_type,
            mode="CNA",
        )

        self.f_HR_conv0 = B.conv_block(
            nf, nf, kernel_size=3, norm_type=None, act_type=act_type
        )
        self.f_HR_conv1 = B.conv_block(
            nf, out_nc, kernel_size=3, norm_type=None, act_type=None
        )

    def forward(self, x):
        x_grad = self.get_g_nopadding(x)
        x = self.model[0](x)

        x, block_list = self.model[1](x)

        x_ori = x
        for i in range(5):
            x = block_list[i](x)
        x_fea1 = x

        for i in range(5):
            x = block_list[i + 5](x)
        x_fea2 = x

        for i in range(5):
            x = block_list[i + 10](x)
        x_fea3 = x

        for i in range(5):
            x = block_list[i + 15](x)
        x_fea4 = x

        x = block_list[20:](x)
        # short cut
        x = x_ori + x
        x = self.model[2:](x)
        x = self.HR_conv1_new(x)

        x_b_fea = self.b_fea_conv(x_grad)
        x_cat_1 = torch.cat([x_b_fea, x_fea1], dim=1)

        x_cat_1 = self.b_block_1(x_cat_1)
        x_cat_1 = self.b_concat_1(x_cat_1)

        x_cat_2 = torch.cat([x_cat_1, x_fea2], dim=1)

        x_cat_2 = self.b_block_2(x_cat_2)
        x_cat_2 = self.b_concat_2(x_cat_2)

        x_cat_3 = torch.cat([x_cat_2, x_fea3], dim=1)

        x_cat_3 = self.b_block_3(x_cat_3)
        x_cat_3 = self.b_concat_3(x_cat_3)

        x_cat_4 = torch.cat([x_cat_3, x_fea4], dim=1)

        x_cat_4 = self.b_block_4(x_cat_4)
        x_cat_4 = self.b_concat_4(x_cat_4)

        x_cat_4 = self.b_LR_conv(x_cat_4)

        # short cut
        x_cat_4 = x_cat_4 + x_b_fea
        x_branch = self.b_module(x_cat_4)

        # x_out_branch = self.conv_w(x_branch)
        ########
        x_branch_d = x_branch
        x_f_cat = torch.cat([x_branch_d, x], dim=1)
        x_f_cat = self.f_block(x_f_cat)
        x_out = self.f_concat(x_f_cat)
        x_out = self.f_HR_conv0(x_out)
        x_out = self.f_HR_conv1(x_out)

        #########
        # return x_out_branch, x_out, x_grad
        return x_out
