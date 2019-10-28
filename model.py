#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:31:54 2019

@author: Pablo Navarrete Michelini, Wenbin Chen
"""
import torch
import numpy as np
import torch.utils.data as data
from torch import nn
from tqdm import tqdm


def Cutborder(input, border, feature=None):
    """Cut borders considering that some borders can be zero (no border)
    """
    assert isinstance(border, tuple)
    if border == (0, 0, 0, 0):
        return input
    left, right, top, bottom = border
    assert top >= 0 and bottom >= 0 and \
        left >= 0 and right >= 0
    if np.all(np.asarray(border) == 0) and feature is None:
        return input
    if bottom == 0:
        bottom = None
    else:
        bottom = -bottom
    if right == 0:
        right = None
    else:
        right = -right
    if feature is None:
        return input[:, :, top:bottom, left:right]
    else:
        return input[:, feature:feature+1, top:bottom, left:right]


def weight_bias_init(model):
    if isinstance(model, nn.Module):
        for k, m in model._modules.items():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.fill_(0)
            else:
                weight_bias_init(m)


class MergingBlocks(data.Dataset):
    def __init__(self, x, noise, block_size, skip, output_channels):
        super().__init__()
        self.input = x
        self.noise = noise
        self.block_size = block_size
        block_h, block_v = self.block_size

        nstep_v = (x.shape[2] - block_v)//skip + 1
        nstep_v = max(nstep_v, 2)
        self.corner_v = np.asarray(np.linspace(
            0, x.shape[2] - block_v, nstep_v
        ).round(), dtype=np.int)
        assert len(self.corner_v) > 0
        nstep_h = (x.shape[3] - block_h)//skip
        nstep_h = max(nstep_h, 2)
        self.corner_h = np.asarray(np.linspace(
            0, x.shape[3] - block_h, nstep_h
        ).round(), dtype=np.int)
        assert len(self.corner_h) > 0

        self.out = torch.zeros(
            [x.shape[0], output_channels, x.shape[2], x.shape[3]]
        ).double()
        self.count = torch.zeros_like(self.out)
        self.weight = (torch.cat([
            torch.arange(1, block_v//2+1+block_v % 2),
            torch.arange(block_v//2, 0, -1)
        ]).unsqueeze(0).transpose(0, 1) @ torch.cat([
            torch.arange(1, block_h//2+1+block_h % 2),
            torch.arange(block_h//2, 0, -1)
        ]).unsqueeze(0)).to(self.out)
        self.weight /= self.weight.mean()

    def __len__(self):
        return len(self.corner_v) * len(self.corner_h)

    def __getitem__(self, index):
        i = index % len(self.corner_h)
        j = index // len(self.corner_h)
        cx, cy = self.corner_h[i], self.corner_v[j]
        block_h, block_v = self.block_size
        return \
            self.input[0, :, cy: cy+block_v, cx: cx+block_h], \
            self.noise[0, :, cy: cy+block_v, cx: cx+block_h]

    def __setitem__(self, index, item):
        i = index % len(self.corner_h)
        j = index // len(self.corner_h)
        cx, cy = self.corner_h[i], self.corner_v[j]
        block_h, block_v = self.block_size
        self.count[0, :, cy: cy+block_v, cx: cx+block_h] += self.weight
        self.out[0, :, cy: cy+block_v, cx: cx+block_h] += item.to(self.out) * self.weight

    def result(self):
        return self.out / self.count


class MGBPv2(nn.Module):
    def __init__(self,
                 model_id,
                 noise_amp=0.,
                 name='bpp',
                 str_tab=''):
        self.model_id = model_id
        self.name = name
        self.str_tab = str_tab
        super().__init__()

        assert self.model_id.endswith('_mgbp')

        parse = self.model_id.split('_')

        self.input_channels = int([s for s in parse if s.startswith('CH')][0][2:])
        self.output_channels = 3
        self.channels = [int(n) for n in [s for s in parse if s.startswith('FE')][0][2:].split('-')]
        self.mu = int([s for s in parse if s.startswith('MU')][0][2:])
        self.nu = 1
        if len([s for s in parse if s.startswith('NU')]) > 0:
            self.nu = int([s for s in parse if s.startswith('NU')][0][2:])
        self._levels = int([s for s in parse if s.startswith('LE')][0][2:])
        self.use_bias = (len([s for s in parse if s.startswith('BIAS')]) > 0)
        self.noise_channels = int([s for s in parse if s.startswith('NOISE')][0][5:])
        self._noise_amp = noise_amp
        assert self._levels == len(self.channels)
        self.factor = 2 * [2**(self._levels-1)]
        self.border = None

        self.net = {}

        parse = [s for s in parse if s.startswith('Filter')][0].split('#')[1:]
        self.kernel_size = int([s for s in parse if s.startswith('K')][0][1:])
        assert (self.kernel_size-1) % 2 == 0
        self._step = np.zeros(self._levels-1, dtype=np.int)
        self.mgbp_dry_run(self._levels-1)

        self.analysis = []
        for k in range(self._levels-1):
            s = 2**(self._levels-1-k)
            self.net['Analysis_%d' % k] = nn.Conv2d(
                self.input_channels, self.channels[k],
                (self.kernel_size-1)*(s-1)+1,
                stride=s, padding=0, dilation=1, bias=self.use_bias
            )
            self.analysis.append(self.net['Analysis_%d' % k])
        k = self._levels-1
        self.net['Analysis_%d' % k] = nn.Conv2d(
            self.input_channels, self.channels[self._levels-1], 1,
            stride=1, padding=0, dilation=1, bias=self.use_bias
        )
        self.analysis.append(self.net['Analysis_%d' % k])

        self.analysisN = []
        for k in range(self._levels-1):
            s = 2**(self._levels-1-k)
            self.net['AnalysisN_%d' % k] = nn.Conv2d(
                self.input_channels, self.noise_channels,
                (self.kernel_size-1)*(s-1)+1,
                stride=s, padding=0, dilation=1, bias=True
            )
            self.analysisN.append(self.net['AnalysisN_%d' % k])
        k = self._levels-1
        self.net['AnalysisN_%d' % k] = nn.Conv2d(
            self.input_channels, self.noise_channels, 1,
            stride=1, padding=0, dilation=1, bias=True
        )
        self.analysisN.append(self.net['AnalysisN_%d' % k])

        self.net['Synthesis'] = nn.Conv2d(
            self.channels[-1], self.output_channels, 1,
            padding=0, dilation=1, bias=self.use_bias
        )
        self.synthesis = self.net['Synthesis']

        for name, layer in self.net.items():
            self.add_module(name, layer)

        self.stat_nparam = 0
        for name, par in self.named_parameters():
            if name.endswith('weight'):
                self.stat_nparam += np.prod(par.shape)
            if name.endswith('bias'):
                self.stat_nparam += np.prod(par.shape)

    def mgbp_dry_run(self, level, lowest=0):
        if level > lowest:
            for k in range(self.mu):
                self._step[level-1] = k
                label = str(level) + '-' + '-'.join([str(s) for s in self._step])
                self.net['Downscale_%s' % label] = nn.Conv2d(
                    self.channels[level], self.channels[level-1],
                    self.kernel_size, stride=2,
                    padding=0, groups=1, dilation=1, bias=self.use_bias
                )
                self.mgbp_dry_run(level-1, lowest)
                self.net['Upscale_%s' % label] = nn.Sequential(
                    nn.ReLU(),
                    nn.ConvTranspose2d(
                        2*self.channels[level-1]+self.noise_channels,
                        self.channels[level],
                        self.kernel_size, stride=2,
                        padding=0, output_padding=0,
                        groups=1, dilation=1, bias=self.use_bias
                    ),
                )

    def mgbp_run(self, HR_in, level, lowest=0):
        HR_out = HR_in
        if level > lowest:
            for k in range(self.mu):
                self._step[level-1] = k
                label = str(level) + '-' + '-'.join([str(s) for s in self._step])
                LR = self.net['Downscale_%s' % label](HR_out)
                correction = self.mgbp_run(LR, level-1, lowest)
                HR_out = HR_out + self.net['Upscale_%s' % label](
                    torch.cat([
                        self._x[level-1],
                        correction,
                        self._noise_amp * self._noise[level-1]
                    ], 1)
                )
            return HR_out
        else:
            return HR_in

    def init_border(self, size):
        p = size[0]
        tmp = s0 = 1
        while s0 < size[0]:
            tmp += 1
            p = tmp
            for k in range(self._levels-1):
                p = (p-1)*2 + self.kernel_size
            s0 = p
        tmp = s1 = 1
        while s1 < size[1]:
            tmp += 1
            p = tmp
            for k in range(self._levels-1):
                p = (p-1)*2 + self.kernel_size
            s1 = p
        self.border = (
            (s1-size[1]) - (s1-size[1])//2, (s1-size[1])//2,
            (s0-size[0]) - (s0-size[0])//2, (s0-size[0])//2
        )
        if np.all(np.asarray(self.border) == 0):
            self.border = None

    def forward(self, input, noise=None):
        if noise is None:
            noise = torch.randn(input.shape).to(input)
        if self.border is not None:
            x_pad = torch.nn.functional.pad(input, pad=self.border, mode='reflect')
            n_pad = torch.nn.functional.pad(noise, pad=self.border, mode='reflect')
        else:
            x_pad = input
            n_pad = noise

        out = {}
        self._x = [None] * (self._levels-1) + [x_pad]
        for k in range(self._levels):
            self._x[k] = self.analysis[k](x_pad)

        self._noise = [None] * (self._levels-1) + [n_pad]
        for k in range(self._levels):
            self._noise[k] = self.analysisN[k](n_pad)

        out = self._x[self._levels-1]
        for _ in range(self.nu):
            self._step = np.zeros(self._levels-1, dtype=np.int)
            out = self.mgbp_run(out, self._levels-1, lowest=0)

        out = self.synthesis(out)

        return out if self.border is None else Cutborder(out, self.border)

    def noise_amp(self, val):
        self._noise_amp = val

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def block_average(self, x, block_size, skip, extended_px, output_channels, device, target_gb, num_workers, verbose):
        if extended_px is None:
            x_pad = x
        else:
            x_pad = torch.nn.functional.pad(
                x, pad=[extended_px, ] * 4, mode='reflect'
            )
        noise = self._noise_amp * torch.randn(x_pad.shape)

        batch_size = int(
            round(0.5*2108*(target_gb/11.)*(100./block_size[0])*(100./block_size[1]))
        )
        dset = MergingBlocks(x_pad, noise, block_size, skip, output_channels)
        block_sampler = torch.utils.data.DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=None,
            sampler=None,
            batch_sampler=None,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            timeout=0,
            worker_init_fn=None
        )
        idx_start = 0
        sample = tqdm(
            block_sampler,
            desc='batches=%dx%d' % (batch_size, len(block_sampler)),
            leave=False
        ) if verbose else block_sampler
        for block_x, block_n in sample:
            block_output = self.forward(
                block_x.to(device),
                noise=block_n.to(device),
            )
            idx_stop = idx_start + block_output.shape[0]
            for k in range(idx_start, idx_stop):
                dset[k] = block_output[k-idx_start]
            idx_start = idx_stop
        if extended_px is None:
            return dset.result()
        return torch.nn.functional.pad(
            dset.result(), pad=[-extended_px, ] * 4, mode='reflect'
        )


class mergeModel(nn.Module):
    def __init__(self, device, name='merge'):
        self.name = name
        self._noise_amp = False
        super().__init__()
        self.model0 = MGBPv2(
            "CH3_LE5_MU2_BIAS_NOISE1_FE256-192-128-48-9_Filter#K3_mgbp",
            name='[%s] mgbp' % device,
            str_tab='  >'
        )
        self.model1 = MGBPv2(
            "CH3_LE5_MU2_BIAS_NOISE1_FE256-192-128-48-9_Filter#K5_mgbp",
            name='[%s] mgbp' % device,
            str_tab='  >'
        )
        self.model2 = MGBPv2(
            "CH3_LE5_MU2_BIAS_NOISE1_FE256-192-128-48-9_Filter#K7_mgbp",
            name='[%s] mgbp' % device,
            str_tab='  >'
        )

    def init_border(self, HR_shape):
        self.model0.init_border(HR_shape)
        self.model1.init_border(HR_shape)
        self.model2.init_border(HR_shape)

    def forward(self, x, noise=None):
        x0 = self.model0(x)
        x1 = self.model1(x)
        x2 = self.model2(x)
        out = torch.add(torch.add(x0, x1), x2)
        return out

    def block_average(self, x, block_size, skip, extended_px, output_channels, device, target_gb, num_workers, verbose):
        if extended_px is None:
            x_pad = x
        else:
            x_pad = torch.nn.functional.pad(
                x, pad=[extended_px, ] * 4, mode='reflect'
            )
        noise = self._noise_amp * torch.randn(x_pad.shape)

        batch_size = int(
            round(0.5*2108*(target_gb/11.)*(100./block_size[0])*(100./block_size[1]))
        )
        dset = MergingBlocks(x_pad, noise, block_size, skip, output_channels)

        block_sampler = torch.utils.data.DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=None,
            sampler=None,
            batch_sampler=None,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            timeout=0,
            worker_init_fn=None
        )
        idx_start = 0
        sample = tqdm(
            block_sampler,
            desc='batches=%dx%d' % (batch_size, len(block_sampler)),
            leave=False
        ) if verbose else block_sampler
        for block_x, block_n in sample:
            block_output = self.forward(
                block_x.to(device),
                noise=block_n.to(device),
            )
            idx_stop = idx_start + block_output.shape[0]
            for k in range(idx_start, idx_stop):
                dset[k] = block_output[k-idx_start]
            idx_start = idx_stop

        if verbose:
            sample.close()

        if extended_px is None:
            return dset.result()
        return torch.nn.functional.pad(
            dset.result(), pad=[-extended_px, ] * 4, mode='reflect'
        )
