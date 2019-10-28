#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:31:54 2019

@author: Pablo Navarrete Michelini, Wenbin Chen
"""
import cv2
import os
from os.path import basename
import shutil
import time
import numpy as np
import torch
import PIL
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import torchvision.transforms.functional as TF

from model import MGBPv2, mergeModel, weight_bias_init


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [
        '.png', '.tif', '.jpg', '.jpeg', '.bmp', '.pgm', '.PNG'
    ])


if __name__ == '__main__':
    # ############################# CHANGE HERE ###############################
    target_gb = 5.  # Approximate device memory usage in GB
    device = torch.device('cuda:0')  # Device used to run the model
    # Model file used for Fidelity track (AIM 2019 Winner):
    # model_file = "BOE_Fidelity_CH3_LE5_MU2_BIAS_NOISE1_FE256-192-128-48-9_Filter#K357_merge.model"
    # Model file used for Perceptual track (AIM 2019 5th place):
    model_file = "BOE_Perceptual_CH3_LE5_MU2_BIAS_NOISE1_FE256-192-128-48-9_Filter#K3_mgbp.model"
    # #########################################################################

    parse = model_file.split('_')
    if 'Fidelity' in parse:
        target_gb *= 0.727272727
        set_block_average = [667, 667]
        extended_px = 0
        skip = 128
        noise = False
    elif 'Perceptual' in parse:
        set_block_average = [767, 767]
        extended_px = 0
        skip = 64
        noise = True
    input_dir = 'input'
    out_dir = 'output'
    model_id = 'CH' + '.'.join(model_file.split('_CH')[1].split('.')[:-1])
    torch.backends.cudnn.benchmark = True
    with torch.cuda.device(device):
        with torch.no_grad():
            print('\n- Load model')
            if model_id.endswith('mgbp'):
                model = MGBPv2(model_id)
                model.noise_amp(noise * 1.)
            elif model_id.endswith('merge'):
                model = mergeModel(
                    name='[%s] merge' % device, device=device
                )
                model.init_border(set_block_average)
            else:
                assert False
            weight_bias_init(model)
            model.load_state_dict(torch.load(
                model_file, map_location=lambda storage, loc: storage
            ))
            model.train(False)
            model.to(device)

            print('\n- Running', flush=True)
            out_ext = '.png'
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            Path(out_dir).mkdir(parents=True)

            time_exec = []
            input_list = [
                str(f) for f in Path(input_dir).iterdir()
                if is_image_file(f.name)
            ]
            tqdm_input_list = tqdm(input_list)
            for input_file in tqdm_input_list:
                input_pil = Image.open(input_file).convert('RGB')
                input_pil = input_pil.resize(
                    (input_pil.size[0] * 16,
                     input_pil.size[1] * 16),
                    resample=PIL.Image.BICUBIC
                )

                h, w = input_pil.size
                padding = [0, 0, 0, 0]
                if set_block_average is not None:
                    if input_pil.size[1] < set_block_average[0]:
                        padding[1] = (set_block_average[0] - input_pil.size[1]) // 2
                        padding[3] = set_block_average[0] - input_pil.size[1] - padding[1]
                    if input_pil.size[0] < set_block_average[1]:
                        padding[0] = (set_block_average[1] - input_pil.size[0]) // 2
                        padding[2] = set_block_average[1] - input_pil.size[0] - padding[0]
                    input_pil = TF.pad(
                        input_pil,
                        padding=tuple(padding),
                        padding_mode='reflect'
                    )
                input_tensor = TF.to_tensor(input_pil).unsqueeze(0)

                tqdm_input_list.set_postfix(
                    H=input_tensor.shape[2], W=input_tensor.shape[3],
                )
                block_average = None
                if set_block_average is not None:
                    if input_tensor.shape[2] > set_block_average[0] or \
                       input_tensor.shape[3] > set_block_average[1]:
                        block_average = set_block_average

                t0 = time.time()
                if block_average is not None:
                    output_rgb = model.block_average(
                        input_tensor, block_size=block_average, skip=skip,
                        extended_px=extended_px,
                        output_channels=3,
                        device=device, target_gb=target_gb,
                        num_workers=2,
                        verbose=False,
                    ).data[0].permute(1, 2, 0).cpu().clamp(0, 1).numpy()
                else:
                    input_tensor = input_tensor.to(device)
                    model.init_border((input_tensor.shape[2], input_tensor.shape[3]))
                    output_rgb = model(
                        input_tensor
                    ).data[0].permute(1, 2, 0).cpu().clamp(0, 1).numpy()
                del input_tensor
                t1 = time.time()
                time_exec.append(t1-t0)
                out_filename8 = out_dir + '/' + basename(input_file)[:-4] + out_ext
                im_cv = cv2.cvtColor(np.uint8(np.round(output_rgb*(2.**8-1.))), cv2.COLOR_RGB2BGR)
                im_cv = im_cv[
                    padding[1]:padding[1]+w,
                    padding[0]:padding[0]+h, :
                ]
                cv2.imwrite(out_filename8, im_cv,  [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print('\n  > Average execution time: %.2f [s]' % np.asarray(time_exec).mean())
