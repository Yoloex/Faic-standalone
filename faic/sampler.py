#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-07-13 16:59:27

import os, math, random, time

import numpy as np
from contextlib import nullcontext

from utils import util_net
from utils import util_common

import torch
import torch.nn.functional as F


class BaseSampler:
    def __init__(
        self,
        configs,
        sf=4,
        use_amp=True,
        chop_size=128,
        chop_stride=128,
        chop_bs=1,
        padding_offset=16,
        seed=10000,
    ):
        """
        Input:
            configs: config, see the yaml file in folder ./configs/
            sf: int, super-resolution scale
            seed: int, random seed
        """
        self.configs = configs
        self.sf = sf
        self.chop_size = chop_size
        self.chop_stride = chop_stride
        self.chop_bs = chop_bs
        self.seed = seed
        self.use_amp = use_amp
        self.padding_offset = padding_offset

        self.setup_dist()  # setup distributed training: self.num_gpus, self.rank

        self.setup_seed()

        self.build_model()

    def setup_seed(self, seed=None):
        seed = self.seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_dist(self, gpu_id=None):
        num_gpus = torch.cuda.device_count()

        if num_gpus > 1:
            # if mp.get_start_method(allow_none=True) is None:
            # mp.set_start_method('spawn')
            # rank = int(os.environ['LOCAL_RANK'])
            # torch.cuda.set_device(rank % num_gpus)
            # dist.init_process_group(backend='nccl', init_method='env://')
            rank = 0
            torch.cuda.set_device(rank)

        self.num_gpus = num_gpus
        self.rank = int(os.environ["LOCAL_RANK"]) if num_gpus > 1 else 0

    def write_log(self, log_str):
        if self.rank == 0:
            print(log_str, flush=True)

    def build_model(self):
        # diffusion model
        log_str = f"Building the diffusion model with length: {self.configs.diffusion.params.steps}..."
        self.write_log(log_str)
        self.base_diffusion = util_common.instantiate_from_config(
            self.configs.diffusion
        )
        model = util_common.instantiate_from_config(self.configs.model).cuda()
        ckpt_path = self.configs.model.ckpt_path
        assert ckpt_path is not None
        self.write_log(f"Loading Diffusion model from {ckpt_path}...")
        self.load_model(model, ckpt_path)
        self.freeze_model(model)
        self.model = model.eval()

        # autoencoder model
        if self.configs.autoencoder is not None:
            ckpt_path = self.configs.autoencoder.ckpt_path
            assert ckpt_path is not None
            self.write_log(f"Loading AutoEncoder model from {ckpt_path}...")
            autoencoder = util_common.instantiate_from_config(
                self.configs.autoencoder
            ).cuda()
            self.load_model(autoencoder, ckpt_path)
            autoencoder.eval()
            self.autoencoder = autoencoder
        else:
            self.autoencoder = None

    def load_model(self, model, ckpt_path=None):
        state = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if "state_dict" in state:
            state = state["state_dict"]
        util_net.reload_model(model, state)

    def freeze_model(self, net):
        for params in net.parameters():
            params.requires_grad = False


class ResShiftSampler(BaseSampler):
    def sample_func(self, y0, noise_repeat=False, mask=None):
        """
        Input:
            y0: n x c x h x w torch tensor, low-quality image, [-1, 1], RGB
            mask: image mask for inpainting
        Output:
            sample: n x c x h x w, torch tensor, [-1, 1], RGB
        """
        if noise_repeat:
            self.setup_seed()

        offset = self.padding_offset
        ori_h, ori_w = y0.shape[2:]
        if not (ori_h % offset == 0 and ori_w % offset == 0):
            flag_pad = True
            pad_h = (math.ceil(ori_h / offset)) * offset - ori_h
            pad_w = (math.ceil(ori_w / offset)) * offset - ori_w
            y0 = F.pad(y0, pad=(0, pad_w, 0, pad_h), mode="reflect")
        else:
            flag_pad = False

        if self.configs.model.params.cond_lq and mask is not None:
            model_kwargs = {
                "lq": y0,
                "mask": mask,
            }
        elif self.configs.model.params.cond_lq:
            model_kwargs = {
                "lq": y0,
            }
        else:
            model_kwargs = None

        results = self.base_diffusion.p_sample_loop(
            y=y0,
            model=self.model,
            first_stage_model=self.autoencoder,
            noise=None,
            noise_repeat=noise_repeat,
            clip_denoised=(self.autoencoder is None),
            denoised_fn=None,
            model_kwargs=model_kwargs,
            progress=False,
        )  # This has included the decoding for latent space

        if flag_pad:
            results = results[:, :, : ori_h * self.sf, : ori_w * self.sf]

        return results.clamp_(-1.0, 1.0)

    def inference(
        self,
        im_lq_tensor,
        noise_repeat=False,
    ):

        with torch.no_grad():
            im_sr_tensor = self.sample_func(
                ((im_lq_tensor + 1) / 2 - 0.5) / 0.5,
                noise_repeat=noise_repeat,
            )  # 1 x c x h x w, [-1, 1]

        im_sr_tensor = im_sr_tensor * 0.5 + 0.5

        return im_sr_tensor


if __name__ == "__main__":
    pass
