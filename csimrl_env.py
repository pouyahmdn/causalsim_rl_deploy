# This RL-env simulator is adapted from:
# https://github.com/sagar-pa/abr_rl_test/blob/e03d209603cc241910e607015cac9e22684ffab5/tara_env.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

from base_env import BaseEnv
from typing import Callable
import numpy as np
from collections import deque
from csim_model import PermInvPolicyNet

OBS_HISTORY = 8
NUM_CHOICES_SCALE = 12
MAX_BUFFER_S = 15                   # Seconds
# REBUF_BASE_PEN = 60
# Q_BAR_BAR = 1 - np.power(10, -15.07/10)
# REBUF_DYN_PEN = REBUF_BASE_PEN / 10 * np.log(10) / 2.002 * (1 - Q_BAR_BAR)
REBUF_DYN_PEN = 0.215
THR_SCALE = 1e6 / 8                 # Mbps
SSIM_DB_SCALE = 60
MAX_THR = 40                        # Mbps
MAX_DTIME = 20                      # Seconds
MAX_CSIZE = 35                      # Mb
CXT_WIDTH = 3 * OBS_HISTORY + 3
OBS_WIDTH = CXT_WIDTH + 2


class CsimRLEnv(BaseEnv):
    def setup_env(self, model_path: str) -> Callable:
        self.device = torch.device('cpu')
        self.norm_past_throughputs = deque(maxlen=OBS_HISTORY)
        self.norm_past_download_times = deque(maxlen=OBS_HISTORY)
        self.norm_past_ssims = deque(maxlen=OBS_HISTORY)

        for _ in range(OBS_HISTORY):
            self.norm_past_throughputs.append(0)
            self.norm_past_download_times.append(0)
            self.norm_past_ssims.append(0)

        model = torch.jit.script(PermInvPolicyNet(obs_width=OBS_WIDTH, cxt_width=CXT_WIDTH,
                                                  device=self.device).to(self.device))
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        # We don't have norm layers or dropout, so the line below shouldn't change anything
        model.eval()
        return model

    def process_env_info(self, env_info: dict) -> dict:
        valid_past_action_norm = 0
        if self.past_action is not None:  # avoid doing extra work on zeros for very first action
            self.norm_past_download_times.append(min(env_info["past_chunk"]["delay"], MAX_DTIME))
            c_hat = env_info["past_chunk"]["size"] / max(env_info["past_chunk"]["delay"], 1e-6) / THR_SCALE
            self.norm_past_throughputs.append(min(c_hat, MAX_THR))
            self.norm_past_ssims.append(env_info["past_chunk"]["ssim"])
            valid_past_action_norm = self.past_action / NUM_CHOICES_SCALE - 1
        buffer_norm = env_info["buffer"] / MAX_BUFFER_S * 2 - 1

        video_sizes = np.array(env_info["sizes"][0]) / THR_SCALE
        video_sizes = np.clip(video_sizes, a_max=MAX_CSIZE, a_min=0)
        ssim_indices = np.array(env_info["ssims"][0])

        assert video_sizes.ndim == 1
        assert video_sizes.shape == ssim_indices.shape

        context = list(self.norm_past_throughputs) + \
            list(self.norm_past_download_times) + \
            list(self.norm_past_ssims) + [buffer_norm, valid_past_action_norm, REBUF_DYN_PEN]

        options = np.c_[video_sizes, ssim_indices]

        obs = {"context": torch.as_tensor(context, dtype=torch.float, device=self.device),
               "options": torch.as_tensor(options, dtype=torch.float, device=self.device)}

        return obs
