# Add ControlNet folder to path and import 
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), "../ldm/ControlNet"))

import cv2
import einops
import numpy as np
import torch
from numpy.typing import NDArray

from share import *
from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

save_memory = False

class DiffusionModel():
    def __init__(self, model_path: str):
        self.model = create_model(os.path.join(os.path.dirname(__file__), '../ldm/ControlNet/models/cldm_v15.yaml')).cpu()
        self.model.load_state_dict(load_state_dict(model_path, location='cuda'))
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)
        self.model.eval()

    @torch.no_grad()
    def forward(self, 
                masks: list[NDArray],
                ddim_steps: int = 50
                ):
        assert len(masks) > 0

        n_samples = len(masks)
        guess_mode = False
        n_prompt = ""
        strength = 1.0
        eta = 0.0
        scale = 9.0 # TODO: experiment with the effect of scale changes (or any of these values)?

        for idx, mask in enumerate(masks):
            masks[idx] = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            masks[idx] = torch.from_numpy(masks[idx].copy()).float().cuda() / 255.0
        
        H, W, C = masks[0].shape
        control = torch.stack(masks, dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([""] * n_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * n_samples)]}
        shape = (4, H // 8, W // 8)

        self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = self.ddim_sampler.sample(ddim_steps, n_samples,
                                                    shape, cond, verbose=False, eta=eta,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=un_cond)
        
        # TODO: intermediates could provide cool visualisation for presentation?

        x_samples = self.model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(n_samples)]

        return results