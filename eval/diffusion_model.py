from ...ldm.ControlNet.share import *

import cv2
import einops
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from ...ldm.ControlNet.cldm.model import create_model, load_state_dict
from ...ldm.ControlNet.cldm.ddim_hacked import DDIMSampler

save_memory = False

class DiffusionModel():
    def __init__(self, model_path: str):
        pass