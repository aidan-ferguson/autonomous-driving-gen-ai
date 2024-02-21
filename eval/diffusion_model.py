# Add ControlNet folder to path and import 
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), "../ldm/ControlNet"))

import cv2
import einops
import numpy as np
import torch
import random


from ...ldm.ControlNet.share import *
from pytorch_lightning import seed_everything
from ...ldm.ControlNet.cldm.model import create_model, load_state_dict
from ...ldm.ControlNet.cldm.ddim_hacked import DDIMSampler

save_memory = False

class DiffusionModel():
    def __init__(self, model_path: str):
        pass