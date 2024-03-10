# Add ControlNet folder to path and import 
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), "../gan/contrastive-unpaired-translation"))

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from models import networks

class CUTModel:
    def __init__(self, model_path: str):
        # Below are the options used when training the final CUT model
        input_nc  = 3 # the number of channels in input images
        output_nc  = 3 # the number of channels in output images
        ngf  = 64 # the number of filters in the last conv layer
        netGArch  = "resnet_9blocks" # the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm  = "instance" # the name of normalization layers used in the network: batch | instance | none
        use_dropout = False # if use dropout layers.
        init_type = "xavier" # the name of our initialization method.
        init_gain = 0.02 # scaling factor for normal, xavier and orthogonal.
        self.gpu_ids = [0] # which GPUs the network runs on: e.g., 0,1,2
        no_antialias=False
        no_antialias_up=False

        # Define image transformation we will need later
        self.resize = transforms.Resize((256, 256), Image.BICUBIC)
        self.to_tensor = transforms.ToTensor()
        

        self.model = networks.define_G(input_nc, output_nc, ngf, netGArch, norm, use_dropout, init_type, init_gain, no_antialias, no_antialias_up, self.gpu_ids)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    @torch.no_grad()
    def forward(self,
                sim_frames: list[Image.Image]):
            
            sim_frames = torch.tensor([self.to_tensor(self.resize(im)) for im in sim_frames]).to(self.gpu_ids[0])
            fake = self.model(sim_frames)

            numpy_img = fake.to("cpu").detach().numpy()
            numpy_img = np.einsum("bcxy->bxyc", numpy_img)

            # Normalise to 0-1
            numpy_img = (numpy_img + 1.0) / 2.0
            numpy_img = np.array(numpy_img*255.0, dtype=np.uint8)

            return numpy_img