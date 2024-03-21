# Add ControlNet folder to path and import 
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), "../gan/pytorch-CycleGAN-and-pix2pix"))

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from collections import OrderedDict

from models import networks

class Pix2PixModel:
    def __init__(self, model_path: str):
        # Below are the options used when training the final CUT model
        input_nc  = 3 # the number of channels in input images
        output_nc  = 3 # the number of channels in output images
        ngf  = 64 # the number of filters in the last conv layer
        netGArch  = "unet_256" # the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm  = "batch" # the name of normalization layers used in the network: batch | instance | none
        use_dropout = True # if use dropout layers.
        init_type = "normal" # the name of our initialization method.
        init_gain = 0.02 # scaling factor for normal, xavier and orthogonal.
        self.gpu_ids = [0] # which GPUs the network runs on: e.g., 0,1,2

        # Define image transformation we will need later
        self.resize = transforms.Resize((256, 256))
        self.to_tensor = transforms.ToTensor()
        
        # opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids
        self.model = networks.define_G(input_nc, output_nc, ngf, netGArch, norm, use_dropout, init_type, init_gain, self.gpu_ids)
        state_dict = torch.load(model_path)
        corrected_state_dict = OrderedDict()
        for key in state_dict.keys():
             corrected_state_dict[f"module.{key}"] = state_dict[key]
        self.model.load_state_dict(corrected_state_dict)
        self.model.eval()

        print(f"Loaded Pix2Pix model {model_path}")

    @torch.no_grad()
    def forward(self,
                masks: list[Image.Image]):
            
            masks = [self.to_tensor(self.resize(im)) for im in masks]
            masks = [(mask*2.0) - 1.0 for mask in masks]

            if len(masks) > 1:
                masks = torch.stack(masks).to(self.gpu_ids[0])
            else:
                masks = masks[0][None, :].to(self.gpu_ids[0])
            
            fake = self.model(masks)

            numpy_img = fake.to("cpu").detach().numpy()
            numpy_img = np.einsum("bcxy->bxyc", numpy_img)

            # Normalise to 0-1
            numpy_img = (numpy_img + 1.0) / 2.0
            numpy_img = np.array(numpy_img*255.0, dtype=np.uint8)

            return numpy_img