"""
This file is the main file for inferencing simulator images and producing novel images
"""

import torch
import numpy as np
import argparse, os
from PIL import Image
from typing import Tuple
from einops import rearrange
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.ddpm import LatentDiffusion

# Paths for relevant files from the ldm/ directory 
CONFIG_FILEPATH = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
MODEL_PATH = "models/txt2img-large.ckpt"
OUTPUT_DIR = "output"

def load_model_from_config(config: str, ckpt: str) -> LatentDiffusion:
    """
    Load a TorchLightning model given a configuration file and the corresponding model weights

    :param config: The path to the configuration file
    :parma ckpt: The path to the model weights 
    :returns: The latent diffusion model
    """

    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0:
        print(f"Missing keys: {m}")
    if len(u) > 0:
        print(f"Unexpected keys {u}")

    model.cuda()
    model.eval()
    return model


def load_image(image_file: str, dimensions: Tuple[int, int]) -> torch.tensor:
    """
    Read an image from a file and convert it to a tensor representation ready for encoding into latent space

    :param image_file: The filename of the image to be loaded
    :param dimensions: The width and height that the image should be
    :returns: A tensor representing the image that can be encoded into latent space
    """

    # Load image and normalise
    image = Image.open(image_file).convert('RGB').resize((dimensions[0], dimensions[1]), resample=Image.Resampling.LANCZOS)
    image = np.float32(image) / 255.0

    # Convert to tensor and the convert from range [0, 1] to range [-1, 1]
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddim_steps", type=int,   default=50,   help="number of ddim sampling steps")
    parser.add_argument("--ddim_eta",   type=float, default=0.0,  help="ddim eta (eta=0.0 corresponds to deterministic sampling")
    parser.add_argument("--height",     type=int,   default=256,  help="desired image height")
    parser.add_argument("--width",      type=int,   default=256,  help="desired image width")
    parser.add_argument("--n_samples",  type=int,   default=1,    help="how many samples to produce for the given prompt")
    parser.add_argument("--scale",      type=float, default=5.0,  help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))")
    parser.add_argument("--p_image",    type=float, default=0.25, help="The weight of the initial image between 0 and 1. 1 means the image is the initial latent image and 0 means the image has no impact")
    parser.add_argument("--init_image", type=str,                 help="The filepath of the initial image")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Initialise the file output directory
    ldm_dir = os.path.abspath(os.path.join((os.path.realpath(__file__)), os.pardir))
    output_dir = os.path.join(ldm_dir, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # Load both the config and the model itself
    model_config = OmegaConf.load(os.path.join(ldm_dir, CONFIG_FILEPATH))
    model = load_model_from_config(model_config, os.path.join(ldm_dir, MODEL_PATH)).to(device)

    # Create the sampler
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=args.ddim_steps, ddim_eta=args.ddim_eta, verbose=False)

    # Load the initial image and convert it into latent space
    image = load_image(args.init_image, (args.width, args.height)).to(device=device)
    initial_latent_image = model.get_first_stage_encoding(model.encode_first_stage(image))

    # Create the initial latent space representation by adding noise 
    t_enc = int((1 - args.p_image) * args.ddim_steps)
    initial_latent_image = sampler.stochastic_encode(initial_latent_image, torch.tensor([t_enc]).to(model.device.type))

    # Run the stable diffusion process
    shape = [4, args.height//8, args.width//8]
    with torch.no_grad():
        with model.ema_scope():
            
            # Create prompt embeddings  
            uc = None
            if args.scale != 1.0:
                uc = model.get_learned_conditioning(args.n_samples * [""])
            c = model.get_learned_conditioning(args.n_samples * ["A POV dash cam image. a white car body with black wheels is visible in the bottom of the image. A blue sky is visible in the top of the image"])

            # Run diffusion process
            samples_ddim, _ = sampler.sample(S=args.ddim_steps,
                                                conditioning=c,
                                                batch_size=args.n_samples,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=args.scale,
                                                unconditional_conditioning=uc,
                                                eta=args.ddim_eta,
                                                x_T=initial_latent_image
                                                )

            # Decode latent space image and convert back to range [0, 1]
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

            # Save decoded images to file
            for idx, x_sample in enumerate(x_samples_ddim):
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(output_dir, f"{idx:04}.png"))
