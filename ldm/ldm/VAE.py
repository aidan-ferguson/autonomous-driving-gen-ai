import torch
from diffusers import DiffusionPipeline
# import cv2
import numpy as np
# import matplotlib.pyplot as plt

# TODO: cite all libraries included
# TODO: use accelerate for easy multi-gpu training

class VAE():
    # Note, I *think* we only use the VAE from pre-trained stable diffusion
    base_model = "stabilityai/stable-diffusion-2-base"

    def __init__(self) -> None:
        self.vae = DiffusionPipeline.from_pretrained(VAE.base_model).vae

    @torch.no_grad()
    def vae_encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Encode n images into latent space using the pre-trained VAE

        :param imgs: The images you want to encode, must be a float32 tensor with pixel
            values in range [0, 1] and be of shape (n, 3, h, w)
        :returns: The encoded latents, will be of shape (n, 4, h/f, w/f), where f is the 
            scaling factor (in our case 8)
        """
        return self.vae.encode(imgs).latent_dist.sample()
    
    @torch.no_grad()
    def vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode n latents into pixel space images using the pre-trained VAE

        :param latents: Latent space tensors representing images, must be of shape
            (n, 4, h/f, w/f), where f is the scaling factor (8 for us)
        """
        return torch.clip(self.vae.decode(latents).sample, 0, 1)

# temp
def load_image(filepath: str) -> torch.Tensor:
    # Arbitrary size for now, can be non-square
    IMAGE_SIZE = (416, 416)

    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    img = np.einsum("whc->chw", img)
    img = np.array(img, dtype=np.float32)/255.0
    return torch.tensor(img, dtype=torch.float32)

if __name__ == "__main__":
    # Simple test to verify functionality of the variational auto-encoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("Warning, running with CPU")

    ldm = VAE()
    img = load_image("/home/aidan/dissertation/dissertation/data/frames/real_world/0.png").unsqueeze(0)
    latent = ldm.vae_encode(img.to(device=device))
    img_prime = ldm.vae_decode(latent)

    plt.imshow(np.einsum("chw->whc", img_prime.cpu().detach().numpy()[0]))
    plt.show()