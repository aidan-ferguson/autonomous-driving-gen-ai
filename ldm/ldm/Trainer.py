from accelerate import Accelerator 
import math
from torch.optim import Adam, lr_scheduler
from ema_pytorch import EMA
import torch
from tqdm import tqdm
from torchvision import transforms as utils
from pathlib import Path

from VAE import VAE
from dataloader import ComposeState, RandomRotate90, cycle, import_dataset

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        *,
        train_batch_size = 16,
        cond_scale = 3.0,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        save_loss_every = 100,
        num_workers = 0,
        num_samples = 4,
        data_folder = None,
        results_folder = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
        convert_image_to = None,
        out_size=None,
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no',
            gradient_accumulation_steps=gradient_accumulate_every
        )

        self.accelerator.native_amp = amp

        self.model = diffusion_model

        assert (math.sqrt(num_samples) ** 2) == num_samples, 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.save_loss_every = save_loss_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        self.cond_scale = cond_scale

        if data_folder:
            transform=ComposeState([
                        utils.ToTensor(),
                        utils.RandomHorizontalFlip(),
                        utils.RandomVerticalFlip(),
                        RandomRotate90(),
                        ])

            train_loader, test_loader = import_dataset(data_folder,
                                                batch_size=train_batch_size,   
                                                transform=transform)

            train_loader, test_loader = self.accelerator.prepare(train_loader, test_loader)
            self.dl = cycle(train_loader)
            self.test_loader= cycle(test_loader)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0
        self.running_loss=[]
        self.running_lr=[]

        # prepare model, optimizer with accelerator

        self.scheduler = lr_scheduler.OneCycleLR(self.opt, max_lr=train_lr, total_steps=train_num_steps)
        self.model, self.opt, self.ema, self.scheduler = self.accelerator.prepare(self.model, self.opt, self.ema, self.scheduler)

        self.vae = VAE()
        self.vae.vae = self.accelerator.prepare(self.vae.vae)
        
    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'loss': self.running_loss,
            'lr': self.running_lr,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'ema': self.accelerator.get_state_dict(self.ema),
            'scaler': self.accelerator.scaler.state_dict() if self.accelerator.scaler is not None else None,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=self.accelerator.device)

        self.model = self.accelerator.unwrap_model(self.model)
        self.model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.scheduler.load_state_dict(data['scheduler'])
        self.ema = self.accelerator.unwrap_model(self.ema)
        self.ema.load_state_dict(data['ema'])
        self.running_loss = data['loss']
        self.running_lr = data['lr']

        if data['scaler'] is not None:
            self.accelerator.scaler.load_state_dict(data['scaler'])
            
        self.model, self.opt, self.ema, self.scheduler = self.accelerator.prepare(self.model, self.opt, self.ema, self.scheduler)
            
    def train_loop(self, imgs, masks):
        with torch.no_grad():
            # this is the og line from diffinfinite - no clue why scaled by 1/50, same thing happens elsewhere when transitioning to/from latents
            # imgs=self.vae.module.encode(imgs).latent_dist.sample()/50
            imgs = self.vae.vae_encode(imgs)/50
            # masks = self.vae.vae_encode(masks)

        with self.accelerator.autocast():
            loss = self.model(img=imgs,classes=masks)
            
        self.accelerator.backward(loss)
                        
        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

        self.opt.step()
        self.opt.zero_grad()
        self.scheduler.step()

        return loss
    
    def eval_loop(self):
        
        if self.accelerator.is_main_process:
            self.ema.to(self.accelerator.device)
            self.ema.module.update()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                self.ema.module.ema_model.eval()

                with torch.no_grad():
                    milestone = self.step // self.save_and_sample_every
                    test_images,test_masks=next(self.test_loader)
                    z = self.vae.vae_encode(
                        test_images[:self.num_samples])/50
                    z = self.ema.module.ema_model.sample(z,test_masks[:self.num_samples])*50
                    test_samples=torch.clip(self.vae.vae_decode(z),0,1)
                    
                utils.save_image(test_images[:self.num_samples], 
                                 str(self.results_folder / f'images-{milestone}.png'), 
                                 nrow = int(math.sqrt(self.num_samples)))   
                
                utils.save_image((test_masks>0).float()[:self.num_samples], 
                                 str(self.results_folder / f'masks-{milestone}.png'), 
                                 nrow = int(math.sqrt(self.num_samples)))         
                
                utils.save_image(test_samples, 
                                 str(self.results_folder / f'sample-{milestone}.png'), 
                                 nrow = int(math.sqrt(self.num_samples)))
                
                self.save(milestone)

    def train(self):

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not self.accelerator.is_main_process) as pbar:
            
            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data,masks=next(self.dl)
                    
                    with self.accelerator.accumulate(self.model):
                        loss = self.train_loop(data,masks)
                        total_loss += loss.item()

                total_loss/=self.gradient_accumulate_every
                if self.step % self.save_loss_every == 0:
                    self.running_loss.append(total_loss)
                    self.running_lr.append(self.scheduler.get_lr()[0])

                pbar.set_description(f'loss: {total_loss:.4f}')

                self.step += 1
                self.eval_loop()
                pbar.update(1)

        self.accelerator.print('training complete')

from GaussianDiffusion import GaussianDiffusion
from unet import UNet
from dataloader import IMAGE_SIZE

if __name__ == "__main__":
    image_size: int = IMAGE_SIZE
    dim: int = 256
    # Yellow, Blue, Orange, Large Orange, Unknown, Background
    num_classes: int = 6
    # dim_mults = [1,2,4]
    channels: int = 4
    resnet_block_groups: int = 2
    block_per_layer: int = 2
    timesteps: int = 1000
    sampling_timesteps: int = 250
    batch_size: int = 32
    lr: float = 1e-4
    train_num_steps: int = 250000
    save_sample_every: int = 25000
    gradient_accumulate_every: int = 1
    save_loss_every: int = 100
    num_samples: int = 4
    num_workers: int = 32
    results_folder: str = './results/run_name'
    milestone: int = None

    z_size=image_size//8

    # dim_mults=[int(mult) for mult in dim_mults.split(' ')]
    
    unet = UNet(
            dim=dim,
            num_classes=num_classes,
            # dim_mults=dim_mults,
            channels=channels,
            resnet_block_groups = resnet_block_groups,
            block_per_layer=block_per_layer,
        )

    model = GaussianDiffusion(
            unet,
            image_size=z_size,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            loss_type='l2')

    trainer = Trainer(
            model,
            train_batch_size=batch_size,
            train_lr=lr,
            train_num_steps=train_num_steps,
            save_and_sample_every=save_sample_every,
            gradient_accumulate_every=gradient_accumulate_every,
            save_loss_every=save_loss_every,
            num_samples=num_samples,
            num_workers=num_workers,
            results_folder=results_folder,
            data_folder="/home/ugrdv/dissertation/parsed_fsoco")

    if milestone:
        trainer.load(milestone)
        
    trainer.train()