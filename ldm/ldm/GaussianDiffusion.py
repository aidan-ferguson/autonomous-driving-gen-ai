from torch import nn
import torch
import torch.nn.functional as F
from functools import partial
import math
from collections import namedtuple
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms as T
from einops import reduce

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 1.
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        # default num sampling timesteps to number of timesteps at training
        if sampling_timesteps is not None:
            self.sampling_timesteps = sampling_timesteps
        else:
            self.sampling_timesteps = timesteps 

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def identity(t, *args, **kwargs):
        return t

    def model_predictions(self, x, t, classes, cond_scale = 3., clip_x_start = False):
        model_output = self.model.forward_with_cond_scale(x, t, classes, cond_scale = cond_scale)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else GaussianDiffusion.identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, classes, cond_scale, clip_denoised = True):
        preds = self.model_predictions(x, t, classes, cond_scale)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, 
                                                                                  x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, classes, cond_scale = 3., clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, classes = classes, cond_scale = cond_scale, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, classes, shape, cond_scale = 3., verbose=False):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None

        if verbose:
            print('Start Sampling\n')
            iterator = tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps)
        else:
            iterator = reversed(range(0, self.num_timesteps))
            
        for t in iterator:
            img, x_start = self.p_sample(img, t, classes, cond_scale)
        return img

    
    @torch.no_grad()
    def ddim_onemask(self, x_t, labels, masks, time, time_next, cond_scale):
        
        masks=torch.cat([(mask*labels[i])[None] for i,mask in enumerate(masks)],0).to(x_t.device)
        cond_scale = 1.0 if (labels == 0).all().item() else cond_scale
        time_cond = torch.full((x_t.shape[0],), time, device=x_t.device, dtype=torch.long)
        
        pred_noise, x_start, *_ = self.model_predictions(x_t, time_cond, masks, 
                                                         cond_scale = cond_scale, 
                                                         clip_x_start = True)
        
        if time_next > 0:
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x_t, device=x_t.device)

            x_next = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
        else:
            x_next = x_start

        return x_next 
    
    @torch.no_grad()
    def ddim_multimask(self, x_t, masks, time, time_next, cond_scale=3.0):
        
        
        labels=[torch.unique(mask) for mask in masks]
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1).int()
        
        x_next = torch.zeros_like(x_t, device=x_t.device)
        
        for i in range(len(padded_labels[0])):
            labels=padded_labels[:,i]
            indices = torch.where(labels != -1)[0]
            sub_images, sub_masks, sub_labels=map(lambda x: x[indices].clone(), (x_t,masks,labels)) 
            #exclude other labels from the sub_masks
            sub_masks = (sub_masks == sub_labels[:, None, None, None]).float() 
            
            x_next[indices] += self.ddim_onemask(sub_images, sub_labels, sub_masks, time, 
                                        time_next, cond_scale=cond_scale)*sub_masks      
            
        return x_next

    @torch.no_grad()
    def ddim_sample(self, images, classes, shape, cond_scale = 3., 
                    sampling_timesteps=None,
                    clip_denoised=True, 
                    inp_mask=None, verbose=False):
        
        sampling_timesteps = self.sampling_timesteps if not sampling_timesteps else sampling_timesteps
        
        # Prepare (time, time_next) pairs
        times = torch.linspace(-1, self.num_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) 

        # Initialise step t=T
        x_t = torch.randn(shape, device = images.device)
        
        # Downsample masks for latent space
        masks=classes.clone().float()
        vmin,vmax=masks.min(),masks.max()
        masks=T.Lambda(lambda x: F.interpolate(x,size=self.image_size))(masks)
        masks=torch.clamp(torch.round(masks),vmin,vmax)

        if verbose:
            print('Start Sampling\n')
            iterator = tqdm(time_pairs, desc = 'sampling loop time step')
        else:
            iterator = time_pairs
            
        for time, time_next in iterator:  
            x_t = self.ddim_multimask(x_t, masks, time, time_next, cond_scale=cond_scale)
            
            noise = torch.randn(x_t.shape, device=x_t.device)
            time_cond = torch.full((x_t.shape[0],), time, device=x_t.device, dtype=torch.long)
            if time_next>0:
                x_0_noised = self.q_sample(x_start = images.clone(), 
                                           t = time_cond, noise = noise)
                if inp_mask is not None:
                    x_t = x_0_noised*(1-inp_mask) + x_t*inp_mask
    
        return x_t

    @torch.no_grad()
    def sample(self, images, classes, inp_mask=None, sampling_timesteps=250, cond_scale = 3., verbose=False):
        batch_size, image_size, channels = classes.shape[0], self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(images=images, classes=classes, inp_mask=inp_mask, shape=(batch_size, channels, image_size, image_size),  cond_scale=cond_scale, sampling_timesteps=sampling_timesteps, clip_denoised=True, verbose=verbose)
    
    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        if t is None:
            t = self.num_timesteps - 1

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device = device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, *, classes, noise = None):
        b, c, h, w = x_start.shape

        if noise is None:
            noise = torch.randn_like(x_start)

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)
        
        # predict and take gradient step

        model_out = self.model(x, t, classes)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        
        model_out=torch.nan_to_num(model_out)
        target=torch.nan_to_num(target)
        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # Defined as identity in diffinfinite?
        # img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, *args, **kwargs)