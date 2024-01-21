# diffusion u-net implementation
from utils.modules import ResnetBlock, SinusoidalPosEmb, Residual, PreNorm, CrossAttention, Downsample, Upsample, RandomOrLearnedSinusoidalPosEmb

import torch
from torch import nn


class UNet(nn.Module):
    def __init__(self,
               dim,
               num_classes,
               dim_mults=[1, 2, 4, 8],
               channels=3,
               resnet_block_groups = 8,
               learned_variance = False,
               learned_sinusoidal_cond = False,
               random_fourier_features = False,
               block_per_layer = 2,
               learned_sinusoidal_dim = 16,
        ) -> None:
        super().__init__()
        
        self.channels = channels

        self.init_conv = nn.Conv2d(self.channels, dim, 7, padding = 3)

        self.out_dim = channels * (1 if not learned_variance else 2) 

        # Setup the dimensions of the u-net layers
        dims = [dim] + list(map(lambda x: x*dim, dim_mults))
        in_out = list(zip(dims[:-1], dims[1:]))

        # Time embeddings
        # TODO: I removed the capability for self learned embeddings - maybe re-enable?
        time_dim = dim * 4
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        # Simple MLP for time embedding
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # Class embeddings
        self.class_embedding = nn.Embedding(num_classes, dim)
        classes_dim = dim * 4

        # Simple MLP for further encoding class embeddings
        self.classes_mlp = nn.Sequential(
            nn.Linear(dim, classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim, classes_dim)
        )

        # Layers for u-net itself
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = (idx >= len(in_out) - 1)

            blocks = []
            for _ in range(block_per_layer):
                blocks.append(
                    ResnetBlock(
                        dim = dim_in,
                        dim_out = dim_in,
                        time_emb_dim = time_dim,
                        classes_emb_dim = classes_dim,
                        groups = resnet_block_groups,

                    )
                )
            
            blocks.append(
                Residual(
                    PreNorm(
                        dim_in, CrossAttention(dim_in)
                    )
                )
            )
            blocks.append(
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            )
            self.downs.append(nn.ModuleList(blocks))

        # Now in bottleneck of u-net
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(
            dim = mid_dim,
            dim_out = mid_dim,
            time_emb_dim = time_dim,
            classes_emb_dim = classes_dim,
            groups = resnet_block_groups
        )
        self.mid_attn = Residual(
            PreNorm(
                mid_dim, CrossAttention(mid_dim)
            )
        )
        self.mid_block2 = ResnetBlock(
            dim = mid_dim,
            dim_out = mid_dim,
            time_emb_dim = time_dim,
            classes_emb_dim = classes_dim,
            groups = resnet_block_groups
        )

        # Going back up the u-net
        for idx, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = (idx >= len(in_out) - 1)

            blocks = []
            for _ in range(block_per_layer):
                blocks.append(
                    ResnetBlock(
                        dim = dim_out + dim_in,
                        dim_out = dim_out,
                        time_emb_dim = time_dim,
                        classes_emb_dim = classes_dim,
                        groups = resnet_block_groups,

                    )
                )
            
            blocks.append(
                Residual(
                    PreNorm(
                        dim_out, CrossAttention(dim_out)
                    )
                )
            )
            blocks.append(
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            )
            self.ups.append(nn.ModuleList(blocks))

        self.final_res_block = ResnetBlock(
            dim = dim * 2,
            dim_out = dim,
            time_emb_dim = time_dim,
            classes_emb_dim = classes_dim,
            groups = resnet_block_groups,
        )
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits
        
        # condition zero classifier free
        args = tuple(arg if i!=2 else torch.zeros_like(arg, device=arg.device).int() for i,arg in enumerate(args))
        null_logits = self.forward(*args, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        classes
    ):
        batch, device = x.shape[0], x.device

        # derive condition, with condition dropout for classifier free guidance        
        masks = classes.clone()
        # Get the class values (I think)
        classes = (torch.max(classes.reshape(classes.shape[0],-1),-1).values).int()
        
        # Calculate class embeddings
        classes_emb = self.class_embedding(classes)
        c = self.classes_mlp(classes_emb)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []
        print("unet forward")

        for *blocks, attn, downsample in self.downs:
            for i, block in enumerate(blocks):
                x = block(x, t, c)
                if i < len(blocks)-1:
                    h.append(x)

            x = attn(x, masks)
            print(x.shape)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x, masks)
        x = self.mid_block2(x, t, c)

        for *blocks, attn, upsample in self.ups:
            for block in blocks:
                x = torch.cat((x, h.pop()), dim = 1)
                x = block(x, t, c)
            x = attn(x, masks)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t, c)
        return self.final_conv(x)
