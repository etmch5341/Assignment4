import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DiffusionModel(nn.Module):
    def __init__(self, image_size, channels, hidden_dims=[32, 64, 128]):
        """
        Initialize the diffusion model.
        Args:
            image_size: Size of input images (assumed square)
            channels: Number of input channels (1 for MNIST)
            hidden_dims: List of channel dimensions for each level of the U-Net
        """
        super().__init__()
        
        # Save parameters
        self.image_size = image_size
        self.channels = channels
        self.hidden_dims = hidden_dims
        
        # Time embedding dimension
        time_emb_dim = hidden_dims[0] * 4
        
        # Time embedding module - sinusoidal positional encoding followed by MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dims[0], time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Initial convolution layer
        self.init_conv = nn.Conv2d(channels, hidden_dims[0], kernel_size=3, padding=1)
        
        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()
        
        in_channels = hidden_dims[0]
        for hidden_dim in hidden_dims:
            self.encoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.SiLU(),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.SiLU()
                )
            )
            self.encoder_pools.append(nn.MaxPool2d(2))
            in_channels = hidden_dim
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], hidden_dims[-1] * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dims[-1] * 2),
            nn.SiLU(),
            nn.Conv2d(hidden_dims[-1] * 2, hidden_dims[-1], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.SiLU()
        )
        
        # Decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()
        
        # Time embedding projections for each decoder level
        self.time_projections = nn.ModuleList()
        
        for i in reversed(range(len(hidden_dims))):
            # Upsample layer - use scale_factor=2 for 2x upsampling
            self.decoder_upsamples.append(
                nn.Upsample(scale_factor=2, mode='nearest')
            )
            
            # Time projection for this level
            self.time_projections.append(
                nn.Linear(time_emb_dim, hidden_dims[i])
            )
            
            # Decoder block (input is upsampled features + skip connection)
            # So input channels is hidden_dims[i] * 2
            if i > 0:
                out_channels = hidden_dims[i-1]
            else:
                out_channels = hidden_dims[0]
            
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dims[i] * 2, hidden_dims[i], kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_dims[i]),
                    nn.SiLU(),
                    nn.Conv2d(hidden_dims[i], out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU()
                )
            )
        
        # Final output layer
        self.final_conv = nn.Conv2d(hidden_dims[0], channels, kernel_size=1)
    
    def pos_encoding(self, t, channels):
        """
        Sinusoidal positional encoding for timesteps.
        Args:
            t: Timesteps tensor of shape [batch_size, 1]
            channels: Number of channels for encoding
        Returns:
            Positional encoding of shape [batch_size, channels]
        """
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=t.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
    def forward(self, x, t):
        """
        Forward pass through the U-Net model.
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            t: Timesteps tensor of shape [batch_size]
        Returns:
            Tensor of shape [batch_size, channels, height, width]
        """
        # 1. Embed the timestep
        t = t.unsqueeze(-1).float()  # [batch_size, 1]
        t_emb = self.pos_encoding(t, self.hidden_dims[0])  # [batch_size, hidden_dims[0]]
        t_emb = self.time_mlp(t_emb)  # [batch_size, time_emb_dim]
        
        # 2. Process input through initial convolution
        x = self.init_conv(x)
        
        # 3. Store residuals for skip connections
        residuals = []
        
        # 4. Process through encoder blocks
        for encoder_block, pool in zip(self.encoder_blocks, self.encoder_pools):
            x = encoder_block(x)
            residuals.append(x)
            x = pool(x)
        
        # 5. Process through bottleneck
        x = self.bottleneck(x)
        
        # 6. Process through decoder blocks with time injection and skip connections
        for i, (upsample, decoder_block, time_proj) in enumerate(
            zip(self.decoder_upsamples, self.decoder_blocks, self.time_projections)
        ):
            # Upsample
            x = upsample(x)
            
            # Get skip connection from encoder
            residual = residuals[-(i+1)]
            
            # CRITICAL FIX: Match dimensions if there's a size mismatch (due to odd-sized pooling)
            if x.shape[-2:] != residual.shape[-2:]:
                x = F.interpolate(x, size=residual.shape[-2:], mode='nearest')
            
            # Concatenate with skip connection
            x = torch.cat([x, residual], dim=1)
            
            # Apply decoder block
            x = decoder_block(x)
            
            # Inject time embedding
            time_features = time_proj(t_emb)  # [batch_size, channels]
            time_features = time_features[:, :, None, None]  # [batch_size, channels, 1, 1]
            x = x + time_features
        
        # 7. Apply final convolution
        x = self.final_conv(x)
        
        # 8. Return the output tensor
        return x