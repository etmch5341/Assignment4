import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DiffusionProcess:
    def __init__(self, image_size, channels, hidden_dims=[64, 128, 256, 512], beta_start=1e-4, beta_end=0.02, noise_steps=1000, device=torch.device('cpu')):
        """
        Initialize the diffusion process.
        Args:
            beta_start: Initial noise variance
            beta_end: Final noise variance
            noise_steps: Number of diffusion steps
        """
        
        def cosine_beta_schedule(timesteps, s=0.008):
            """
            cosine schedule as proposed in https://arxiv.org/abs/2102.09672
            """
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            
            # --- FIX: Reduce the max clip value ---
            # Changing 0.9999 -> 0.95 prevents the division-by-zero instability
            return torch.clip(betas, 0.0001, 0.95)
        
        super().__init__()
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.noise_steps = noise_steps
    
        # TODO: Define beta schedule and calculate derived quantities
        # Create a linear noise schedule from beta_start to beta_end
        # Calculate alpha, alpha_cumprod, and their square roots
        
        # TODO: Initialize the model and optimizer
        # Create an instance of the DiffusionModel
        # Set up the optimizer
        
        self.image_size = image_size
        self.channels = channels
        self.noise_steps = noise_steps
        self.device = device
        self.hidden_dims = hidden_dims
    
        # 1. Define Beta Schedule (Linear)
        # self.betas = torch.linspace(beta_start, beta_end, noise_steps).to(device)
        
        # Define Cosine Beta Schedule (alternative)
        self.betas = cosine_beta_schedule(noise_steps).to(device)
        
        # 2. Calculate Alphas
        self.alphas = 1.0 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=0) # Cumulative product
        
        # 3. Pre-calculate values for Forward Diffusion (add_noise)
        self.sqrt_alpha_hats = torch.sqrt(self.alpha_hats)
        self.sqrt_one_minus_alpha_hats = torch.sqrt(1.0 - self.alpha_hats)
        
        # 4. Initialize Model and Optimizer
        self.model = DiffusionModel(image_size, channels, hidden_dims).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
        self.criterion = nn.MSELoss()
        
    def add_noise(self, x, t):
        """
        Add noise to the input images according to the diffusion process.
        Args:
            x: Clean images tensor of shape [batch_size, channels, height, width]
            t: Timesteps tensor of shape [batch_size]
        Returns:
            Tuple of (noisy_images, noise)
        """
        # TODO: Implement the forward diffusion process
        # 1. Get the appropriate alpha_cumprod values for the timesteps
        # 2. Generate Gaussian noise
        # 3. Combine the clean images and noise according to the diffusion equation
        # 4. Return the noisy images and the noise
        # pass
        sqrt_alpha_hat = self.sqrt_alpha_hats[t][:, None, None, None]
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hats[t][:, None, None, None]
        
        # Generate Gaussian noise
        epsilon = torch.randn_like(x)
        
        # Apply formula
        noisy_images = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon
        
        return noisy_images, epsilon
    
    def sample(self, num_samples=16):
        """
        Generate new samples by reversing the diffusion process.
        Args:
            num_samples: Number of samples to generate
        Returns:
            Generated images tensor
        """
        # TODO: Implement the reverse diffusion sampling process
        # 1. Start with random noise
        # 2. Gradually denoise the samples by iterating through timesteps in reverse
        # 3. For each step, predict noise and perform denoising
        # 4. Return the generated samples
        # pass
        self.model.eval() # Set model to evaluation mode
        
        with torch.no_grad():
            # 1. Start with pure random noise
            x = torch.randn(num_samples, self.channels, self.image_size, self.image_size).to(self.device)
            
            # 2. Iterate backwards from T-1 to 0
            for i in reversed(range(self.noise_steps)):
                # Create a batch of timesteps (all equal to i)
                t = (torch.ones(num_samples) * i).long().to(self.device)
                
                # Predict the noise using the model
                predicted_noise = self.model(x, t)
                
                # Get the alpha/beta values for this step
                alpha = self.alphas[t][:, None, None, None]
                alpha_hat = self.alpha_hats[t][:, None, None, None]
                beta = self.betas[t][:, None, None, None]
                
                # Add noise (z) for Langevin dynamics, except for the very last step (i=0)
                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                # 3. Denoising Step Formula (DDPM)
                # x_{t-1} = 1/sqrt(alpha) * (x_t - (1-alpha)/sqrt(1-alpha_hat) * epsilon) + sigma * z
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                
        self.model.train() # Set back to train mode
        
        # Clamp values to valid image range [-1, 1] (optional but good practice)
        x = x.clamp(-1, 1)
        
        return x
    
    def train_step(self, x):
        """
        Perform one training step for the diffusion model.
        Args:
            x: Clean images tensor of shape [batch_size, channels, height, width]
        Returns:
            Loss value for the step
        """
        # TODO: Implement one training step
        # 1. Sample random timesteps
        # 2. Add noise to images
        # 3. Predict the noise using the model
        # 4. Calculate loss between predicted and actual noise
        # 5. Perform backpropagation
        # 6. Return the loss value
        # pass
        self.optimizer.zero_grad()
        
        # 1. Sample random timesteps
        t = torch.randint(0, self.noise_steps, (x.shape[0],)).to(self.device)
        
        # 2. Add noise to images
        x_noisy, noise = self.add_noise(x, t)
        
        # 3. Predict the noise using the model
        noise_pred = self.model(x_noisy, t)
        
        # 4. Calculate Loss (MSE between actual noise and predicted noise)
        loss = self.criterion(noise_pred, noise)
        
        # 5. Backpropagation
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


class DiffusionModel(nn.Module):
    def __init__(self, image_size, channels, hidden_dims=[64, 128, 256, 512]):#[32, 64, 128]
        """
        Initialize the diffusion model.
        Args:
            config: Configuration object with model parameters
        """
        super().__init__()
        
        # TODO: Check the parameters and save up necessary ones
        
        # TODO: Implement the time embedding module
        # Create a time embedding MLP to encode the timestep
        # This should consist of linear layers with SiLU activation
        
        # TODO: Implement the initial convolution layer
        # Create an initial convolution layer to process the input image
        
        # TODO: Implement the encoder (downsampling path)
        # Create a list of down blocks for the encoder path
        # Each block should include convolutions, batch normalization, and activation
        # Don't forget to include a downsampling mechanism (e.g., MaxPool2d)
        
        # TODO: Implement the bottleneck
        # Create a bottleneck block with additional processing
        
        # TODO: Implement the decoder (upsampling path)
        # Create a list of up blocks for the decoder path
        # Each block should include upsampling, concatenation with skip connections,
        # and convolutions with batch normalization and activation
        
        # TODO: Implement time embedding projections
        # Create projections for injecting time features into each decoder layer
        
        # TODO: Implement the final output layer
        # Create a final convolution to map to the output channels
        
        # # Save parameters
        # self.image_size = image_size
        # self.channels = channels
        # self.hidden_dims = hidden_dims
        
        # # 1. Time Embedding
        # time_emb_dim = hidden_dims[0] # i.e. 32 (smallest channel size)
        # self.time_mlp = nn.Sequential(
        #     nn.Linear(1, time_emb_dim),
        #     nn.SiLU(),
        #     nn.Linear(time_emb_dim, time_emb_dim),
        # )

        # # 2. Initial Convolution
        # self.inc = nn.Conv2d(channels, hidden_dims[0], kernel_size=3, padding=1)

        # # 3. Encoder
        # self.down1 = nn.Sequential(
        #     nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=3, padding=1),
        #     nn.BatchNorm2d(hidden_dims[1]),
        #     nn.SiLU()
        # )
        # self.down2 = nn.Sequential(
        #     nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel_size=3, padding=1),
        #     nn.BatchNorm2d(hidden_dims[2]),
        #     nn.SiLU()
        # )
        # self.pool = nn.MaxPool2d(2)

        # # 4. Bottleneck
        # self.bot1 = nn.Conv2d(hidden_dims[2], hidden_dims[2], kernel_size=3, padding=1)
        # self.bot2 = nn.Conv2d(hidden_dims[2], hidden_dims[2], kernel_size=3, padding=1)

        # # 5. Decoder
        # self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # # --- FIX 1: Correct Input Channels for dec1 ---
        # # Input is concat of upsampled bottleneck (128) + skip connection x3 (128) = 256
        # self.dec1 = nn.Sequential(
        #     nn.Conv2d(hidden_dims[2] * 2, hidden_dims[1], kernel_size=3, padding=1),
        #     nn.BatchNorm2d(hidden_dims[1]),
        #     nn.SiLU()
        # )
        
        # # dec2 input is concat of upsampled dec1 (64) + skip connection x1 (32) = 96
        # # This matches hidden_dims[1] + hidden_dims[0]
        # self.dec2 = nn.Sequential(
        #     nn.Conv2d(hidden_dims[1] + hidden_dims[0], hidden_dims[0], kernel_size=3, padding=1),
        #     nn.BatchNorm2d(hidden_dims[0]),
        #     nn.SiLU()
        # )

        # # --- FIX 2 & 3: Correct Time Projection Dimensions ---
        # # time_proj1 adds to bottleneck features (128 ch)
        # self.time_proj1 = nn.Linear(time_emb_dim, hidden_dims[2]) 
        
        # # time_proj2 adds to dec1 output features (64 ch)
        # self.time_proj2 = nn.Linear(time_emb_dim, hidden_dims[1]) 

        # # 7. Final Output
        # self.outc = nn.Conv2d(hidden_dims[0], channels, kernel_size=1)
        
        # === 4 layer model with 64,128,256,512 hidden dims ===
        self.image_size = image_size
        self.channels = channels
        self.hidden_dims = hidden_dims
        
        # 1. Time Embedding - CHANGE 2: Larger embedding dimension
        time_emb_dim = hidden_dims[0]  # Now 64 instead of 32
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # 2. Initial Convolution - CHANGE 3: Output to 64 channels
        self.inc = nn.Conv2d(channels, hidden_dims[0], kernel_size=3, padding=1)

        # 3. Encoder - ADD NEW down3 block
        self.down1 = nn.Sequential(
            nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=3, padding=1),  # 64 -> 128
            nn.BatchNorm2d(hidden_dims[1]),
            nn.SiLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel_size=3, padding=1),  # 128 -> 256
            nn.BatchNorm2d(hidden_dims[2]),
            nn.SiLU()
        )
        # CHANGE 4: ADD down3
        self.down3 = nn.Sequential(
            nn.Conv2d(hidden_dims[2], hidden_dims[3], kernel_size=3, padding=1),  # 256 -> 512
            nn.BatchNorm2d(hidden_dims[3]),
            nn.SiLU()
        )
        self.pool = nn.MaxPool2d(2)

        # 4. Bottleneck - CHANGE 5: Now operates on 512 channels
        self.bot1 = nn.Conv2d(hidden_dims[3], hidden_dims[3], kernel_size=3, padding=1)
        self.bot2 = nn.Conv2d(hidden_dims[3], hidden_dims[3], kernel_size=3, padding=1)

        # 5. Decoder - ADD up0 and dec0
        # CHANGE 6: ADD first upsampling layer
        # self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # CHANGE 7: ADD dec0 for first decoder block
        # Input: concat of upsampled bottleneck (512) + skip x4 (512) = 1024
        # self.dec0 = nn.Sequential(
        #     nn.Conv2d(hidden_dims[3] * 2, hidden_dims[2], kernel_size=3, padding=1),  # 1024 -> 256
        #     nn.BatchNorm2d(hidden_dims[2]),
        #     nn.SiLU()
        # )
        
        # CHANGE 8: Update dec1 dimensions
        # Input: concat of upsampled dec0 (256) + skip x3 (256) = 512
        self.dec1 = nn.Sequential(
            nn.Conv2d(768, hidden_dims[1], kernel_size=3, padding=1),  # 768 -> 128
            nn.BatchNorm2d(hidden_dims[1]),
            nn.SiLU()
        )
        
        # CHANGE 9: Update dec2 dimensions
        # Input: concat of upsampled dec1 (128) + skip x1 (64) = 192
        self.dec2 = nn.Sequential(
            nn.Conv2d(hidden_dims[1] + hidden_dims[0], hidden_dims[0], kernel_size=3, padding=1),  # 192 -> 64
            nn.BatchNorm2d(hidden_dims[0]),
            nn.SiLU()
        )

        # 6. Time Projections - ADD time_proj0 and update others
        # CHANGE 10: ADD time_proj0
        # self.time_proj0 = nn.Linear(time_emb_dim, hidden_dims[3])  # 64 -> 512
        
        # CHANGE 11: Update time_proj1
        self.time_proj1 = nn.Linear(time_emb_dim, hidden_dims[3])  # 64 -> 512
        
        # CHANGE 12: Update time_proj2
        self.time_proj2 = nn.Linear(time_emb_dim, hidden_dims[1])  # 64 -> 128

        # 7. Final Output - channels stay the same
        self.outc = nn.Conv2d(hidden_dims[0], channels, kernel_size=1)
    
    def forward(self, x, t):
        """
        Forward pass through the U-Net model.
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            t: Timesteps tensor of shape [batch_size]
        Returns:
            Tensor of shape [batch_size, channels, height, width]
        """
        # TODO: Implement the forward pass
        # 1. Embed the timestep
        # 2. Process input through initial convolution
        # 3. Store residuals for skip connections
        # 4. Process through encoder blocks
        # 5. Process through bottleneck
        # 6. Process through decoder blocks with time injection and skip connections
        # 7. Apply final convolution
        # 8. Return the output tensor
        # pass
        # # 1. Embed Time
        # t = t.float().unsqueeze(-1) 
        # t_emb = self.time_mlp(t)

        # # 2. Initial Conv
        # x1 = self.inc(x)            # (B, 32, 28, 28)

        # # 3. Encoder
        # x2 = self.down1(x1)         # (B, 64, 28, 28)
        # x2_p = self.pool(x2)        # (B, 64, 14, 14)
        
        # x3 = self.down2(x2_p)       # (B, 128, 14, 14)
        # x3_p = self.pool(x3)        # (B, 128, 7, 7)

        # # 4. Bottleneck
        # x_bot = self.bot1(x3_p)
        # x_bot = self.bot2(x_bot)    # (B, 128, 7, 7)

        # # 5. Decoder Step 1
        # x_up1 = self.up1(x_bot)     # (B, 128, 14, 14)
        
        # # Fix: time_emb1 is now 128 channels, matches x_up1
        # time_emb1 = self.time_proj1(t_emb).unsqueeze(-1).unsqueeze(-1)
        # x_up1 = x_up1 + time_emb1 
        
        # # Concatenation: 128 + 128 = 256 channels
        # x_cat1 = torch.cat([x_up1, x3], dim=1) 
        # x_dec1 = self.dec1(x_cat1)  # Output is 64 channels

        # # 6. Decoder Step 2
        # x_up2 = self.up2(x_dec1)    # (B, 64, 28, 28)
        
        # # Fix: time_emb2 is now 64 channels, matches x_up2
        # time_emb2 = self.time_proj2(t_emb).unsqueeze(-1).unsqueeze(-1)
        # x_up2 = x_up2 + time_emb2
        
        # # Concatenation: 64 + 32 = 96 channels
        # x_cat2 = torch.cat([x_up2, x1], dim=1) 
        # x_dec2 = self.dec2(x_cat2)  # Output is 32 channels

        # # 7. Final
        # output = self.outc(x_dec2)  # Output is 1 channel
        
        # return output
        
        # === 4 layer model forward pass with 64,128,256,512 hidden dims ===
        # 1. Embed Time
        t = t.float().unsqueeze(-1) 
        t_emb = self.time_mlp(t)

        # 2. Initial Conv
        x1 = self.inc(x)            # (B, 64, 28, 28)

        # 3. Encoder
        x2 = self.down1(x1)         # (B, 128, 28, 28)
        x2_p = self.pool(x2)        # (B, 128, 14, 14)
        
        x3 = self.down2(x2_p)       # (B, 256, 14, 14)
        x3_p = self.pool(x3)        # (B, 256, 7, 7)
        
        x4 = self.down3(x3_p)       # (B, 512, 7, 7)
        # FIX: DON'T POOL HERE - keep at 7×7
        # x4_p = self.pool(x4)      # ← REMOVE THIS LINE

        # 4. Bottleneck - operates on x4 (7×7), not x4_p
        x_bot = self.bot1(x4)       # (B, 512, 7, 7)
        x_bot = self.bot2(x_bot)    # (B, 512, 7, 7)

        # 5. Decoder
        # FIX: Skip the up0 step entirely since we're already at 7×7
        # x_up0 = self.up0(x_bot)   # ← REMOVE THIS
        # time_emb0 = self.time_proj0(t_emb).unsqueeze(-1).unsqueeze(-1)
        # x_up0 = x_up0 + time_emb0 
        # x_cat0 = torch.cat([x_up0, x4], dim=1)
        # x_dec0 = self.dec0(x_cat0)
        
        # Start decoder from bottleneck directly at 7×7
        x_up1 = self.up1(x_bot)     # (B, 512, 14, 14)
        
        time_emb1 = self.time_proj1(t_emb).unsqueeze(-1).unsqueeze(-1)
        x_up1 = x_up1 + time_emb1 
        
        x_cat1 = torch.cat([x_up1, x3], dim=1)  # 512 + 256 = 768
        x_dec1 = self.dec1(x_cat1)  # Output: 128 channels

        # 6. Decoder Step 2
        x_up2 = self.up2(x_dec1)    # (B, 128, 28, 28)
        
        time_emb2 = self.time_proj2(t_emb).unsqueeze(-1).unsqueeze(-1)
        x_up2 = x_up2 + time_emb2
        
        x_cat2 = torch.cat([x_up2, x1], dim=1)  # 128 + 64 = 192
        x_dec2 = self.dec2(x_cat2)  # Output: 64 channels

        # 7. Final
        output = self.outc(x_dec2)  # Output: 1 channel
        
        return output
