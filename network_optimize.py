class DiffusionModel(nn.Module):
    def __init__(self, image_size, channels, hidden_dims=[64, 128, 256, 512]):  # CHANGE 1: New default
        super().__init__()
        
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
        self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # CHANGE 7: ADD dec0 for first decoder block
        # Input: concat of upsampled bottleneck (512) + skip x4 (512) = 1024
        self.dec0 = nn.Sequential(
            nn.Conv2d(hidden_dims[3] * 2, hidden_dims[2], kernel_size=3, padding=1),  # 1024 -> 256
            nn.BatchNorm2d(hidden_dims[2]),
            nn.SiLU()
        )
        
        # CHANGE 8: Update dec1 dimensions
        # Input: concat of upsampled dec0 (256) + skip x3 (256) = 512
        self.dec1 = nn.Sequential(
            nn.Conv2d(hidden_dims[2] * 2, hidden_dims[1], kernel_size=3, padding=1),  # 512 -> 128
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
        self.time_proj0 = nn.Linear(time_emb_dim, hidden_dims[3])  # 64 -> 512
        
        # CHANGE 11: Update time_proj1
        self.time_proj1 = nn.Linear(time_emb_dim, hidden_dims[2])  # 64 -> 256
        
        # CHANGE 12: Update time_proj2
        self.time_proj2 = nn.Linear(time_emb_dim, hidden_dims[1])  # 64 -> 128

        # 7. Final Output - channels stay the same
        self.outc = nn.Conv2d(hidden_dims[0], channels, kernel_size=1)
    
    def forward(self, x, t):
        # 1. Embed Time
        t = t.float().unsqueeze(-1) 
        t_emb = self.time_mlp(t)

        # 2. Initial Conv
        x1 = self.inc(x)            # (B, 64, 28, 28)

        # 3. Encoder - ADD x4
        x2 = self.down1(x1)         # (B, 128, 28, 28)
        x2_p = self.pool(x2)        # (B, 128, 14, 14)
        
        x3 = self.down2(x2_p)       # (B, 256, 14, 14)
        x3_p = self.pool(x3)        # (B, 256, 7, 7)
        
        # CHANGE 13: ADD down3 encoding
        x4 = self.down3(x3_p)       # (B, 512, 7, 7)
        x4_p = self.pool(x4)        # (B, 512, 3, 3)

        # 4. Bottleneck - CHANGE 14: operates on x4_p
        x_bot = self.bot1(x4_p)
        x_bot = self.bot2(x_bot)    # (B, 512, 3, 3)

        # 5. Decoder - ADD first decoding step
        # CHANGE 15: ADD dec0 step
        x_up0 = self.up0(x_bot)     # (B, 512, 7, 7) - back to 7x7
        
        time_emb0 = self.time_proj0(t_emb).unsqueeze(-1).unsqueeze(-1)
        x_up0 = x_up0 + time_emb0 
        
        x_cat0 = torch.cat([x_up0, x4], dim=1)  # Concatenate with x4 skip: 512 + 512 = 1024
        x_dec0 = self.dec0(x_cat0)  # Output: 256 channels
        
        # CHANGE 16: Update dec1 to use x_dec0 instead of x_bot
        x_up1 = self.up1(x_dec0)    # (B, 256, 14, 14)
        
        time_emb1 = self.time_proj1(t_emb).unsqueeze(-1).unsqueeze(-1)
        x_up1 = x_up1 + time_emb1 
        
        x_cat1 = torch.cat([x_up1, x3], dim=1)  # 256 + 256 = 512
        x_dec1 = self.dec1(x_cat1)  # Output: 128 channels

        # 6. Decoder Step 2 (unchanged logic)
        x_up2 = self.up2(x_dec1)    # (B, 128, 28, 28)
        
        time_emb2 = self.time_proj2(t_emb).unsqueeze(-1).unsqueeze(-1)
        x_up2 = x_up2 + time_emb2
        
        x_cat2 = torch.cat([x_up2, x1], dim=1)  # 128 + 64 = 192
        x_dec2 = self.dec2(x_cat2)  # Output: 64 channels

        # 7. Final
        output = self.outc(x_dec2)  # Output: 1 channel
        
        return output
    
# CHANGE 17: Update hidden_dims in DiffusionProcess initialization
process = DiffusionProcess(
    image_size=28, 
    channels=1, 
    hidden_dims=[64, 128, 256, 512],  # CHANGE THIS LINE
    noise_steps=1000, 
    beta_start=1e-4, 
    beta_end=0.02, 
    device=device
)