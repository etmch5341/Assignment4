import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import math

# --- 1. Enhanced Architecture for CIFAR-10 ---
class CIFARDiffusionModel(nn.Module):
    def __init__(self, image_size=32, channels=3, hidden_dims=[64, 128, 256]):
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        
        # Time Embedding
        time_emb_dim = 128 # Increased for CIFAR
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initial Conv
        self.inc = nn.Conv2d(channels, hidden_dims[0], kernel_size=3, padding=1)

        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.SiLU(),
            nn.Dropout(0.1) # Dropout helps prevent overfitting on color images
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dims[2]),
            nn.SiLU(),
            nn.Dropout(0.1)
        )
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bot1 = nn.Conv2d(hidden_dims[2], hidden_dims[2], kernel_size=3, padding=1)
        self.bot2 = nn.Conv2d(hidden_dims[2], hidden_dims[2], kernel_size=3, padding=1)

        # Upsampling
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.Conv2d(hidden_dims[2] * 2, hidden_dims[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.SiLU(),
            nn.Dropout(0.1)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(hidden_dims[1] + hidden_dims[0], hidden_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(0.1)
        )

        # Time Projections
        self.time_proj1 = nn.Linear(time_emb_dim, hidden_dims[2])
        self.time_proj2 = nn.Linear(time_emb_dim, hidden_dims[1])

        # Final Output
        self.outc = nn.Conv2d(hidden_dims[0], channels, kernel_size=1)

    def forward(self, x, t):
        t = t.float().unsqueeze(-1)
        t_emb = self.time_mlp(t)

        x1 = self.inc(x)            
        x2 = self.down1(x1)         
        x2_p = self.pool(x2)        
        x3 = self.down2(x2_p)       
        x3_p = self.pool(x3)        

        x_bot = self.bot1(x3_p)
        x_bot = self.bot2(x_bot)    

        x_up1 = self.up1(x_bot)     
        time_emb1 = self.time_proj1(t_emb).unsqueeze(-1).unsqueeze(-1)
        x_up1 = x_up1 + time_emb1 
        x_cat1 = torch.cat([x_up1, x3], dim=1) 
        x_dec1 = self.dec1(x_cat1)  

        x_up2 = self.up2(x_dec1)    
        time_emb2 = self.time_proj2(t_emb).unsqueeze(-1).unsqueeze(-1)
        x_up2 = x_up2 + time_emb2
        x_cat2 = torch.cat([x_up2, x1], dim=1) 
        x_dec2 = self.dec2(x_cat2)  

        return self.outc(x_dec2)

# --- 2. Diffusion Process (Standard) ---
class CIFARDiffusionProcess:
    def __init__(self, model, device='cuda'):
        self.noise_steps = 1000
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.device = device
        self.model = model.to(device)
        
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.noise_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_hats = torch.sqrt(self.alpha_hats)
        self.sqrt_one_minus_alpha_hats = torch.sqrt(1.0 - self.alpha_hats)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
        self.criterion = nn.MSELoss()

    def add_noise(self, x, t):
        sqrt_alpha_hat = self.sqrt_alpha_hats[t][:, None, None, None]
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hats[t][:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def train_step(self, x):
        self.optimizer.zero_grad()
        t = torch.randint(0, self.noise_steps, (x.shape[0],)).to(self.device)
        x_noisy, noise = self.add_noise(x, t)
        noise_pred = self.model(x_noisy, t)
        loss = self.criterion(noise_pred, noise)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sample(self, num_samples=16):
        self.model.eval()
        with torch.no_grad():
            x = torch.randn(num_samples, 3, 32, 32).to(self.device) # 3 Channels for RGB
            for i in reversed(range(self.noise_steps)):
                t = (torch.ones(num_samples) * i).long().to(self.device)
                predicted_noise = self.model(x, t)
                alpha = self.alphas[t][:, None, None, None]
                alpha_hat = self.alpha_hats[t][:, None, None, None]
                beta = self.betas[t][:, None, None, None]
                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        self.model.train()
        return x.clamp(-1, 1)

# --- 3. Training Loop ---
def train_cifar(epochs=20, batch_size=64, device='cuda'):
    print(f"Training CIFAR-10 Diffusion on {device}...")
    
    os.makedirs("./results_cifar", exist_ok=True)
    os.makedirs("./models", exist_ok=True)

    # Transform for CIFAR (32x32 RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize 3 channels
    ])
    
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Initialize Enhanced Model
    model = CIFARDiffusionModel(image_size=32, channels=3, hidden_dims=[64, 128, 256])
    process = CIFARDiffusionProcess(model, device)
    
    loss_history = []
    
    for epoch in range(epochs):
        pbar = tqdm(dataloader)
        epoch_loss = 0
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            loss = process.train_step(images)
            epoch_loss += loss
            pbar.set_description(f"Epoch {epoch+1} | Loss: {loss:.4f}")
            
        loss_history.append(epoch_loss / len(dataloader))
        
        # Save Samples
        if (epoch + 1) % 5 == 0 or epoch == 0:
            samples = process.sample(16)
            samples = (samples + 1) / 2 # Denormalize
            save_image(samples, f"./results_cifar/cifar_epoch_{epoch+1}.png", nrow=4)
            
    # Save Model & Curve
    torch.save(process.model.state_dict(), "./models/ddpm_cifar.pth")
    plt.figure()
    plt.plot(loss_history)
    plt.title("CIFAR-10 Diffusion Training Loss")
    plt.savefig("./results_cifar/loss_curve.png")
    print("CIFAR Training Complete!")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Note: CIFAR takes longer than MNIST. 20 epochs is a minimum for recognizable colors.
    train_cifar(epochs=100, batch_size=64, device=device)