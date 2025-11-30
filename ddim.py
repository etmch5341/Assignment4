import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from diffusion import DiffusionProcess, DiffusionModel
import math
import os

class DDIMSampler(DiffusionProcess):
    """
    Extends DiffusionProcess with DDIM sampling capabilities.
    Paper: https://arxiv.org/abs/2010.02502
    """
    def __init__(self, model_path, image_size=28, channels=1, device='cuda'):
        # Initialize the base class
        super().__init__(image_size, channels, hidden_dims=[32, 64, 128], 
                         noise_steps=1000, beta_start=1e-4, beta_end=0.02, device=device)
        
        # Load the pre-trained model state
        # Note: Ensure architecture matches what you trained with!
        # If you used the larger model from previous steps, change hidden_dims here.
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        except RuntimeError:
             # Fallback for the larger model if you used the "improved" version
            self.model = DiffusionModel(image_size, channels, hidden_dims=[64, 128, 256]).to(device)
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            
        self.model.eval()
        print(f"Loaded model from {model_path}")

    def sample_ddim(self, num_samples=16, ddim_steps=50, eta=0.0):
        """
        DDIM Sampling.
        Args:
            ddim_steps: Number of steps to take (e.g., 50 instead of 1000)
            eta: 0.0 for deterministic DDIM, 1.0 for standard DDPM
        """
        self.model.eval()
        with torch.no_grad():
            # 1. Determine the sub-sequence of timesteps
            # e.g., if total=1000, ddim=50, we take steps [0, 20, 40, ..., 980]
            c = self.noise_steps // ddim_steps
            time_seq = list(range(0, self.noise_steps, c)) + [self.noise_steps - 1]
            time_seq = sorted(list(set(time_seq))) # Ensure unique and sorted
            
            # Start from random noise
            x = torch.randn(num_samples, self.channels, self.image_size, self.image_size).to(self.device)
            
            # 2. Reverse Loop through the subset of timesteps
            # We iterate backwards: t_i -> t_{i-1}
            print(f"Sampling with DDIM ({ddim_steps} steps)...")
            
            for i in reversed(range(1, len(time_seq))):
                t_now = time_seq[i]
                t_prev = time_seq[i-1]
                
                # Create batch of timesteps
                t_tensor = (torch.ones(num_samples) * t_now).long().to(self.device)
                
                # Predict Noise
                noise_pred = self.model(x, t_tensor)
                
                # Get alpha values for t and t-1
                alpha_hat_t = self.alpha_hats[t_now]
                alpha_hat_t_prev = self.alpha_hats[t_prev]
                
                # Calculate sigma (noise) based on eta
                # For DDIM (eta=0), sigma is 0.
                sigma_t = eta * torch.sqrt((1 - alpha_hat_t_prev) / (1 - alpha_hat_t) * (1 - alpha_hat_t / alpha_hat_t_prev))
                
                # Predict x0 (original image) from current noisy x
                # "predicted x_0" formula
                pred_x0 = (x - torch.sqrt(1 - alpha_hat_t) * noise_pred) / torch.sqrt(alpha_hat_t)
                
                # Direction pointing to x_t
                dir_xt = torch.sqrt(1 - alpha_hat_t_prev - sigma_t**2) * noise_pred
                
                # Random noise component
                noise = torch.randn_like(x)
                
                # Update x (Equation 12 in DDIM paper)
                x = torch.sqrt(alpha_hat_t_prev) * pred_x0 + dir_xt + sigma_t * noise
                
            return x.clamp(-1, 1)

def compare_samplers():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("./results_ddim", exist_ok=True)
    
    # Initialize Wrapper
    # MAKE SURE this path points to your trained MNIST model
    sampler = DDIMSampler("./models/ddpm_mnist.pth", device=device)
    
    # 1. Standard DDPM Sampling (1000 Steps)
    print("Running standard DDPM (1000 steps)...")
    ddpm_samples = sampler.sample(num_samples=16) # Uses the original .sample() method
    save_image((ddpm_samples + 1) / 2, "./results_ddim/ddpm_1000.png", nrow=4)
    
    # 2. DDIM Sampling (50 Steps)
    print("Running DDIM (50 steps)...")
    ddim_50 = sampler.sample_ddim(num_samples=16, ddim_steps=50, eta=0.0)
    save_image((ddim_50 + 1) / 2, "./results_ddim/ddim_50.png", nrow=4)
    
    # 3. DDIM Sampling (10 Steps - Extreme speed test)
    print("Running DDIM (10 steps)...")
    ddim_10 = sampler.sample_ddim(num_samples=16, ddim_steps=10, eta=0.0)
    save_image((ddim_10 + 1) / 2, "./results_ddim/ddim_10.png", nrow=4)
    
    # 4. Create Comparison Grid
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    def show_img(tensor, ax, title):
        grid = make_grid((tensor + 1) / 2, nrow=4)
        ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
        ax.set_title(title)
        ax.axis('off')
        
    show_img(ddpm_samples, axes[0], "DDPM (1000 Steps)\nSlow, High Quality")
    show_img(ddim_50, axes[1], "DDIM (50 Steps)\n20x Faster, Good Quality")
    show_img(ddim_10, axes[2], "DDIM (10 Steps)\n100x Faster, Lossy")
    
    plt.tight_layout()
    plt.savefig("./results_ddim/comparison_grid.png")
    print("Comparison saved to ./results_ddim/comparison_grid.png")

if __name__ == "__main__":
    compare_samplers()