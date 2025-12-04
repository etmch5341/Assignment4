import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from diffusion import DiffusionProcess, DiffusionModel
import math
import os
import time

class DDIMSampler(DiffusionProcess):
    """
    Extends DiffusionProcess with DDIM sampling capabilities.
    Paper: https://arxiv.org/abs/2010.02502
    
    DDIM allows deterministic sampling with fewer steps than DDPM,
    achieving 10-100x speedup with minimal quality loss.
    """
    def __init__(self, model_path, image_size=28, channels=1, device='cuda'):
        # Initialize with your 4-level architecture
        super().__init__(
            image_size=image_size, 
            channels=channels, 
            hidden_dims=[64, 128, 256, 512],  # Your 4-level architecture
            noise_steps=1000, 
            beta_start=1e-4, 
            beta_end=0.02, 
            device=device
        )
        
        # Load the pre-trained model
        print(f"Loading model from {model_path}...")
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✓ Successfully loaded 4-level model [64, 128, 256, 512]")
        except RuntimeError as e:
            print(f"✗ Error loading model: {e}")
            print("Make sure the model architecture matches your saved checkpoint!")
            raise
            
        self.model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {total_params:,}")

    def sample_ddim(self, num_samples=16, ddim_steps=50, eta=0.0):
        """
        DDIM Sampling - Deterministic generation with fewer steps.
        
        Args:
            num_samples: Number of images to generate
            ddim_steps: Number of denoising steps (e.g., 50 instead of 1000)
            eta: Stochasticity parameter
                 - 0.0: Fully deterministic (DDIM)
                 - 1.0: Equivalent to DDPM (stochastic)
        
        Returns:
            Generated images tensor of shape [num_samples, channels, height, width]
        """
        self.model.eval()
        
        with torch.no_grad():
            # 1. Create subsequence of timesteps
            # Example: If noise_steps=1000 and ddim_steps=50
            # We take every 20th step: [0, 20, 40, ..., 980, 999]
            c = self.noise_steps // ddim_steps
            time_seq = list(range(0, self.noise_steps, c))
            
            # Ensure we include the final timestep
            if time_seq[-1] != self.noise_steps - 1:
                time_seq.append(self.noise_steps - 1)
            
            print(f"DDIM: Using {len(time_seq)} timesteps: {time_seq[:5]}...{time_seq[-3:]}")
            
            # 2. Start from random Gaussian noise
            x = torch.randn(num_samples, self.channels, self.image_size, self.image_size).to(self.device)
            
            # 3. Iteratively denoise using DDIM update rule
            for i in reversed(range(1, len(time_seq))):
                t_now = time_seq[i]      # Current timestep
                t_prev = time_seq[i-1]   # Previous (less noisy) timestep
                
                # Create batch of current timesteps
                t_tensor = (torch.ones(num_samples) * t_now).long().to(self.device)
                
                # Predict noise at current timestep
                noise_pred = self.model(x, t_tensor)
                
                # Get cumulative alpha values
                alpha_hat_t = self.alpha_hats[t_now]
                alpha_hat_t_prev = self.alpha_hats[t_prev] if t_prev >= 0 else torch.tensor(1.0).to(self.device)
                
                # Calculate sigma (controls stochasticity)
                # eta=0.0 (DDIM): deterministic
                # eta=1.0 (DDPM): fully stochastic
                if t_prev > 0:
                    sigma_t = eta * torch.sqrt(
                        (1 - alpha_hat_t_prev) / (1 - alpha_hat_t) * (1 - alpha_hat_t / alpha_hat_t_prev)
                    )
                else:
                    sigma_t = 0.0
                
                # DDIM Update Equations (from paper):
                
                # Step 1: Predict x_0 (clean image) from current noisy x_t
                pred_x0 = (x - torch.sqrt(1 - alpha_hat_t) * noise_pred) / torch.sqrt(alpha_hat_t)
                pred_x0 = pred_x0.clamp(-1, 1)  # Clip to valid range
                
                # Step 2: Direction pointing to x_t
                dir_xt = torch.sqrt(1 - alpha_hat_t_prev - sigma_t**2) * noise_pred
                
                # Step 3: Random noise component (scaled by sigma)
                if sigma_t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                # Step 4: DDIM update (Equation 12 from paper)
                x = torch.sqrt(alpha_hat_t_prev) * pred_x0 + dir_xt + sigma_t * noise
                
            return x.clamp(-1, 1)

def compare_samplers():
    """
    Compare DDPM (1000 steps) vs DDIM (50 and 10 steps) for speed/quality tradeoff.
    """
    # Setup
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    os.makedirs("./results_ddim", exist_ok=True)
    
    # Initialize DDIM sampler with your trained model
    MODEL_PATH = "./model/ddpm_mnist_4layer_linear.pth"  # Update if different
    
    try:
        sampler = DDIMSampler(MODEL_PATH, image_size=28, channels=1, device=device)
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please train a model first or update MODEL_PATH")
        return
    
    print("\n" + "="*60)
    print("COMPARING SAMPLING METHODS")
    print("="*60)
    
    # 1. Standard DDPM Sampling (1000 steps) - BASELINE
    print("\n[1/3] Running DDPM (1000 steps)...")
    start = time.time()
    ddpm_samples = sampler.sample(num_samples=16)
    ddpm_time = time.time() - start
    save_image((ddpm_samples + 1) / 2, "./results_ddim/ddpm_1000.png", nrow=4)
    print(f"  ✓ Complete in {ddpm_time:.2f}s ({ddpm_time/16:.2f}s per image)")
    
    # 2. DDIM Sampling (50 steps) - 20x FASTER
    print("\n[2/3] Running DDIM (50 steps, eta=0.0)...")
    start = time.time()
    ddim_50 = sampler.sample_ddim(num_samples=16, ddim_steps=50, eta=0.0)
    ddim_50_time = time.time() - start
    save_image((ddim_50 + 1) / 2, "./results_ddim/ddim_50.png", nrow=4)
    print(f"  ✓ Complete in {ddim_50_time:.2f}s ({ddim_50_time/16:.2f}s per image)")
    print(f"  → Speedup: {ddpm_time/ddim_50_time:.1f}x faster than DDPM")
    
    # 3. DDIM Sampling (10 steps) - 100x FASTER (extreme speed)
    print("\n[3/3] Running DDIM (10 steps, eta=0.0)...")
    start = time.time()
    ddim_10 = sampler.sample_ddim(num_samples=16, ddim_steps=10, eta=0.0)
    ddim_10_time = time.time() - start
    save_image((ddim_10 + 1) / 2, "./results_ddim/ddim_10.png", nrow=4)
    print(f"  ✓ Complete in {ddim_10_time:.2f}s ({ddim_10_time/16:.2f}s per image)")
    print(f"  → Speedup: {ddpm_time/ddim_10_time:.1f}x faster than DDPM")
    
    # 4. Create Comparison Visualization
    print("\n" + "="*60)
    print("Creating comparison grid...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    def show_img(tensor, ax, title, time_taken):
        grid = make_grid((tensor + 1) / 2, nrow=4)
        img = grid.permute(1, 2, 0).cpu().numpy()
        ax.imshow(img, cmap='gray' if tensor.shape[1] == 1 else None)
        ax.set_title(f"{title}\n{time_taken:.2f}s total", fontsize=12)
        ax.axis('off')
    
    show_img(ddpm_samples, axes[0], "DDPM (1000 Steps)\nBaseline Quality", ddpm_time)
    show_img(ddim_50, axes[1], f"DDIM (50 Steps)\n{ddpm_time/ddim_50_time:.1f}x Faster", ddim_50_time)
    show_img(ddim_10, axes[2], f"DDIM (10 Steps)\n{ddpm_time/ddim_10_time:.1f}x Faster", ddim_10_time)
    
    plt.suptitle("DDPM vs DDIM: Speed/Quality Tradeoff", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("./results_ddim/comparison_grid.png", dpi=300, bbox_inches='tight')
    
    print(f"✓ Comparison grid saved to ./results_ddim/comparison_grid.png")
    
    # 5. Summary Statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"DDPM (1000 steps):  {ddpm_time:.2f}s  [Baseline]")
    print(f"DDIM (50 steps):    {ddim_50_time:.2f}s  [{ddpm_time/ddim_50_time:.1f}x speedup]")
    print(f"DDIM (10 steps):    {ddim_10_time:.2f}s  [{ddpm_time/ddim_10_time:.1f}x speedup]")
    print("="*60)
    print("\n✓ All results saved to ./results_ddim/")

if __name__ == "__main__":
    compare_samplers()