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
        # Initialize with 4-level architecture
        super().__init__(
            image_size=image_size, 
            channels=channels, 
            hidden_dims=[64, 128, 256, 512],  # 4-level architecture
            noise_steps=1000, 
            beta_start=1e-4, 
            beta_end=0.02, 
            device=device
        )
        
        # Load the pre-trained model
        print(f"Loading model from {model_path}...")
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"  Successfully loaded 4-level model [64, 128, 256, 512]")
        except RuntimeError as e:
            print(f"Error loading model: {e}")
            print("Check model architecture matches saved checkpoint")
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
        # def get_quadratic_schedule(num_steps, max_t):
        #     """Allocate more steps to high-noise regions"""
        #     # Quadratic spacing: more steps at beginning (high noise)
        #     t_values = torch.linspace(0, 1, num_steps) ** 2
        #     timesteps = (t_values * max_t).long()
        #     return timesteps
        
        self.model.eval()
        x = torch.randn(num_samples, self.channels, self.image_size, self.image_size).to(self.device)

        # make sampling schedule
        steps = torch.linspace(0, self.noise_steps-1, ddim_steps).long().to(self.device)
        # steps = get_quadratic_schedule(ddim_steps, self.noise_steps - 1).to(self.device)

        with torch.no_grad():
            for i in reversed(range(1, ddim_steps)):
                t = steps[i]
                t_prev = steps[i-1]

                t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)

                eps = self.model(x, t_batch)

                alpha_t = self.alpha_hats[t]
                alpha_prev = self.alpha_hats[t_prev]

                # predict x0
                x0 = (x - torch.sqrt(1-alpha_t)*eps) / torch.sqrt(alpha_t)

                # DDIM update (eta=0 deterministic)
                x = torch.sqrt(alpha_prev)*x0 + torch.sqrt(1-alpha_prev)*eps

        return x.clamp(-1,1)
        
def test_model_predictions(sampler):
    """Test if model gives reasonable predictions"""
    with torch.no_grad():
        # Create test noisy image at t=500 (middle timestep)
        x_test = torch.randn(1, 1, 28, 28).to(sampler.device) * 0.5  # Moderate noise
        t_test = torch.tensor([500]).to(sampler.device)
        
        pred_noise = sampler.model(x_test, t_test)
        
        print(f"\nModel Test at t=500:")
        print(f"  Input range: [{x_test.min():.2f}, {x_test.max():.2f}]")
        print(f"  Predicted noise range: [{pred_noise.min():.2f}, {pred_noise.max():.2f}]")
        print(f"  Predicted noise mean: {pred_noise.mean():.4f}")
        print(f"  Predicted noise std: {pred_noise.std():.4f}")
        
        # Should be roughly Gaussian: mean≈0, std≈1
        if abs(pred_noise.mean()) > 0.5 or pred_noise.std() < 0.5:
            print("    WARNING: Model predictions look unusual!")
        else:
            print("    Model predictions look reasonable")

def compare_samplers():
    """
    Compare DDPM (1000 steps) vs DDIM (50 and 10 steps) for speed/quality tradeoff.
    """
    # Setup
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    os.makedirs("./results_ddim", exist_ok=True)
    
    # Initialize DDIM sampler with trained model
    MODEL_PATH = "./model/ddpm_mnist_4layer_linear.pth"
    
    try:
        sampler = DDIMSampler(MODEL_PATH, image_size=28, channels=1, device=device)
        test_model_predictions(sampler)
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
    
    print("\n" + "="*60)
    print("COMPARING SAMPLING METHODS")
    print("="*60)
    
    # 1. Standard DDPM Sampling (1000 steps) - Baseline
    print("\n[1/4] Running DDPM (1000 steps)...")
    start = time.time()
    ddpm_samples = sampler.sample(num_samples=16)
    ddpm_time = time.time() - start
    save_image((ddpm_samples + 1) / 2, "./results_ddim/ddpm_1000.png", nrow=4)
    print(f"    Complete in {ddpm_time:.2f}s")
    
    # 2. DDIM Sampling (100 steps)
    print("\n[2/4] Running DDIM (100 steps, eta=0.0)...")
    start = time.time()
    ddim_100 = sampler.sample_ddim(num_samples=16, ddim_steps=100, eta=0.0)
    ddim_100_time = time.time() - start
    save_image((ddim_100 + 1) / 2, "./results_ddim/ddim_100.png", nrow=4)
    print(f"    Complete in {ddim_100_time:.2f}s")
    print(f"     Speedup: {ddpm_time/ddim_100_time:.1f}x faster than DDPM")

    # 3. DDIM Sampling (50 steps)
    print("\n[3/4] Running DDIM (50 steps, eta=0.0)...")
    start = time.time()
    ddim_50 = sampler.sample_ddim(num_samples=16, ddim_steps=50, eta=0.0)
    ddim_50_time = time.time() - start
    save_image((ddim_50 + 1) / 2, "./results_ddim/ddim_50.png", nrow=4)
    print(f"    Complete in {ddim_50_time:.2f}s")
    print(f"     Speedup: {ddpm_time/ddim_50_time:.1f}x faster than DDPM")
    
    # 4. DDIM Sampling (10 steps)
    print("\n[4/4] Running DDIM (10 steps, eta=0.0)...")
    start = time.time()
    ddim_10 = sampler.sample_ddim(num_samples=16, ddim_steps=10, eta=0.0)
    ddim_10_time = time.time() - start
    save_image((ddim_10 + 1) / 2, "./results_ddim/ddim_10.png", nrow=4)
    print(f"    Complete in {ddim_10_time:.2f}s")
    print(f"     Speedup: {ddpm_time/ddim_10_time:.1f}x faster than DDPM")
    
    # 5. Create Comparison Visualization (w/ 4 columns)
    print("\n" + "="*60)
    print("Creating comparison grid...")
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    def show_img(tensor, ax, title, time_taken):
        grid = make_grid((tensor + 1) / 2, nrow=4)
        img = grid.permute(1, 2, 0).cpu().numpy()
        ax.imshow(img, cmap='gray' if tensor.shape[1] == 1 else None)
        ax.set_title(f"{title}\n{time_taken:.2f}s total", fontsize=12)
        ax.axis('off')
    
    show_img(ddpm_samples, axes[0], "DDPM (1000 Steps)\nBaseline Quality", ddpm_time)
    show_img(ddim_100, axes[1], f"DDIM (100 Steps)\n{ddpm_time/ddim_100_time:.1f}x Faster", ddim_100_time)
    show_img(ddim_50, axes[2], f"DDIM (50 Steps)\n{ddpm_time/ddim_50_time:.1f}x Faster", ddim_50_time)
    show_img(ddim_10, axes[3], f"DDIM (10 Steps)\n{ddpm_time/ddim_10_time:.1f}x Faster", ddim_10_time)
    
    plt.suptitle("DDPM vs DDIM: Speed/Quality Tradeoff", fontsize=20, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig("./results_ddim/comparison_grid.png", dpi=300, bbox_inches='tight')
    
    print(f"  Comparison grid saved to ./results_ddim/comparison_grid.png")
    
    # 5. Summary Statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"DDPM (1000 steps):  {ddpm_time:.2f}s  [Baseline]")
    print(f"DDIM (100 steps):   {ddim_100_time:.2f}s  [{ddpm_time/ddim_100_time:.1f}x speedup]")
    print(f"DDIM (50 steps):    {ddim_50_time:.2f}s  [{ddpm_time/ddim_50_time:.1f}x speedup]")
    print(f"DDIM (10 steps):    {ddim_10_time:.2f}s  [{ddpm_time/ddim_10_time:.1f}x speedup]")
    print("="*60)
    print("\n  All results saved to ./results_ddim/")

if __name__ == "__main__":
    compare_samplers()