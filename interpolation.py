import torch
import torch.nn as nn
from torchvision.utils import save_image
import os
from diffusion import DiffusionModel

def visualize_interpolation(model_path, device='cpu', num_interpolations=8):
    """
    Visualize interpolation between two random noise vectors.
    
    Args:
        model_path: Path to the saved model checkpoint (.pth file)
        device: Device to run on ('cpu', 'cuda', or 'mps')
        num_interpolations: Number of interpolation steps between the two noise vectors
    """
    # Setup
    os.makedirs("./interpolation_results", exist_ok=True)
    
    # Initialize model architecture (must match training)
    model = DiffusionModel(
        image_size=28,
        channels=1,
        # hidden_dims=[32, 64, 128]
        hidden_dims=[64, 128, 256, 512]
    ).to(device)
    
    # Load trained weights
    print(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully!")
    
    # Initialize noise schedule (must match training)
    noise_steps = 1000
    beta_start = 1e-4
    beta_end = 0.02
    
    # Choose schedule type (change this to match what you used in training)
    # Option 1: Linear schedule
    # betas = torch.linspace(beta_start, beta_end, noise_steps).to(device)
    
    # Option 2: Cosine schedule (uncomment if you used this)
    import math
    def cosine_beta_schedule(timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.99)
    betas = cosine_beta_schedule(noise_steps).to(device)
    
    # Calculate alphas
    alphas = 1.0 - betas
    alpha_hats = torch.cumprod(alphas, dim=0)
    
    print(f"Generating {num_interpolations} interpolated samples...")
    
    # 1. Generate two random noise vectors
    z1 = torch.randn(1, 1, 28, 28).to(device)
    z2 = torch.randn(1, 1, 28, 28).to(device)
    
    # 2. Create interpolation steps between z1 and z2
    alphas_interp = torch.linspace(0, 1, num_interpolations).to(device)
    interpolated_z = []
    
    for alpha in alphas_interp:
        z_interp = (1 - alpha) * z1 + alpha * z2
        interpolated_z.append(z_interp)
    
    # Stack into a batch
    batch_z = torch.cat(interpolated_z, dim=0)
    
    # 3. Denoise the batch using the reverse diffusion process
    with torch.no_grad():
        x = batch_z
        
        # Standard DDPM sampling loop
        for i in reversed(range(noise_steps)):
            t = (torch.ones(num_interpolations) * i).long().to(device)
            predicted_noise = model(x, t)
            
            # Get alpha/beta values for this timestep
            alpha = alphas[t][:, None, None, None]
            alpha_hat = alpha_hats[t][:, None, None, None]
            beta = betas[t][:, None, None, None]
            
            # Add noise for Langevin dynamics (except at final step)
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            # Denoising step
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
            
            # Print progress
            if (i + 1) % 100 == 0 or i == 0:
                print(f"Denoising step {noise_steps - i}/{noise_steps}")
    
    # 4. Denormalize and save
    x = x.clamp(-1, 1)  # Clamp to valid range
    x = (x + 1) / 2      # Convert from [-1, 1] to [0, 1]
    
    output_path = "./interpolation_results/interpolation.png"
    save_image(x, output_path, nrow=num_interpolations)
    print(f"Interpolation saved to {output_path}")
    
    return x

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "./model/ddpm_mnist_4layer_cosine.pth"  # Update this path if needed
    
    # Auto-detect best available device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Run interpolation
    visualize_interpolation(
        model_path=MODEL_PATH,
        device=device,
        num_interpolations=8
    )