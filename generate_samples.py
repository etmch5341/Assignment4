import torch
from torchvision.utils import save_image
from diffusion import DiffusionProcess
import os

def generate_final_samples():
    # Setup
    os.makedirs("./cosine_results", exist_ok=True)
    
    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Initialize process (must match training config)
    process = DiffusionProcess(
        image_size=28, 
        channels=1, 
        # hidden_dims=[32, 64, 128], 
        hidden_dims=[64, 128, 256, 512],
        noise_steps=1000, 
        beta_start=1e-4, 
        beta_end=0.02, 
        device=device
    )
    
    # Load trained model
    print("Loading trained model...")
    process.model.load_state_dict(torch.load("./model/ddpm_mnist_4layer_linear.pth", map_location=device))
    process.model.eval()
    print("Model loaded successfully!")
    
    # Generate 8x8 grid
    print("Generating 64 samples...")
    with torch.no_grad():
        final_samples = process.sample(num_samples=64)
        final_samples = (final_samples + 1) / 2  # Denormalize
        save_image(final_samples, "./final_comparison_8x8.png", nrow=8)
    
    print("Final 8x8 grid saved to ./final_comparison_8x8.png")
if __name__ == "__main__":
    generate_final_samples()