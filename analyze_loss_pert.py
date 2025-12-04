import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from diffusion import DiffusionModel, DiffusionProcess

def analyze_loss_per_timestep(model_path, device='cpu', num_bins=10):
    """
    Measure loss for different timestep ranges using a saved model.
    
    Args:
        model_path: Path to saved model .pth file
        device: Device to run on ('cpu', 'cuda', or 'mps')
        num_bins: Number of timestep bins to analyze
    """
    # 1. Load the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # 2. Initialize the diffusion process (must match training config)
    diffusion = DiffusionProcess(
        image_size=28,
        channels=1,
        hidden_dims=[64, 128, 256, 512],  # Use 4-layer config
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device=device
    )
    
    # 3. Load the trained model weights
    print(f"Loading model from {model_path}...")
    diffusion.model.load_state_dict(torch.load(model_path, map_location=device))
    diffusion.model.eval()
    print("Model loaded successfully!")
    
    # 4. Create bins for timesteps
    timestep_bins = torch.linspace(0, 1000, num_bins + 1).long()
    bin_losses = [[] for _ in range(num_bins)]
    
    print(f"Analyzing loss across {num_bins} timestep bins...")
    
    # 5. Evaluate loss for each timestep range
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(test_loader):
            images = images.to(device)
            
            # Sample timesteps uniformly across full range
            t = torch.randint(0, 1000, (images.shape[0],)).to(device)
            
            # Add noise according to timestep
            noisy_x, noise = diffusion.add_noise(images, t)
            
            # Predict noise
            noise_pred = diffusion.model(noisy_x, t)
            
            # Calculate loss for each sample (per-sample MSE)
            losses = F.mse_loss(noise_pred, noise, reduction='none')
            losses = losses.view(losses.shape[0], -1).mean(dim=1)  # Average per sample
            
            # Bin the losses by timestep
            for i in range(len(t)):
                timestep = t[i].item()
                bin_idx = min((timestep // (1000 // num_bins)), num_bins - 1)
                bin_losses[bin_idx].append(losses[i].item())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # 6. Calculate average loss per bin
    avg_bin_losses = []
    bin_labels = []
    
    for i in range(num_bins):
        if bin_losses[i]:
            avg_loss = sum(bin_losses[i]) / len(bin_losses[i])
            avg_bin_losses.append(avg_loss)
            bin_labels.append(f"{timestep_bins[i]}-{timestep_bins[i+1]}")
            print(f"Timesteps {timestep_bins[i]:4d}-{timestep_bins[i+1]:4d}: Loss = {avg_loss:.6f} ({len(bin_losses[i])} samples)")
        else:
            avg_bin_losses.append(0)
            bin_labels.append(f"{timestep_bins[i]}-{timestep_bins[i+1]}")
    
    # 7. Plot the results
    plt.figure(figsize=(12, 6))
    
    bin_centers = [(timestep_bins[i] + timestep_bins[i+1]) / 2 for i in range(num_bins)]
    
    plt.plot(bin_centers, avg_bin_losses, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Timestep (t)', fontsize=12)
    plt.ylabel('Average MSE Loss', fontsize=12)
    plt.title('Loss vs Timestep', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./loss_per_timestep.png', dpi=300)
    print(f"\n Plot saved to ./loss_per_timestep.png")
    
    return bin_centers, avg_bin_losses

if __name__ == "__main__":
    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}\n")
    
    # Analyze best model (linear 4-layer)
    MODEL_PATH = "./model/ddpm_mnist_4layer_cosine.pth"
    
    timesteps, losses = analyze_loss_per_timestep(
        model_path=MODEL_PATH,
        device=device,
        num_bins=20  # More bins = finer granularity
    )
    
    print("\n Analysis complete!")