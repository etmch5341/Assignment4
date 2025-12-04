import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Import your existing diffusion classes
from diffusion import DiffusionModel, DiffusionProcess

def train_cifar(epochs=100, batch_size=64, device='cuda'):
    """
    Train DDPM on CIFAR-10 using the same architecture that worked for MNIST.
    
    Key changes from MNIST:
    - Image size: 28x28 -> 32x32
    - Channels: 1 (grayscale) -> 3 (RGB)
    - Normalization: 3 channels instead of 1
    """
    print(f"Training CIFAR-10 Diffusion on {device}...")
    
    os.makedirs("./results_cifar", exist_ok=True)
    os.makedirs("./models", exist_ok=True)

    # CIFAR-10 Transform (32x32 RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 3 channels
    ])
    
    # Load CIFAR-10
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)

    # Initialize Diffusion Process with CIFAR-10 parameters
    # Using your proven configuration: 4-level architecture, linear schedule, lr=1e-4
    process = DiffusionProcess(
        image_size=32,              # CIFAR is 32x32 (vs 28x28 for MNIST)
        channels=3,                 # RGB (vs 1 for MNIST)
        hidden_dims=[64, 128, 256, 512],  # Your best architecture
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device=device
    )
    
    # Print model info
    total_params = sum(p.numel() for p in process.model.parameters())
    print(f"Model Parameters: {total_params:,}")
    print(f"Architecture: 4-level U-Net [64, 128, 256, 512]")
    print(f"Schedule: Linear")
    print(f"Learning Rate: 1e-4")
    print(f"Training for {epochs} epochs...\n")
    
    loss_history = []
    
    for epoch in range(epochs):
        process.model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0
        
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            
            # Perform training step
            loss = process.train_step(images)
            
            epoch_loss += loss
            pbar.set_postfix({'loss': f'{loss:.4f}'})
            
        # Log average loss for this epoch
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
        
        # Generate and save samples periodically
        if (epoch + 1) % 5 == 0 or epoch == 0:
            process.model.eval()
            with torch.no_grad():
                sampled_images = process.sample(num_samples=16)
                # Denormalize from [-1, 1] to [0, 1]
                sampled_images = (sampled_images + 1) / 2
                save_image(sampled_images, f"./results_cifar/sample_epoch_{epoch+1}.png", nrow=4)
            process.model.train()
            print(f"  Saved samples to ./results_cifar/sample_epoch_{epoch+1}.png")

    # Save final model
    torch.save(process.model.state_dict(), "./models/ddpm_cifar10.pth")
    print("\nTraining Complete. Model saved to ./models/ddpm_cifar10.pth")
    
    # Plot and save training curve
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.title('CIFAR-10 Diffusion Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("./results_cifar/training_loss.png")
    print("Loss curve saved to ./results_cifar/training_loss.png")
    
    # Generate final comparison grid
    print("\nGenerating final 8x8 sample grid...")
    process.model.eval()
    with torch.no_grad():
        final_samples = process.sample(num_samples=64)
        final_samples = (final_samples + 1) / 2
        save_image(final_samples, "./results_cifar/final_comparison_8x8.png", nrow=8)
    print("Final grid saved to ./results_cifar/final_comparison_8x8.png")
    
    return loss_history

if __name__ == "__main__":
    # Auto-detect best device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}\n")
    
    # Train on CIFAR-10
    # Note: CIFAR-10 is more complex than MNIST
    # - More colors/textures to learn
    # - Higher resolution (32x32 vs 28x28)
    # - Expect slower convergence and potentially higher final loss
    
    train_cifar(epochs=100, batch_size=64, device=device)