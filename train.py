import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from diffusion import DiffusionProcess
import matplotlib.pyplot as plt
import os
from tqdm import tqdm # For progress bars

def train(epochs=10, batch_size=64, learning_rate=3e-4, device='cuda'):
    # 1. Setup Environment
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    print(f"Training on {device}...")
    
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./models", exist_ok=True)

    # 2. Prepare Data (MNIST)
    # Transform: Resize to 28x28 (standard), Convert to Tensor, Normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) 
    ])
    
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 3. Initialize Diffusion Process
    # Note: ensure hidden_dims matches what you used in DiffusionModel
    process = DiffusionProcess(
        image_size=28, 
        channels=1, 
        hidden_dims=[32, 64, 128], 
        noise_steps=1000, 
        beta_start=1e-4, 
        beta_end=0.02, 
        device=device
    )
    
    # 4. Training Loop
    loss_history = []
    
    for epoch in range(epochs):
        process.model.train()
        pbar = tqdm(dataloader)
        epoch_loss = 0
        
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            
            # Perform single training step
            loss = process.train_step(images)
            
            epoch_loss += loss
            pbar.set_description(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")
            
        # Log average loss for this epoch
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        
        # 5. Sampling (Validation)
        # Generate samples every few epochs to visualize progress
        if (epoch + 1) % 1 == 0 or epoch == epochs - 1:
            print(f"Sampling epoch {epoch+1}...")
            sampled_images = process.sample(num_samples=16)
            
            # Denormalize from [-1, 1] to [0, 1] for saving
            sampled_images = (sampled_images + 1) / 2
            
            save_image(sampled_images, f"./results/sample_epoch_{epoch+1}.png", nrow=4)

    # 6. Save Model
    torch.save(process.model.state_dict(), "./models/ddpm_mnist.pth")
    print("Training Complete. Model saved.")
    
    return loss_history

if __name__ == "__main__":
    # Check for GPU, MPS (Mac), or CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    loss_values = train(epochs=5, batch_size=128, device=device)
    
    # Save loss curve data
    with open("./results/loss_history.txt", "w") as f:
        for loss in loss_values:
            f.write(f"{loss}\n")