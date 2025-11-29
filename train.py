import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from diffusion import DiffusionProcess
import matplotlib.pyplot as plt
from IPython import display
import os
import math
from tqdm import tqdm

# --- Reused Visualization Code from Assignment 2 ---
def visualize_progress(metrics, display_handle=None, display_id='visualization'):
    """
    Real-time visualization of training metrics and generated images.
    Now saves the plot to disk as 'latest_progress.png'.
    """
    max_cols = 4
    n_plots = len(metrics)
    n_cols = min(n_plots, max_cols)
    n_rows = math.ceil(n_plots / n_cols)
    figsize = (15, 3 * n_rows)
    
    # Create figure
    fig = plt.figure(num=200, figsize=figsize)
    plt.clf()
    
    # Plot Metrics (Loss)
    for idx, name in enumerate(list(metrics.keys())):
        if name != 'images':
            plt.subplot(n_rows, n_cols, idx + 1)
            if len(metrics[name]) > 0:
                plt.plot(metrics[name], label=name)
            plt.title(name)
            plt.xlabel('Iteration')
            plt.ylabel('Value')
            plt.grid(True)

    # Plot Images
    images = metrics.get('images', None)
    if images is not None:
        ax = plt.subplot(n_rows, n_cols, n_plots)
        ax.set_title('Generated Samples')
        if isinstance(images, torch.Tensor):
            images = images.cpu().detach().numpy()
        plt.imshow(images, cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    
    # --- NEW: Save the plot to disk ---
    # This overwrites the file every time so you always have the latest plot
    plt.savefig("./results/training_summary.png") 
    # ----------------------------------

    if display_handle is None:
        display_handle = display.display(display.HTML(''), display_id=display_id)
    display_handle.update(fig)
    plt.close()
    return display_handle

# --- Training Loop ---
def train(epochs=10, batch_size=64, device='cuda'):
    # 1. Setup
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    print(f"Training on {device}...")
    
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./models", exist_ok=True)

    # 2. Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) 
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 3. Model
    process = DiffusionProcess(
        image_size=28, 
        channels=1, 
        hidden_dims=[32, 64, 128], 
        noise_steps=1000, 
        device=device
    )
    
    # 4. Visualization Setup
    visual_metric = {
        'Loss': [],
        'images': None
    }
    display_handle = None
    
    # 5. Loop
    for epoch in range(epochs):
        process.model.train()
        pbar = tqdm(dataloader)
        
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            
            # Train Step
            loss = process.train_step(images)
            
            # Update Metrics
            visual_metric['Loss'].append(loss)
            pbar.set_description(f"Epoch {epoch+1} | Loss: {loss:.4f}")
            
            # Update Plot (Loss only) every 100 iterations
            if i % 100 == 0:
                display_handle = visualize_progress(visual_metric, display_handle, display_id='diffusion_train')
        
        # --- End of Epoch: Sample Images ---
        print(f"Sampling epoch {epoch+1}...")
        sampled_images = process.sample(num_samples=16)
        
        # Manual Reshape to 4x4 Grid (Matches Assignment 2 style)
        # 16 images -> 4x4 grid of 28x28
        grid = sampled_images.reshape(4, 4, 1, 28, 28).permute(2, 0, 3, 1, 4).reshape(4*28, 4*28)
        
        # Denormalize: [-1, 1] -> [0, 1]
        grid = (grid + 1) / 2
        grid = grid.clamp(0, 1)
        
        # Update Visual Metric with new images
        visual_metric['images'] = grid.cpu().numpy()
        
        # Final update for the epoch
        display_handle = visualize_progress(visual_metric, display_handle, display_id='diffusion_train')
        
        # Save to file as well
        save_image(sampled_images, f"./results/sample_epoch_{epoch+1}.png", nrow=4, normalize=True, value_range=(-1, 1))

    # Save Model
    torch.save(process.model.state_dict(), "./models/ddpm_mnist.pth")
    print("Training Complete.")
    
    return visual_metric['Loss']