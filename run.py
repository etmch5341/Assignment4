import torch

from diffusion import DiffusionModel

def check_diffusion_model(model_class):
    """Verify that the DiffusionModel class is correctly implemented."""
    try:
        channels = 1
        image_size = 28
        noise_steps = 1000
        model = model_class(image_size=image_size, channels=channels)
        
        # Test forward pass with random inputs
        batch_size = 4
        x = torch.randn(batch_size, channels, image_size, image_size)
        t = torch.randint(0, noise_steps, (batch_size,))
        
        output = model(x, t)
        
        # Check output shape
        expected_shape = (batch_size, channels, image_size, image_size)
        assert output.shape == expected_shape, f"Expected output shape {expected_shape}, got {output.shape}"
        
        print("DiffusionModel implementation is correct!")
        return True
    except Exception as e:
        print(f"DiffusionModel check failed: {str(e)}")
        return False
    
check_diffusion_model(DiffusionModel)

from diffusion import DiffusionProcess

def check_diffusion_process(diffusion_class):
    """Verify that the DiffusionProcess class is correctly implemented."""
    try:
        channels = 1
        image_size = 28
        noise_steps = 1000
        diffusion = diffusion_class(image_size=image_size, channels=channels, noise_steps=noise_steps)
        
        # Test add_noise function
        batch_size = 4
        x = torch.randn(batch_size, channels, image_size, image_size)
        t = torch.randint(0, noise_steps, (batch_size,))
        
        noisy_x, noise = diffusion.add_noise(x, t)
        assert noisy_x.shape == x.shape, f"Expected noisy_x shape {x.shape}, got {noisy_x.shape}"
        assert noise.shape == x.shape, f"Expected noise shape {x.shape}, got {noise.shape}"
        
        # Test train_step function
        loss = diffusion.train_step(x)
        assert isinstance(loss, float), f"Expected loss to be a float, got {type(loss)}"
        
        print("DiffusionProcess implementation is correct!")
        return True
    except Exception as e:
        print(f"DiffusionProcess check failed: {str(e)}")
        return False
    
check_diffusion_process(DiffusionProcess)