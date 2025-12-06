
import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline
from torchvision.utils import save_image

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load SD2.1
    model_id = "Manojb/stable-diffusion-2-base"
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    vae = pipe.vae
    vae_scale_factor = vae.config.scaling_factor
    print(f"VAE Scale Factor: {vae_scale_factor}")

    # LINEAR_RGB_ESTIMATOR
    LINEAR_RGB_ESTIMATOR = torch.tensor([
        [ 0.298,  0.207,  0.208],
        [ 0.187,  0.286,  0.173],
        [-0.158,  0.189,  0.264],
        [-0.184, -0.271, -0.473],
    ]).t().to(device)

    # Method 1: Matrix Init (Current)
    print("Testing Matrix Init...")
    A = LINEAR_RGB_ESTIMATOR # [3, 4]
    regularizer = 1e-2
    term = A.T @ A + regularizer * torch.eye(4, device=device)
    projector = torch.linalg.pinv(term) @ A.T # [4, 3]
    
    target_rgb = torch.tensor([0.5, 0.5, 0.5], device=device)
    target_latent_mean = projector @ target_rgb # [4]
    print(f"Matrix Init Mean: {target_latent_mean}")
    
    # Simulate texture creation
    texture_map = target_latent_mean.view(1, 4, 1, 1).repeat(1, 1, 64, 64)
    # Add noise
    texture_map = texture_map * 0.3 + torch.randn_like(texture_map) * 0.4
    
    # Decode
    with torch.no_grad():
        # Unscale
        latents = 1 / vae_scale_factor * texture_map
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
    save_image(image, "debug_init_matrix.png")
    print(f"Matrix Init Image Mean: {image.mean()}")

    # Method 2: VAE Init
    print("Testing VAE Init...")
    gray_image = torch.ones(1, 3, 64, 64, device=device) * 0.5
    gray_image = 2.0 * gray_image - 1.0
    with torch.no_grad():
        latent_dist = vae.encode(gray_image).latent_dist
        target_latent = latent_dist.sample() * vae_scale_factor
        target_latent_mean_vae = target_latent.mean(dim=[2, 3])
    print(f"VAE Init Mean: {target_latent_mean_vae}")
    
    # Simulate texture creation
    texture_map_vae = target_latent_mean_vae.view(1, 4, 1, 1).repeat(1, 1, 64, 64)
    # Add noise
    texture_map_vae = texture_map_vae * 0.3 + torch.randn_like(texture_map_vae) * 0.4
    
    # Decode
    with torch.no_grad():
        # Unscale
        latents = 1 / vae_scale_factor * texture_map_vae
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
    save_image(image, "debug_init_vae.png")
    print(f"VAE Init Image Mean: {image.mean()}")

if __name__ == "__main__":
    main()
