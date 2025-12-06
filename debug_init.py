
import torch
import torch.nn.functional as F
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

    # 1. Check Target Latent
    target_rgb = torch.tensor([0.5, 0.5, 0.5], device=device)
    target_latent = target_rgb @ LINEAR_RGB_ESTIMATOR
    print(f"Target Latent: {target_latent}")

    # 2. Decode Target Latent (simulate visualize_latent_texture)
    # Reshape to [1, 4, 1, 1]
    latent_tensor = target_latent.view(1, 4, 1, 1)
    
    # Unscale
    latent_unscaled = latent_tensor / vae_scale_factor
    print(f"Latent Unscaled: {latent_unscaled}")

    # Decode
    with torch.no_grad():
        image = vae.decode(latent_unscaled).sample
        image = (image / 2 + 0.5).clamp(0, 1)
    
    print(f"Decoded Image Mean: {image.mean()}")
    print(f"Decoded Image Shape: {image.shape}")
    save_image(image, "debug_init_decode.png")

    # 3. Check VAE Encode -> Decode consistency
    # Create gray image
    img_gray = torch.ones(1, 3, 64, 64, device=device) * 0.5
    img_gray_norm = 2.0 * img_gray - 1.0
    
    with torch.no_grad():
        encoded_latents = vae.encode(img_gray_norm).latent_dist.sample() * vae_scale_factor
    
    print(f"Encoded Gray Latent Mean: {encoded_latents.mean()}")
    print(f"Encoded Gray Latent Std: {encoded_latents.std()}")
    
    # Compare with Linear Estimator
    print(f"Difference (Estimator - Encoded): {(target_latent.mean() - encoded_latents.mean()).item()}")

if __name__ == "__main__":
    main()
