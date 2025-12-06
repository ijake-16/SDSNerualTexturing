
import torch
from diffusers import StableDiffusionPipeline

model_id = "Manojb/stable-diffusion-2-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
vae_scale_factor = pipe.vae.config.scaling_factor
print(f"VAE Scale Factor: {vae_scale_factor}")

LINEAR_RGB_ESTIMATOR = torch.tensor([
    [ 0.298,  0.207,  0.208], # Channel 1
    [ 0.187,  0.286,  0.173], # Channel 2
    [-0.158,  0.189,  0.264], # Channel 3
    [-0.184, -0.271, -0.473], # Channel 4
]).t()

target_rgb = torch.tensor([0.5, 0.5, 0.5])
target_latent = target_rgb @ LINEAR_RGB_ESTIMATOR
print(f"Target Latent (from matrix): {target_latent}")
print(f"Target Latent Mean: {target_latent.mean()}")
print(f"Target Latent Std: {target_latent.std()}")

# Check what actual VAE produces
# Create a gray image
img = torch.ones(1, 3, 512, 512) * 0.5
# Normalize to [-1, 1]
img = 2.0 * img - 1.0
latents = pipe.vae.encode(img).latent_dist.sample()
print(f"Actual VAE Latent (unscaled) Mean: {latents.mean()}")
print(f"Actual VAE Latent (unscaled) Std: {latents.std()}")

scaled_latents = latents * vae_scale_factor
print(f"Actual VAE Latent (scaled) Mean: {scaled_latents.mean()}")
print(f"Actual VAE Latent (scaled) Std: {scaled_latents.std()}")
