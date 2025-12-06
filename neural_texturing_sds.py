"""
Neural Texturing with Score Distillation Sampling (SDS)
========================================================

This script generates high-quality textures for a procedural 3D mesh using
gradients from a pre-trained Stable Diffusion XL (SDXL) model.

The optimization target is the texture map (UV map) of an ico-sphere,
which is iteratively refined using SDS loss.

Requirements:
    pip install torch diffusers pytorch3d accelerate transformers

Author: Deep Learning Engineer
"""

import math
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# Diffusers for SDXL
from diffusers import StableDiffusionXLPipeline, DDPMScheduler

# PyTorch3D for differentiable rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    PointLights,
    look_at_view_transform,
)
from pytorch3d.utils import ico_sphere


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Hyperparameters and settings for the neural texturing pipeline."""
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # SDXL settings
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    dtype = torch.float16
    
    # Texture settings
    texture_resolution = 1024  # UV texture map resolution
    render_resolution = 512    # Rendered image resolution for SDS
    
    # Mesh settings
    ico_sphere_level = 4  # Higher = more subdivisions = smoother sphere
    
    # Optimization settings
    num_iterations = 1000
    learning_rate = 0.01
    
    # SDS settings
    guidance_scale = 100.0  # CFG scale for SDS
    min_timestep = 20       # Minimum diffusion timestep
    max_timestep = 980      # Maximum diffusion timestep
    
    # Camera sampling
    camera_distance = 2.5
    min_elevation = -30.0   # degrees
    max_elevation = 60.0    # degrees
    
    # Text prompt for texture generation
    prompt = "A peeling rusty metal sphere, highly detailed, 8k, photorealistic"
    negative_prompt = "blurry, low quality, distorted, ugly"


# =============================================================================
# SDXL Pipeline Wrapper
# =============================================================================

class SDXLWrapper:
    """
    Wrapper for Stable Diffusion XL pipeline optimized for SDS.
    The model is frozen and used only for computing gradients.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        
        print("Loading SDXL pipeline...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            config.model_id,
            torch_dtype=config.dtype,
            use_safetensors=True,
            variant="fp16",
        )
        self.pipe.to(self.device)
        
        # Freeze all parameters
        self.pipe.unet.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.text_encoder_2.requires_grad_(False)
        
        # Enable memory efficient attention if available
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("xformers memory efficient attention enabled")
        except Exception:
            print("xformers not available, using default attention")
        
        # Get scheduler for noise scheduling
        self.scheduler = self.pipe.scheduler
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        
        # Pre-encode text prompts
        self._encode_prompts()
        
        # VAE scaling factor
        self.vae_scale_factor = 2 ** (len(self.pipe.vae.config.block_out_channels) - 1)
        
    def _encode_prompts(self):
        """Pre-encode text prompts to avoid repeated computation."""
        print("Encoding text prompts...")
        
        # Encode positive prompt
        (
            self.prompt_embeds,
            self.negative_prompt_embeds,
            self.pooled_prompt_embeds,
            self.negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=self.config.prompt,
            prompt_2=self.config.prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=self.config.negative_prompt,
            negative_prompt_2=self.config.negative_prompt,
        )
        
    @torch.no_grad()
    def encode_image_to_latents(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode a rendered image to VAE latent space.
        
        Args:
            image: Tensor of shape [B, 3, H, W] in range [0, 1]
            
        Returns:
            Latent tensor of shape [B, 4, H//8, W//8]
        """
        # Normalize to [-1, 1] for VAE
        image = 2.0 * image - 1.0
        image = image.to(dtype=self.dtype)
        
        # Encode
        latent_dist = self.pipe.vae.encode(image).latent_dist
        latents = latent_dist.sample() * self.pipe.vae.config.scaling_factor
        
        return latents
    
    def compute_sds_gradient(
        self, 
        latents: torch.Tensor,
        timestep: int,
    ) -> torch.Tensor:
        """
        Compute the SDS gradient for the given latents.
        
        SDS Gradient: ∇ = w(t) * (ε_pred - ε)
        
        Args:
            latents: Clean latents from rendered image [B, 4, H//8, W//8]
            timestep: Diffusion timestep
            
        Returns:
            SDS gradient tensor
        """
        batch_size = latents.shape[0]
        
        # Sample noise
        noise = torch.randn_like(latents)
        
        # Add noise to latents
        timesteps = torch.tensor([timestep], device=self.device).long()
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # Expand timesteps for batch
        timesteps = timesteps.expand(batch_size * 2)
        
        # Prepare latent model input (for CFG, we need both conditional and unconditional)
        latent_model_input = torch.cat([noisy_latents] * 2)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, timesteps[0])
        
        # Prepare added conditions for SDXL
        add_time_ids = self._get_add_time_ids(
            original_size=(self.config.render_resolution, self.config.render_resolution),
            crops_coords_top_left=(0, 0),
            target_size=(self.config.render_resolution, self.config.render_resolution),
            dtype=self.dtype,
            batch_size=batch_size,
        )
        
        added_cond_kwargs = {
            "text_embeds": torch.cat([self.negative_pooled_prompt_embeds, self.pooled_prompt_embeds]),
            "time_ids": torch.cat([add_time_ids, add_time_ids]),
        }
        
        # Predict noise
        with torch.no_grad():
            noise_pred = self.pipe.unet(
                latent_model_input,
                timesteps[0],
                encoder_hidden_states=torch.cat([self.negative_prompt_embeds, self.prompt_embeds]),
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
        
        # Classifier-free guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Compute SDS gradient: w(t) * (ε_pred - ε)
        # w(t) is typically set to 1 or based on SNR
        # TODO: Experiment with different weighting schemes (e.g., SNR-based)
        w_t = 1.0
        grad = w_t * (noise_pred - noise)
        
        return grad
    
    def _get_add_time_ids(
        self,
        original_size: Tuple[int, int],
        crops_coords_top_left: Tuple[int, int],
        target_size: Tuple[int, int],
        dtype: torch.dtype,
        batch_size: int,
    ) -> torch.Tensor:
        """Get additional time embeddings for SDXL."""
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=self.device)
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        return add_time_ids


# =============================================================================
# 3D Scene Setup with PyTorch3D
# =============================================================================

class DifferentiableRenderer:
    """
    PyTorch3D-based differentiable renderer for the textured mesh.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        
        # Setup rasterization
        self.raster_settings = RasterizationSettings(
            image_size=config.render_resolution,
            blur_radius=0.0,
            faces_per_pixel=1,
            # TODO: Adjust perspective_correct and cull_backfaces as needed
            perspective_correct=True,
            cull_backfaces=False,
        )
        
        # Setup lighting
        self.lights = PointLights(
            device=self.device,
            location=[[2.0, 2.0, 2.0]],
            ambient_color=[[0.5, 0.5, 0.5]],
            diffuse_color=[[0.7, 0.7, 0.7]],
            specular_color=[[0.2, 0.2, 0.2]],
        )
        
    def create_camera(self, elevation: float, azimuth: float) -> FoVPerspectiveCameras:
        """
        Create a camera at the specified elevation and azimuth.
        
        Args:
            elevation: Elevation angle in degrees
            azimuth: Azimuth angle in degrees
            
        Returns:
            PyTorch3D camera object
        """
        R, T = look_at_view_transform(
            dist=self.config.camera_distance,
            elev=elevation,
            azim=azimuth,
            device=self.device,
        )
        
        camera = FoVPerspectiveCameras(
            device=self.device,
            R=R,
            T=T,
            fov=60.0,
        )
        
        return camera
    
    def render(self, mesh: Meshes, camera: FoVPerspectiveCameras) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render the mesh from the given camera viewpoint.
        
        Args:
            mesh: PyTorch3D Meshes object with texture
            camera: Camera for rendering
            
        Returns:
            Rendered image tensor [B, 3, H, W] in range [0, 1]
            Alpha mask tensor [B, 1, H, W] where 1 = geometry, 0 = background
        """
        # Build rasterizer and shader separately so we can access fragments (for alpha)
        rasterizer = MeshRasterizer(
            cameras=camera,
            raster_settings=self.raster_settings,
        )
        fragments = rasterizer(mesh)

        # Alpha mask: where a face is hit (pix_to_face >= 0)
        alpha = (fragments.pix_to_face[..., 0] >= 0).float()  # [B, H, W]
        alpha = alpha.unsqueeze(1)  # [B, 1, H, W]

        shader = SoftPhongShader(
            device=self.device,
            cameras=camera,
            lights=self.lights,
        )
        images = shader(fragments, mesh)  # [B, H, W, 4] (RGBA)

        # Extract RGB and transpose to [B, 3, H, W]
        images = images[..., :3].permute(0, 3, 1, 2)

        # Clamp to valid range
        images = torch.clamp(images, 0.0, 1.0)

        return images, alpha


# =============================================================================
# Procedural Mesh Generation
# =============================================================================

def create_textured_ico_sphere(
    level: int,
    texture_map: torch.Tensor,
    device: torch.device,
) -> Meshes:
    """
    Create an ico-sphere mesh with UV texture mapping.
    
    Args:
        level: Subdivision level for ico_sphere
        texture_map: Learnable texture tensor [1, H, W, 3]
        device: Torch device
        
    Returns:
        PyTorch3D Meshes object with UV texture
    """
    # Generate ico-sphere
    mesh = ico_sphere(level=level, device=device)
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    
    # Generate UV coordinates using spherical mapping
    # Normalize vertices to unit sphere
    verts_normalized = F.normalize(verts, p=2, dim=1)
    
    # Spherical UV mapping
    # u = 0.5 + atan2(z, x) / (2*pi)
    # v = 0.5 - asin(y) / pi
    x, y, z = verts_normalized[:, 0], verts_normalized[:, 1], verts_normalized[:, 2]
    u = 0.5 + torch.atan2(z, x) / (2 * math.pi)
    v = 0.5 - torch.asin(torch.clamp(y, -1.0, 1.0)) / math.pi
    
    verts_uvs = torch.stack([u, v], dim=1)  # [V, 2]
    
    # For simplicity, use per-vertex UVs (faces_uvs same as faces)
    faces_uvs = faces
    
    # Create texture
    textures = TexturesUV(
        maps=texture_map,        # [1, H, W, 3]
        faces_uvs=[faces_uvs],   # List of [F, 3]
        verts_uvs=[verts_uvs],   # List of [V, 2]
    )
    
    # Create mesh with texture
    mesh = Meshes(
        verts=[verts],
        faces=[faces],
        textures=textures,
    )
    
    return mesh


# =============================================================================
# Main Optimization Loop
# =============================================================================

def sample_camera_position(config: Config) -> Tuple[float, float]:
    """
    Randomly sample camera position (elevation, azimuth).
    
    TODO: Refine camera sampling distribution (e.g., stratified sampling,
          bias towards certain views, or importance sampling)
    
    Returns:
        Tuple of (elevation, azimuth) in degrees
    """
    elevation = random.uniform(config.min_elevation, config.max_elevation)
    azimuth = random.uniform(0.0, 360.0)
    return elevation, azimuth


def main():
    """Main function for neural texturing with SDS."""
    
    print("=" * 60)
    print("Neural Texturing with Score Distillation Sampling")
    print("=" * 60)
    
    config = Config()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script")
    
    print(f"Using device: {config.device}")
    print(f"Texture resolution: {config.texture_resolution}")
    print(f"Render resolution: {config.render_resolution}")
    print(f"Prompt: {config.prompt}")
    
    # -------------------------------------------------------------------------
    # Step 1: Initialize SDXL Pipeline
    # -------------------------------------------------------------------------
    sdxl = SDXLWrapper(config)
    
    # -------------------------------------------------------------------------
    # Step 2: Initialize Learnable Texture Map
    # -------------------------------------------------------------------------
    print("\nInitializing learnable texture map...")
    
    # Initialize texture as grey with small random noise
    # Shape: [1, H, W, 3] for TexturesUV
    texture_map = torch.ones(
        1, 
        config.texture_resolution, 
        config.texture_resolution, 
        3,
        device=config.device,
        dtype=torch.float32,
    ) * 0.5
    
    # Add small noise for initialization diversity
    texture_map = texture_map + torch.randn_like(texture_map) * 0.1
    texture_map = torch.clamp(texture_map, 0.0, 1.0)
    
    # Make it a learnable parameter
    texture_map = nn.Parameter(texture_map, requires_grad=True)
    
    # -------------------------------------------------------------------------
    # Step 3: Setup Renderer
    # -------------------------------------------------------------------------
    print("Setting up differentiable renderer...")
    renderer = DifferentiableRenderer(config)
    
    # -------------------------------------------------------------------------
    # Step 4: Setup Optimizer
    # -------------------------------------------------------------------------
    optimizer = Adam([texture_map], lr=config.learning_rate)
    
    # -------------------------------------------------------------------------
    # Step 5: Optimization Loop
    # -------------------------------------------------------------------------
    print(f"\nStarting optimization for {config.num_iterations} iterations...")
    print("-" * 60)
    
    # Gradient flow verification flag
    gradient_verified = False
    
    for iteration in range(config.num_iterations):
        optimizer.zero_grad()
        
        # ----- Camera Sampling -----
        # TODO: Refine camera sampling distribution
        elevation, azimuth = sample_camera_position(config)
        camera = renderer.create_camera(elevation, azimuth)
        
        # ----- Create Mesh with Current Texture -----
        # Clamp texture to valid range
        texture_clamped = torch.clamp(texture_map, 0.0, 1.0)
        mesh = create_textured_ico_sphere(
            level=config.ico_sphere_level,
            texture_map=texture_clamped,
            device=config.device,
        )
        
        # ----- Render -----
        rendered_image, alpha = renderer.render(mesh, camera)  # rendered_image: [1, 3, H, W], alpha: [1, 1, H, W]

        # ----- Background Handling -----
        # Solution 2: Randomized background to prevent background bleeding
        bg_color = torch.rand(1, 3, 1, 1, device=config.device)
        rendered_image = rendered_image * alpha + bg_color * (1.0 - alpha)
        
        # ----- SDS Loss Computation -----
        # Encode rendered image to latents (with gradient tracking)
        # IMPORTANT: We need gradients to flow from loss -> latents -> rendered_image -> texture
        
        # Resize if needed (KEEP GRADIENTS - no torch.no_grad()!)
        if rendered_image.shape[-1] != config.render_resolution:
            rendered_image_resized = F.interpolate(
                rendered_image,
                size=(config.render_resolution, config.render_resolution),
                mode='bilinear',
                align_corners=False,
            )
        else:
            rendered_image_resized = rendered_image
        
        # Normalize to [-1, 1] for VAE (keep gradients!)
        rendered_for_vae = 2.0 * rendered_image_resized - 1.0
        rendered_for_vae = rendered_for_vae.to(dtype=config.dtype)
        
        # Encode to latents - use .mode() instead of .sample() for deterministic 
        # encoding that properly passes gradients (sample() has stochastic issues)
        latent_dist = sdxl.pipe.vae.encode(rendered_for_vae).latent_dist
        # Use mean (mode) for deterministic gradient flow
        latents = latent_dist.mean * sdxl.pipe.vae.config.scaling_factor
        
        # Sample random timestep
        # TODO: Experiment with timestep sampling strategies
        timestep = random.randint(config.min_timestep, config.max_timestep)
        
        # Compute SDS gradient (this is computed without gradients - it's our "target")
        sds_grad = sdxl.compute_sds_gradient(latents.detach(), timestep)

        # Apply alpha mask to SDS gradient to avoid background bleeding
        alpha_latents = F.interpolate(
            alpha, size=latents.shape[-2:], mode="bilinear", align_corners=False
        )
        sds_grad = sds_grad * alpha_latents
        
        # ----- Backpropagation -----
        # SDS gradient update: we want to move latents in the direction of sds_grad
        # The trick: loss = latents · stop_gradient(sds_grad)
        # Taking gradient of this w.r.t. texture gives us the SDS update direction
        # 
        # Mathematically: ∂L/∂texture = ∂latents/∂texture · sds_grad
        # This backprops the SDS gradient through: latents -> VAE -> rendered_image -> texture
        loss = (latents * sds_grad.detach()).sum()
        loss.backward()
        
        # ----- Verify Gradient Flow (first iteration only) -----
        if not gradient_verified:
            if texture_map.grad is not None and texture_map.grad.abs().sum() > 0:
                print(f"✓ Gradient flow verified! Texture grad norm: {texture_map.grad.norm().item():.6f}")
                gradient_verified = True
            else:
                print("✗ WARNING: No gradients flowing to texture! Check the computation graph.")
                print(f"  - rendered_image requires_grad: {rendered_image.requires_grad}")
                print(f"  - latents requires_grad: {latents.requires_grad}")
        
        # ----- Optimizer Step -----
        optimizer.step()
        
        # ----- Logging -----
        if iteration % 50 == 0 or iteration == config.num_iterations - 1:
            grad_norm = texture_map.grad.norm().item() if texture_map.grad is not None else 0.0
            texture_mean = texture_map.data.mean().item()
            texture_std = texture_map.data.std().item()
            print(
                f"Iter {iteration:4d}/{config.num_iterations} | "
                f"t={timestep:3d} | "
                f"elev={elevation:5.1f}° az={azimuth:5.1f}° | "
                f"loss={loss.item():.2e} | "
                f"grad={grad_norm:.2e} | "
                f"tex μ={texture_mean:.3f} σ={texture_std:.3f}"
            )
        
        # ----- Save Intermediate Results -----
        # TODO: Add checkpointing and visualization saving
        if iteration % 100 == 0:
            # Save texture map
            texture_save = torch.clamp(texture_map.detach(), 0.0, 1.0)
            # Could save to file here
            pass
    
    print("-" * 60)
    print("Optimization complete!")
    
    # -------------------------------------------------------------------------
    # Step 6: Save Final Results
    # -------------------------------------------------------------------------
    print("\nSaving final texture...")
    
    final_texture = torch.clamp(texture_map.detach(), 0.0, 1.0)
    
    # Save as image
    try:
        from torchvision.utils import save_image
        # Permute to [1, 3, H, W] for save_image
        texture_for_save = final_texture.permute(0, 3, 1, 2)
        save_image(texture_for_save, "final_texture.png")
        print("Saved: final_texture.png")
    except ImportError:
        print("torchvision not available, skipping image save")
    
    # Save rendered views
    print("\nRendering final views...")
    for i, azim in enumerate([0, 90, 180, 270]):
        camera = renderer.create_camera(elevation=20.0, azimuth=azim)
        mesh = create_textured_ico_sphere(
            level=config.ico_sphere_level,
            texture_map=final_texture,
            device=config.device,
        )
        rendered = renderer.render(mesh, camera)
        
        try:
            from torchvision.utils import save_image
            save_image(rendered, f"final_view_{azim:03d}.png")
            print(f"Saved: final_view_{azim:03d}.png")
        except ImportError:
            pass
    
    print("\nDone!")
    return texture_map


if __name__ == "__main__":
    main()

