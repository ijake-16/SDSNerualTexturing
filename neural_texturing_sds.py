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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
# Diffusers for SDXL
from diffusers import StableDiffusionXLPipeline, DDPMScheduler, StableDiffusionPipeline, DDIMScheduler
from torchvision.transforms import GaussianBlur
# PyTorch3D for differentiable rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    HardPhongShader,
    TexturesUV,
    PointLights,
    look_at_view_transform,
)
from pytorch3d.utils import ico_sphere

def compute_tv_loss(img):
    """이미지의 가로/세로 인접 픽셀 차이를 계산 (매끄러움 강제)"""
    h_diff = img[:, :-1, :, :] - img[:, 1:, :, :]
    w_diff = img[:, :, :-1, :] - img[:, :, 1:, :]
    return torch.sum(torch.abs(h_diff)) + torch.sum(torch.abs(w_diff))
# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Hyperparameters and settings for the neural texturing pipeline."""
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # SDXL settings
    #model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    model_id = "Manojb/stable-diffusion-2-base"
    
    dtype = torch.float32
    
    # Texture settings
    texture_resolution = 1024  # UV texture map resolution
    render_resolution = 512   # Rendered image resolution for SDS
    
    # Mesh settings
    ico_sphere_level = 4  # Higher = more subdivisions = smoother sphere
    
    # Optimization settings
    num_iterations = 2000
    learning_rate = 1e-2
    
    # SDS settings
    guidance_scale = 50.0   # CFG scale for SDS (Reduced from 100)
    min_timestep = 200       # Minimum diffusion timestep
    max_timestep = 980      # Maximum diffusion timestep
    
    # Camera sampling
    camera_distance = 2.5
    min_elevation = -30.0   # degrees
    max_elevation = 60.0    # degrees
    
    # Text prompt for texture generation
    prompt = "a human head, highly detailed, 8k, photorealistic"
    negative_prompt = "blurry, low quality, distorted, ugly"


# =============================================================================
# SDXL Pipeline Wrapper
# =============================================================================
class SD2Wrapper:
    """
    Wrapper for Stable Diffusion 2.1 Base pipeline optimized for SDS.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        
        print(f"Loading SD2 pipeline: {config.model_id}...")
        # SDXLPipeline -> StableDiffusionPipeline으로 변경
        self.pipe = StableDiffusionPipeline.from_pretrained(
            config.model_id,
            torch_dtype=config.dtype,
            use_safetensors=True,
        )
        self.pipe.to(self.device)
        
        # 메모리 절약을 위한 컴포넌트 고정
        self.pipe.vae.requires_grad_(False)
        self.pipe.unet.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        
        # xformers 적용 (OOM 방지에 중요)
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("xformers enabled")
        except Exception:
            print("xformers not available")
        
        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        
        self._encode_prompts()
        
        # SD2는 VAE 스케일링 팩터가 보통 0.18215
        self.vae_scale_factor = self.pipe.vae.config.scaling_factor
        
    def _encode_prompts(self):
        """Pre-encode text prompts."""
        print("Encoding text prompts...")
        
        # SD2는 encode_prompt 반환값이 SDXL과 다름 (심플함)
        # prompt_embeds shape: [B, 77, 768] (SDXL은 2048+)
        self.prompt_embeds = self.pipe._encode_prompt(
            self.config.prompt,
            self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=self.config.negative_prompt,
        )
        
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
        noise = torch.randn_like(latents)
        timesteps = torch.tensor([timestep], device=self.device).long()
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # 2. Input 준비
        latent_model_input = torch.cat([noisy_latents] * 2)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, timesteps)
        
        # 3. 예측
        with torch.no_grad():
            noise_pred = self.pipe.unet(
                latent_model_input,
                timesteps,
                encoder_hidden_states=self.prompt_embeds,
                return_dict=False,
            )[0]

        # 4. CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_text - noise_pred_uncond)

        # 5. SDS Gradient
        w_t = 1.0
        grad = w_t * (noise_pred - noise)

        # ★ [추가] 디버깅을 위해 noise_pred도 함께 반환
        return grad, noise_pred, noisy_latents

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Latent를 RGB 이미지로 디코딩 (현재 렌더링 상태 확인용)"""
        # 스케일링 복구
        latents = 1 / self.vae_scale_factor * latents
        image = self.pipe.vae.decode(latents).sample
        # [-1, 1] -> [0, 1]
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    @torch.no_grad()
    def decode_prediction(self, latents: torch.Tensor, noise_pred: torch.Tensor, timestep: int) -> torch.Tensor:
        """
        [수정 2] scheduler.step 대신 수식으로 직접 x0 추정 (디버그용)
        Scheduler 설정이 꼬이는 것을 방지하기 위해 직접 수식 사용
        x0 = (xt - sqrt(1-alpha_bar) * epsilon) / sqrt(alpha_bar)
        """
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        
        # x0 prediction formula
        pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
        
        return self.decode_latents(pred_original_sample)

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
        self.pipe.vae.to(dtype=torch.float32)
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
        # Use 1024x1024 for original_size to encourage high-res features
        add_time_ids = self._get_add_time_ids(
            original_size=(1024, 1024),
            crops_coords_top_left=(0, 0),
            target_size=(Config.render_resolution, Config.render_resolution),
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
            cull_backfaces=True,
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
    
    def render(self, mesh: Meshes, camera: FoVPerspectiveCameras) -> torch.Tensor:
        """
        Render the mesh from the given camera viewpoint.
        
        Args:
            mesh: PyTorch3D Meshes object with texture
            camera: Camera for rendering
            
        Returns:
            Rendered image tensor [1, 3, H, W] in range [0, 1]
        """
        # Create renderer with current camera
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera,
                raster_settings=self.raster_settings,
            ),
            shader=HardPhongShader(
                device=self.device,
                cameras=camera,
                lights=self.lights,
            ),
        )
        
        # Render
        images = renderer(mesh)  # [B, H, W, 4] (RGBA)
        
        # Permute to [B, 4, H, W]
        images = images.permute(0, 3, 1, 2)
        
        # Clamp to valid range
        images = torch.clamp(images, 0.0, 1.0)
        
        return images


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
    
    # Explicitly set vertex normals for smooth shading
    # We set the internal cache directly to ensure it's used
    mesh._verts_normals_packed = verts_normalized
    
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
    sdxl = SD2Wrapper(config)
    
    # -------------------------------------------------------------------------
    # Step 2: Initialize Learnable Texture Map
    # -------------------------------------------------------------------------
    print("\nInitializing learnable texture map...")
    
    # Initialize texture as green (tennis ball prior)
    # Shape: [1, H, W, 3] for TexturesUV
    texture_map = torch.ones(
        1, 
        config.texture_resolution, 
        config.texture_resolution, 
        3,
        device=config.device,
        dtype=torch.float32,
    )*0.5
    
    
    # Add small noise for initialization diversity
    # texture_map = texture_map + torch.randn_like(texture_map) * 0.1
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
    # optimizer = Adam([texture_map], lr=config.learning_rate)
    optimizer = AdamW([texture_map], lr=config.learning_rate, weight_decay=0.0)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_iterations, eta_min=0.001)
    blurrer = GaussianBlur(kernel_size=(9, 9), sigma=(2.0, 2.0))
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
        # texture_flipped = torch.flip(texture_clamped, [1])
        mesh = create_textured_ico_sphere(
            level=config.ico_sphere_level,
            texture_map=texture_clamped,
            device=config.device,
        )
        
        # ----- Render -----
        # rendered_image = renderer.render(mesh, camera)  # [1, 4, H, W]
        rendered_image_raw = renderer.render(mesh, camera)
        
        rgb = rendered_image_raw[:, :3, :, :]
        mask = rendered_image_raw[:, 3:4, :, :]
        bg_color = torch.tensor([0.5, 0.5, 0.5], device=config.device).view(1, 3, 1, 1)
        rendered_image = rgb * mask + bg_color * (1 - mask)
        # ----- SDS Loss Computation -----
        # Encode rendered image to latents (with gradient tracking)
        # IMPORTANT: We need gradients to flow from loss -> latents -> rendered_image -> texture
        
        # Resize if needed (KEEP GRADIENTS - no torch.no_grad()!)
        rendered_image_resized = F.interpolate(
            rendered_image,
            size=(Config.render_resolution,Config.render_resolution),
            mode='bilinear',
            align_corners=False,
        )

        # Normalize to [-1, 1] for VAE (keep gradients!)
        rendered_for_vae = 2.0 * rendered_image_resized - 1.0
        rendered_for_vae = rendered_for_vae.to(dtype=config.dtype).contiguous()
        
        # Encode to latents - use .mode() instead of .sample() for deterministic 
        # encoding that properly passes gradients (sample() has stochastic issues)
        latent_dist = sdxl.pipe.vae.encode(rendered_for_vae).latent_dist
        # Use mean (mode) for deterministic gradient flow
        latents = latent_dist.mean * sdxl.pipe.vae.config.scaling_factor
        
        # Sample random timestep
        # TODO: Experiment with timestep sampling strategies
        timestep = random.randint(config.min_timestep, config.max_timestep)

        # t_ratio = iteration / config.num_iterations
        # t_range = config.max_timestep - config.min_timestep
        # current_t = config.max_timestep - int(t_range * t_ratio * 0.8) # 너무 작은 t는 피함
        # timestep = random.randint(max(config.min_timestep, current_t - 200), current_t)
        
        # Compute SDS gradient (this is computed without gradients - it's our "target")
        sds_grad, noise_pred, noisy_latents = sdxl.compute_sds_gradient(latents.detach(), timestep)
        sds_grad = sds_grad.to(dtype=torch.float32)
        # sds_grad = blurrer(sds_grad)  # Removed aggressive blur
        # ----- Backpropagation -----
        # SDS gradient update: we want to move latents in the direction of sds_grad
        # The trick: loss = latents · stop_gradient(sds_grad)
        # Taking gradient of this w.r.t. texture gives us the SDS update direction
        # 
        # Mathematically: ∂L/∂texture = ∂latents/∂texture · sds_grad
        # This backprops the SDS gradient through: latents -> VAE -> rendered_image -> texture
        #####################
        # [수정 전]
        # loss = (latents * sds_grad.detach()).sum()
        # loss.backward()

        # [수정 후] -> 이 방식으로 변경!
        # SDS 그래디언트를 latents에 직접 주입하여 텍스처까지 역전파

        # grad_mag = sds_grad.norm()
        # if grad_mag > 1.0:
        #     sds_grad = sds_grad / grad_mag  # Normalize to unit norm if too large
        
        # Apply a constant scale factor to control update size
        sds_grad = sds_grad * 0.3  # Reduced scale to prevent noisy updates
        loss_sds = (latents * sds_grad.detach()).sum()

        # TV Loss 가중치 설정 (보통 0.01 ~ 0.1 사이, 노이즈가 심하면 키우세요)
        # tv_weight = 1e-5
        # loss_tv = compute_tv_loss(texture_map) * tv_weight

        # 최종 Loss 합산
        # loss = loss_tv
        optimizer.zero_grad() # 기울기 초기화
        # latents.backward(gradient=sds_grad, retain_graph=True)
        tv_weight = 1e-5
        loss_tv = compute_tv_loss(texture_map) * tv_weight
        loss_total = loss_tv+loss_sds
        loss_total.backward()
        optimizer.step()
        scheduler.step()
        loss = loss_total

        
        ##################
        
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
        # optimizer.step()
        
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
            print(f"Saving debug images at iter {iteration}...")
            with torch.no_grad():
                # 1. 현재 렌더링된 모습 (VAE가 잘 동작하는지 확인)
                # 3D 렌더러 -> VAE Encode -> VAE Decode 과정을 거친 이미지
                # 이게 노이즈라면 렌더러나 VAE 입력 범위(-1~1) 문제
                current_view_img = sdxl.decode_latents(latents.detach())
                
                # 2. AI가 상상하는 목표 모습 (x0 prediction)
                # "이 노이즈를 제거하면 이런 얼굴이 될 거야"라고 모델이 생각하는 이미지
                # 이게 괴상하면 프롬프트가 이상하거나 CFG가 너무 높은 것
                target_view_img = sdxl.decode_prediction(noisy_latents, noise_pred, timestep).cpu()
                
                # 저장
                from torchvision.utils import save_image
                save_image(current_view_img, f"debug_iter_{iteration:04d}_current.png")
                save_image(target_view_img, f"debug_iter_{iteration:04d}_target_prediction.png")
                
                # 텍스처 맵도 저장
                save_image(texture_map.detach().permute(0, 3, 1, 2), f"debug_iter_{iteration:04d}_texture.png")

    
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

