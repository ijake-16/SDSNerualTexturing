# SDS Neural Texturing

2025F CS492 Diffusion and Flow Model - Visual Content Generation

Generate high-quality textures for 3D meshes using Score Distillation Sampling (SDS) with Stable Diffusion XL.

## 🔧 Environment Setup

### Prerequisites
- NVIDIA GPU with CUDA support (recommended: 16GB+ VRAM for SDXL)
- CUDA 11.8 or 12.1 (check with `nvcc --version` or `nvidia-smi`)
- Python 3.9 or 3.10 (recommended)

### Step 1: Check Your CUDA Version

```bash
# Check CUDA version
nvidia-smi
# Look at the top right for "CUDA Version: XX.X"

# Or check nvcc
nvcc --version
```

### Step 2: Create Conda Environment (Recommended)

```bash
# Create new environment
conda create -n sds-texture python=3.10 -y
conda activate sds-texture
```

### Step 3: Install PyTorch (Match Your CUDA Version!)

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verify PyTorch CUDA:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

### Step 4: Install PyTorch3D (The Tricky Part!)

PyTorch3D must match your PyTorch and CUDA versions exactly.

**Option A: Pre-built Wheels (Easiest - if available)**

```bash
# For PyTorch 2.1+ with CUDA 11.8
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt211/download.html

# For PyTorch 2.1+ with CUDA 12.1  
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt211/download.html
```

Check available wheels at: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

**Option B: Install from Source (Most Reliable)**

```bash
# Install build dependencies
pip install fvcore iopath

# For CUDA - need ninja for faster compilation
pip install ninja

# Clone and install pytorch3d
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
cd ..
```

**Option C: Conda Install (Alternative)**

```bash
conda install pytorch3d -c pytorch3d
```

**Verify PyTorch3D:**
```bash
python -c "from pytorch3d.structures import Meshes; print('PyTorch3D installed successfully!')"
```

### Step 5: Install Other Dependencies

```bash
pip install diffusers transformers accelerate
pip install xformers  # Optional but recommended for memory efficiency
pip install numpy scipy matplotlib tqdm
```

### Step 6: Login to Hugging Face (for SDXL)

```bash
# Install huggingface CLI
pip install huggingface_hub

# Login (you need a HuggingFace account)
huggingface-cli login
```

## 📋 Full Setup Script

Create `setup.sh` and run it:

```bash
#!/bin/bash

# Detect CUDA version
CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
echo "Detected CUDA: $CUDA_VERSION"

# Create conda environment
conda create -n sds-texture python=3.10 -y
conda activate sds-texture

# Install PyTorch based on CUDA version
if [[ "$CUDA_VERSION" == 12.* ]]; then
    echo "Installing PyTorch for CUDA 12.x"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
elif [[ "$CUDA_VERSION" == 11.* ]]; then
    echo "Installing PyTorch for CUDA 11.x"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

# Install PyTorch3D from source (most reliable)
pip install fvcore iopath ninja
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e . && cd ..

# Install other dependencies
pip install diffusers transformers accelerate xformers
pip install numpy scipy matplotlib tqdm huggingface_hub

echo "Setup complete! Run: conda activate sds-texture"
```

## 🚀 Usage

```bash
conda activate sds-texture
python neural_texturing_sds.py
```

## 🐛 Troubleshooting

### "CUDA out of memory"
- Reduce `texture_resolution` (try 512 instead of 1024)
- Reduce `render_resolution` (try 256 instead of 512)
- Use `torch.cuda.empty_cache()` between iterations

### PyTorch3D compilation errors
```bash
# Make sure you have the right CUDA toolkit
conda install -c nvidia cuda-toolkit=11.8  # or 12.1

# Set CUDA paths
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### "No module named pytorch3d"
```bash
# Try conda install as fallback
conda install pytorch3d -c pytorch3d -c conda-forge
```

### xformers issues
```bash
# xformers is optional - you can skip it
# The code will fall back to standard attention
```

## 📁 Output Files

After running, you'll get:
- `final_texture.png` - The optimized UV texture map
- `final_view_000.png` to `final_view_270.png` - Rendered views from different angles

## 🎨 Customization

Edit the `Config` class in `neural_texturing_sds.py`:

```python
class Config:
    prompt = "A peeling rusty metal sphere"  # Change this!
    texture_resolution = 1024  # Lower if OOM
    num_iterations = 1000  # More = better quality
    guidance_scale = 100.0  # Higher = more prompt adherence
```
