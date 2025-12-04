#!/bin/bash
# SDS Neural Texturing - Setup Script
# Run: bash setup.sh

set -e  # Exit on error

echo "=============================================="
echo "SDS Neural Texturing - Environment Setup"
echo "=============================================="

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

# Detect CUDA version
CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
echo "Detected CUDA Version: $CUDA_VERSION"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Anaconda or Miniconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Environment name
ENV_NAME="sds-texture"

echo ""
echo "Step 1: Creating conda environment '$ENV_NAME' with Python 3.10..."
conda create -n $ENV_NAME python=3.10 -y

# Activate environment
echo ""
echo "Step 2: Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install PyTorch based on CUDA version
echo ""
echo "Step 3: Installing PyTorch..."
if [[ "$CUDA_VERSION" == 12.* ]]; then
    echo "Installing PyTorch for CUDA 12.x"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
elif [[ "$CUDA_VERSION" == 11.8* ]] || [[ "$CUDA_VERSION" == 11.7* ]]; then
    echo "Installing PyTorch for CUDA 11.8"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch for CUDA 11.8 (default)"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

# Verify PyTorch installation
echo ""
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()} | CUDA Version: {torch.version.cuda}')"

# Install PyTorch3D dependencies
echo ""
echo "Step 4: Installing PyTorch3D build dependencies..."
pip install fvcore iopath ninja

# Install PyTorch3D from source
echo ""
echo "Step 5: Installing PyTorch3D from source (this may take a while)..."
if [ -d "pytorch3d" ]; then
    echo "pytorch3d directory exists, removing..."
    rm -rf pytorch3d
fi

git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
cd ..

# Verify PyTorch3D
echo ""
echo "Verifying PyTorch3D installation..."
python -c "from pytorch3d.structures import Meshes; from pytorch3d.renderer import MeshRenderer; print('PyTorch3D OK!')"

# Install Diffusers and other dependencies
echo ""
echo "Step 6: Installing diffusers and other dependencies..."
pip install diffusers transformers accelerate
pip install numpy scipy matplotlib tqdm
pip install huggingface_hub

# Try to install xformers (optional)
echo ""
echo "Step 7: Installing xformers (optional, for memory efficiency)..."
pip install xformers || echo "xformers installation failed - continuing without it (optional)"

# Final verification
echo ""
echo "=============================================="
echo "Final Verification"
echo "=============================================="
python << 'EOF'
import torch
print(f"✓ PyTorch {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

from pytorch3d.structures import Meshes
from pytorch3d.renderer import MeshRenderer
print("✓ PyTorch3D")

from diffusers import StableDiffusionXLPipeline
print("✓ Diffusers")

print("\n✅ All dependencies installed successfully!")
EOF

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "To use this environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To run the neural texturing:"
echo "  python neural_texturing_sds.py"
echo ""
echo "Note: You may need to login to HuggingFace for SDXL access:"
echo "  huggingface-cli login"
echo ""

