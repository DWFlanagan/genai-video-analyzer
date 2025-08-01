#!/bin/bash
# Setup script for RTX 5080 PyTorch compatibility
# Based on: https://docs.astral.sh/uv/guides/integration/pytorch/

set -e

echo "🚀 Setting up PyTorch for RTX 5080 GPU compatibility..."

# Check if we're in a uv environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Error: No virtual environment detected. Please run 'uv sync' first."
    exit 1
fi

echo "📦 Installing PyTorch with automatic backend selection for RTX 5080..."
uv pip install torch torchvision torchaudio --torch-backend=auto

echo "🧪 Testing PyTorch installation..."
python -c "
import torch
print('✅ PyTorch version:', torch.__version__)
print('✅ CUDA version:', torch.version.cuda)
print('✅ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ Device name:', torch.cuda.get_device_name(0))
    # Test GPU computation
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.mm(x, y)
    print('✅ GPU computation successful! Result shape:', z.shape)
    print('✅ GPU memory allocated:', torch.cuda.memory_allocated() / 1024**2, 'MB')
else:
    print('❌ CUDA not available')
    exit 1
"

echo "✅ RTX 5080 PyTorch setup completed successfully!"
echo ""
echo "📋 Summary:"
echo "   - PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "   - CUDA support: Enabled"
echo "   - GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo ""
echo "🎬 You can now run the video-summarizer with GPU acceleration:"
echo "   uv run video-summarizer /path/to/your/video.mp4"
