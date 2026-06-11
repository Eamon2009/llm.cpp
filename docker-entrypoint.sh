#!/bin/bash
set -e
VARIANT="${QUADTRIX_VARIANT:-auto}"

if [ "$VARIANT" = "auto" ]; then
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
        VARIANT="cuda"
    else
        VARIANT="cpu"
    fi
fi

echo ""
echo "  ██████╗ ██╗   ██╗ █████╗ ██████╗ ████████╗██████╗ ██╗██╗  ██╗"
echo "  ██╔═══██╗██║   ██║██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██║╚██╗██╔╝"
echo "  ██║   ██║██║   ██║███████║██║  ██║   ██║   ██████╔╝██║ ╚███╔╝ "
echo "  ██║▄▄ ██║██║   ██║██╔══██║██║  ██║   ██║   ██╔══██╗██║ ██╔██╗ "
echo "  ╚██████╔╝╚██████╔╝██║  ██║██████╔╝   ██║   ██║  ██║██║██╔╝ ██╗"
echo "   ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝"
echo ""
echo "  Starting Quadtrix.cpp..."
echo ""
echo "  FastAPI backend  at http://localhost:3001"
echo "  React frontend   at  http://localhost:8080"
echo "  Models volume    at  /app/models"
echo ""

if [ "$VARIANT" = "cuda" ]; then
    echo "  Mode: CUDA / GPU"
    echo ""
    if command -v nvidia-smi &>/dev/null; then
        if nvidia-smi &>/dev/null 2>&1; then
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
            GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "?")
            echo "  GPU detected: $GPU_NAME ($GPU_MEM)"
        else
            echo "  WARNING: nvidia-smi present but GPU is not accessible."
            echo ""
            echo "      Running on WSL2? Make sure you have:"
            echo "        1. NVIDIA Game/Studio/Data-Centre driver ≥ 515 on Windows"
            echo "        2. WSL2 (not WSL1):  wsl --set-version <distro> 2"
            echo "        3. The CUDA toolkit installed inside WSL is optional;"
            echo "           the driver bridge handles it automatically."
            echo "        Docs: https://developer.nvidia.com/cuda/wsl"
        fi
    else
        echo "  WARNING: No NVIDIA driver found. CUDA image started without GPU."
        echo ""
        echo "      On WSL2, install the NVIDIA driver on Windows (not inside WSL)."
        echo "      GPU passthrough is handled automatically by WSL2."
        echo "      Docs: https://developer.nvidia.com/cuda/wsl"
        echo ""
        echo "      To run without a GPU, use the CPU image instead:"
        echo "        docker pull ghcr.io/eamon2009/quadtrix-cpu:sha-3b65553"
    fi
    echo ""
else
    echo "  Mode: CPU"
    echo ""
fi
WEIGHTS_FOUND=0

if [ -f "/app/models/best_model.pt" ]; then
    echo "  PyTorch checkpoint found: /app/models/best_model.pt"
    WEIGHTS_FOUND=1
fi

if [ -f "/app/models/best_model.bin" ]; then
    echo "  C++ checkpoint found:     /app/models/best_model.bin"
    WEIGHTS_FOUND=1
fi

if [ "$WEIGHTS_FOUND" = "0" ]; then
    echo ""
    echo "  WARNING: No model weights found in /app/models"
    echo "      The backend will start but inference will fail until weights are mounted."
    echo ""
    echo "      Mount your weights at run-time:"
    if [ "$VARIANT" = "cuda" ]; then
        echo "        docker run --gpus all -v /path/to/models:/app/models ghcr.io/eamon2009/quadtrix-cuda:sha-3b65553"
    else
        echo "        docker run -v /path/to/models:/app/models ghcr.io/eamon2009/quadtrix-cpu:sha-3b65553"
    fi
    echo ""
fi
exec /usr/bin/supervisord -c /etc/supervisor/supervisord.conf