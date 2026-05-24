set -e
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
echo "  FastAPI backend  at  http://localhost:3001"
echo "  React frontend   at  http://localhost:8080"
echo "  Models volume    at  /app/models"
echo ""
WEIGHTS_FOUND=0

if [ -f "/app/models/best_model.pt" ]; then
    echo " PyTorch checkpoint found: /app/models/best_model.pt"
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
    echo "      Mount your weights directory:"
    echo "        docker run -v /path/to/your/models:/app/models ..."
    echo ""
fi
exec /usr/bin/supervisord -c /etc/supervisor/supervisord.conf