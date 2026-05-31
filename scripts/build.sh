
# Quadtrix.cpp — build.sh  
# Usage
#   ./scripts/build.sh               # full stack, CPU
#   ./scripts/build.sh dev           # hot-reload dev mode
#   ./scripts/build.sh gpu           # CUDA backend
#   ./scripts/build.sh cpp-only      # compile + run C++ engine
#   ./scripts/build.sh train-cpp     # train with C++ backend
#   ./scripts/build.sh train-torch   # train with PyTorch backend
#   ./scripts/build.sh bench         # run benchmark
#   ./scripts/build.sh clean         # remove containers + volumes
#   ./scripts/build.sh logs          # tail all service logs

set -euo pipefail

BOLD="\033[1m"
GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
RESET="\033[0m"

info()    { echo -e "${CYAN}[quadtrix]${RESET} $*"; }
success() { echo -e "${GREEN}[quadtrix]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[quadtrix]${RESET} $*"; }
error()   { echo -e "${RED}[quadtrix] ERROR:${RESET} $*" >&2; }

COMPOSE_BASE="docker compose -f docker-compose.yml"
COMPOSE_DEV="${COMPOSE_BASE} -f docker-compose.dev.yml"
COMPOSE_GPU="${COMPOSE_BASE} -f docker-compose.gpu.yml"

check_docker() {
    if ! docker info &>/dev/null; then
        error "Docker daemon is not running. Start Docker Desktop or the Docker service."
        exit 1
    fi
}

check_nvidia() {
    if ! command -v nvidia-smi &>/dev/null; then
        warn "nvidia-smi not found — GPU mode may not work."
    else
        info "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    fi
}

pull_cache() {
    info "Pulling build cache images (if available)..."
    $COMPOSE_BASE pull --ignore-pull-failures 2>/dev/null || true
}

cmd_up() {
    check_docker
    info "Starting full stack (CPU)..."
    $COMPOSE_BASE up --build -d
    success "Stack is up."
    echo ""
    echo -e "  ${BOLD}Frontend:${RESET}  http://localhost:5173"
    echo -e "  ${BOLD}API:${RESET}       http://localhost:3001/api/health"
    echo -e "  ${BOLD}Docs:${RESET}      http://localhost:3001/docs"
}

cmd_dev() {
    check_docker
    info "Starting in DEV mode (hot-reload)..."
    $COMPOSE_DEV up --build
}

cmd_gpu() {
    check_docker
    check_nvidia
    info "Starting with CUDA GPU support..."
    $COMPOSE_GPU up --build -d
    success "GPU stack is up."
}

cmd_cpp_only() {
    check_docker
    info "Compiling and running C++ engine..."
    $COMPOSE_BASE --profile cpp run --rm cpp "$@"
}

cmd_train_cpp() {
    check_docker
    info "Training with C++ backend..."
    $COMPOSE_BASE --profile train run --rm train-cpp
    success "C++ training complete. Checkpoint saved in 'models' volume."
}

cmd_train_torch() {
    check_docker
    info "Training with PyTorch backend..."
    $COMPOSE_BASE --profile train run --rm train-torch
    success "PyTorch training complete. Checkpoint saved in 'models' volume."
}

cmd_bench() {
    check_docker
    info "Running benchmark..."
    $COMPOSE_BASE --profile benchmark run --rm benchmark
}

cmd_logs() {
    check_docker
    $COMPOSE_BASE logs -f --tail=100
}

cmd_clean() {
    check_docker
    warn "This will remove all containers and volumes (including saved models!)"
    read -r -p "Are you sure? [y/N] " confirm
    if [[ "${confirm,,}" == "y" ]]; then
        $COMPOSE_BASE down -v --remove-orphans
        docker image prune -f --filter "label=org.opencontainers.image.source=https://github.com/Eamon2009/Quadtrix.cpp"
        success "Cleaned."
    else
        info "Aborted."
    fi
}

cmd_ps() {
    $COMPOSE_BASE ps
}

cmd_shell() {
    service="${1:-backend}"
    info "Opening shell in '${service}'..."
    $COMPOSE_BASE exec "${service}" /bin/sh
}
CMD="${1:-up}"
shift || true

case "${CMD}" in
    up)           cmd_up "$@" ;;
    dev)          cmd_dev "$@" ;;
    gpu)          cmd_gpu "$@" ;;
    cpp-only)     cmd_cpp_only "$@" ;;
    train-cpp)    cmd_train_cpp "$@" ;;
    train-torch)  cmd_train_torch "$@" ;;
    bench)        cmd_bench "$@" ;;
    logs)         cmd_logs "$@" ;;
    clean)        cmd_clean "$@" ;;
    ps)           cmd_ps "$@" ;;
    shell)        cmd_shell "$@" ;;
    *)
        echo -e "Usage: ./scripts/build.sh ${BOLD}[command]${RESET}"
        echo ""
        echo "Commands:"
        echo "  up           Full stack (CPU) — default"
        echo "  dev          Hot-reload dev mode"
        echo "  gpu          CUDA GPU stack"
        echo "  cpp-only     Run C++ engine CLI"
        echo "  train-cpp    Train with C++ backend"
        echo "  train-torch  Train with PyTorch"
        echo "  bench        Benchmark"
        echo "  logs         Tail logs"
        echo "  ps           Show container status"
        echo "  shell [svc]  Shell into service (default: backend)"
        echo "  clean        Remove all containers + volumes"
        ;;
esac
