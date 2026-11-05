#!/bin/bash
# Prepara uma instância EC2 nova (Amazon Linux 2023 ou Ubuntu) para o treino: Docker + git.
# Uso, dentro da instância:  bash setup_ec2.sh   (depois saia e entre de novo no SSH para valer o grupo docker)
set -euo pipefail

if command -v dnf >/dev/null; then           # Amazon Linux 2023
    sudo dnf install -y docker git
elif command -v apt-get >/dev/null; then     # Ubuntu
    sudo apt-get update
    sudo apt-get install -y docker.io git
else
    echo "Distribuição não suportada: instale Docker e git manualmente." >&2
    exit 1
fi

sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
echo "Pronto: $(docker --version 2>/dev/null || sudo docker --version), $(nproc) vCPU, $(free -g | awk '/Mem/{print $2}') GB RAM."
echo "Saia e entre de novo no SSH, clone o repositório e rode aws/benchmark.sh."
