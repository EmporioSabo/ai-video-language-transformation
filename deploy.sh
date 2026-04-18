#!/usr/bin/env bash
# deploy.sh — One-shot setup for a fresh Linode (Ubuntu 22.04/24.04)
# Run as root: bash deploy.sh

set -euo pipefail

APP_DIR="/opt/aivlt"
REPO_URL="https://github.com/emporiosabo/AIVideoLanguageTransformation.git"
BRANCH="feature/multilang-voxtral"

echo "=== [1/5] Installing Docker ==="
apt-get update -qq
apt-get install -y -qq ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update -qq
apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
systemctl enable --now docker

echo "=== [2/5] Cloning repository ==="
git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
cd "$APP_DIR"

echo "=== [3/5] Creating .env ==="
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo ">>> EDIT $APP_DIR/.env with your real API keys, then re-run:"
    echo ">>>   cd $APP_DIR && docker compose up -d --build"
    echo ""
    echo "Required keys: DEEPL_API_KEY, GEMINI_API_KEY, MISTRAL_API_KEY,"
    echo "               RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID, HF_TOKEN"
    exit 0
fi

echo "=== [4/5] Creating data directories ==="
mkdir -p data/jobs output nginx/certs
touch users.json
echo "{}" > users.json

echo "=== [5/5] Building and starting containers ==="
docker compose up -d --build

echo ""
echo "=== Done! ==="
SERVER_IP=$(curl -s https://ipinfo.io/ip 2>/dev/null || hostname -I | awk '{print $1}')
echo "App is running at: http://$SERVER_IP"
echo "Logs: docker compose logs -f webapp"
