#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/workspace/dart-gui"
CONTAINER_NAME="dart-rollouter"
DOCKERD_LOG="/root/dockerd.log"

if [ ! -S /var/run/docker.sock ]; then
  nohup dockerd --iptables=false --bridge=none --ip-forward=false --ip-masq=false > "${DOCKERD_LOG}" 2>&1 &
fi

for _ in $(seq 1 30); do
  if [ -S /var/run/docker.sock ]; then
    break
  fi
  sleep 1
done

if ! docker info >/dev/null 2>&1; then
  echo "docker daemon is not ready; skip container start"
  exit 0
fi

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  docker start "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  exit 0
fi

if [ -f "${REPO_ROOT}/docker/Dockerfile.dart-rollouter" ]; then
  docker build -t dart-rollouter:local -f "${REPO_ROOT}/docker/Dockerfile.dart-rollouter" "${REPO_ROOT}" || true
  docker run -d \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --network host \
    --restart unless-stopped \
    -v "${REPO_ROOT}:${REPO_ROOT}" \
    dart-rollouter:local >/dev/null 2>&1 || true
fi
