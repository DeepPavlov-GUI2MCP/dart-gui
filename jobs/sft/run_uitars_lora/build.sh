#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-job-dart-uitars-sft:1.0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

docker build -f "${SCRIPT_DIR}/Dockerfile" -t "${IMAGE_NAME}" "${REPO_ROOT}"

echo "Built ${IMAGE_NAME}"
echo "Submit from the notebook base env: python jobs/submit.py sft"
echo "Tag and push per jobs/README.md, e.g.:"
echo '  REGISTRY_PATH="cr.ai.cloud.ru/<workspace-uuid>"'
echo "  docker tag \"${IMAGE_NAME}\" \"\${REGISTRY_PATH}/${IMAGE_NAME}\""
echo "  docker push \"\${REGISTRY_PATH}/${IMAGE_NAME}\""
echo ""
echo "Base image: cr.ai.cloud.ru/aicloud-base-images/py3.11-torch2.9.0:0.0.42"
