---
name: dart-rollouter
description: Build, persist, and run the DART rollouter model service in Docker on a GPU VM. Use when setting up rollouter container deployment, restart policy, reboot autostart, and external port availability for model endpoints.
---

# DART Rollouter Container

Use this skill to run the rollouter as a Docker service instead of a native Python process.

## Prerequisites

- GPU VM with Docker daemon support (privileged mount/overlay operations enabled).
- NVIDIA runtime support (`--gpus all`).
- Repository checked out at `/workspace/dart-gui`.
- Model path available to container (local path or model hub ID).

If Docker operations fail with `operation not permitted` during image extraction/build, the VM/container is missing required Docker privileges; use a privileged VM/container runtime.

For no-Docker launch, create the project virtualenv first:

```bash
cd /workspace/dart-gui
python3 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Use `requirements.txt` as the source of truth for runtime package versions.

## Build Image

Use a wrapper Dockerfile:

```dockerfile
FROM crpi-iwtwdoj3ikoon38c.cn-beijing.personal.cr.aliyuncs.com/pengxiangli1999/dart-gui:v0
WORKDIR /workspace/dart-gui/validation
ENV PYTHONUNBUFFERED=1
CMD ["bash", "-lc", "source /workspace/dart-gui/.venv/bin/activate && python model_service.py --config-name config_singleapp model.ckpt_path=ByteDance-Seed/UI-TARS-1.5-7B model.replicas=1 model.base_port=8010 model.host=0.0.0.0 model.service_port=15961 model.vllm_params.gpu_memory_utilization=0.92 +model.vllm_params.max_model_len=4096"]
```

Build:

```bash
docker build -t dart-rollouter:local -f docker/Dockerfile.dart-rollouter .
```

## Launch Container

Run with host network and restart policy:

```bash
docker run -d \
  --name dart-rollouter \
  --gpus all \
  --network host \
  --restart unless-stopped \
  -v /workspace/dart-gui:/workspace/dart-gui \
  dart-rollouter:local
```

## Verify Service

On VM:

```bash
curl -s http://127.0.0.1:15961/
curl -s http://127.0.0.1:15961/status
curl -s http://127.0.0.1:8010/health
```

Container/process checks:

```bash
docker ps --filter name=dart-rollouter
docker logs --tail=200 dart-rollouter
```

## Reboot Autostart

`--restart unless-stopped` restarts container after Docker daemon starts.

Use the managed Docker startup script in this skill folder:

```bash
chmod +x /workspace/dart-gui/agents/skills/dart-rollouter/docker.sh
```

If your VM provider runs `/root/onstart.sh` at boot, rebind it:

```bash
cat >/root/onstart.sh <<'EOF'
#!/usr/bin/env bash
exec /workspace/dart-gui/agents/skills/dart-rollouter/docker.sh
EOF
chmod +x /root/onstart.sh
```

If you use systemd instead of `/root/onstart.sh`, point `ExecStart` to:

```bash
/workspace/dart-gui/agents/skills/dart-rollouter/docker.sh
```

For no-Docker startup with a persistent tmux session:

```bash
chmod +x /workspace/dart-gui/agents/skills/dart-rollouter/no_docker.sh
cat >/root/onstart.sh <<'EOF'
#!/usr/bin/env bash
exec /workspace/dart-gui/agents/skills/dart-rollouter/no_docker.sh
EOF
chmod +x /root/onstart.sh
```

Then launch manually once to verify:

```bash
/root/onstart.sh
tmux ls
```

## External Availability Warning

The service being up on `0.0.0.0:15961` and `0.0.0.0:8010` does not guarantee internet reachability.

You must expose/map VM public ports to these internal ports in your VM provider networking settings (for example, Vast `VAST_TCP_PORT_*` mappings), or use SSH tunneling:

```bash
ssh -L 15961:127.0.0.1:15961 -L 8010:127.0.0.1:8010 <vm-host>
```

## Local Access via SSH Tunnel

If provider networking does not expose `15961`/`8010`, forward them to **the same ports on localhost** (matches `~/.ssh/config` `Host dart-rollouter` and `GUI-Docker-Env` defaults).

Use a clean SSH config (`-F /dev/null`) only if you need to avoid conflicts with another `LocalForward` on the same ports.

Start tunnel (example for `dart-rollouter` host `84.50.156.126:17110`):

```bash
ssh -F /dev/null -N \
  -o ExitOnForwardFailure=yes \
  -o StrictHostKeyChecking=accept-new \
  -i /home/pitchblack/.ssh/id_rsa \
  -p 17110 \
  -L 15961:127.0.0.1:15961 \
  -L 8010:127.0.0.1:8010 \
  root@84.50.156.126
```

**GUI-Docker-Env / UITARS:** set `OPENAI_BASE_URL=http://127.0.0.1:8010` (see `GUI-Docker-Env/.env-default`). The OpenAI client appends `/v1` for chat completions; vLLM listens on **8010**, not 8000.

Health checks from local machine (with tunnel up):

```bash
curl -s http://127.0.0.1:15961/status
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8010/health
```

This maps:
- local `127.0.0.1:15961` -> remote `127.0.0.1:15961` (model service API)
- local `127.0.0.1:8010` -> remote `127.0.0.1:8010` (vLLM OpenAI-compatible API and `/health`)
