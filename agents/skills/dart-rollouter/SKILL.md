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
