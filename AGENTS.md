# DART-GUI Infrastructure

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  This Machine (CPU)                                     │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐ │
│  │ MySQL :3306  │  │ Monitor :80  │  │ Docker Server │ │
│  │ (container)  │  │ (container)  │  │   :50003      │ │
│  └──────────────┘  └──────────────┘  └───────┬───────┘ │
│                                              │         │
│                              ┌───────────────┘         │
│                              ▼                         │
│                    ┌───────────────────┐               │
│                    │ Desktop Emulators │               │
│                    │ (Docker containers│               │
│                    │  happysixd/       │               │
│                    │  osworld-docker)  │               │
│                    │                   │               │
│                    │ VNC  :8006+       │               │
│                    │ API  :5000+       │               │
│                    │ Chrome:9222+      │               │
│                    └───────────────────┘               │
└─────────────────────────────────────────────────────────┘
```

## Components

### 1. MySQL Database (Docker container)

Stores rollout run tracking and model checkpoint metadata.

- **Image:** `mysql:8.0.44-debian`
- **Container name:** `mysql-server`
- **Port:** 3306
- **Data directory:** `/mnt/data/mysql` (persistent volume)
- **Database:** `dart_database`
- **App credentials:** `dart_rollouter` / `password`
- **Root password:** `admin`

**Launch:**

```bash
sudo docker run -dit \
  --name mysql-server \
  -p 3306:3306 \
  -e MYSQL_ROOT_PASSWORD=admin \
  -v /mnt/data/mysql:/var/lib/mysql \
  mysql:8.0.44-debian
```

**Schema init** (only needed on first run — tables: `rollout_run`, `checkpoint`):

```bash
sudo docker exec mysql-server mysql -u root -padmin dart_database < schema.sql
```

### 2. Monitor Dashboard (Docker container via docker-compose)

Flask web dashboard for viewing evaluation tasks, results, and agent trajectories.

- **Image:** Built from `GUI-Docker-Env/monitor/Dockerfile`
- **Container name:** `monitor-monitor-1`
- **Port:** 80 (configured via `FLASK_PORT` in `GUI-Docker-Env/monitor/.env`)
- **Config:** `GUI-Docker-Env/monitor/.env`
- **Compose file:** `GUI-Docker-Env/monitor/docker-compose.yml`

**Launch:**

```bash
cd GUI-Docker-Env/monitor
sudo docker compose up -d --build
```

**Restart:**

```bash
cd GUI-Docker-Env/monitor
sudo docker compose restart
```

### 3. Desktop Server (Python process — NOT a container)

Flask API that manages desktop emulator containers (Ubuntu VMs running in Docker via QEMU). Each emulator is a `happysixd/osworld-docker` container with a QCOW2 system image.

- **Port:** 50003
- **Config:** `GUI-Docker-Env/configs/config.yaml`
- **VM image:** `/mnt/data/VMs/Ubuntu.qcow2` (~24GB)
- **Docker image:** `happysixd/osworld-docker`
- **Auth token:** `dart` (max 2 concurrent emulators)

**Launch** (requires docker group access):

```bash
cd GUI-Docker-Env
sg docker -c "source .venv/bin/activate && python -m desktop_env.docker_server.server"
```

> The `sg docker` wrapper is required because the user must have Docker socket
> access. Alternatively, add the user to the `docker` group and open a new shell:
> `sudo usermod -aG docker $USER`

**Key endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ping` | GET | Health check |
| `/status` | GET | Server stats, token quotas, emulator count |
| `/dashboard` | GET | Web dashboard UI |
| `/window` | GET | Browser-based VNC viewer (select emulator from dropdown) |
| `/start_emulator` | POST | Start a new desktop emulator (needs `Authorization: Bearer <token>`) |
| `/stop_emulator` | POST | Stop emulator by `emulator_id` |
| `/emulators` | GET | List running emulators with ports |
| `/stop_all_emulators` | POST | Stop all emulators |

**Start an emulator:**

```bash
curl -X POST http://localhost:50003/start_emulator \
  -H "Authorization: Bearer dart"
```

Returns `vnc_port`, `server_port`, `chromium_port`, `vlc_port`.

**View the desktop in a browser:**

- noVNC viewer: `http://localhost:<vnc_port>` (default starts at 8006)
- Built-in viewer: `http://localhost:50003/window`

### 3b. Remote inference (dart-rollouter GPU VM over SSH)

When UITARS / `run_uitars.py` runs on this machine but vLLM lives on the **dart-rollouter** GPU host, keep an SSH session open so **local** ports forward to the VM:

| Local (after tunnel) | Remote (on GPU VM) | Purpose |
|----------------------|--------------------|---------|
| `127.0.0.1:8010` | `127.0.0.1:8010` | **vLLM** — OpenAI-compatible `OPENAI_BASE_URL` (not 8000) |
| `127.0.0.1:15961` | `127.0.0.1:15961` | dart_rollouter **model service** API |

- **`~/.ssh/config`** `Host dart-rollouter` should include `LocalForward` for **8010** and **15961** (same port numbers both sides).
- **`GUI-Docker-Env`:** `OPENAI_BASE_URL=http://127.0.0.1:8010` in `.env-default` / `.env` (loaded by `mm_agents.env_loader`).
- **Background tunnel:** `ssh -N dart-rollouter` (leave running while evaluating).
- **Verify:** `curl -s http://127.0.0.1:8010/health` and `curl -s http://127.0.0.1:15961/status`.

See `.cursor/skills/dart-rollouter/SKILL.md` for ad-hoc `ssh -L` examples.

### 4. Desktop Emulator (Docker containers, managed by Desktop Server)

Each emulator is a Docker container running QEMU with an Ubuntu desktop inside. Created/destroyed by the Desktop Server API.

- **Image:** `happysixd/osworld-docker`
- **System disk:** `/mnt/data/VMs/Ubuntu.qcow2` mounted read-only as `/System.qcow2`
- **Ports per emulator** (dynamically allocated starting from):
  - `8006+` — noVNC web viewer (open in browser for desktop access)
  - `5000+` — OSWorld API server (screenshot, execute, accessibility tree)
  - `9222+` — Chromium DevTools
  - `8080+` — VLC

These containers are NOT launched directly — use the Desktop Server API.

## Disk Layout

| Path | Mount | Usage |
|------|-------|-------|
| `/` | `/dev/root` (8.6GB) | OS, Docker images, venvs |
| `/mnt/data` | `/dev/nvme0n2` (98GB) | MySQL data, VM images, large files |
| `/mnt/data/VMs/Ubuntu.qcow2` | — | Desktop VM disk image |
| `/mnt/data/mysql` | — | MySQL persistent data |

Root disk is small (~3GB free). Large files must go on `/mnt/data`.

## Full Restart Sequence

```bash
# 1. MySQL
sudo docker restart mysql-server

# 2. Monitor
cd GUI-Docker-Env/monitor && sudo docker compose restart

# 3. Desktop Server (kill existing, then relaunch)
pkill -f "desktop_env.docker_server.server"
cd GUI-Docker-Env && sg docker -c "source .venv/bin/activate && python -m desktop_env.docker_server.server" &

# 4. Verify
curl -s http://localhost:50003/ping        # Desktop Server
curl -s -o /dev/null -w "%{http_code}" http://localhost:80/  # Monitor
sudo docker exec mysql-server mysqladmin ping -u root -padmin --silent  # MySQL
```

## First-Time Setup

```bash
# 1. Init and sync git submodules (GUI-Docker-Env, dart_rollouter)
git submodule update --init --recursive

# 2. Docker group access (once, then re-login or use sg)
sudo usermod -aG docker $USER

# 3. Pull images
sudo docker pull mysql:8.0.44-debian
sudo docker pull happysixd/osworld-docker

# 4. Desktop Server venv
cd GUI-Docker-Env
python3 -m venv .venv
source .venv/bin/activate
uv pip install flask docker psutil omegaconf filelock requests pyyaml python-dotenv

# 5. Download VM image (if not present at /mnt/data/VMs/Ubuntu.qcow2)
mkdir -p /mnt/data/VMs && cd /mnt/data/VMs
wget https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip
unzip Ubuntu.qcow2.zip
```

## Sanity Check (single-task smoke test)

Use `validation/run.py` (Hydra-based runner with Ray) against
`validation/evaluation_examples/test_single.json` to verify the environment + agent
pipeline. The task file contains a single Chrome task.

**Venv:** `/mnt/dart-gui/.venv` — contains deps for the validation runner.

**Config:** Create or copy a Hydra config under `validation/config/` that points to the
local Desktop Server. Use `config_example_openai.yaml` as a starting template and
override `task.task_file`, `env.*`, and `runner.env.*` for the local environment:

```bash
source /mnt/dart-gui/.venv/bin/activate
cd /mnt/dart-gui/validation

# Run with a config override pointing to test_single.json and the local desktop server
python run.py --config-name config_example_openai \
  task.task_file=evaluation_examples/test_single.json \
  env.server_url=http://localhost:50003 \
  env.user_token=dart \
  runner.env.server_url=http://localhost:50003 \
  runner.env.user_token=dart \
  coordinator.max_concurrent_envs=1 \
  coordinator.rollout_n=1 \
  storage.root=results/sanity
```

Results are written to `validation/results/sanity/` (override with `storage.root`).

**Install / refresh deps:**

```bash
source /mnt/dart-gui/.venv/bin/activate
uv pip install tqdm gymnasium wrapt_timeout_decorator \
  docker flask psutil omegaconf requests pyyaml python-dotenv filelock \
  requests-toolbelt lxml cssselect xmltodict \
  openai tiktoken Pillow backoff \
  openpyxl python-docx python-pptx pypdf rapidfuzz \
  playwright pandas pyacoustid librosa fastdtw pytz \
  ray hydra-core
```

## GPU Components (not on this machine)

The rollouter and trainer containers require GPUs and run on separate GPU machines.
See `DOC.md` for their setup (`docker pull crpi-iwtwdoj3ikoon38c.cn-beijing.personal.cr.aliyuncs.com/pengxiangli1999/dart-gui:v0`).
