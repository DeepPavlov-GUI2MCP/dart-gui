---
name: connect-dart-rollouter
description: Start and verify an SSH port-forward tunnel from the local CPU machine to the dart-rollouter GPU VM in a dedicated tmux session so vLLM and the model service pool are reachable on localhost. Use when the user asks to connect to dart-rollouter, start the SSH tunnel, forward ports 8010/15961, check remote service aliveness, or run local eval against remote inference.
---

# Connect dart-rollouter

Use this skill to expose remote inference on the local machine. Run eval or UITARS against `http://127.0.0.1:8010` only after the tunnel is up and health checks pass.

## Tmux session (required)

Launch `ssh -N` in a **separate** tmux session named `dart-rollouter-tunnel`. Do not:

- run `ssh -N ... &` in the current shell
- reuse the eval / desktop-server / monitor tmux session for the tunnel

The tunnel session owns only the port-forward process. Keep eval and other work in other sessions.

```bash
.cursor/skills/connect-dart-rollouter/scripts/tunnel.sh start   # creates dart-rollouter-tunnel
tmux ls                                                        # confirm separate session
tmux attach -t dart-rollouter-tunnel                           # inspect; Ctrl+B then D to detach
```

## SSH host

`~/.ssh/config` should define `Host dart-rollouter` (hostname, user, port, identity file). Do not commit keys.

## Port map

Remote services may listen on different ports than local docs assume. **Discover first**, then forward both actual and alias ports.

| Local | Remote | Service |
|-------|--------|---------|
| `8000` | `8000` | vLLM (actual on current VM) |
| `8010` | `8000` | vLLM alias for `OPENAI_BASE_URL` |
| `15959` | `15959` | model service pool (actual) |
| `15961` | `15959` | model service alias |

If remote ports differ, set `REMOTE_VLLM_PORT` / `REMOTE_MODEL_PORT` when starting the tunnel script.

## Workflow

Copy this checklist:

```
- [ ] Discover remote ports and services
- [ ] Start tunnel in separate tmux session dart-rollouter-tunnel
- [ ] Verify local listeners
- [ ] Health-check model service + vLLM
- [ ] Optional: one inference request
```

### 1. Discover remote services

```bash
.cursor/skills/connect-dart-rollouter/scripts/tunnel.sh discover
```

Confirm vLLM and model service are listening before forwarding.

### 2. Start tunnel in a separate tmux session

Use the helper script (preferred) or create the dedicated session manually:

```bash
.cursor/skills/connect-dart-rollouter/scripts/tunnel.sh start
```

Manual equivalent — note the **new session** (`-s dart-rollouter-tunnel`), not a window in an existing session:

```bash
tmux new-session -d -s dart-rollouter-tunnel -n ssh \
  'ssh -N -o ExitOnForwardFailure=yes -o BatchMode=yes \
    -L 8000:127.0.0.1:8000 \
    -L 15959:127.0.0.1:15959 \
    -L 8010:127.0.0.1:8000 \
    -L 15961:127.0.0.1:15959 \
    dart-rollouter'
```

Inspect without stopping the tunnel: `tmux attach -t dart-rollouter-tunnel` (Ctrl+B, D to detach).

### 3. Verify tunnel status

```bash
.cursor/skills/connect-dart-rollouter/scripts/tunnel.sh status
```

Expect local listeners on `127.0.0.1` for the forwarded ports.

### 4. Health checks

```bash
.cursor/skills/connect-dart-rollouter/scripts/tunnel.sh check
```

Manual checks:

```bash
curl -s http://127.0.0.1:15961/status
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8010/health
```

Both must succeed before starting eval.

### 5. Sample inference request

Resolve the model id from `/v1/models` (full checkpoint path on the VM), then:

```bash
MODEL="$(curl -s http://127.0.0.1:8010/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")"
curl -s -X POST http://127.0.0.1:8010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word.\"}],\"max_tokens\":16,\"temperature\":0}"
```

Prefer direct vLLM `/v1/chat/completions` for smoke tests. `POST /generate` on the model pool may fail if the remote pool expects multimodal message content.

## Stop tunnel

```bash
.cursor/skills/connect-dart-rollouter/scripts/tunnel.sh stop
```

## Local eval config

After the tunnel is healthy, point UITARS / CoAct at:

- `OPENAI_BASE_URL=http://127.0.0.1:8010` in `GUI-Docker-Env/.env`

Before choosing an eval runner, inspect the live model id:

```bash
curl -s http://127.0.0.1:8010/v1/models | python3 -c "import sys,json; print([m['id'] for m in json.load(sys.stdin)['data']])"
```

Use that to choose the matching OSWorld runner:

- `uitars_run/run_uitars.py` for UI-TARS models
- `holo_run/run_holo.py` for Holo / Holotron / `Hcompany/*` models

See also: `.cursor/skills/launch-dart-rollouter/SKILL.md` (launch services on the GPU VM), `.cursor/skills/eval-osworld/scripts/preflight.sh` (full eval preflight).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Local port listens, curl gets empty reply | Remote service not running | `tunnel.sh discover`; start model service on VM |
| Connection refused locally | Tunnel not running | `tunnel.sh start` (separate tmux session) |
| Tunnel dies when terminal closes | Started outside tmux | Restart with `tunnel.sh start` |
| vLLM 404 on model name | Wrong model id | Use id from `GET /v1/models` |
| Docs say 8010/15961 but discover shows 8000/15959 | Port drift on VM | Use alias forwards in this skill |
