# DART-GUI systemd services

Systemd units for the local OSWorld desktop environment server that
`run.py` connects to on port `4999`.

## Units

### `xvfb.service`
Starts an Xvfb virtual framebuffer on display `:99` (1920x1080x24).
Optional when you want systemd to manage the X display separately.

### `dart-desktop-server.service`
Runs `validation/start_desktop.sh`, which starts the full local desktop
stack:

- Xvfb on display `:99` if no display is already active
- Xfce session
- `x11vnc` on port `5900`
- `websockify`/noVNC on port `6080`
- `local_desktop_server.py` on port `4999`

## Install & enable

```bash
sudo cp /mnt/dart-gui/services/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now dart-desktop-server
```

## Check status

```bash
systemctl status dart-desktop-server
curl -s http://localhost:4999/server/list
```

To verify the viewer path itself:

```bash
curl -I http://localhost:6080/
```

## Disable

```bash
sudo systemctl disable --now dart-desktop-server
```
